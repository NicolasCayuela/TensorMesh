"""
Assembly Benchmark: Single-GPU vs Multi-GPU
=============================================

Scales up mesh size until OOM, recording assembly time and peak GPU memory.

Usage:
    python benchmark_assembly.py                     # auto-detect GPUs
    python benchmark_assembly.py --partitions 4      # specify partition count
    python benchmark_assembly.py --start 50 --step 50 --max 500
"""

import argparse
import time
import gc
import numpy as np
import torch
import meshio
import tensormesh as tm
from tensormesh.mesh import Mesh
from tensormesh.distributed import DistributedMesh, distributed_element_assemble_to_sparse


# ─── Structured mesh generation ─────────────────────────────────────

def gen_structured_cube(n: int) -> Mesh:
    """Structured tet mesh on [0,1]^3.  Points=(n+1)^3, tets=5*n^3."""
    lin = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    n1 = n + 1
    def idx(i, j, k):
        return i * n1 * n1 + j * n1 + k

    ii, jj, kk = np.mgrid[0:n, 0:n, 0:n]
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
    v0 = idx(ii, jj, kk);     v1 = idx(ii+1, jj, kk)
    v2 = idx(ii+1, jj+1, kk); v3 = idx(ii, jj+1, kk)
    v4 = idx(ii, jj, kk+1);   v5 = idx(ii+1, jj, kk+1)
    v6 = idx(ii+1, jj+1, kk+1); v7 = idx(ii, jj+1, kk+1)

    tets = np.concatenate([
        np.stack([v0, v1, v3, v4], axis=1),
        np.stack([v1, v2, v3, v6], axis=1),
        np.stack([v1, v4, v5, v6], axis=1),
        np.stack([v3, v4, v6, v7], axis=1),
        np.stack([v1, v3, v4, v6], axis=1),
    ], axis=0)

    m_io = meshio.Mesh(points=points, cells=[("tetra", tets)])
    mesh = Mesh(m_io)
    pts = mesh.points
    is_boundary = (
        (pts[:, 0] == 0) | (pts[:, 0] == 1) |
        (pts[:, 1] == 0) | (pts[:, 1] == 1) |
        (pts[:, 2] == 0) | (pts[:, 2] == 1)
    )
    mesh.register_point_data("is_boundary", is_boundary)
    return mesh


# ─── GPU helpers ────────────────────────────────────────────────────

def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_mb(device_idx=0):
    return torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)


def fmt_mb(mb):
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


# ─── Benchmark routines ────────────────────────────────────────────

def bench_single_gpu(mesh):
    """Single-GPU assembly.  Returns (time_s, peak_mb) or None on OOM."""
    reset_gpu()
    try:
        mesh_gpu = mesh.clone().cuda()
        torch.cuda.synchronize()
        reset_gpu()

        t0 = time.perf_counter()
        K = tm.LaplaceElementAssembler.from_mesh(mesh_gpu)()
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        mem = peak_mb(0)

        del K, mesh_gpu
        reset_gpu()
        return t, mem
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
            reset_gpu()
            return None  # OOM
        raise


def bench_multi_gpu(mesh, num_partitions, devices):
    """Multi-GPU assembly.  Returns (time_s, per_gpu_peak_mb_list) or None on OOM."""
    reset_gpu()
    try:
        dmesh = DistributedMesh(mesh, num_partitions=num_partitions, devices=devices)
        torch.cuda.synchronize()
        reset_gpu()

        t0 = time.perf_counter()
        K = distributed_element_assemble_to_sparse(
            tm.LaplaceElementAssembler, dmesh, quadrature_order=2
        )
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        mems = [peak_mb(d.index if d.index is not None else 0) for d in devices if d.type == 'cuda']

        del K, dmesh
        reset_gpu()
        return t, mems
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
            reset_gpu()
            return None  # OOM
        raise


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Assembly benchmark: 1-GPU vs N-GPU")
    parser.add_argument("--partitions", type=int, default=None,
                        help="Number of GPU partitions (default: all GPUs)")
    parser.add_argument("--start", type=int, default=30,
                        help="Starting n (divisions per axis)")
    parser.add_argument("--step", type=int, default=20,
                        help="Step size for n")
    parser.add_argument("--max", type=int, default=1000,
                        help="Max n to try")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    num_partitions = args.partitions or num_gpus
    devices = [torch.device(f'cuda:{i}') for i in range(num_partitions)]

    print(f"GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"Partitions: {num_partitions}")
    print()

    # Table header
    header = (
        f"{'n':>5} | {'Points':>12} | {'Tets':>12} | "
        f"{'1-GPU time':>10} {'1-GPU mem':>10} | "
        f"{'N-GPU time':>10} {'N-GPU mem/card':>14} {'N-GPU total':>12} | "
        f"{'Speedup':>7} {'Mem save':>9}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    single_oom = False
    multi_oom = False

    n = args.start
    while n <= args.max:
        # Generate mesh
        mesh = gen_structured_cube(n)
        n_pts = mesh.n_points
        n_tets = sum(mesh.cells[k].shape[0] for k in mesh.cells.keys())

        # Single GPU
        if not single_oom:
            res_single = bench_single_gpu(mesh)
            if res_single is None:
                single_oom = True
                s_time_str = "OOM"
                s_mem_str = "OOM"
            else:
                s_time, s_mem = res_single
                s_time_str = f"{s_time:.2f}s"
                s_mem_str = fmt_mb(s_mem)
        else:
            s_time_str = "OOM"
            s_mem_str = "OOM"
            res_single = None

        # Multi GPU
        if not multi_oom:
            res_multi = bench_multi_gpu(mesh, num_partitions, devices)
            if res_multi is None:
                multi_oom = True
                m_time_str = "OOM"
                m_mem_str = "OOM"
                m_total_str = "OOM"
            else:
                m_time, m_mems = res_multi
                m_time_str = f"{m_time:.2f}s"
                m_max = max(m_mems)
                m_total = sum(m_mems)
                m_mem_str = fmt_mb(m_max)
                m_total_str = fmt_mb(m_total)
        else:
            m_time_str = "OOM"
            m_mem_str = "OOM"
            m_total_str = "OOM"
            res_multi = None

        # Speedup & memory saving
        if res_single and res_multi:
            speedup = f"{s_time / m_time:.2f}x"
            mem_save = f"{s_mem / max(m_mems):.1f}x"
        elif res_single is None and res_multi:
            speedup = "inf"
            mem_save = "inf"
        else:
            speedup = "-"
            mem_save = "-"

        print(
            f"{n:>5} | {n_pts:>12,} | {n_tets:>12,} | "
            f"{s_time_str:>10} {s_mem_str:>10} | "
            f"{m_time_str:>10} {m_mem_str:>14} {m_total_str:>12} | "
            f"{speedup:>7} {mem_save:>9}",
            flush=True
        )

        del mesh
        reset_gpu()

        if multi_oom:
            print("\nMulti-GPU hit OOM. Stopping.")
            break

        n += args.step

    print(sep)
    print("Done!")


if __name__ == "__main__":
    main()
