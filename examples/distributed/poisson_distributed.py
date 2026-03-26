"""
Distributed Poisson Equation Solver (3D)
=========================================

Solves -Δu = 1 on a unit cube with u = 0 on the boundary,
using multi-GPU parallel assembly and distributed solve.

Usage:
    python poisson_distributed.py                    # auto-detect GPUs
    python poisson_distributed.py --partitions 4     # specify partition count
    python poisson_distributed.py --cpu              # CPU-only mode
    python poisson_distributed.py --n 100 --partitions 4 --no-solve  # structured 100^3 mesh
"""

import argparse
import time
import gc
import numpy as np
import torch
import meshio
import tensormesh as tm
from tensormesh.mesh import Mesh
from tensormesh.distributed import (
    DistributedMesh,
    distributed_element_assemble_to_sparse,
    distributed_node_assemble,
)
from tensormesh.assemble import const_node_assembler


# ─── Fast structured mesh generation ───────────────────────────────

def gen_structured_cube(n: int) -> Mesh:
    """Generate a structured tetrahedral mesh on [0,1]^3.

    Splits the domain into n×n×n hexahedra, each subdivided into 6
    tetrahedra.  O(n^3) and extremely fast — no gmsh needed.

    Parameters
    ----------
    n : int
        Number of divisions per axis. Total points = (n+1)^3,
        total tetrahedra = 6*n^3.

    Returns
    -------
    Mesh
        TensorMesh Mesh with ``is_boundary`` point data.
    """
    # Grid points
    lin = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    n1 = n + 1

    def idx(i, j, k):
        return i * n1 * n1 + j * n1 + k

    # 6 tets per hex (standard decomposition)
    # Hex vertices: v0..v7
    #   v0=(i,j,k)     v1=(i+1,j,k)   v2=(i+1,j+1,k)   v3=(i,j+1,k)
    #   v4=(i,j,k+1)   v5=(i+1,j,k+1) v6=(i+1,j+1,k+1) v7=(i,j+1,k+1)
    ii, jj, kk = np.mgrid[0:n, 0:n, 0:n]
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    v0 = idx(ii, jj, kk)
    v1 = idx(ii+1, jj, kk)
    v2 = idx(ii+1, jj+1, kk)
    v3 = idx(ii, jj+1, kk)
    v4 = idx(ii, jj, kk+1)
    v5 = idx(ii+1, jj, kk+1)
    v6 = idx(ii+1, jj+1, kk+1)
    v7 = idx(ii, jj+1, kk+1)

    # 6-tet split (Freudenthal triangulation)
    tets = np.concatenate([
        np.stack([v0, v1, v3, v4], axis=1),
        np.stack([v1, v2, v3, v6], axis=1),
        np.stack([v1, v4, v5, v6], axis=1),
        np.stack([v3, v4, v6, v7], axis=1),
        np.stack([v1, v3, v4, v6], axis=1),
        np.stack([v4, v5, v1, v6], axis=1) if False else np.zeros((0, 4), dtype=np.int64),
    ], axis=0)

    # Actually use the 5-tet + 1 alternative: standard 5-tet decomposition
    # Simpler and correct: use the well-known 5-tet split
    tets = np.concatenate([
        np.stack([v0, v1, v3, v4], axis=1),
        np.stack([v1, v2, v3, v6], axis=1),
        np.stack([v1, v4, v5, v6], axis=1),
        np.stack([v3, v4, v6, v7], axis=1),
        np.stack([v1, v3, v4, v6], axis=1),
    ], axis=0)

    m_io = meshio.Mesh(points=points, cells=[("tetra", tets)])
    mesh = Mesh(m_io)

    # Boundary: any face on x=0, x=1, y=0, y=1, z=0, z=1
    pts = mesh.points
    is_boundary = (
        (pts[:, 0] == 0) | (pts[:, 0] == 1) |
        (pts[:, 1] == 0) | (pts[:, 1] == 1) |
        (pts[:, 2] == 0) | (pts[:, 2] == 1)
    )
    mesh.register_point_data("is_boundary", is_boundary)
    return mesh


# ─── GPU memory helpers ─────────────────────────────────────────────

def reset_gpu_memory():
    """Reset peak memory stats on all CUDA devices."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def get_peak_memory(devices=None):
    """Get peak GPU memory (MB) for given devices since last reset.

    Returns dict {device_index: peak_MB}.
    """
    if not torch.cuda.is_available():
        return {}
    if devices is None:
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    result = {}
    for d in devices:
        if d.type != 'cuda':
            continue
        idx = d.index if d.index is not None else 0
        peak = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
        result[idx] = peak
    return result


def fmt_memory(mem_dict):
    """Format memory dict as string."""
    if not mem_dict:
        return "  (CPU mode, no GPU memory)"
    parts = []
    for idx in sorted(mem_dict):
        mb = mem_dict[idx]
        if mb > 1024:
            parts.append(f"GPU {idx}: {mb / 1024:.2f} GB")
        else:
            parts.append(f"GPU {idx}: {mb:.1f} MB")
    total = sum(mem_dict.values())
    total_str = f"{total / 1024:.2f} GB" if total > 1024 else f"{total:.1f} MB"
    return ", ".join(parts) + f"  (total: {total_str})"


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distributed 3D Poisson solver")
    parser.add_argument("--n", type=int, default=None,
                        help="Structured mesh: divisions per axis (fast, no gmsh). "
                             "Points=(n+1)^3, tets=5*n^3. E.g. --n 100 → ~1M points")
    parser.add_argument("--chara-length", type=float, default=0.1,
                        help="Gmsh mesh characteristic length (ignored if --n is set)")
    parser.add_argument("--partitions", type=int, default=None,
                        help="Number of partitions (default: num GPUs or 2)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip single-device reference assembly")
    parser.add_argument("--no-solve", action="store_true",
                        help="Skip solve (only benchmark assembly)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the solution")
    args = parser.parse_args()

    # ─── Mesh generation ───────────────────────────────────────────
    t0 = time.perf_counter()
    if args.n is not None:
        print(f"Generating structured 3D mesh (n={args.n})...")
        mesh = gen_structured_cube(args.n)
    else:
        print(f"Generating 3D mesh (chara_length={args.chara_length})...")
        mesh = tm.Mesh.gen_cube(chara_length=args.chara_length)
    t_mesh = time.perf_counter() - t0
    n_tets = sum(mesh.cells[k].shape[0] for k in mesh.cells.keys())
    print(f"  Mesh: {mesh.n_points:,} points, {n_tets:,} tets, generation time: {t_mesh:.3f}s")

    ConstLoad = const_node_assembler()

    # ─── Single-device reference assembly ────────────────────────
    if not args.no_ref:
        print("\n[Single-device assembly]")
        mesh_gpu = mesh.clone().cuda()
        reset_gpu_memory()

        t0 = time.perf_counter()
        K_ref = tm.LaplaceElementAssembler.from_mesh(mesh_gpu)()
        f_ref = ConstLoad.from_mesh(mesh_gpu)()
        torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0

        mem_ref_asm = get_peak_memory([torch.device('cuda:0')])
        print(f"  Assembly time: {t_ref:.3f}s")
        print(f"  Peak memory:   {fmt_memory(mem_ref_asm)}")

        # Move results to CPU, free GPU
        K_ref = K_ref.cpu(); f_ref = f_ref.cpu()
        del mesh_gpu; gc.collect(); torch.cuda.empty_cache()

    # ─── Partition ─────────────────────────────────────────────────
    num_partitions = args.partitions
    if num_partitions is None:
        num_partitions = torch.cuda.device_count() if (torch.cuda.is_available() and not args.cpu) else 2

    if args.cpu:
        devices = [torch.device('cpu')] * num_partitions
    else:
        devices = None

    print(f"\nPartitioning into {num_partitions} parts...")
    t0 = time.perf_counter()
    dmesh = DistributedMesh(mesh, num_partitions=num_partitions, devices=devices)
    t_partition = time.perf_counter() - t0
    print(f"  Partition time: {t_partition:.3f}s")
    print(f"  {dmesh}")

    # ─── Distributed assembly ──────────────────────────────────────
    print("\n[Multi-device assembly]")
    reset_gpu_memory()

    t0 = time.perf_counter()
    K = distributed_element_assemble_to_sparse(
        tm.LaplaceElementAssembler, dmesh, quadrature_order=2
    )
    f = distributed_node_assemble(ConstLoad, dmesh, quadrature_order=2)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_dist = time.perf_counter() - t0

    mem_dist_asm = get_peak_memory(dmesh.devices)
    print(f"  Assembly time: {t_dist:.3f}s  (K: {K.shape})")
    print(f"  Peak memory:   {fmt_memory(mem_dist_asm)}")

    if not args.no_ref:
        print(f"  Speedup:       {t_ref / t_dist:.2f}x")
        if mem_ref_asm and mem_dist_asm:
            ref_total = sum(mem_ref_asm.values())
            dist_max = max(mem_dist_asm.values()) if mem_dist_asm else 0
            print(f"  Memory/GPU:    {dist_max:.1f} MB vs {ref_total:.1f} MB single "
                  f"({ref_total / max(dist_max, 1):.1f}x reduction per GPU)")

    if args.no_solve:
        print("\nSkipping solve (--no-solve).")
        print("Done!")
        return

    # ─── Solve: single-device reference ────────────────────────────
    if not args.no_ref:
        print("\n[Single-device solve]")
        condenser_ref = tm.Condenser(mesh.boundary_mask)
        K_c_ref, f_c_ref = condenser_ref(K_ref, f_ref)

        reset_gpu_memory()
        t0 = time.perf_counter()
        u_ref = condenser_ref.recover(K_c_ref.solve(f_c_ref))
        t_solve_ref = time.perf_counter() - t0

        mem_ref_solve = get_peak_memory([torch.device('cuda:0')])
        print(f"  Solve time:  {t_solve_ref:.3f}s")
        print(f"  Peak memory: {fmt_memory(mem_ref_solve)}")
        print(f"  u_ref max = {u_ref.max().item():.6f}")

    # ─── Solve: distributed assembled matrix ───────────────────────
    print("\n[Distributed-assembled solve]")
    condenser = tm.Condenser(mesh.boundary_mask)
    K_c, f_c = condenser(K, f)

    reset_gpu_memory()
    t0 = time.perf_counter()
    u = condenser.recover(K_c.solve(f_c))
    t_solve = time.perf_counter() - t0

    mem_dist_solve = get_peak_memory(dmesh.devices)
    print(f"  Solve time:  {t_solve:.3f}s")
    print(f"  Peak memory: {fmt_memory(mem_dist_solve)}")
    print(f"  u max = {u.max().item():.6f}")

    if not args.no_ref:
        error = (u - u_ref).abs().max().item()
        print(f"  Max |u_dist - u_ref| = {error:.2e}")

    # ─── Plot ──────────────────────────────────────────────────────
    if args.plot:
        mesh.point_data['u'] = u
        mesh.plot('u', show=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
