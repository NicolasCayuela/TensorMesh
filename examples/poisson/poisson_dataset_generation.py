"""
Benchmark PoissonMultiFrequency.source_term & solution.

What it does
------------
- Sweeps over a grid of (n_points, batch_size, K).
- **dof is defined as n_points** (number of spatial points) for this benchmark.
- Measures runtime and memory on CPU and (optionally) GPU.
- Caches results to disk so you can plot later without recomputing.

Usage
-----
Run benchmark (CPU + GPU if available):
  python poisson_dataest_generation.py bench

Plot from cache (no compute):
  python poisson_dataest_generation.py plot
"""

import os
import sys
import time
import gc
import argparse
import csv
import json
import shutil
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append("../..")

import torch
import numpy as np
from tqdm import tqdm

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib

# IMPORTANT:
# Importing the `tensormesh` package can pull in optional visualization deps (e.g. pyvista/vtk).
# For benchmarking PoissonMultiFrequency we only need the implementation in
# `tensormesh/dataset/equation/poisson.py`. We load it directly from the source file to avoid
# forcing optional deps in minimal benchmark environments.
import importlib.util


def _load_poisson_multifrequency_cls():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "../.."))
    mod_path = os.path.join(root, "tensormesh", "dataset", "equation", "poisson.py")
    spec = importlib.util.spec_from_file_location("tensormesh_dataset_equation_poisson", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "PoissonMultiFrequency")


PoissonMultiFrequency = _load_poisson_multifrequency_cls()


def _apply_icml_style():
    """Apply a clean, ICML-like matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


@dataclass
class BenchResult:
    device: str
    n_points: int
    batch_size: int
    K: int  # frequency dimension (coefficients are [K, K])
    dof: int  # dof := n_points
    repeats: int

    # timings (seconds)
    t_source_mean: float
    t_source_std: float
    t_solution_mean: float
    t_solution_std: float
    t_total_mean: float
    t_total_std: float

    # memory
    cpu_rss_after_mean_bytes: Optional[float] = None
    cpu_rss_after_std_bytes: Optional[float] = None
    cpu_rss_peak_mean_bytes: Optional[float] = None
    cpu_rss_peak_std_bytes: Optional[float] = None
    cuda_peak_mean_bytes: Optional[float] = None
    cuda_peak_std_bytes: Optional[float] = None


def _now() -> float:
    return time.perf_counter()


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


def _bench_one_cpu_peak_rss(
    *,
    n_points: int,
    batch_size: int,
    K: int,
    repeats: int,
    dtype: torch.dtype,
    dim: int = 2,
) -> BenchResult:
    """CPU benchmark with peak RSS measurement via a sampling thread (run in a fresh subprocess)."""
    if psutil is None:
        raise RuntimeError("psutil is required for subprocess peak RSS measurement.")

    # Deterministic points on CPU
    rng = torch.Generator(device="cpu").manual_seed(0)
    points = torch.rand((int(n_points), dim), dtype=dtype, generator=rng, device="cpu")

    a = torch.rand((int(batch_size), int(K), int(K)), device="cpu", dtype=dtype) * 2 - 1
    eq = PoissonMultiFrequency(a=a)

    # Warmup
    domain = "rectangle" if dim == 2 else "cube"
    _ = eq.source_term(points, domain=domain)
    _ = eq.solution(points)

    proc = psutil.Process(os.getpid())

    t_source: List[float] = []
    t_solution: List[float] = []
    rss_after: List[float] = []
    rss_peak: List[float] = []

    for _ in range(int(repeats)):
        gc.collect()

        peak_box = {"peak": float(proc.memory_info().rss)}
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    peak_box["peak"] = max(peak_box["peak"], float(proc.memory_info().rss))
                except Exception:
                    pass
                time.sleep(0.005)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()

        t0 = _now()
        f = eq.source_term(points, domain=domain)
        t1 = _now()
        u = eq.solution(points)
        t2 = _now()

        _ = float(f.mean().detach().cpu())
        _ = float(u.mean().detach().cpu())

        stop.set()
        th.join(timeout=1.0)

        t_source.append(t1 - t0)
        t_solution.append(t2 - t1)
        rss_after.append(float(proc.memory_info().rss))
        rss_peak.append(float(peak_box["peak"]))

    t_source_mean = float(np.mean(t_source))
    t_source_std = float(np.std(t_source, ddof=1)) if len(t_source) >= 2 else 0.0
    t_solution_mean = float(np.mean(t_solution))
    t_solution_std = float(np.std(t_solution, ddof=1)) if len(t_solution) >= 2 else 0.0
    t_total = [a + b for a, b in zip(t_source, t_solution)]
    t_total_mean = float(np.mean(t_total))
    t_total_std = float(np.std(t_total, ddof=1)) if len(t_total) >= 2 else 0.0

    after_mean = float(np.mean(rss_after)) if rss_after else None
    after_std = float(np.std(rss_after, ddof=1)) if len(rss_after) >= 2 else 0.0 if rss_after else None
    peak_mean = float(np.mean(rss_peak)) if rss_peak else None
    peak_std = float(np.std(rss_peak, ddof=1)) if len(rss_peak) >= 2 else 0.0 if rss_peak else None

    return BenchResult(
        device="cpu",
        n_points=int(n_points),
        batch_size=int(batch_size),
        K=int(K),
        dof=int(n_points),
        repeats=int(repeats),
        t_source_mean=t_source_mean,
        t_source_std=t_source_std,
        t_solution_mean=t_solution_mean,
        t_solution_std=t_solution_std,
        t_total_mean=t_total_mean,
        t_total_std=t_total_std,
        cpu_rss_after_mean_bytes=after_mean,
        cpu_rss_after_std_bytes=after_std,
        cpu_rss_peak_mean_bytes=peak_mean,
        cpu_rss_peak_std_bytes=peak_std,
    )


def _cpu_peak_rss_subprocess(*, n_points: int, batch_size: int, K: int, repeats: int, dtype: torch.dtype, dim: int = 2) -> Tuple[float, float]:
    """Measure CPU peak RSS in a subprocess; returns (mean_bytes, std_bytes)."""
    here = os.path.abspath(__file__)
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = env.get("PYTHONNOUSERSITE", "1")
    cmd = [
        sys.executable,
        here,
        "bench_child",
        "--n_points",
        str(int(n_points)),
        "--batch_size",
        str(int(batch_size)),
        "--K",
        str(int(K)),
        "--repeats",
        str(int(repeats)),
        "--dtype",
        "float32" if dtype == torch.float32 else "float64",
        "--dim",
        str(int(dim)),
    ]
    out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True)
    payload = json.loads(out.strip().splitlines()[-1])
    mean_b = float(payload.get("cpu_rss_peak_mean_bytes"))
    std_b = float(payload.get("cpu_rss_peak_std_bytes") or 0.0)
    return mean_b, std_b


def _bench_one(
    *,
    points: torch.Tensor,
    batch_size: int,
    K: int,
    repeats: int,
    device: torch.device,
) -> BenchResult:
    # Generate coefficients a (dof = K^2) and build equation
    a = torch.rand((batch_size, K, K), device=device, dtype=points.dtype) * 2 - 1
    eq = PoissonMultiFrequency(a=a)

    # Warmup (especially for CUDA) to avoid first-call overhead dominating
    _sync(device)
    _ = eq.source_term(points, domain="rectangle" if points.shape[-1] == 2 else "cube")
    _ = eq.solution(points)
    _sync(device)

    t_source: List[float] = []
    t_solution: List[float] = []
    mem_samples: List[float] = []
    for _ in range(repeats):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        _sync(device)
        t0 = _now()
        f = eq.source_term(points, domain="rectangle" if points.shape[-1] == 2 else "cube")
        _sync(device)
        t1 = _now()
        u = eq.solution(points)
        _sync(device)
        t2 = _now()

        # Touch results so nothing gets optimized away
        _ = float(f.mean().detach().cpu())
        _ = float(u.mean().detach().cpu())

        t_source.append(t1 - t0)
        t_solution.append(t2 - t1)

        # memory sample after the run
        if device.type == "cuda":
            mem_samples.append(float(torch.cuda.max_memory_allocated(device)))
        else:
            rss = _cpu_rss_bytes()
            if rss is not None:
                mem_samples.append(float(rss))

    t_source_mean = float(np.mean(t_source))
    t_source_std = float(np.std(t_source, ddof=1)) if len(t_source) >= 2 else 0.0
    t_solution_mean = float(np.mean(t_solution))
    t_solution_std = float(np.std(t_solution, ddof=1)) if len(t_solution) >= 2 else 0.0
    t_total = [a + b for a, b in zip(t_source, t_solution)]
    t_total_mean = float(np.mean(t_total))
    t_total_std = float(np.std(t_total, ddof=1)) if len(t_total) >= 2 else 0.0

    cpu_mem_mean = None
    cpu_mem_std = None
    cuda_mem_mean = None
    cuda_mem_std = None
    if mem_samples:
        if device.type == "cuda":
            cuda_mem_mean = float(np.mean(mem_samples))
            cuda_mem_std = float(np.std(mem_samples, ddof=1)) if len(mem_samples) >= 2 else 0.0
        else:
            cpu_mem_mean = float(np.mean(mem_samples))
            cpu_mem_std = float(np.std(mem_samples, ddof=1)) if len(mem_samples) >= 2 else 0.0

    return BenchResult(
        device=str(device),
        n_points=int(points.shape[0]),
        batch_size=int(batch_size),
        K=int(K),
        dof=int(points.shape[0]),
        repeats=int(repeats),
        t_source_mean=t_source_mean,
        t_source_std=t_source_std,
        t_solution_mean=t_solution_mean,
        t_solution_std=t_solution_std,
        t_total_mean=t_total_mean,
        t_total_std=t_total_std,
        cpu_rss_after_mean_bytes=cpu_mem_mean,
        cpu_rss_after_std_bytes=cpu_mem_std,
        cuda_peak_mean_bytes=cuda_mem_mean,
        cuda_peak_std_bytes=cuda_mem_std,
    )


def _default_cache_path(out_dir: str) -> str:
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    # v3 schema: repeats stats + dof := n_points
    return os.path.join(cache_dir, "poisson_multifreq_bench.csv")


CSV_FIELDS: Tuple[str, ...] = (
    "key",
    "device",
    "n_points",
    "batch_size",
    "K",
    "dof",
    "repeats",
    "t_source_mean",
    "t_source_std",
    "t_solution_mean",
    "t_solution_std",
    "t_total_mean",
    "t_total_std",
    "cpu_rss_after_mean_bytes",
    "cpu_rss_after_std_bytes",
    "cpu_rss_peak_mean_bytes",
    "cpu_rss_peak_std_bytes",
    "cuda_peak_mean_bytes",
    "cuda_peak_std_bytes",
)


def _read_existing_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row.get("key")
            if k:
                keys.add(k)
    return keys


def _ensure_csv_schema(csv_path: str, desired_fields: Sequence[str]):
    """Ensure the CSV has the given header fields; if not, rewrite in-place with a backup."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)
    if not existing:
        return

    missing = [c for c in desired_fields if c not in existing]
    if not missing:
        return

    # Rewrite with upgraded schema.
    backup = csv_path + ".bak_schema"
    if not os.path.exists(backup):
        shutil.copy2(csv_path, backup)

    new_fields = list(existing) + [c for c in desired_fields if c not in existing]
    tmp = csv_path + ".tmp_schema"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in new_fields})
    os.replace(tmp, csv_path)


def _append_csv(csv_path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    _ensure_csv_schema(csv_path, CSV_FIELDS)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def _is_pow4(x: int) -> bool:
    """Return True if x is 1,4,16,... (i.e., x = 4^k)."""
    if x <= 0:
        return False
    while x % 4 == 0:
        x //= 4
    return x == 1


def prune_cache_csv(
    *,
    cache_path: str,
    backup_suffix: str = ".bak",
    keep_cuda_pow4: bool = True,
    cuda_top_n_points: int = 5,
) -> None:
    """
    Prune a cached CSV in-place (with backup) to keep only the latest-looking CUDA sweep.

    Heuristic used (no timestamps available):
    - For CUDA rows: keep only batch_size on the 4^k grid (1,4,16,...) if keep_cuda_pow4=True.
    - Then keep only the top-N largest n_points among remaining CUDA rows (default N=5).
    - Non-CUDA rows are kept unchanged.
    - If duplicate keys exist, keep the last occurrence.
    """
    if not os.path.exists(cache_path):
        raise RuntimeError(f"CSV cache not found: {cache_path}")

    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows or not fieldnames:
        return

    def to_i(v: Any) -> int:
        try:
            return int(float(v))
        except Exception:
            return 0

    cuda_rows = [r for r in rows if "cuda" in str(r.get("device", ""))]
    if keep_cuda_pow4:
        cuda_rows = [r for r in cuda_rows if _is_pow4(to_i(r.get("batch_size", 0)))]

    cuda_n_points = sorted({to_i(r.get("n_points", 0)) for r in cuda_rows if to_i(r.get("n_points", 0)) > 0})
    keep_cuda_n = set(cuda_n_points[-cuda_top_n_points:]) if cuda_top_n_points > 0 else set(cuda_n_points)

    pruned: List[Dict[str, Any]] = []
    for r in rows:
        dev = str(r.get("device", ""))
        if "cuda" in dev:
            b = to_i(r.get("batch_size", 0))
            npt = to_i(r.get("n_points", 0))
            if keep_cuda_pow4 and not _is_pow4(b):
                continue
            if keep_cuda_n and npt not in keep_cuda_n:
                continue
        pruned.append(r)

    # De-duplicate by key, keeping last occurrence
    last_idx: Dict[str, int] = {}
    last_row: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(pruned):
        k = str(r.get("key", ""))
        if not k:
            # Keep rows without key as-is
            k = f"__nokey__{i}"
        last_idx[k] = i
        last_row[k] = r

    # Emit rows ordered by their last occurrence
    ordered_keys = sorted(last_idx.keys(), key=lambda kk: last_idx[kk])
    out_rows = [last_row[k] for k in ordered_keys]

    backup = cache_path + backup_suffix
    if not os.path.exists(backup):
        shutil.copy2(cache_path, backup)

    tmp = cache_path + ".tmp_prune"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, cache_path)


def _logspace_int(min_v: float, max_v: float, num: int) -> List[int]:
    """Log-spaced integer list, unique and sorted."""
    if num <= 1:
        return [int(max(1, round(min_v)))]
    xs = np.logspace(np.log10(min_v), np.log10(max_v), num=num)
    out = sorted({int(max(1, round(x))) for x in xs})
    return out


def _key(device: str, n_points: int, batch_size: int, K: int) -> str:
    return f"{device}|n{n_points}|b{batch_size}|K{K}"


def run_bench(
    *,
    devices: Sequence[str],
    n_points_list: Sequence[int],
    batch_sizes: Sequence[int],
    Ks: Sequence[int],
    repeats: int,
    cache_path: str,
    max_x: int,
    max_output_elems: int,
    dtype: torch.dtype,
    cuda_device: int,
    max_output_elems_cuda: int,
    cpu_mem_method: str,
    dim: int = 2,
):
    # Generate a deterministic base set of points on CPU in [0,1]^dim.
    # This avoids gmsh overhead and makes dof scaling (n_points) explicit.
    max_n = int(max(n_points_list))
    rng = torch.Generator(device="cpu").manual_seed(0)
    base_pts = torch.rand((max_n, dim), dtype=dtype, generator=rng)

    existing_keys = _read_existing_keys(cache_path)

    # Expand devices list
    dev_list: List[torch.device] = []
    for d in devices:
        if d == "cpu":
            dev_list.append(torch.device("cpu"))
        elif d in ("cuda", "gpu"):
            if torch.cuda.is_available():
                dev_list.append(torch.device(f"cuda:{cuda_device}"))
            else:
                print("[bench] CUDA requested but torch.cuda.is_available() == False; skipping CUDA.")
        elif d == "both":
            dev_list.append(torch.device("cpu"))
            if torch.cuda.is_available():
                dev_list.append(torch.device(f"cuda:{cuda_device}"))
            else:
                print("[bench] CUDA not available; running CPU-only.")
        else:
            raise ValueError(f"Unknown device spec: {d}")

    # Deduplicate while preserving order
    seen = set()
    uniq_devs: List[torch.device] = []
    for dev in dev_list:
        if str(dev) not in seen:
            uniq_devs.append(dev)
            seen.add(str(dev))

    # Sweep
    sweep: List[Tuple[torch.device, int, int, int]] = []
    for dev in uniq_devs:
        for npt in n_points_list:
            for b in batch_sizes:
                for K in Ks:
                    if int(npt) * int(b) <= max_x:
                        sweep.append((dev, int(npt), int(b), int(K)))

    pbar = tqdm(sweep, desc="bench", total=len(sweep))
    for dev, npt, b, K in pbar:
        pbar.set_postfix({"device": str(dev), "n": npt, "b": b, "K": K})
        pts = base_pts[:npt]
        pts_dev = pts.to(dev)

        # Safety: output tensors are [batch_size, n_points] for both source_term and solution.
        # Skip combinations that would allocate too much memory.
        out_elems = int(npt) * int(b)
        if dev.type == "cuda":
            if out_elems > max_output_elems_cuda:
                continue
        else:
            if out_elems > max_output_elems:
                continue

        k = _key(str(dev), int(pts_dev.shape[0]), b, K)
        if k in existing_keys:
            continue

        try:
            # Time is always measured in-process (方案2). Peak RSS can optionally be measured in a subprocess.
            res = _bench_one(points=pts_dev, batch_size=b, K=K, repeats=repeats, device=dev)
            if dev.type == "cpu" and cpu_mem_method == "subprocess":
                peak_mean, peak_std = _cpu_peak_rss_subprocess(
                    n_points=int(pts_dev.shape[0]),
                    batch_size=b,
                    K=K,
                    repeats=repeats,
                    dtype=dtype,
                    dim=dim,
                )
                res.cpu_rss_peak_mean_bytes = peak_mean
                res.cpu_rss_peak_std_bytes = peak_std
        except torch.OutOfMemoryError:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rec = asdict(res)
        rec["key"] = k
        _append_csv(cache_path, rec)
        existing_keys.add(k)


def plot_cache(cache_path: str, out_dir: str, no_title: bool = False, save_pdf: bool = False):
    _apply_icml_style()
    if not os.path.exists(cache_path):
        raise RuntimeError(f"CSV cache not found: {cache_path}")

    rows: List[Dict[str, Any]] = []
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No cached rows found at {cache_path}")

    os.makedirs(out_dir, exist_ok=True)

    def to_i(v: Any) -> int:
        return int(float(v))

    def to_f(v: Any) -> float:
        return float(v)

    devices = sorted({r["device"] for r in rows})
    n_points_vals = sorted({to_i(r["n_points"]) for r in rows})
    Ks = sorted({to_i(r["K"]) for r in rows})

    # Color encodes n_points via Spectral
    nmin, nmax = min(n_points_vals), max(n_points_vals)
    norm = mcolors.LogNorm(vmin=max(1, nmin), vmax=max(1, nmax))
    # Use reversed spectral as requested (low->high mapped opposite of default)
    # Avoid deprecated cm.get_cmap on newer matplotlib.
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    def color_for_n(n: int):
        return cmap(norm(max(1, n)))

    def is_cuda(dev: str) -> bool:
        return "cuda" in dev

    def marker_for_device(dev: str) -> str:
        return "s" if is_cuda(dev) else "o"

    # Select representative n_points values (5 values evenly spaced in log scale)
    def select_representative(vals: List[int], n: int = 5) -> List[int]:
        if len(vals) <= n:
            return vals
        indices = np.linspace(0, len(vals) - 1, n).astype(int)
        return [vals[i] for i in indices]

    n_points_repr = select_representative(n_points_vals, 8)

    # Device styles: marker to differentiate CPU vs CUDA (all solid lines)
    device_styles = {
        "cpu": {"linestyle": "-", "marker": "o", "label": "CPU"},
        "cuda": {"linestyle": "-", "marker": "s", "label": "CUDA"},
    }

    def get_device_style(dev: str):
        if "cuda" in dev:
            return device_styles["cuda"]
        return device_styles["cpu"]

    def draw_slope_triangle(ax, slope: float, x_pos: float, y_pos: float, width_decades: float = 0.5, color: str = "black", label: str = None):
        """
        Draw a slope triangle annotation in log-log space.
        slope: the slope value (dy/dx in log-log)
        x_pos, y_pos: position of the bottom-left corner in data coordinates
        width_decades: width of the triangle in decades (log10 units)
        """
        # In log-log space: log(y) = slope * log(x) + const
        # Triangle: horizontal run = width_decades, vertical rise = slope * width_decades
        x0 = x_pos
        x1 = x_pos * (10 ** width_decades)
        y0 = y_pos
        y1 = y_pos * (10 ** (slope * width_decades))

        # Draw the triangle (right angle at bottom-right)
        ax.plot([x0, x1], [y0, y0], color=color, linewidth=1.5, linestyle="-")  # horizontal
        ax.plot([x1, x1], [y0, y1], color=color, linewidth=1.5, linestyle="-")  # vertical
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5, linestyle="-")  # hypotenuse

        # Add slope label
        if label is None:
            label = f"slope={slope:.2f}"
        # Position label above the triangle
        ax.text(x0 * (10 ** (width_decades * 0.5)), y1 * 1.3, label, fontsize=9, ha="center", va="bottom", color=color)

    def compute_slope(xs: np.ndarray, ys: np.ndarray) -> float:
        """Compute slope in log-log space using linear regression."""
        if len(xs) < 2:
            return 1.0
        log_x = np.log10(xs)
        log_y = np.log10(ys)
        # Simple linear regression
        slope = np.polyfit(log_x, log_y, 1)[0]
        return slope

    # If multiple Ks exist, write plots per K.
    for K in Ks:
        has_cpu = any(not is_cuda(d) for d in devices)
        has_cuda = any(is_cuda(d) for d in devices)

        # TIME: Single plot with style/marker to differentiate devices
        fig, ax = plt.subplots(figsize=(8, 5.5))

        # Minimum time threshold: skip points with mean < 5ms (too noisy)
        min_time_threshold = 5e-3
        # For CUDA, only show x >= 5*10^6 to avoid noisy small-scale region
        cuda_min_x = 5e6

        # Collect all points for slope calculation
        all_xs_time: Dict[str, List[float]] = {"cpu": [], "cuda": []}
        all_ys_time: Dict[str, List[float]] = {"cpu": [], "cuda": []}

        for dev in devices:
            dev_style = get_device_style(dev)
            min_x = cuda_min_x if is_cuda(dev) else 0
            dev_key = "cuda" if is_cuda(dev) else "cpu"

            for npt in n_points_repr:
                xs = []
                ys = []
                yerr = []
                for r in rows:
                    if r["device"] != dev:
                        continue
                    if to_i(r["K"]) != K:
                        continue
                    if to_i(r["n_points"]) != npt:
                        continue
                    b = to_i(r["batch_size"])
                    x = float(b * npt)
                    y = to_f(r["t_total_mean"])
                    e = to_f(r.get("t_total_std") or 0.0)
                    # Skip very short timings and below min_x threshold
                    if x >= max(1, min_x) and y >= min_time_threshold:
                        xs.append(x)
                        ys.append(y)
                        yerr.append(e)
                if len(xs) < 2:
                    continue
                order = np.argsort(np.array(xs))
                xs_arr = np.array(xs)[order]
                ys_arr = np.array(ys)[order]
                yerr_arr = np.array(yerr)[order]

                # Collect for slope calculation
                all_xs_time[dev_key].extend(xs_arr.tolist())
                all_ys_time[dev_key].extend(ys_arr.tolist())

                c = color_for_n(npt)
                ax.errorbar(
                    xs_arr,
                    ys_arr,
                    yerr=yerr_arr,
                    linestyle=dev_style["linestyle"],
                    color=c,
                    linewidth=2.0,
                    alpha=0.85,
                    marker=dev_style["marker"],
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    capsize=2,
                    capthick=1,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("batch_size × n_points")
        ax.set_ylabel("time (s)")
        ax.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("n_points (dof)", fontsize=11)

        # Add device legend
        dev_handles = []
        dev_labels = []
        if has_cpu:
            dev_handles.append(Line2D([0], [0], color="gray", linestyle="-", marker="o", linewidth=2, markersize=6))
            dev_labels.append("CPU")
        if has_cuda:
            dev_handles.append(Line2D([0], [0], color="gray", linestyle="-", marker="s", linewidth=2, markersize=6))
            dev_labels.append("CUDA")
        ax.legend(dev_handles, dev_labels, title="Device", loc="upper left", frameon=True, fancybox=True, framealpha=0.9)

        # Draw slope triangles
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # CPU slope - lower-center area (avoid overlapping with lines)
        if len(all_xs_time["cpu"]) >= 3:
            slope_cpu = compute_slope(np.array(all_xs_time["cpu"]), np.array(all_ys_time["cpu"]))
            tri_x_cpu = xlim[0] * (10 ** 1.5)  # moved right
            tri_y_cpu = ylim[0] * (10 ** 0.3)
            draw_slope_triangle(ax, slope_cpu, tri_x_cpu, tri_y_cpu, width_decades=0.5, color="#2E86AB", label=f"CPU ≈{slope_cpu:.2f}")

        # CUDA slope - upper-right area
        if len(all_xs_time["cuda"]) >= 3:
            slope_cuda = compute_slope(np.array(all_xs_time["cuda"]), np.array(all_ys_time["cuda"]))
            tri_x_cuda = xlim[1] * (10 ** -1.2)
            tri_y_cuda = ylim[1] * (10 ** -1.0)
            draw_slope_triangle(ax, slope_cuda, tri_x_cuda, tri_y_cuda, width_decades=0.5, color="#A23B72", label=f"CUDA ≈{slope_cuda:.2f}")

        if not no_title:
            ax.set_title(f"PoissonMultiFrequency Total Time (K={K})", fontsize=13, fontweight="bold")

        out_time = os.path.join(out_dir, f"bench_time_K{K}_loglog.png")
        fig.savefig(out_time, dpi=170, bbox_inches="tight")
        print(f"[plot] wrote: {out_time}")
        if save_pdf:
            out_time_pdf = os.path.join(out_dir, f"bench_time_K{K}_loglog.pdf")
            fig.savefig(out_time_pdf, bbox_inches="tight")
            print(f"[plot] wrote: {out_time_pdf}")
        plt.close(fig)

        # MEMORY: Single plot with style/marker to differentiate devices
        fig_mem, ax_mem = plt.subplots(figsize=(8, 5.5))

        # Collect all points for slope calculation
        all_xs_mem: Dict[str, List[float]] = {"cpu": [], "cuda": []}
        all_ys_mem: Dict[str, List[float]] = {"cpu": [], "cuda": []}

        for dev in devices:
            dev_style = get_device_style(dev)
            min_x = cuda_min_x if is_cuda(dev) else 0
            dev_key = "cuda" if is_cuda(dev) else "cpu"

            for npt in n_points_repr:
                xs = []
                ys = []
                yerr = []
                for r in rows:
                    if r["device"] != dev:
                        continue
                    if to_i(r["K"]) != K:
                        continue
                    if to_i(r["n_points"]) != npt:
                        continue
                    b = to_i(r["batch_size"])
                    x = float(b * npt)
                    if is_cuda(dev):
                        y_raw = r.get("cuda_peak_mean_bytes") or ""
                        e_raw = r.get("cuda_peak_std_bytes") or ""
                    else:
                        y_raw = r.get("cpu_rss_peak_mean_bytes") or r.get("cpu_rss_after_mean_bytes") or ""
                        e_raw = r.get("cpu_rss_peak_std_bytes") or r.get("cpu_rss_after_std_bytes") or ""
                    if y_raw == "" or y_raw is None:
                        continue
                    y = float(y_raw) / (1024**3)  # GB
                    e = float(e_raw) / (1024**3) if e_raw not in ("", None) else 0.0
                    if x >= max(1, min_x) and y > 0:
                        xs.append(x)
                        ys.append(y)
                        yerr.append(e)
                if len(xs) < 2:
                    continue
                order = np.argsort(np.array(xs))
                xs_arr = np.array(xs)[order]
                ys_arr = np.array(ys)[order]
                yerr_arr = np.array(yerr)[order]

                # Collect for slope calculation
                all_xs_mem[dev_key].extend(xs_arr.tolist())
                all_ys_mem[dev_key].extend(ys_arr.tolist())

                c = color_for_n(npt)
                ax_mem.errorbar(
                    xs_arr,
                    ys_arr,
                    yerr=yerr_arr,
                    linestyle=dev_style["linestyle"],
                    color=c,
                    linewidth=2.0,
                    alpha=0.85,
                    marker=dev_style["marker"],
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    capsize=2,
                    capthick=1,
                )

        ax_mem.set_xscale("log")
        ax_mem.set_yscale("log")
        ax_mem.set_xlabel("batch_size × n_points")
        ax_mem.set_ylabel("Memory (GB)")
        ax_mem.grid(True, which="major", alpha=0.3, linestyle="-", linewidth=0.5)

        # Add colorbar
        sm_mem = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm_mem.set_array([])
        cbar_mem = fig_mem.colorbar(sm_mem, ax=ax_mem, pad=0.02)
        cbar_mem.set_label("n_points (dof)", fontsize=11)

        # Add device legend
        dev_handles_mem = []
        dev_labels_mem = []
        if has_cpu:
            dev_handles_mem.append(Line2D([0], [0], color="gray", linestyle="-", marker="o", linewidth=2, markersize=6))
            dev_labels_mem.append("CPU")
        if has_cuda:
            dev_handles_mem.append(Line2D([0], [0], color="gray", linestyle="-", marker="s", linewidth=2, markersize=6))
            dev_labels_mem.append("CUDA")
        ax_mem.legend(dev_handles_mem, dev_labels_mem, title="Device", loc="upper left", frameon=True, fancybox=True, framealpha=0.9)

        # Draw slope triangles for memory
        xlim_mem = ax_mem.get_xlim()
        ylim_mem = ax_mem.get_ylim()

        # CPU slope - lower-center area (avoid overlapping with lines)
        if len(all_xs_mem["cpu"]) >= 3:
            slope_cpu_mem = compute_slope(np.array(all_xs_mem["cpu"]), np.array(all_ys_mem["cpu"]))
            tri_x_cpu_mem = xlim_mem[0] * (10 ** 1.5)  # moved right
            tri_y_cpu_mem = ylim_mem[0] * (10 ** 0.3)
            draw_slope_triangle(ax_mem, slope_cpu_mem, tri_x_cpu_mem, tri_y_cpu_mem, width_decades=0.5, color="#2E86AB", label=f"CPU ≈{slope_cpu_mem:.2f}")

        # CUDA slope - upper-right area
        if len(all_xs_mem["cuda"]) >= 3:
            slope_cuda_mem = compute_slope(np.array(all_xs_mem["cuda"]), np.array(all_ys_mem["cuda"]))
            tri_x_cuda_mem = xlim_mem[1] * (10 ** -1.2)
            tri_y_cuda_mem = ylim_mem[1] * (10 ** -1.0)
            draw_slope_triangle(ax_mem, slope_cuda_mem, tri_x_cuda_mem, tri_y_cuda_mem, width_decades=0.5, color="#A23B72", label=f"CUDA ≈{slope_cuda_mem:.2f}")

        if not no_title:
            ax_mem.set_title(f"PoissonMultiFrequency Peak Memory (K={K})", fontsize=13, fontweight="bold")

        out_mem = os.path.join(out_dir, f"bench_mem_K{K}_loglog.png")
        fig_mem.savefig(out_mem, dpi=170, bbox_inches="tight")
        print(f"[plot] wrote: {out_mem}")
        if save_pdf:
            out_mem_pdf = os.path.join(out_dir, f"bench_mem_K{K}_loglog.pdf")
            fig_mem.savefig(out_mem_pdf, bbox_inches="tight")
            print(f"[plot] wrote: {out_mem_pdf}")
        plt.close(fig_mem)

    # Intentionally do not write any sidecar artifacts in cache/; plots are the deliverable.


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))  # allow 1e6, etc.
    return out


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path_default = _default_cache_path(out_dir)

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bench = sub.add_parser("bench")
    ap_bench.add_argument("--devices", default="both", help="cpu,cuda,both")
    ap_bench.add_argument("--n_points_list", default="", help="comma-separated n_points values (dof). If empty, use logspace params below.")
    ap_bench.add_argument("--batch_sizes", default="", help="comma-separated batch sizes. If empty, use logspace params below.")
    ap_bench.add_argument("--n_points_min", type=float, default=1e3)
    ap_bench.add_argument("--n_points_max", type=float, default=1e6)
    ap_bench.add_argument("--n_points_num", type=int, default=20)
    ap_bench.add_argument("--batch_min", type=float, default=1)
    # Allow batch_size up to 64*1024 by default (user-requested)
    ap_bench.add_argument("--batch_max", type=float, default=64 * 1024)
    ap_bench.add_argument("--batch_num", type=int, default=20)
    ap_bench.add_argument("--Ks", default="16", help="comma-separated K values (keep one for the final 2-figure deliverable)")
    ap_bench.add_argument("--repeats", type=int, default=5)
    ap_bench.add_argument("--max_x", type=float, default=1e9, help="max batch_size*n_points to include in sweep")
    ap_bench.add_argument("--max_output_elems", type=float, default=2e7, help="CPU: skip if batch_size*n_points exceeds this (prevents OOM)")
    ap_bench.add_argument("--max_output_elems_cuda", type=float, default=2e8, help="CUDA: skip if batch_size*n_points exceeds this (prevents OOM)")
    ap_bench.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="dtype for points (affects memory/time)")
    ap_bench.add_argument("--cuda_device", type=int, default=0, help="CUDA device index to use (e.g., 1 for cuda:1)")
    ap_bench.add_argument("--cache", default=cache_path_default)
    ap_bench.add_argument(
        "--cpu_mem_method",
        default="subprocess",
        choices=["subprocess", "rss_after"],
        help="CPU memory method: subprocess=peak RSS (recommended), rss_after=RSS after run (legacy).",
    )
    ap_bench.add_argument("--dim", type=int, default=2, choices=[2, 3], help="Spatial dimension: 2 for rectangle, 3 for cube")

    ap_plot = sub.add_parser("plot")
    ap_plot.add_argument("--cache", default=cache_path_default)
    # Put final PNGs in the parent example directory by default.
    ap_plot.add_argument("--out_dir", default=out_dir)
    ap_plot.add_argument("--no-title", action="store_true", help="Generate plots without titles (for paper figures)")
    ap_plot.add_argument("--pdf", action="store_true", help="Also generate PDF output")

    ap_prune = sub.add_parser("prune")
    ap_prune.add_argument("--cache", default=cache_path_default)
    ap_prune.add_argument("--backup_suffix", default=".bak")
    ap_prune.add_argument("--cuda_top_n_points", type=int, default=5, help="Keep only the top-N largest n_points for CUDA rows.")

    ap_child = sub.add_parser("bench_child")
    ap_child.add_argument("--n_points", type=int, required=True)
    ap_child.add_argument("--batch_size", type=int, required=True)
    ap_child.add_argument("--K", type=int, required=True)
    ap_child.add_argument("--repeats", type=int, required=True)
    ap_child.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap_child.add_argument("--dim", type=int, default=2, choices=[2, 3])

    args = ap.parse_args()

    if args.cmd == "bench":
        n_points_list = _parse_int_list(args.n_points_list) if args.n_points_list.strip() else _logspace_int(args.n_points_min, args.n_points_max, args.n_points_num)
        batch_sizes = _parse_int_list(args.batch_sizes) if args.batch_sizes.strip() else _logspace_int(args.batch_min, args.batch_max, args.batch_num)
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        run_bench(
            devices=[d.strip() for d in args.devices.split(",")],
            n_points_list=n_points_list,
            batch_sizes=batch_sizes,
            Ks=_parse_int_list(args.Ks),
            repeats=int(args.repeats),
            cache_path=str(args.cache),
            max_x=int(float(args.max_x)),
            max_output_elems=int(float(args.max_output_elems)),
            dtype=dtype,
            cuda_device=int(args.cuda_device),
            max_output_elems_cuda=int(float(args.max_output_elems_cuda)),
            cpu_mem_method=str(args.cpu_mem_method),
            dim=int(args.dim),
        )
        print(f"[bench] cache: {args.cache}")
    elif args.cmd == "plot":
        plot_cache(str(args.cache), str(args.out_dir), no_title=getattr(args, "no_title", False), save_pdf=getattr(args, "pdf", False))
    elif args.cmd == "prune":
        prune_cache_csv(
            cache_path=str(args.cache),
            backup_suffix=str(args.backup_suffix),
            keep_cuda_pow4=True,
            cuda_top_n_points=int(args.cuda_top_n_points),
        )
        print(f"[prune] wrote: {args.cache}")
    elif args.cmd == "bench_child":
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        res = _bench_one_cpu_peak_rss(
            n_points=int(args.n_points),
            batch_size=int(args.batch_size),
            K=int(args.K),
            repeats=int(args.repeats),
            dtype=dtype,
            dim=int(args.dim),
        )
        print(json.dumps(asdict(res)))
    else:
        raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()


