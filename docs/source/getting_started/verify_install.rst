Verify Install
==============

A self-contained smoke test that confirms TensorMesh, ``torch-sla``, and
your PyTorch build are wired up correctly. The script:

* solves a tiny Poisson problem on the **CPU** (always),
* repeats the solve on the **GPU** if CUDA is available,
* prints version information, and
* reports which sparse-solver backends are usable on your machine.

Save the snippet below as ``verify_install.py`` and run
``python verify_install.py``. Total runtime is a couple of seconds.


The script
----------

.. code-block:: python

   """Verify a TensorMesh install: solve a tiny Poisson problem on CPU
   (and on GPU if available), then report which sparse-solver backends
   are wired up."""

   import math
   import platform
   import sys
   import time

   import torch


   def solve_poisson(device):
       from tensormesh import ElementAssembler, NodeAssembler, Mesh, Condenser

       mesh = Mesh.gen_rectangle(chara_length=0.1).to(device)

       class Laplace(ElementAssembler):
           def forward(self, gradu, gradv):
               return gradu @ gradv

       class Source(NodeAssembler):
           def forward(self, v, f):
               return f * v

       x, y = mesh.points[:, 0], mesh.points[:, 1]
       f_vals = 2 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

       K = Laplace.from_mesh(mesh)()
       b = Source.from_mesh(mesh)(point_data={"f": f_vals})

       cond = Condenser(mesh.boundary_mask)
       K_, b_ = cond(K, b)
       u = cond.recover(K_.solve(b_))

       u_exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)
       return float((u - u_exact).norm() / u_exact.norm())


   def probe_backends():
       out = {"scipy": (True, ""), "torch": (True, "")}
       try:
           import torch_sla  # noqa: F401
           out["torch_sla"] = (True, "")
       except ImportError:
           out["torch_sla"] = (False, "pip install torch-sla")
       try:
           import pyamg  # noqa: F401
           out["amg"] = (True, "")
       except ImportError:
           out["amg"] = (False, "pip install pyamg")
       from tensormesh.sparse import is_petsc_available, is_cupy_available
       out["petsc"] = (is_petsc_available,
                       "" if is_petsc_available else "pip install petsc4py")
       out["cupy"]  = (is_cupy_available and torch.cuda.is_available(),
                       "" if is_cupy_available else "pip install cupy")
       out["cudss"] = (torch.cuda.is_available(),
                       "" if torch.cuda.is_available() else "requires CUDA + libcudss")
       return out


   def main():
       import tensormesh
       print("TensorMesh smoke test")
       print("=" * 40)
       print(f"tensormesh : {tensormesh.__version__}")
       print(f"torch      : {torch.__version__}")
       try:
           import torch_sla
           print(f"torch-sla  : {getattr(torch_sla, '__version__', 'unknown')}")
       except ImportError:
           print(f"torch-sla  : MISSING")
       print(f"python     : {sys.version.split()[0]} on {platform.system().lower()}")
       print()

       t0 = time.perf_counter()
       err = solve_poisson("cpu")
       print(f"[CPU ] Poisson 2D ... OK   L2 error = {err:.3e}   {time.perf_counter()-t0:.2f} s")

       if torch.cuda.is_available():
           t0 = time.perf_counter()
           err = solve_poisson("cuda")
           print(f"[CUDA] Poisson 2D ... OK   L2 error = {err:.3e}   {time.perf_counter()-t0:.2f} s")
       else:
           print(f"[CUDA] not available, skipping GPU test")

       print()
       print("Sparse solver backends:")
       for name, (ok, hint) in probe_backends().items():
           mark = "OK " if ok else "-- "
           suffix = f"  ({hint})" if (not ok and hint) else ""
           print(f"  {name:<10s}: {mark}{suffix}")

       print()
       print("All required checks passed.")


   if __name__ == "__main__":
       main()


Expected output
---------------

On a CPU-only macOS / Linux laptop with a fresh ``pip install tensor-mesh``,
you should see something close to:

.. code-block:: text

   TensorMesh smoke test
   ========================================
   tensormesh : 0.1.0
   torch      : 2.11.0
   torch-sla  : 0.2.0
   python     : 3.13.4 on darwin

   [CPU ] Poisson 2D ... OK   L2 error = 1.187e-02   0.04 s
   [CUDA] not available, skipping GPU test

   Sparse solver backends:
     scipy     : OK
     torch     : OK
     torch_sla : OK
     amg       : --   (pip install pyamg)
     petsc     : --   (pip install petsc4py)
     cupy      : --   (pip install cupy)
     cudss     : --   (requires CUDA + libcudss)

   All required checks passed.

The exact L2 error depends on the mesh, but should be on the order of
:math:`10^{-2}` — anything more than a few percent indicates a numerical
problem.

.. note::

   **Why is the GPU run sometimes *slower* than the CPU run?**
   The smoke-test mesh is intentionally tiny (:math:`\sim` 100 DOFs). At
   this scale, the CUDA solve is dominated by one-time overheads —
   context creation, JIT kernel compilation, host↔device transfers, and
   ``cuSPARSE``/``cuSOLVER`` workspace allocation — rather than by
   actual floating-point work. The first GPU call also pays for CUDA
   driver initialization. Run a real-sized problem
   (:math:`\geq 10^4`–:math:`10^5` DOFs) to see the GPU pull ahead;
   see :doc:`../performance/index` for benchmarks.

.. note::

   **Which solver does ``A.solve(b)`` actually use by default?**
   ``SparseMatrix.solve`` is inherited from ``torch_sla.SparseTensor``
   and called with ``backend="auto"``, ``method="auto"``. torch-sla's
   auto-selector then picks, based on device and problem size:

   * **CPU** → SciPy / **SuperLU** (``backend="scipy"``, ``method="lu"``) —
     a direct factorization, fast and machine-precision for the sizes
     reachable on CPU.
   * **CUDA, DOF < 2M** → **cuDSS** if available
     (``method="cholesky"`` when SPD, else ``ldlt`` / ``lu``); falls
     back to CuPy, and finally to the PyTorch-native iterative CG.
   * **CUDA, DOF ≥ 2M** → PyTorch-native iterative solver
     (``backend="pytorch"``, ``method="cg"`` for SPD or
     ``bicgstab`` otherwise) with Jacobi preconditioning, to stay
     within GPU memory.

   So the smoke test above runs **SuperLU on CPU** and **cuDSS Cholesky
   on GPU** (Poisson is SPD), not an iterative solver. To pick a
   different solver explicitly, pass ``backend=...`` / ``method=...``
   to ``solve`` — see :doc:`../user_guide/linear_solvers`.

The optional backends marked ``--`` are *not* a failure: they simply
aren't installed. Pick the ones you need:

* **petsc** — distributed direct / iterative solvers, useful for very
  large 3D problems on CPU clusters.
* **cupy** / **cudss** — GPU sparse-direct solvers for fast factorization
  of small-to-medium 3D problems on CUDA.
* **amg** — algebraic multigrid preconditioner; recommended for 3D
  elasticity.

See :doc:`../user_guide/linear_solvers` for guidance on choosing among
them.


Troubleshooting
---------------

**``ModuleNotFoundError: No module named 'tensormesh'``** — the package
isn't installed in the active Python environment. Re-check
:doc:`installation`.

**``torch-sla : MISSING``** — ``pip install "torch-sla>=0.2.0"``. This is
a hard, import-time dependency: :mod:`tensormesh.sparse` will refuse to
import without it, so every solver-touching code path fails until it is
installed.

**Stuck on ``[CPU ] Poisson 2D``** — your PyTorch build may be downloading
``gmsh`` cache files on first use; subsequent runs are instant.

**``L2 error`` ≫ ``1e-2``** — likely an issue with the mesh or PyTorch
build. Open an issue at
`github.com/camlab-ethz/TensorMesh/issues
<https://github.com/camlab-ethz/TensorMesh/issues>`_ with the full output
of this script.
