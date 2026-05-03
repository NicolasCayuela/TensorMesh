Linear Solvers
==============

After assembly and boundary-condition condensation you have a sparse
linear system to solve. TensorMesh delegates this work to a separate
package, `torch-sla <https://pypi.org/project/torch-sla/>`_, which
ships a portable set of sparse-linear-algebra backends with autograd
support. The same FEM code therefore retargets between CPU and GPU,
and between iterative and direct solvers, by changing one keyword
argument.


The ``torch-sla`` package
-------------------------

``torch-sla`` is the engine behind every sparse solve in TensorMesh.
It provides:

* :class:`~tensormesh.sparse.SparseTensor` — the data type that
  :class:`~tensormesh.sparse.SparseMatrix` extends.
* A unified ``solve`` op with a custom backward pass (an adjoint
  sparse solve), so gradients flow through the linear system.
* A pluggable backend layer that dispatches the solve to one of
  several sparse-linear-algebra implementations.

Install or upgrade with:

.. code-block:: bash

   pip install "torch-sla>=0.1.4"

Without ``torch-sla`` installed, :func:`tensormesh.sparse.spsolve`
falls back to a built-in mini-stack inside :mod:`tensormesh.sparse`
that wraps SciPy / SciPy + ILU / SuperLU / CuPy / PETSc directly. The
fallback path works but lacks the autograd guarantees and unified
backend interface — install ``torch-sla`` for serious use.


Supported backends
------------------

``torch-sla`` ships five backends today:

.. list-table::
   :header-rows: 1
   :widths: 18 14 14 16 38

   * - Backend
     - String
     - Device
     - Direct / Iter
     - Notes
   * - SciPy
     - ``"scipy"``
     - CPU
     - both
     - Default for CPU. Wraps ``scipy.sparse.linalg``.
   * - Eigen
     - ``"eigen"``
     - CPU
     - both
     - C++ Eigen via pybind11. Fast direct + iterative on CPU.
   * - Native PyTorch
     - ``"pytorch"``
     - CPU / GPU
     - iterative
     - Pure-torch CG / BiCGSTAB / GMRES. GPU-friendly, autograd-clean.
   * - cuDSS
     - ``"cudss"``
     - GPU
     - direct
     - NVIDIA cuDSS sparse direct solver. Fastest GPU direct path.
   * - CuPy
     - ``"cupy"``
     - GPU
     - both
     - CuPy's iterative solvers and SuperLU.

PETSc and Hypre are on the ``torch-sla`` roadmap. Until they ship
inside ``torch-sla``, the legacy fallback paths in
:mod:`tensormesh.sparse` provide best-effort PETSc and CuPy support
for users who already have those libraries installed locally — see
the ``[petsc]`` and ``[cupy]`` extras in :doc:`../getting_started/installation`.


The ``spsolve`` entry point
---------------------------

:func:`tensormesh.sparse.spsolve` is the dispatch entry point. The
quickstart-style :meth:`~tensormesh.sparse.SparseMatrix.solve` is a thin convenience
wrapper around it.

.. code-block:: python

   from tensormesh.sparse import spsolve

   x = spsolve(edata, row, col, shape, b,
               backend="auto", method="cg",
               preconditioner="jacobi",
               tol=1e-5, max_iter=10000,
               x0=None, is_spd=True)

Key options:

* ``backend``: ``"auto"`` (the default — picks SciPy on CPU, PyTorch
  on CUDA) or one of the strings in the table above.
* ``method``: iterative algorithm — ``"cg"``, ``"bicgstab"``,
  ``"minres"``, ``"gmres"``, ``"lgmres"`` — or ``"superlu"`` for a
  direct factorization.
* ``preconditioner``: ``"jacobi"`` (default), ``"ilu"``, or ``"none"``.
* ``tol`` and ``max_iter`` control iterative convergence.
* ``is_spd=True`` (default) is a hint that lets the solver pick CG;
  set ``False`` for indefinite or non-symmetric systems.

For most TensorMesh code you don't call :func:`~tensormesh.sparse.spsolve`
directly:
:meth:`~tensormesh.sparse.SparseMatrix.solve` does it for you, taking the same kwargs:

.. code-block:: python

   K = LaplaceElementAssembler.from_mesh(mesh)()
   x = K.solve(b)                              # SPD CG via torch-sla
   x = K.solve(b, method="superlu")            # direct
   x = K.solve(b, backend="cudss")             # NVIDIA GPU direct
   x = K.solve(b, is_spd=False, method="bicgstab")


Batched right-hand sides
------------------------

When ``b`` has shape ``[n_dof, n_batch]``, ``spsolve`` automatically
routes to a SuperLU direct solve — one factorization, ``n_batch``
back-substitutions — instead of running iterative CG independently
per column. The speedup over a Python loop is typically large; this
is the workhorse of the ``tensormesh.dataset`` ML workflow.

.. code-block:: python

   B = torch.randn(K.shape[0], 64)             # 64 right-hand sides
   X = K.solve(B)                              # one factorization

See :doc:`batched_workflows` for the broader story on batching.


Nonlinear systems
-----------------

For problems where the residual ``F(u, params) = 0`` is nonlinear in
``u`` — hyperelasticity, plasticity, phase-field — use
:func:`tensormesh.sparse.nonlinear_solve`:

.. code-block:: python

   from tensormesh.sparse import nonlinear_solve

   def residual(u, *params):
       ...
       return F                              # torch.Tensor [n_dof]

   def jacobian(u, *params):
       ...
       return J                              # SparseMatrix [n_dof, n_dof]

   u = nonlinear_solve(residual, jacobian, u0, params=(p1, p2),
                       max_iter=100, tol=1e-6, verbose=False)

Internally this is Newton-Raphson: at each iteration, ``J(u, params)``
is solved against the residual to update ``u``. The whole call is
implemented as a custom :class:`torch.autograd.Function` with an
*adjoint* backward — gradients of ``u`` with respect to ``params``
flow through the implicit-function theorem rather than through the
Newton iterations themselves, so backprop cost is roughly one extra
linear solve regardless of how many Newton iterations the forward
pass took.


Choosing a backend
------------------

A pragmatic first cut:

* **SPD + CPU + small-medium** → ``backend="auto"``, ``method="cg"``,
  ``preconditioner="jacobi"``. The TensorMesh default.
* **SPD + GPU** → ``backend="pytorch"`` (fully differentiable) or
  ``backend="cudss"`` (fastest direct, gradients still flow through
  the autograd-aware wrapper).
* **Indefinite / non-symmetric** → ``is_spd=False`` plus
  ``method="bicgstab"`` or ``"gmres"``. For tough conditioning, drop
  to a direct factorization with ``method="superlu"`` (CPU) or
  ``backend="cudss"`` (GPU).
* **Many right-hand sides, same matrix** → batch the RHS and let the
  SuperLU auto-routing kick in.

For autodiff workflows (parameter identification, topology
optimization, neural-network couplings) the safe defaults are
``backend="auto"`` on CPU and ``backend="pytorch"`` on GPU. The
SciPy and Eigen backends route through ``torch-sla``'s autograd
wrapper too, but the pure-PyTorch path is the most painless if you
hit any rough edges.


What's next
-----------

* :doc:`time_integration` — wrap a linear solve in a time-stepping
  loop for transient problems.
* :doc:`batched_workflows` — three different axes of batching, and
  when each one applies.
* :doc:`differentiability` — how the adjoint backward works and how
  to wire a learnable parameter through a sparse solve.
