Linear Solvers
==============

After assembly and boundary-condition condensation you have a sparse
linear system to solve. TensorMesh delegates this work to
`torch-sla <https://pypi.org/project/torch-sla/>`_, a standalone
differentiable sparse-linear-algebra library. The same FEM code
retargets between CPU and GPU, and between iterative and direct
solvers, by changing one keyword argument.


Install ``torch-sla``
---------------------

.. important::

   ``torch-sla`` is a **hard, import-time** dependency of
   :mod:`tensormesh.sparse`. The module will not import without it.
   It is also the library where all current and future solver work
   lands — TensorMesh tracks ``torch-sla`` releases as the canonical
   sparse-linear-algebra layer and we recommend keeping it up to date.

.. code-block:: bash

   pip install "torch-sla>=0.2.0"

For GPU sparse solves install the CUDA extra, which pulls in CuPy and
the NVIDIA cuDSS bindings:

.. code-block:: bash

   pip install "torch-sla[cuda]>=0.2.0"

What ``torch-sla`` gives TensorMesh:

* :class:`~tensormesh.sparse.SparseTensor` — the COO data type that
  :class:`~tensormesh.sparse.SparseMatrix` extends.
* A unified ``solve`` op with a custom backward pass (an adjoint
  sparse solve), so gradients flow through every linear system.
* A pluggable backend layer covering CPU, GPU, iterative and direct
  solvers behind a single dispatch entry point.


Supported backends
------------------

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
     - Pure-torch CG / BiCGSTAB / GMRES. Default for CUDA.
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


The ``spsolve`` entry point
---------------------------

:func:`tensormesh.sparse.spsolve` is the dispatch entry point;
:meth:`~tensormesh.sparse.SparseMatrix.solve` is a thin convenience
wrapper around it that you'll see in most quickstart-style code.

.. code-block:: python

   K = LaplaceElementAssembler.from_mesh(mesh)()
   x = K.solve(b)                              # SPD CG via torch-sla
   x = K.solve(b, method="superlu")            # direct
   x = K.solve(b, backend="cudss")             # NVIDIA GPU direct
   x = K.solve(b, is_spd=False, method="bicgstab")

Key keyword arguments:

* ``backend``: ``"auto"`` (default — SciPy on CPU, native PyTorch on
  CUDA) or one of the strings in the table above.
* ``method``: ``"cg"``, ``"bicgstab"``, ``"minres"``, ``"gmres"``,
  ``"lgmres"``, or ``"superlu"`` (direct factorization).
* ``preconditioner``: ``"jacobi"`` (default), ``"ilu"``, or ``"none"``.
* ``is_spd=True`` (default) tells the solver it can use CG. Set
  ``False`` for indefinite / non-symmetric ``A`` and pair with
  ``method="bicgstab"`` or ``"gmres"``.
* ``tol``, ``max_iter`` control iterative convergence.


Batched right-hand sides
------------------------

When ``b`` has shape ``[n_dof, n_batch]``, ``spsolve`` automatically
routes to a SuperLU direct solve — one factorization,
``n_batch`` back-substitutions — instead of running iterative CG
independently per column. This is the workhorse of the
:mod:`tensormesh.dataset` ML workflow.

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

   def residual(u, *params):  ...   # torch.Tensor [n_dof]
   def jacobian(u, *params):  ...   # SparseMatrix [n_dof, n_dof]

   u = nonlinear_solve(residual, jacobian, u0, params=(p1, p2),
                       max_iter=100, tol=1e-6)

Forward pass is Newton-Raphson; backward pass uses the implicit-function
theorem, so gradients of ``u`` w.r.t. ``params`` cost roughly one extra
linear solve regardless of how many Newton iterations the forward took.


What's next
-----------

* :doc:`time_integration` — wrap a linear solve in a time-stepping
  loop for transient problems.
* :doc:`batched_workflows` — three different axes of batching, and
  when each one applies.
* :doc:`differentiability` — how the adjoint backward works and how
  to wire a learnable parameter through a sparse solve.
