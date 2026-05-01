Batched Workflows
=================

"Batched" means three different things in TensorMesh, and they
operate at different layers of the pipeline. Picking the right one
for your problem matters more than turning every knob at once. This
page walks through what's actually supported today.


Three axes of batching
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Axis
     - Where it lives
     - When to reach for it
   * - **Memory chunking**
     - ``Assembler(batch_size=N)``
     - One assembly that doesn't fit in GPU memory.
   * - **Batched right-hand sides**
     - ``b: [n_dof, n_batch]``
     - Many loads, one mesh, one stiffness matrix.
   * - **Multi-problem datasets**
     - :mod:`tensormesh.dataset`
     - ML training data: many ``(K_i, b_i)`` pairs.

What is **not** built in: vmap-style assembly that vectorizes
``K`` construction across many parameter values. If you need that,
loop in Python or wrap your assembler call in ``torch.vmap``
manually — see :ref:`batched-not-built-in` below.


Axis 1 — Memory chunking with ``batch_size``
--------------------------------------------

A single :class:`~tensormesh.ElementAssembler` call holds the
per-element, per-quadrature-point integrand tensor in memory. For
fine meshes or high-order elements, that tensor can be too large.
The ``batch_size`` argument splits the quadrature dimension into
chunks, accumulating the contribution one chunk at a time:

.. code-block:: python

   K = LaplaceElementAssembler.from_mesh(mesh)(batch_size=4)

The assembled matrix is bit-identical to the un-batched call —
``batch_size`` is purely a memory knob. Default is ``-1`` (no
chunking, fastest if memory allows). A typical recipe: start with
the default; if you OOM, halve ``batch_size`` until the assembly
fits.

This axis has nothing to do with batching across *problems*. The
result is one matrix, not a batch of matrices.


Axis 2 — Batched right-hand sides
---------------------------------

When you have one stiffness matrix ``K`` and many load vectors
``b_i``, stack the loads into a single ``[n_dof, n_batch]`` tensor
and let the solver factor ``K`` once:

.. code-block:: python

   B = torch.stack([b_1, b_2, ..., b_64], dim=1)   # [n_dof, 64]
   X = K.solve(B)                                  # [n_dof, 64]

:func:`~tensormesh.sparse.spsolve` detects the 2D RHS and
auto-routes to a SuperLU direct solve: one factorization, ``n_batch``
back-substitutions. The speedup over a Python loop of independent
CG solves is typically large, especially on CPU.

Condensation works the same way. :meth:`Condenser.condense_rhs`
accepts ``[n_dof, ...]`` shapes — the leading DOF axis is sliced
and the trailing axes pass through:

.. code-block:: python

   condenser = Condenser(mesh.boundary_mask)
   K_, _ = condenser(K, torch.zeros(mesh.n_points))
   B_inner = condenser.condense_rhs(B)             # [n_inner, 64]
   X_inner = K_.solve(B_inner)                     # [n_inner, 64]
   X_full  = condenser.recover(X_inner)            # [n_dof, 64]

This is the workhorse pattern for ML training-data generation and
for parametric studies where every problem shares geometry.


Axis 3 — Multi-problem datasets
-------------------------------

For ML workloads where you want many ``(input, output)`` pairs,
:mod:`tensormesh.dataset` provides:

* :class:`~tensormesh.MeshGen` — programmatic Gmsh-backed mesh
  generation from CSG primitives (rectangles, circles, holes).
* Multi-frequency equation classes that emit batched source terms
  with known analytical solutions, for supervised training:

  * :class:`~tensormesh.dataset.PoissonMultiFrequency` (and ``…3D``)
  * :class:`~tensormesh.dataset.HeatMultiFrequency`
  * :class:`~tensormesh.dataset.WaveMultiFrequency`
  * ``LinearElasticityMultiFrequency`` (and ``…3D``)

Each ``...MultiFrequency`` class lets you draw random Fourier
coefficients for the source term and (where applicable) returns the
analytical solution at the same nodes — handy for both training and
validation.

The standard "same mesh, varying source" workflow combines this
generator with batched RHS:

.. code-block:: python

   from tensormesh import Mesh, LaplaceElementAssembler, MassElementAssembler, Condenser
   from tensormesh.dataset import PoissonMultiFrequency

   mesh = Mesh.gen_rectangle(chara_length=0.02)
   K = LaplaceElementAssembler.from_mesh(mesh)()
   M = MassElementAssembler.from_mesh(mesh)()
   condenser = Condenser(mesh.boundary_mask)
   K_, _ = condenser(K, torch.zeros(mesh.n_points))

   # 64 random Poisson source terms, all evaluated at the same nodes.
   eq = PoissonMultiFrequency(a=torch.rand(64, 8, 8) * 2 - 1)
   f  = eq.source_term(mesh.points, domain="rectangle")    # [64, n_points]
   b  = (M @ f.T)                                          # [n_points, 64]
   b_ = condenser.condense_rhs(b)                           # [n_inner, 64]
   u_ = K_.solve(b_)                                        # [n_inner, 64]
   u  = condenser.recover(u_)                               # [n_points, 64]

The full version of this loop, with timing benchmarks across
``n_batch`` from 1 to 1024, lives in
``examples/poisson/poisson_batch_solver.py``. On a workstation GPU
the per-problem cost drops by an order of magnitude or more
relative to one-at-a-time solves.

For *varying mesh* workflows, regenerate ``Mesh`` per problem and
pay the assembly cost each time — there is no shortcut for that
case today.


.. _batched-not-built-in:

What's not built in
-------------------

There is **no** built-in vmap-style assembler that vectorizes
``K`` construction across many parameter values (e.g. one ``K``
per realization of a stochastic coefficient field). For now, two
patterns work in user code:

* **Python loop** — simplest, fine when assembly is cheap relative
  to solve.
* **``torch.vmap`` around the assembler call** — works when the
  varying parameter enters via ``point_data`` or ``element_data``
  and the result is uniform in shape. Note that vmap's interaction
  with sparse output is fragile; test on small problems first.

A first-class batched-assembly API may show up in a future release;
the dataset workflow above covers the most common case (same mesh,
many sources) without it.


What's next
-----------

* :doc:`linear_solvers` — the SuperLU auto-routing and the
  ``backend`` knobs that affect batched solves.
* :doc:`differentiability` — backprop a loss through a batched
  pipeline.
* :doc:`../example_gallery/index` — ``examples/poisson/poisson_batch_solver.py``
  for the full benchmarked driver.
