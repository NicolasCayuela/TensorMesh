ML Datasets
===========

Three scripts in ``examples/dataset/`` generate large, ML-ready
batches of PDE solutions for training neural operators (DeepONet,
FNO, Geo-FNO, …). The recipe is the one outlined in
:doc:`../user_guide/batched_workflows`: same mesh, many right-hand
sides, one factorization, multi-RHS back-substitution.

* ``poisson/poisson_dataset.py`` — steady Poisson over many source
  terms; the cleanest illustration of factor-once / back-sub-many.
* ``heat/heat_dataset.py`` — 1000 heat-equation samples, 100
  timesteps, on an L-shaped domain.
* ``wave/wave_dataset.py`` — 100 wave-equation samples, 100
  timesteps, on a circular domain.

All three run on GPU when one is visible (preferring CuPy for the
sparse solve), fall back to SciPy on CPU, and benchmark the two
paths against each other.


Poisson batched RHS — ``poisson_dataset.py``
--------------------------------------------

The steady-state warm-up: same stiffness matrix, many source
fields. The script generates a batch of multi-frequency Fourier
sources via :class:`~tensormesh.dataset.PoissonMultiFrequency`,
assembles a single ``K``, and solves all loads simultaneously:

.. code-block:: python
   :caption: examples/dataset/poisson/poisson_dataset.py (essence)

   from tensormesh import (LaplaceElementAssembler,
                           MassElementAssembler,
                           Mesh, Condenser)
   from tensormesh.dataset import PoissonMultiFrequency

   mesh      = Mesh.gen_rectangle(chara_length=0.02).to(device)
   K_full    = LaplaceElementAssembler.from_mesh(mesh)()
   M         = MassElementAssembler.from_mesh(mesh)()
   condenser = Condenser(mesh.boundary_mask)
   K, _      = condenser(K_full, torch.zeros(mesh.n_points, device=device))

   batch_size, K_modes = 64, 8
   a  = torch.rand((batch_size, K_modes, K_modes), device=device) * 2 - 1
   eq = PoissonMultiFrequency(a=a)
   f  = eq.source_term(mesh.points, domain="rectangle")  # [batch, n_points]

   F  = M @ f.T                                          # [n_points, batch]
   F_ = condenser.condense_rhs(F)                        # [n_inner, batch]
   u_ = K.solve(F_)                                      # [n_inner, batch]
   u  = condenser.recover(u_).T                          # [batch, n_points]

The 2D RHS shape ``[n_inner, batch]`` is auto-routed to a SuperLU
direct solve: one factorization, ``batch`` back-substitutions. A
``--mode bench`` benchmark sweeps batch sizes from 1 to 1024 and
reports per-problem timing on CPU and GPU; the per-problem cost
drops sharply with batch size as the factorization overhead is
amortized.

The CLI exposes 2D and 3D variants on the unit square / cube:

.. code-block:: bash

   python poisson_dataset.py --mode 2d      # rectangle
   python poisson_dataset.py --mode 3d      # cube (writes .vtu)
   python poisson_dataset.py --mode bench   # batch-size sweep
   python poisson_dataset.py --mode all     # all of the above

Because :class:`~tensormesh.dataset.PoissonMultiFrequency` also
provides the analytical solution from the same Fourier coefficients,
the script logs max / mean / relative error per batch.


Heat dataset on the L-shape — ``heat_dataset.py``
-------------------------------------------------

Same backward-Euler scheme as :doc:`diffusion`, but the time loop
operates on a 2D snapshot ``U`` of shape ``[n_dofs, batch_size]``:

.. code-block:: python
   :caption: examples/dataset/heat/heat_dataset.py

   mesh = Mesh.gen_L(chara_length=0.008, element_type="tri").to(device)

   batch_size, d, n = 1000, 16, 100
   mus     = torch.rand((batch_size, d))            # random Fourier coeffs
   dataset = HeatMultiFrequency(mu=mus)
   u0s     = dataset.initial_condition(mesh.points)  # [batch_size, n_dofs]

   M = MAssembler.from_mesh(mesh, quadrature_order=2)()
   A = AAssembler.from_mesh(mesh, quadrature_order=2)()
   condenser = Condenser(mesh.boundary_mask)

   dt, D = 5e-5, 1.0
   K_    = condenser(M + dt * D * D * A)[0]          # condense once

   U = u0s.T.clone()                                  # [n_dofs, batch_size]
   Us = [U]
   for _ in range(n - 1):
       F  = M @ U                                     # [n_dofs, batch_size]
       F_ = condenser.condense_rhs(F)                 # [n_inner, batch_size]
       U_ = K_.solve(F_)                              # [n_inner, batch_size]
       U  = condenser.recover(U_)
       Us.append(U)

A few details that make the GPU path fast:

* ``F`` is a 2D tensor at every step; the sparse
  matrix-multiplication ``M @ U`` broadcasts over the batch axis on
  GPU.
* ``K_.solve(F_)`` with ``F_`` shape ``[n_inner, batch_size]``
  routes to ``torch-sla``'s batched-RHS direct-solve path
  (``K_`` is a :class:`~tensormesh.sparse.SparseMatrix`, so
  ``.solve`` is the inherited ``torch_sla.SparseTensor.solve``).
  On CPU this is SciPy's SuperLU; on GPU it is CuPy / cuDSS. One
  factorization is reused across all 1000 samples.
* The L-shape is meshed at ``chara_length=0.008``, giving on the
  order of 50–100k DOFs — large enough that the batched solve is
  much faster than 1000 independent CG calls.

Output is a single ``.npz`` file with the full snapshot tensor:

.. code-block:: text

   heat_dataset.npz
       snapshots:  (100, n_dofs, 1000)   # n_steps × n_dofs × batch_size
       points:     (n_dofs, 2)
       mus:        (1000, 16)            # Fourier coefficients
       dt, D, n:   scalars

The script also writes ``heat_dataset.mp4`` showing the first
sample evolving on GPU and CPU side-by-side, and prints a speedup
ratio at the end.

*(figure: GPU vs CPU snapshot at t = 50 dt for sample 0; will be added in a follow-up)*


Wave dataset on the disk — ``wave_dataset.py``
----------------------------------------------

Same idea, central-difference scheme. The mesh is a circular
domain (centered at :math:`(0.5, 0.5)`, radius :math:`0.5`,
``chara_length=0.015``), and the per-sample initial condition is
parameterized by a 16×16 random Fourier-coefficient matrix
:math:`a`:

.. code-block:: python

   batch_size, K, n = 100, 16, 100
   a = torch.zeros((batch_size, K, K)).uniform_(-1, 1)
   dataset = WaveMultiFrequency(a=a, c=c)
   u0s = dataset.initial_condition(mesh.points)        # [batch_size, n_dofs]

The ``[n_dofs, batch_size]``-shaped time loop is identical in
spirit to the heat case, with the wave-equation 3-term recurrence
replacing the Euler step (see :doc:`wave` for the per-sample
derivation). Output schema:

.. code-block:: text

   wave_dataset.npz
       snapshots:  (100, n_dofs, 100)
       points:     (n_dofs, 2)
       a:          (100, 16, 16)
       dt, c, n:   scalars


Where the speedup comes from
----------------------------

The wins on these scripts are not from any clever batching at the
Python level — they come from two TensorMesh defaults:

1. **Single factorization.** Each script assembles its system
   matrix (the stiffness ``K`` for Poisson, ``M + dt D² A`` for
   heat, ``2 M`` for wave) once and reuses it for every batch
   element — and, in the time-dependent cases, every timestep.
   SciPy's SuperLU and CuPy's sparse direct solver both expose
   this factor-once / back-sub-many pattern through
   ``SparseMatrix.solve`` (inherited from
   ``torch_sla.SparseTensor``).
2. **Native batched RHS.** ``[n_inner, batch_size]`` lands in the
   solver as a 2D dense matrix; back-substitution iterates over
   the batch axis at the C / CUDA level, with no Python loop in
   between.

For a deeper discussion of the three batching axes (memory
chunking via ``Assembler(batch_size=N)``, batched RHS, multi-
problem datasets) see :doc:`../user_guide/batched_workflows`.


Running the examples
--------------------

.. code-block:: bash

   cd examples/dataset/poisson
   python poisson_dataset.py --mode 2d    # writes poisson_batch_solver_2d.png

   cd ../heat
   python heat_dataset.py                 # writes heat_dataset.{npz,mp4}

   cd ../wave
   python wave_dataset.py                 # writes wave_dataset.{npz,mp4}

CuPy is used automatically when CUDA is available; install it via
``pip install -e ".[cupy]"``.


What's next
-----------

* :doc:`../user_guide/batched_workflows` — the three axes of
  batching and when each one applies.
* :doc:`../user_guide/linear_solvers` — backend choice (cudss /
  cupy / scipy / pytorch / eigen) and how SuperLU is selected
  for batched RHS.
* :doc:`solid/mousavi2026` — a production-grade dataset suite
  (18 configurations) for training neural operators on solid-
  mechanics problems.
