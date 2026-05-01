Time Integration
================

Transient problems — heat, wave, transient elasticity — extend the
static FEM pipeline with a time-stepping loop. After assembling
mass and stiffness matrices once, you advance the solution in time
by repeatedly solving a linear (or nonlinear) update at each step.

TensorMesh supports two styles:

1. **Manual stepping**, where you assemble the time-stepped operator
   yourself and reuse the :class:`~tensormesh.Condenser` machinery
   inside a Python ``for`` loop. Lean and explicit, often the
   shortest path for a one-off problem.

2. **The integrator classes** in :mod:`tensormesh.ode`, where you
   plug ``M``, ``A``, ``B`` (or an explicit RHS) into a built-in
   Runge-Kutta scheme and call ``step(t, u, dt)``. Useful when you
   want to swap schemes or write a generic transient driver.

Both compose with autograd, so gradients flow through every step.


The integrators
---------------

:mod:`tensormesh.ode` ships three built-in schemes plus two
extensible base classes:

.. list-table::
   :header-rows: 1
   :widths: 28 8 22 42

   * - Class
     - Order
     - Form
     - Use case
   * - :class:`~tensormesh.ode.ExplicitEuler`
     - 1
     - :math:`\dot u = f(t, u)`
     - Wave-like, explicit dynamics, fast non-stiff problems.
   * - :class:`~tensormesh.ode.ImplicitLinearEuler`
     - 1
     - :math:`M\dot u = A u + B`
     - Heat / diffusion. Unconditionally stable.
   * - :class:`~tensormesh.ode.MidPointLinearEuler`
     - 2
     - :math:`M\dot u = A u + B`
     - Same family, second-order accurate (trapezoidal rule).
   * - :class:`~tensormesh.ode.ExplicitRungeKutta`
     - s-stage
     - :math:`\dot u = f(t, u)`
     - Base class — supply your own Butcher tableau ``(a, b)``.
   * - :class:`~tensormesh.ode.ImplicitLinearRungeKutta`
     - s-stage
     - :math:`M\dot u = A u + B`
     - Same, for linear-implicit schemes.

For the explicit family you override ``forward(t, u)`` to return
the RHS. For the linear-implicit family you override three methods
that return the operators at the current time:

.. code-block:: python

   class MyScheme(ImplicitLinearEuler):
       def forward_M(self, t):  return M_matrix          # SparseMatrix or scalar
       def forward_A(self, t):  return -K_matrix         # SparseMatrix or scalar
       def forward_B(self, t):  return zero_load          # Tensor or scalar

Each call to ``step(t0, u0, dt)`` returns the new ``u`` advanced by
``dt``. ``u`` must be 1D (``[D]``); if your problem has multiple
fields, flatten them before stepping.


Worked example: 2D heat equation
--------------------------------

Solve

.. math::

   \frac{\partial u}{\partial t} \;=\; \kappa\, \Delta u
   \quad\text{in } \Omega,
   \qquad u = 0 \text{ on } \partial\Omega,

with backward Euler. The semi-discrete weak form is :math:`M\dot u = -\kappa K u`,
which after one step gives

.. math::

   (M + \Delta t\,\kappa\, K)\, u^{n+1} \;=\; M u^n.

The full driver is a few lines once ``M`` and ``K`` are assembled:

.. code-block:: python

   import torch
   from tensormesh import (Mesh, ElementAssembler,
                           MassElementAssembler, LaplaceElementAssembler,
                           Condenser)

   mesh = Mesh.gen_rectangle(chara_length=0.02, order=2)
   M = MassElementAssembler.from_mesh(mesh)()
   K = LaplaceElementAssembler.from_mesh(mesh)()

   kappa = 1.0
   dt    = 5e-5

   # Build the time-stepped operator once and condense it once.
   A    = M + dt * kappa * K
   condenser = Condenser(mesh.boundary_mask)
   A_, _ = condenser(A, torch.zeros(mesh.n_points))

   # Initial condition.
   x, y = mesh.points[:, 0], mesh.points[:, 1]
   u    = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

   # Time stepping: each iteration is one mass-matrix-vector product,
   # one RHS condensation, one linear solve, and one recovery.
   snapshots = [u]
   for _ in range(100):
       b      = M @ u
       b_     = condenser.condense_rhs(b)
       u_in   = A_.solve(b_)
       u      = condenser.recover(u_in)
       snapshots.append(u)

   mesh.plot({"u(t)": snapshots}, save_path="heat.mp4", dt=dt)

The two ingredients that make this loop fast: assembling and
condensing ``A`` only once, and reusing the cached layout via
:meth:`Condenser.condense_rhs` per step.


The same problem via ``ImplicitLinearEuler``
--------------------------------------------

Wrapped in the integrator class, the per-step boilerplate becomes
plumbing inside the framework. You provide ``M``, ``A``, and ``B``,
and call :meth:`step`:

.. code-block:: python

   from tensormesh.ode import ImplicitLinearEuler

   class HeatEq(ImplicitLinearEuler):
       def __post_init__(self, M, K, kappa, condenser):
           self.M, self.K, self.kappa, self.condenser = M, K, kappa, condenser

       def forward_M(self, t): return self.M
       def forward_A(self, t): return -self.kappa * self.K
       def forward_B(self, t): return 0.0

       # Hooks: condense the system before solve, recover after.
       def pre_solve_lhs(self, K):  return self.condenser(K)[0]
       def pre_solve_rhs(self, f):  return self.condenser.condense_rhs(f)
       def post_solve(self, u):     return self.condenser.recover(u)

   integrator = HeatEq(M, K, kappa, condenser)
   for n in range(100):
       u = integrator.step(n * dt, u, dt)

The ``pre_solve_lhs`` / ``pre_solve_rhs`` / ``post_solve`` hooks are
exactly where Dirichlet BCs are applied inside the integrator.
Without these hooks the integrator solves the un-condensed system.

The wrapper is more code than the manual loop, but it pays off when
you want to swap to :class:`~tensormesh.ode.MidPointLinearEuler` or
a custom Butcher tableau without rewriting the driver.


Choosing a scheme
-----------------

A pragmatic guide:

* **Diffusive (heat, viscous flow)** → implicit. Use
  :class:`~tensormesh.ode.ImplicitLinearEuler` for robustness, or
  :class:`~tensormesh.ode.MidPointLinearEuler` when you need
  second-order accuracy and the system is reasonably linear.

* **Hyperbolic / explicit dynamics (wave, mass-spring)** →
  :class:`~tensormesh.ode.ExplicitEuler` is fine for a quick check,
  but the CFL limit is tight; for serious work, build a higher-order
  scheme via :class:`~tensormesh.ode.ExplicitRungeKutta` with a
  custom tableau.

* **Nonlinear transient (hyperelastic dynamics, plasticity)** →
  combine an implicit linear integrator with a Newton iteration per
  step using :func:`~tensormesh.sparse.nonlinear_solve`. See the
  :doc:`../example_gallery/index` for end-to-end recipes.


What's next
-----------

* :doc:`linear_solvers` — the solver behind every step.
* :doc:`differentiability` — backprop through a transient solve to
  optimize parameters or initial conditions.
* :doc:`../example_gallery/index` — heat, wave, and transient
  elasticity demos.
