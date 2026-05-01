Boundary Conditions
===================

Once you have a stiffness matrix ``K`` and a load vector ``b`` from
:doc:`forms`, you still need to enforce the problem's boundary
conditions. In TensorMesh today this is done via *static condensation*
of Dirichlet DOFs — a single call to :class:`~tensormesh.Condenser`
takes a full system and returns a reduced system on the interior
DOFs, ready to solve.

Neumann conditions don't need a separate operator: they appear
naturally in the weak form as a boundary integral, assembled with
:class:`~tensormesh.FacetAssembler`.


Why static condensation
-----------------------

Splitting the DOFs into "inner" (free) and "outer" (Dirichlet-fixed)
turns the linear system

.. math::

   \begin{bmatrix}
       K_{ii} & K_{io} \\
       K_{oi} & K_{oo}
   \end{bmatrix}
   \begin{bmatrix} u_i \\ u_o \end{bmatrix}
   =
   \begin{bmatrix} b_i \\ b_o \end{bmatrix}

into a smaller solve on just the interior block:

.. math::

   K_{ii} \, u_i \;=\; b_i - K_{io} \, u_o,
   \qquad u_o \text{ prescribed.}

Condensation avoids Lagrange multipliers, keeps an SPD operator SPD,
and is differentiable end-to-end — the boundary values flow through
the right-hand-side correction and pick up gradients in the usual way.


The Condenser API
-----------------

Construct a :class:`~tensormesh.Condenser` from a boolean mask over
all DOFs:

.. code-block:: python

   from tensormesh import Condenser

   condenser = Condenser(mesh.boundary_mask)              # zero values (default)
   condenser = Condenser(mesh.boundary_mask, values)      # prescribed values

* ``dirichlet_mask: [n_dof]`` — bool tensor; ``True`` where the DOF is fixed.
* ``dirichlet_value`` — optional 1D tensor giving the prescribed values.
  May be ``[n_dof]`` (the constructor slices it down to the boundary
  entries) or already-sliced ``[n_outer_dof]``. Defaults to zeros.

Three methods cover the common workflow:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - What it does
   * - ``condenser(K, b)``
     - Condense both at once. Returns ``(K_inner, b_inner)``. **Run this first** — it caches the layout for subsequent calls.
   * - ``condenser.condense_rhs(b_new)``
     - Re-condense only the right-hand side, reusing the cached layout. Useful when ``K`` is fixed but ``b`` changes (transient problems, batched RHS).
   * - ``condenser.recover(u_inner)``
     - Glue the interior solution and the prescribed boundary values back together into a full ``[n_dof]`` vector.

For time-dependent BCs, ``condenser.update_dirichlet(new_values)``
swaps in new boundary values without rebuilding the layout.

The first ``condenser(K, b)`` call computes and caches the inner /
outer index split for ``K``'s sparsity pattern; subsequent calls on
matrices with the same layout reuse that work.


Homogeneous Dirichlet
---------------------

The bread-and-butter case — zero values on the boundary — is exactly
what the :doc:`../getting_started/quickstart` does:

.. code-block:: python

   condenser = Condenser(mesh.boundary_mask)
   K_, b_ = condenser(K, b)
   u_inner = K_.solve(b_)
   u = condenser.recover(u_inner)

Three lines, including the solve. The recovered ``u`` has zeros at
boundary DOFs and the FEM solution everywhere else.


Non-homogeneous Dirichlet
-------------------------

Prescribe per-DOF values by passing them to the constructor. Two
shapes are accepted — choose whichever is more convenient:

.. code-block:: python

   # Option A: full-length [n_dof] vector (Condenser slices internally)
   values = torch.zeros(mesh.n_points)
   x, y = mesh.points[:, 0], mesh.points[:, 1]
   values[(x == 1) & mesh.boundary_mask] = 1.0       # u = 1 on right edge
   condenser = Condenser(mesh.boundary_mask, values)

   # Option B: pre-sliced [n_outer_dof] vector
   sliced = values[mesh.boundary_mask]
   condenser = Condenser(mesh.boundary_mask, sliced)

The condensed RHS picks up the term :math:`-K_{io} u_o` automatically,
so the rest of the pipeline is unchanged.


Time-varying boundary values
----------------------------

Build the condenser once, then push new values per timestep without
recomputing the layout:

.. code-block:: python

   condenser = Condenser(mesh.boundary_mask, initial_values)
   K_, _ = condenser(K, torch.zeros(mesh.n_points))    # caches the split

   for t in time_steps:
       condenser.update_dirichlet(values_at(t))         # cheap
       b_new = build_rhs_for(t)
       b_inner = condenser.condense_rhs(b_new)          # reuses K_'s layout
       u = condenser.recover(K_.solve(b_inner))


Mixed regions
-------------

For separate Dirichlet, Neumann, and traction regions, build masks
from coordinates and combine them as needed:

.. code-block:: python

   x, y = mesh.points[:, 0], mesh.points[:, 1]
   left_mask  = (x == 0) & mesh.boundary_mask
   right_mask = (x == 1) & mesh.boundary_mask

   # Fix left edge to zero, prescribe ramp on right edge
   dirichlet_mask = left_mask | right_mask
   values = torch.zeros(mesh.n_points)
   values[right_mask] = 1.0
   condenser = Condenser(dirichlet_mask, values)


Neumann (natural) conditions
----------------------------

Non-zero traction or flux on a portion of :math:`\partial\Omega` is a
boundary integral in the weak form's right-hand side:

.. math::

   l(v) \;=\; \int_{\Omega} f \cdot v \, \mathrm{d}\Omega
   \;+\; \int_{\Gamma_N} g \cdot v \, \mathrm{d}S

Assemble that surface integral with a :class:`~tensormesh.FacetAssembler`
subclass and add the result to your load vector before solving. No
Condenser involvement — natural BCs are handled entirely by the
weak form.


What's not built in today
-------------------------

The :mod:`tensormesh.operator` module ships exactly one operator —
:class:`~tensormesh.Condenser` for Dirichlet via static condensation.
A few neighborhood techniques are *not* first-class operators:

* **Lagrange multipliers** for Dirichlet — alternative to condensation
  that keeps the original DOF layout. Workable by hand if you build
  the saddle-point system yourself, but no helper class.
* **Periodic boundary conditions** — typically wired up by identifying
  matched DOFs and merging rows/columns in the assembled matrix.
* **Contact** — the ``ContactAssembler`` base in :mod:`tensormesh.assemble`
  provides penalty-based contact surface integrals. See the example
  gallery for full recipes.

If you need any of these, the example gallery has worked patterns,
and they all compose with the rest of the FEM pipeline.


What's next
-----------

* :doc:`linear_solvers` — solve the condensed system with the right
  backend.
* :doc:`time_integration` — transient problems where the same
  condenser is reused per step.
* :doc:`../getting_started/quickstart` — the full pipeline including
  homogeneous BCs.
