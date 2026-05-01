Differentiability
=================

Every component along the ``Mesh → Assembler → SparseMatrix → Condenser → Solve``
pipeline is a :class:`torch.nn.Module` or a custom
:class:`torch.autograd.Function`, and the linear solve at the end
has an analytic adjoint backward. As a result, a loss computed on
the FEM solution can be backpropagated all the way to *anything*
that touched the pipeline: a material coefficient at every node,
a Dirichlet value, a neural network's prediction of a stiffness
modifier — without writing any gradient code yourself.

This page explains how the gradient flow works, what the cost is,
and the two canonical workflows (parameter identification and
topology optimization).


How it works
------------

Three pieces of TensorMesh are wired into the autograd graph:

* :class:`~tensormesh.Mesh` extends :class:`torch.nn.Module`. Its
  ``points`` and per-node fields are tensors that can carry
  ``requires_grad=True``. Moving the mesh with ``mesh.to(device)``
  and saving with ``state_dict`` work as you'd expect.
* :class:`~tensormesh.ElementAssembler` and
  :class:`~tensormesh.NodeAssembler` are also ``nn.Module`` s. Any
  parameter that flows into a ``forward(...)`` integrand becomes a
  graph input to the assembled matrix or vector.
* :meth:`tensormesh.sparse.SparseMatrix.solve` is implemented as a
  :class:`torch.autograd.Function` with a custom ``backward``. The
  backward solves the *adjoint* system

  .. math::

     A^{T}\, \boldsymbol{\lambda} \;=\; \frac{\partial L}{\partial u},

  then computes the gradient of any nonzero matrix entry as
  :math:`\partial L / \partial A_{ij} = -\lambda_i\, u_j` and the
  gradient of the right-hand side as
  :math:`\partial L / \partial b = \boldsymbol{\lambda}`.

The whole forward call ``loss = criterion(K.solve(b), target).sum()``
followed by ``loss.backward()`` therefore gives correct gradients
for every leaf tensor that fed into ``K``, ``b``, or the loss —
matrix entries, source terms, prescribed boundary values, and any
upstream parameter (NN weights, design variables, …).


Adjoint cost
------------

Backprop through one linear solve costs **one additional sparse
solve** (with the transposed system). For SPD problems
:math:`A^T = A`, so the backward uses the same factorization or
preconditioner pattern as the forward. The total per-iteration
cost in a gradient-based optimizer is therefore roughly
``2 × forward_solve_cost`` regardless of how many degrees of
freedom or how complex the assembly logic was.


What's differentiable, what isn't
---------------------------------

**Differentiable through ``solve`` today:**

* Matrix entries (``edata``) — gradients land on the upstream
  tensor whose values fed each non-zero.
* The right-hand side ``b``.
* Any tensor in ``point_data`` / ``element_data`` / ``scalar_data``
  passed to an assembler — gradients flow through the integrand.
* Dirichlet values (via the condensed RHS contribution).

**Backend caveats:**

* The ``torch-sla`` ``backend="pytorch"`` and ``backend="auto"``
  paths are the safest defaults for autograd. They route the
  forward through a pure-PyTorch iterative solver and use the
  analytic adjoint for backward.
* The SciPy / Eigen / cuDSS / CuPy backends are wrapped by the same
  custom ``autograd.Function``, so gradients still flow — but the
  forward computation lives in NumPy/CuPy/C++ and detaches from the
  graph for that span. This is correct for the linear solve but
  means you cannot "see into" the solver from autograd.
* The legacy fallback paths in :mod:`tensormesh.sparse` (used when
  ``torch-sla`` is not installed) implement the same custom
  ``autograd.Function`` with adjoint backward, so gradients work
  there too.

When in doubt for a research workflow involving autodiff, install
``torch-sla`` and stick with ``backend="auto"``.


Worked example: parameter identification
----------------------------------------

Suppose we observe a "ground truth" Poisson solution and want to
recover the unknown coefficient field :math:`\kappa(x)` at every
mesh node by gradient descent:

.. math::

   -\nabla \cdot (\kappa(x)\, \nabla u) \;=\; f \quad \text{in } \Omega,
   \qquad u = 0 \text{ on } \partial\Omega.

The forward map is "FEM-solve given :math:`\kappa`"; the loss is
the L² distance between the FEM solution and the observation; the
gradient is computed by autograd; SGD updates :math:`\kappa`.

.. code-block:: python

   import torch
   from tensormesh import (Mesh, ElementAssembler,
                           NodeAssembler, Condenser)

   mesh = Mesh.gen_rectangle(chara_length=0.05)
   condenser = Condenser(mesh.boundary_mask)

   class WeightedLaplace(ElementAssembler):
       def forward(self, gradu, gradv, kappa):
           return kappa * (gradu @ gradv)

   class Source(NodeAssembler):
       def forward(self, v, f):
           return f * v

   def fem_solve(kappa, f_vals):
       K = WeightedLaplace.from_mesh(mesh)(point_data={"kappa": kappa})
       b = Source.from_mesh(mesh)(point_data={"f": f_vals})
       K_, b_ = condenser(K, b)
       return condenser.recover(K_.solve(b_))

   # Synthetic ground-truth data.
   torch.manual_seed(0)
   kappa_true = 1.0 + 0.5 * torch.sin(3 * torch.pi * mesh.points[:, 0])
   f_vals     = torch.ones(mesh.n_points)
   with torch.no_grad():
       u_obs = fem_solve(kappa_true, f_vals)

   # Recover kappa from u_obs by gradient descent.
   kappa = torch.ones(mesh.n_points, requires_grad=True)
   optim = torch.optim.Adam([kappa], lr=5e-2)

   for step in range(200):
       optim.zero_grad()
       u    = fem_solve(kappa, f_vals)
       loss = ((u - u_obs) ** 2).sum()
       loss.backward()
       optim.step()
       if step % 20 == 0:
           rel_err = (kappa - kappa_true).abs().max() / kappa_true.abs().max()
           print(f"step {step:3d}  loss={loss.item():.3e}  max_rel_err={rel_err:.3e}")

A few hundred Adam steps recover ``kappa_true`` to within a few
percent. The whole gradient computation goes through the
``WeightedLaplace`` assembler and the linear solve via the adjoint
backward — you never write a sensitivity equation by hand.


Worked example: topology optimization
-------------------------------------

For density-based topology optimization (compliance minimization
under a volume constraint), TensorMesh ships a dedicated
:class:`~tensormesh.optimizer.OCOptimizer` that implements the
classical Optimality Criteria update:

.. code-block:: python

   from tensormesh.optimizer import OCOptimizer

   rho = torch.full((mesh.n_elements,), 0.5, requires_grad=True)
   optimizer = OCOptimizer(rho, vf=0.5, move_limit=0.2)

   for step in range(80):
       compliance = compute_compliance(rho)        # FEM solve + b @ u
       compliance.backward()
       optimizer.step()                            # OC update + bisection
       optimizer.zero_grad()

The OC step uses the gradient just computed by autograd — no
finite differences, no by-hand adjoint code. The full driver
(SIMP penalization, density filter, intermediate plots) is in the
:doc:`../example_gallery/index`.


Wiring a neural network in
--------------------------

Three patterns cover the common ML/PDE couplings:

* **NN predicts a coefficient field.** The NN ingests coordinates
  (or some features) and outputs a per-node value — feed the
  output to an assembler via ``point_data={"kappa": nn_out}``.
  Gradients flow from the FEM loss back through the NN.
* **NN predicts boundary values.** Pass the output as
  ``dirichlet_value`` to a freshly-built
  :class:`~tensormesh.Condenser`. The condensation contribution
  to the RHS keeps the gradients connected.
* **NN predicts per-element stiffness modifiers.** Stack the
  output into a per-element tensor and pass it via
  ``element_data={"alpha": nn_out}``; let the assembler's
  ``forward`` use it inside the integrand.

In all three, the NN is a regular ``nn.Module`` and the FEM
pipeline is a regular sequence of ``nn.Module`` calls; standard
``torch.optim`` optimizers work without any special hooks.


What's next
-----------

* :doc:`linear_solvers` — the autograd-aware solver behind
  ``SparseMatrix.solve``.
* :doc:`batched_workflows` — backprop through a batched pipeline
  for ML training.
* :doc:`../example_gallery/index` — full inverse problem and
  topology-optimization recipes.
* :doc:`../api/index` — the ``solve`` autograd function and the
  ``OCOptimizer`` reference.
