Forms
=====

Solving a PDE in TensorMesh starts with two pieces of math:

.. math::

   a(u, v) \;=\; \int_{\Omega} \cdots \, \mathrm{d}\Omega
   \qquad
   l(v) \;=\; \int_{\Omega} \cdots \, \mathrm{d}\Omega

The bilinear form ``a(u, v)`` becomes a stiffness matrix; the linear
form ``l(v)`` becomes a load vector. In TensorMesh you write each
integrand as a single Python function and the library handles
quadrature, geometry, and the global scatter into a sparse matrix.

Three base classes cover the common cases:

* :class:`~tensormesh.ElementAssembler` — volume bilinear forms (matrices)
* :class:`~tensormesh.NodeAssembler` — volume linear forms (vectors)
* :class:`~tensormesh.FacetAssembler` — boundary integrals (matrices or vectors)


The weak-form contract
----------------------

Every assembler subclass overrides ``forward(...)``. The library
inspects the parameter *names* of your function and supplies a tensor
for each, evaluated at every quadrature point of every element:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Argument name
     - Provided by
     - Shape (per quadrature point, per element)
   * - ``u``, ``v``
     - the basis itself
     - ``[n_basis]``
   * - ``gradu``, ``gradv``
     - the basis gradient (in physical coordinates)
     - ``[n_basis, dim]``
   * - ``x``
     - the spatial coordinate (from ``points``)
     - ``[dim]``
   * - any key in ``point_data``
     - your tensor, interpolated to quadrature points
     - ``[...]`` (per-node trailing dims preserved)
   * - ``grad`` + key in ``point_data``
     - the gradient of that field
     - ``[..., dim]``
   * - any key in ``element_data``
     - per-element constant or quadrature-varying tensor
     - ``[...]``
   * - any key in ``scalar_data``
     - a global scalar
     - scalar

Inside ``forward`` you write the integrand as a normal tensor
expression — TensorMesh's vmap layer takes care of broadcasting it
across elements and quadrature points. You return the integrand;
the base class multiplies by the Jacobian × quadrature weight and
scatters the result into the global matrix or vector.

Construction is uniform: any ``Assembler`` subclass exposes
``Assembler.from_mesh(mesh, **kwargs)`` to build an instance, and the
instance is callable.


ElementAssembler — bilinear forms
---------------------------------

For ``a(u, v)``, write a ``forward(...)`` that returns the integrand.
Calling the assembler returns a :class:`~tensormesh.sparse.SparseMatrix`.

.. code-block:: python

   from tensormesh import Mesh, ElementAssembler

   class LaplaceAssembler(ElementAssembler):
       """a(u, v) = ∫ ∇u · ∇v dΩ"""
       def forward(self, gradu, gradv):
           return gradu @ gradv

   mesh = Mesh.gen_rectangle(chara_length=0.05)
   K = LaplaceAssembler.from_mesh(mesh)()      # SparseMatrix [n_points, n_points]

For vector-valued problems (e.g. linear elasticity in 2D, where each
node has 2 displacement DOFs), the integrand is a block matrix and
the assembled output is a block-CSR-style sparse matrix of shape
``[n_points * dof, n_points * dof]``. The library detects this from
the rank of your ``forward`` return.

Subclass parameters go into ``__post_init__``, which is called after
the base ``__init__`` wires up the projector and quadrature:

.. code-block:: python

   class ElasticityAssembler(ElementAssembler):
       def __post_init__(self, E=1.0, nu=0.3):
           self.E, self.nu = E, nu

       def forward(self, gradu, gradv):
           ...   # use self.E, self.nu

   K = ElasticityAssembler.from_mesh(mesh, E=210e9, nu=0.3)()

Pass per-node, per-element, or scalar parameters at call time:

.. code-block:: python

   kappa_field = torch.rand(mesh.n_points)             # spatially varying coef
   class WeightedLaplace(ElementAssembler):
       def forward(self, gradu, gradv, kappa):
           return kappa * (gradu @ gradv)

   K = WeightedLaplace.from_mesh(mesh)(point_data={"kappa": kappa_field})


NodeAssembler — linear forms
----------------------------

For ``l(v)``, the same pattern but the result is a vector:

.. code-block:: python

   from tensormesh import NodeAssembler

   class SourceAssembler(NodeAssembler):
       """l(v) = ∫ f v dΩ"""
       def forward(self, v, f):
           return f * v

   f_vals = torch.sin(math.pi * mesh.points[:, 0])
   b = SourceAssembler.from_mesh(mesh)(point_data={"f": f_vals})  # Tensor [n_points]

The simplest possible load — a constant body force ``∫ c·v dΩ`` —
ships pre-built; see ``const_node_assembler`` below.


FacetAssembler — boundary integrals
-----------------------------------

For surface terms — Neumann tractions, Robin BCs, penalty contact —
:class:`~tensormesh.FacetAssembler` runs the same dispatch over
facet quadrature instead of volume quadrature. The signature
contract is identical; the only difference is that ``x`` lives on
``∂Ω`` and an outward facet normal is available to subclasses that
need it. See the :doc:`../example_gallery/index` for a worked
penalty-contact example.


Built-in assemblers
-------------------

The most common forms are pre-written so you don't have to. All of
them are importable directly from ``tensormesh``:

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Class / factory
     - Kind
     - What it computes
   * - :class:`~tensormesh.LaplaceElementAssembler`
     - Element
     - ``∫ ∇u · ∇v dΩ`` — Laplacian / diffusion stiffness
   * - :class:`~tensormesh.MassElementAssembler`
     - Element
     - ``∫ u v dΩ`` — mass matrix (transient, L² projection)
   * - :class:`~tensormesh.LinearElasticityElementAssembler`
     - Element
     - small-strain isotropic elasticity (parameters ``E``, ``nu``)
   * - ``NeoHookeanModel``
     - Element (energy)
     - Neo-Hookean hyperelasticity strain energy density
   * - ``ContactAssembler``
     - Facet (energy)
     - base class for penalty / barrier contact and surface terms
   * - :func:`~tensormesh.const_node_assembler`
     - Node (factory)
     - ``∫ c v dΩ`` — uniform body force; returns a class
   * - :func:`~tensormesh.func_node_assembler`
     - Node (factory)
     - ``∫ f(x) v dΩ`` — spatially varying load; returns a class

Factory style for the node loads:

.. code-block:: python

   from tensormesh import const_node_assembler, func_node_assembler

   ConstantLoad = const_node_assembler(c=9.81)
   b = ConstantLoad.from_mesh(mesh)()

   SineSource = func_node_assembler(lambda x: torch.sin(math.pi * x[..., 0]))
   b = SineSource.from_mesh(mesh)()

The hyperelastic and plasticity classes (``NeoHookeanModel``,
``J2Plasticity``) are *energy*-based: they implement
``element_energy(...)`` and you call ``assembler.energy(point_data={...})``
to get a scalar potential whose gradient is the internal force
vector. See :doc:`../example_gallery/index` for end-to-end recipes.


A custom form: reaction-diffusion
---------------------------------

Combining stiffness and mass in one assembler:

.. math::

   a(u, v) \;=\; \int_{\Omega} \nabla u \cdot \nabla v \;+\; \kappa \, u v \, \mathrm{d}\Omega

is just two terms inside ``forward``:

.. code-block:: python

   class ReactionDiffusion(ElementAssembler):
       def __post_init__(self, kappa=1.0):
           self.kappa = kappa

       def forward(self, u, v, gradu, gradv):
           return gradu @ gradv + self.kappa * (u * v)

   K = ReactionDiffusion.from_mesh(mesh, kappa=2.5)()

Order of arguments doesn't matter — the dispatch is by name.


Memory batching with ``batch_size``
-----------------------------------

For very fine meshes or high-order elements, holding all per-element
quadrature tensors in memory at once may not fit. The assembler
``__call__`` accepts a ``batch_size`` argument that splits the
quadrature points into chunks, accumulates the contribution, and
finally returns the same matrix:

.. code-block:: python

   K = LaplaceAssembler.from_mesh(mesh)(batch_size=4)  # 4 quadrature points at a time

This is purely a memory knob; the assembled result is bit-identical
to the un-batched call. It is **not** problem-level vectorization —
for that, see :doc:`batched_workflows`.


What's next
-----------

* :doc:`boundary_conditions` — apply Dirichlet BCs to the matrix and
  vector you just assembled.
* :doc:`linear_solvers` — solve ``K x = b``.
* :doc:`differentiability` — backprop a loss through assembly and
  solve.
* :doc:`../example_gallery/index` — worked examples for elasticity,
  hyperelasticity, plasticity, and contact.
