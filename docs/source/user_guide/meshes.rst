Meshes
======

A :class:`~tensormesh.Mesh` is the geometric and topological foundation
of every problem in TensorMesh. It holds the points, the connectivity
of one or more element types, and any per-node or per-element data
attached to them. Because :class:`~tensormesh.Mesh` extends
:class:`torch.nn.Module`, you move it to a device with ``mesh.to("cuda")``,
serialize it with ``state_dict``, and let autograd track gradients
through anything that touches its tensors.


The Mesh data structure
-----------------------

Every mesh exposes the same six attributes:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Attribute
     - Type
     - Shape / contents
   * - ``points``
     - ``torch.Tensor``
     - ``[n_points, dim]`` — node coordinates.
   * - ``cells``
     - ``BufferDict[str, torch.Tensor]``
     - keyed by element type (e.g. ``"triangle"``, ``"hexahedron27"``);
       each value is ``[n_elements, n_basis]`` long-integer connectivity.
   * - ``point_data``
     - ``BufferDict[str, torch.Tensor]``
     - per-node fields. Keys starting with ``is_`` or ending in ``_mask``
       are auto-cast to ``bool`` on load (e.g. ``is_boundary``).
   * - ``cell_data``
     - ``ModuleDict[str, BufferDict[str, torch.Tensor]]``
     - per-element fields, nested by element type then field name.
   * - ``field_data``
     - ``BufferDict[str, torch.Tensor]``
     - mesh-global metadata (rare).
   * - ``cell_sets``
     - ``dict``
     - meshio-style named subsets, kept opaque on round-trip.

Because ``cells`` is a dict, mixed-element meshes (triangles + quads,
tets + hexes, …) are first-class. Iterating ``mesh.cells.items()``
gives you each element block in turn.

Useful properties: ``mesh.n_points``, ``mesh.n_elements``, ``mesh.dim``
(``= mesh.points.shape[1]``), ``mesh.dtype``, ``mesh.device``, and
``mesh.default_element_type`` (the highest-dimensional type, falling
back to a list for mixed meshes).


Built-in generators
-------------------

For domains with simple shapes, TensorMesh ships a Gmsh-backed
generator family. All return a :class:`~tensormesh.Mesh` ready to use
and accept ``chara_length`` (target element size) and ``order``
(polynomial order) as the two universal knobs:

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Generator
     - Default element
     - Domain
   * - :meth:`Mesh.gen_rectangle`
     - ``"tri"``
     - axis-aligned rectangle on ``[left, right] × [bottom, top]``
   * - :meth:`Mesh.gen_hollow_rectangle`
     - ``"quad"``
     - rectangle with rectangular hole
   * - :meth:`Mesh.gen_circle`
     - ``"tri"``
     - disk of radius ``r`` centered at ``(cx, cy)``
   * - :meth:`Mesh.gen_hollow_circle`
     - ``"quad"``
     - annulus
   * - :meth:`Mesh.gen_L`
     - ``"quad"``
     - L-shaped 2D domain
   * - :meth:`Mesh.gen_cube`
     - tet
     - axis-aligned 3D box
   * - :meth:`Mesh.gen_hollow_cube`
     - tet
     - cube with cubic hole
   * - :meth:`Mesh.gen_sphere`
     - tet
     - solid ball of radius ``r``
   * - :meth:`Mesh.gen_hollow_sphere`
     - tet
     - spherical shell

A typical call:

.. code-block:: python

   from tensormesh import Mesh

   mesh = Mesh.gen_rectangle(chara_length=0.05)
   print(mesh)
   # Mesh(
   #     points: torch.Size([441, 2])
   #     cells: triangle:torch.Size([800, 3])
   #     point_data: is_boundary(torch.bool):1
   #     cell_data:
   #     field_data:
   # )

Smaller ``chara_length`` → finer mesh (and finer means quadratically
more points in 2D, cubically in 3D). Use ``order=2`` to get
``triangle6`` / ``quad9`` / ``tetra10`` / ``hexahedron27`` instead of
the linear default.

For non-trivial geometries — CAD imports, complex CSG, named
boundaries — use Gmsh directly and load the result via meshio
(see :ref:`mesh-io` below).


Per-node and per-element data
-----------------------------

Attach a field to every node:

.. code-block:: python

   import torch
   u = torch.zeros(mesh.n_points)
   mesh.register_point_data("u", u)            # appears in mesh.point_data
   print(mesh.point_data["u"].shape)            # torch.Size([441])

The chained form ``mesh.register_point_data(...)`` returns the mesh,
which is convenient when building up a result before saving.

Per-element fields work the same way, keyed by element type:

.. code-block:: python

   energy = torch.zeros(mesh.n_elements)
   mesh.register_element_data("strain_energy", energy)
   # Equivalent to mesh.cell_data["triangle"]["strain_energy"] = energy

The lower-level :attr:`Mesh.cells`, :attr:`Mesh.point_data`,
:attr:`Mesh.cell_data` are full :class:`~tensormesh.nn.BufferDict`
objects — you can read them with ``[...]``, iterate them, or move
them with ``.to(device)``.


Boundary identification
-----------------------

The generators all populate ``point_data["is_boundary"]`` (a bool
tensor over points), which the convenience property exposes as:

.. code-block:: python

   mesh.boundary_mask        # bool tensor, shape [n_points]
   mesh.boundary_mask.sum()  # number of boundary nodes

Hand-rolled meshes can use either ``is_boundary`` or ``boundary_mask``
as the key — the property accepts both.

For multi-region BCs, derive your own masks from coordinates and
store them as additional ``point_data`` entries:

.. code-block:: python

   x, y = mesh.points[:, 0], mesh.points[:, 1]
   left   = (x == 0)
   right  = (x == 1)
   mesh.register_point_data("left_mask",  left)
   mesh.register_point_data("right_mask", right)

These plug straight into a :class:`~tensormesh.Condenser` for
non-homogeneous Dirichlet BCs (see :doc:`boundary_conditions`).


.. _mesh-io:

I/O — loading and saving
------------------------

TensorMesh round-trips through `meshio
<https://github.com/nschloe/meshio>`_, so any format meshio understands
(``.msh``, ``.vtk``, ``.vtu``, ``.xdmf``, ``.obj``, …) is fair game.

**Loading.** From a path:

.. code-block:: python

   mesh = Mesh.read("plate_with_hole.msh", reorder=True)

Or from an in-memory meshio object:

.. code-block:: python

   import meshio
   raw = meshio.read("plate_with_hole.msh")
   mesh = Mesh.from_meshio(raw, reorder=True)

The ``reorder=True`` flag is **required when ingesting Gmsh or VTK**
data: those formats use a different node-ordering convention for
quads, hexes, and high-order elements than TensorMesh's internal
lexicographic layout. Skipping it produces silently-broken
basis-function evaluations. The built-in generators already handle
this, so you only need ``reorder=True`` on external files.

**Saving.** Whatever format meshio writes:

.. code-block:: python

   mesh.register_point_data("u", u_solution)
   mesh.save("solution.vtu")

For ``.vtk`` and ``.vtu`` outputs, ``save`` automatically reorders
back to VTK convention and pads 2D coordinates to 3D — no flag
needed. The lower-level :meth:`Mesh.to_meshio` returns the meshio
object directly if you need custom write logic.


Inspecting and visualizing
--------------------------

A quick visual check of a 2D mesh and its solution is a one-liner:

.. code-block:: python

   mesh.plot({"u": u_solution}, save_path="u.png")

Pass a dict of ``{label: 1D tensor}`` for static side-by-side panels;
pass ``{label: list_of_tensors}`` to render an MP4/GIF animation
(needs ``pyvista``). For a deep dive on visualization, including 3D
deformation plots, see the :doc:`../example_gallery/index`.


What's next
-----------

* :doc:`elements_and_quadrature` — the reference shapes, basis
  evaluation, and the ordering convention behind ``reorder=True``.
* :doc:`forms` — turn a mesh into a stiffness matrix or load vector
  via the assembler base classes.
* :doc:`../getting_started/quickstart` — a complete worked Poisson
  problem that uses everything on this page.
