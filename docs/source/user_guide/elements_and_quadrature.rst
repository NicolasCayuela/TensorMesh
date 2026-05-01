Elements and Quadrature
=======================

Every entry in :attr:`Mesh.cells` is keyed by an *element type string*
— ``"triangle"``, ``"hexahedron27"``, ``"wedge"``, and so on — that
maps to one of the seven reference shapes in :mod:`tensormesh.element`.
The element classes carry the basis-function definitions and
quadrature rules that the assembler uses internally; you rarely call
them directly, but understanding what they expose makes the rest of
the pipeline (assembly, IO, higher-order solves) less mysterious.


Supported element types
-----------------------

TensorMesh ships seven reference shapes covering 1D, 2D, and 3D, with
linear and higher-order variants for each:

.. list-table::
   :header-rows: 1
   :widths: 24 10 22 44

   * - Class
     - Dim
     - Linear type string
     - Higher-order type strings
   * - :class:`~tensormesh.Line`
     - 1D
     - ``"line"``
     - ``"line3"``, ``"line4"``, …, ``"line11"``
   * - :class:`~tensormesh.Triangle`
     - 2D
     - ``"triangle"``
     - ``"triangle6"``, ``"triangle10"``, …, ``"triangle66"``
   * - :class:`~tensormesh.Quadrilateral`
     - 2D
     - ``"quad"``
     - ``"quad8"``, ``"quad9"``, ``"quad16"``, …, ``"quad121"``
   * - :class:`~tensormesh.Tetrahedron`
     - 3D
     - ``"tetra"``
     - ``"tetra10"``, ``"tetra20"``, …, ``"tetra286"``
   * - :class:`~tensormesh.Hexahedron`
     - 3D
     - ``"hexahedron"``
     - ``"hexahedron20"``, ``"hexahedron27"``, ``"hexahedron64"``, …
   * - :class:`~tensormesh.Prism` (wedge)
     - 3D
     - ``"wedge"``
     - ``"wedge18"``, ``"wedge40"``, ``"wedge75"``, …
   * - :class:`~tensormesh.Pyramid`
     - 3D
     - ``"pyramid"``
     - ``"pyramid14"``

The mapping between strings and dimensions / orders is exposed as two
plain dicts:

.. code-block:: python

   from tensormesh import element_type2order, element_type2dimension
   element_type2order["triangle6"]      # 2
   element_type2dimension["hexahedron"] # 3

A reverse map ``element_type2element`` returns the class:

.. code-block:: python

   from tensormesh.element import element_type2element
   element_type2element("hexahedron27")  # tensormesh.Hexahedron

Type-string naming follows meshio: the lowercase shape (``triangle``)
plus the basis-node count for higher-order variants (``triangle6``,
``hexahedron27``). Note one inconsistency carried over from meshio —
the 3D triangular prism is the *class* :class:`~tensormesh.Prism` but
the *string* ``"wedge"``.


Higher-order elements
---------------------

Higher-order is a generator-time decision, not a different class.
Pass ``order=2`` (or ``3``, …) to a built-in mesh generator:

.. code-block:: python

   from tensormesh import Mesh

   tri  = Mesh.gen_rectangle(chara_length=0.1, order=1)  # "triangle"
   tri6 = Mesh.gen_rectangle(chara_length=0.1, order=2)  # "triangle6"
   hex27 = Mesh.gen_cube(chara_length=0.2, order=2)      # "hexahedron27"

The element type string in ``mesh.cells`` updates accordingly, and
every assembler downstream picks up the higher-order basis without
any further configuration.


The element interface
---------------------

Each class is a thin namespace of class methods. The four most
commonly useful:

.. code-block:: python

   from tensormesh import Triangle

   # Reference-element basis-node coordinates: [n_basis, dim]
   nodes = Triangle.get_basis(order=2)             # [6, 2]

   # Quadrature points and weights for a given polynomial order
   qpts, wts = Triangle.get_quadrature(order=2)    # qpts: [n_q, 2], wts: [n_q]

   # Facet element type — what shape are this element's faces?
   Triangle.get_facet_type()                       # tensormesh.Line

   # Permute connectivity between Gmsh/VTK and TensorMesh ordering
   conn_internal = Triangle.reorder(conn_gmsh, to_gmsh=False)
   conn_gmsh     = Triangle.reorder(conn_internal, to_gmsh=True)

For mixed-facet elements the :meth:`Element.get_facet_type` method
returns a tuple — :class:`Prism` returns ``(Triangle, Quadrilateral)``
and :class:`Pyramid` returns ``(Triangle, Quadrilateral)``, since
those bodies have faces of two kinds.

These methods are what :class:`~tensormesh.ElementAssembler` calls
internally during ``from_mesh``; users typically only see them when
writing custom assemblers that need a non-standard quadrature rule
or when implementing facet integrals directly.


The ordering convention
-----------------------

TensorMesh stores connectivity for tensor-product shapes
(:class:`~tensormesh.Quadrilateral`, :class:`~tensormesh.Hexahedron`)
in **lexicographic** order — the order that matches the underlying
tensor-product polynomial layout. Gmsh and VTK use a different
convention. The two are not interchangeable for higher-order or
tensor-product elements: feeding Gmsh-ordered hex27 connectivity
straight into the assembler produces silently-broken basis
evaluations.

The single conversion point is :meth:`Element.reorder`:

* ``Element.reorder(elements, to_gmsh=False)`` — Gmsh/VTK → TensorMesh
  (used when **reading** external meshes).
* ``Element.reorder(elements, to_gmsh=True)`` — TensorMesh → Gmsh/VTK
  (used when **writing** ``.vtk`` / ``.vtu`` files).

You almost never call :meth:`reorder` directly. The high-level paths
do it for you:

* The built-in mesh generators emit TensorMesh ordering already.
* :meth:`Mesh.read` and :meth:`Mesh.from_meshio` accept ``reorder=True``
  to convert on ingest.
* :meth:`Mesh.save` auto-detects ``.vtk``/``.vtu`` outputs and reorders
  for you.

If you build a mesh from raw arrays (``Mesh(meshio_obj, reorder=True)``)
the same flag controls the behavior — pass ``reorder=True`` when the
source uses Gmsh/VTK convention.


What's next
-----------

* :doc:`forms` — write a weak form against the basis tensors that
  these elements supply.
* :doc:`meshes` — generators, I/O, and the ``reorder=True`` calling
  pattern in context.
* :doc:`../example_gallery/index` — working examples on triangular,
  quad, tet, and hex meshes (including high-order).
