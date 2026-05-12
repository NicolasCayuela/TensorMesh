tensormesh.element
==================

.. py:module:: tensormesh.element

Elements
--------

.. autoclass:: tensormesh.Element
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Line
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Triangle
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Tetrahedron
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Quadrilateral
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Hexahedron
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Prism
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.Pyramid
    :members:
    :show-inheritance:

Transformation
--------------
.. autoclass:: tensormesh.Transformation
    :members:
    :show-inheritance:
    :exclude-members: basis_order, element, elements, points, quadrature_order


Element type registry
---------------------

Lookup tables and helpers that map between element type strings
(``"triangle"``, ``"tetra10"``, …) and the corresponding element class,
spatial dimension, and polynomial order.

.. py:currentmodule:: tensormesh

.. autofunction:: tensormesh.element_type2element

.. py:data:: element_types
   :type: list[str]

   List of every element type string the library understands —
   first-order shapes (``"line"``, ``"triangle"``, ``"quad"``,
   ``"tetra"``, ``"hexahedron"``, ``"wedge"``, ``"pyramid"``) plus their
   higher-order counterparts (``"triangle6"``, ``"quad9"``,
   ``"tetra10"``, ``"hexahedron27"``, ``"triangle10"``, …).

.. py:data:: element_type2dimension
   :type: dict[str, int]

   Map from element type string to spatial dimension
   (``"line": 1``, ``"triangle": 2``, ``"tetra": 3``, …).

.. py:data:: element_type2order
   :type: dict[str, int]

   Map from element type string to polynomial order
   (``"triangle": 1``, ``"triangle6": 2``, ``"triangle10": 3``, …).

.. py:currentmodule:: tensormesh.element

Polynomial (advanced)
---------------------

.. note::

   :class:`~tensormesh.element.Polynomial` and
   :class:`~tensormesh.element.Polynomials` are the low-level building
   blocks used to construct shape functions for new element types. Most
   users should subclass :class:`~tensormesh.Element` and override its
   basis / quadrature hooks rather than calling these classes directly.
   The interface here is **less stable** than the rest of the public
   API and may evolve between releases.

.. autoclass:: tensormesh.element.Polynomial
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.element.Polynomials
    :members:
    :show-inheritance:
    :exclude-members: device, dtype, n_polys, n_terms, n_vars, shape


Internal evaluators
-------------------

The basis-, quadrature-, and normal-evaluation routines live under
``tensormesh/element/{basis,quadrature,normal}.py``. They are
implementation details of :class:`~tensormesh.Element` — the supported
extension path is to subclass ``Element`` and override its hooks rather
than calling these modules directly. They are **not** part of the
public API and may change between releases. If you need them, read the
source.
