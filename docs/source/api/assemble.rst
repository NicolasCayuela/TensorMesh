tensormesh.assemble
===================

.. contents:: contents
    :local:

Element Assembler
-----------------

.. autoclass:: tensormesh.ElementAssembler
    :members:
    :show-inheritance:
    :exclude-members: dimension, edges, element_types, elements, n_points, projector, transformation


Facet Assembler
---------------

.. autoclass:: tensormesh.FacetAssembler
    :members:
    :show-inheritance:
    :exclude-members: dimension, edges, element_types, elements, n_points, projector, transformation, facet_mask

Node Assembler
--------------

.. autoclass:: tensormesh.NodeAssembler
    :members:
    :show-inheritance:
    :exclude-members: dimension, edges, element_types, elements, n_points, projector, transformation


Built-in Assemblers
-------------------

.. autoclass:: tensormesh.LaplaceElementAssembler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.MassElementAssembler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.LinearElasticityElementAssembler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.assemble.NeoHookeanModel
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.assemble.builtin.J2Plasticity
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.assemble.ContactAssembler
    :members:
    :show-inheritance:

.. autofunction:: tensormesh.const_node_assembler

.. autofunction:: tensormesh.func_node_assembler
