Solid Mechanics Examples
========================

This section demonstrates various solid mechanics problems solved using TensorMesh, 
including linear elasticity, hyperelasticity, plasticity, and contact mechanics.


Cantilever Beam Deformation (Linear Elasticity)
------------------------------------------------

.. figure:: ../_static/solid_mechanics/cantilever_steel.png
   :width: 100%
   :align: center
   :alt: Cantilever Beam Deformation

**Physics**

Linear elasticity is governed by the balance of linear momentum:

.. math::

    \nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = \mathbf{0}

where :math:`\boldsymbol{\sigma}` is the Cauchy stress tensor and :math:`\mathbf{f}` is the body force.

The constitutive relation (Hooke's Law) is:

.. math::

    \boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}

where :math:`\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)` is the infinitesimal strain tensor.

For isotropic materials:

.. math::

    \boldsymbol{\sigma} = \lambda \, \text{tr}(\boldsymbol{\varepsilon}) \mathbf{I} + 2\mu \boldsymbol{\varepsilon}

where :math:`\lambda` and :math:`\mu` are Lamé parameters related to Young's modulus :math:`E` and Poisson's ratio :math:`\nu`.

**Code**

.. literalinclude:: ../../../examples/solid mechanics/cantilever_beam.py
   :language: python


Hyperelastic Beam (Neo-Hookean)
-------------------------------

.. figure:: ../_static/solid_mechanics/hyperelastic_rubber.png
   :width: 100%
   :align: center
   :alt: Hyperelastic Rubber Beam under Torsion

**Physics**

Hyperelasticity models large deformations where the stress-strain relationship is nonlinear.
The Neo-Hookean model is a fundamental hyperelastic constitutive law.

The deformation gradient is defined as:

.. math::

    \mathbf{F} = \mathbf{I} + \nabla \mathbf{u}

The strain energy density function for compressible Neo-Hookean material is:

.. math::

    \Psi = \frac{\mu}{2}(I_1 - 3) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2

where:

- :math:`I_1 = \text{tr}(\mathbf{F}^T \mathbf{F})` is the first invariant of the right Cauchy-Green tensor
- :math:`J = \det(\mathbf{F})` is the Jacobian (volume ratio)
- :math:`\mu, \lambda` are Lamé parameters

The total potential energy is minimized:

.. math::

    \Pi(\mathbf{u}) = \int_\Omega \Psi(\mathbf{F}) \, dV - \int_\Omega \mathbf{f} \cdot \mathbf{u} \, dV

**Code**

.. literalinclude:: ../../../examples/solid mechanics/hyperelastic_beam.py
   :language: python


3D Plasticity (J2 Flow Theory)
------------------------------

<div style="text-align: center;">

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/solid_mechanics/plasticity_3d_combined.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

</div>

**Physics**

J2 plasticity (von Mises plasticity) is the most common model for metal plasticity.

The yield function is:

.. math::

    f(\boldsymbol{\sigma}, \alpha) = \sqrt{\frac{3}{2} \mathbf{s} : \mathbf{s}} - (\sigma_y + H \alpha) = 0

where:

- :math:`\mathbf{s} = \boldsymbol{\sigma} - \frac{1}{3}\text{tr}(\boldsymbol{\sigma})\mathbf{I}` is the deviatoric stress
- :math:`\sigma_y` is the initial yield stress
- :math:`H` is the hardening modulus
- :math:`\alpha` is the equivalent plastic strain (internal variable)

The plastic strain evolution follows the associative flow rule:

.. math::

    \dot{\boldsymbol{\varepsilon}}^p = \dot{\gamma} \frac{\partial f}{\partial \boldsymbol{\sigma}} = \dot{\gamma} \sqrt{\frac{3}{2}} \frac{\mathbf{s}}{||\mathbf{s}||}

The simulation demonstrates loading and unloading cycles, showing the characteristic 
hysteresis loop of elasto-plastic materials.

**Code**

.. literalinclude:: ../../../examples/solid mechanics/plasticity_3d.py
   :language: python


Hertzian Contact (Circle on Block)
----------------------------------

.. figure:: ../_static/solid_mechanics/hertzian_contact.png
   :width: 100%
   :align: center
   :alt: Hertzian Contact Stress Distribution

**Physics**

The Hertzian contact stress problem describes the localized stresses that develop as two curved surfaces come in contact and deform slightly under the imposed loads. This example simulates a circular indenter (Slave) pressed against a flat rectangular block (Master).

The contact condition is enforced using the **Point-to-Segment** penalty method. For every slave node :math:`\mathbf{x}_s` on the indenter surface, we find the closest point :math:`\mathbf{x}_c` on the master surface segments.

The gap function is defined as:

.. math::

    g = (\mathbf{x}_s - \mathbf{x}_c) \cdot \mathbf{n}

where :math:`\mathbf{n}` is the normal vector of the master segment. The penalty energy is:

.. math::

    E_{\text{contact}} = \frac{1}{2} k_p \sum_{g < 0} g^2

where :math:`k_p` is a large penalty parameter to enforce non-penetration.

**Code**

.. literalinclude:: ../../../examples/solid mechanics/hertzian_contact.py
   :language: python
