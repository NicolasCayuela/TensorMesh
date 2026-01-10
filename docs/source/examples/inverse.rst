Inverse Problems (Topology Optimization)
========================================

TensorMesh is built from the ground up to be **fully differentiable**. This makes it an ideal platform for solving inverse problems, such as **Topology Optimization**, where gradients of physical quantities (like compliance, stress, or heat flux) with respect to design variables (like shape or density) are required.

Instead of manually deriving and implementing the Adjoint Method, you can simply rely on PyTorch's ``autograd`` to compute sensitivities automatically.

SIMP Topology Optimization
--------------------------

**Solid Isotropic Material with Penalization (SIMP)** is the standard approach for density-based topology optimization. The goal is to distribute a limited amount of material in a domain to maximize stiffness (minimize compliance).

The following animation demonstrates the evolution of the material density distribution during the optimization process:

.. raw:: html

    <div align="center">
        <video width="700" controls autoplay loop muted>
            <source src="../_static/inverse/simp_animation.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <br>

Problem Formulation
^^^^^^^^^^^^^^^^^^^
Minimize compliance :math:`C(\rho)`:

.. math::
    \min_{\rho} \quad & C = \mathbf{U}^T \mathbf{K}(\rho) \mathbf{U} \\
    \text{subject to} \quad & \mathbf{K}(\rho)\mathbf{U} = \mathbf{F} \\
    & \sum \rho_e V_e \le V_{max} \\
    & 0 \le \rho_e \le 1

TensorMesh Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

1.  **Physics Definition**: Define the PDE (Linear Elasticity) and the material model (SIMP). The stiffness matrix :math:`\mathbf{K}` depends on density :math:`\rho` via :math:`E(\rho) = E_{min} + \rho^p (E_0 - E_{min})`.
2.  **Forward Pass**: Solve :math:`\mathbf{K}\mathbf{U} = \mathbf{F}`.
3.  **Objective**: Compute :math:`C = \mathbf{F}^T \mathbf{U}`.
4.  **Sensitivity**: Call ``C.backward()`` to get :math:`\partial C / \partial \rho`.
5.  **Optimization**: Update :math:`\rho` using the **Optimality Criteria (OC)** method (provided in ``tensormesh.optimizer.OCOptimizer``).

.. code-block:: python

    # Conceptual Code
    from tensormesh.optimizer import OCOptimizer

    # Initialize density (design variable)
    rho = torch.full((n_elements,), 0.5, requires_grad=True)
    optimizer = OCOptimizer([rho], vf=0.5, ...)

    for epoch in range(100):
        # 1. Assemble Global Stiffness K(rho)
        K = assemble_stiffness(mesh, rho)
        
        # 2. Solve Forward Problem
        u = solver(K, f)
        
        # 3. Compute Compliance
        compliance = (f * u).sum()
        
        # 4. Compute Gradients
        optimizer.zero_grad()
        compliance.backward()
        
        # 5. Update Design
        optimizer.step()

API Reference
-------------

.. autoclass:: tensormesh.optimizer.OCOptimizer
    :members:
