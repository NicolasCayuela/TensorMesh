Inverse Problems (Topology Optimization)
========================================

TensorMesh is built from the ground up to be **fully differentiable**. This makes it an ideal platform for solving inverse problems, such as **Topology Optimization**, where gradients of physical quantities (like compliance, stress, or heat flux) with respect to design variables (like shape or density) are required.

Instead of manually deriving and implementing the Adjoint Method, you can simply rely on PyTorch's ``autograd`` to compute sensitivities automatically.

SIMP Topology Optimization
--------------------------

**Solid Isotropic Material with Penalization (SIMP)** is the standard approach for density-based topology optimization. The goal is to distribute a limited amount of material in a domain to maximize stiffness (minimize compliance).

Problem Setup
^^^^^^^^^^^^^

Consider a 2D cantilever beam topology optimization problem:

.. image:: ../_static/inverse/boundary_conditions.png
   :width: 60%
   :align: center

**Geometry:**

- Domain: :math:`L_x \times L_y = 60 \times 30` (unit length)
- Mesh: :math:`60 \times 30` quadrilateral elements (QUAD4)
- Nodes: 1891
- Elements: 1800

**Boundary Conditions:**

- **Dirichlet (fixed)**: Left boundary :math:`x = 0`
  
  .. math::
      \mathbf{u} = \mathbf{0} \quad \text{on } \Gamma_D = \{(x, y) : x = 0\}

- **Neumann (load)**: Right-bottom corner with surface traction
  
  .. math::
      \mathbf{t} = [0, -100]^T \text{ N/m} \quad \text{on } \Gamma_N

**Material Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 15

   * - Parameter
     - Symbol
     - Value
     - Unit
   * - Max Young's Modulus
     - :math:`E_{max}`
     - 70,000
     - MPa
   * - Min Young's Modulus
     - :math:`E_{min}`
     - 70
     - MPa
   * - Poisson's Ratio
     - :math:`\nu`
     - 0.3
     - \-
   * - SIMP Penalty
     - :math:`p`
     - 3
     - \-
   * - Target Volume Fraction
     - :math:`\bar{v}`
     - 0.5
     - \-

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

**Objective Function (Compliance Minimization):**

.. math::
    \min_{\rho} \quad C(\rho) = \mathbf{u}^T \mathbf{K}(\rho) \mathbf{u}

**Subject to:**

.. math::
    \begin{aligned}
    & \mathbf{K}(\rho)\mathbf{u} = \mathbf{F} \quad \text{(equilibrium)} \\
    & \frac{1}{|\Omega|} \int_{\Omega} \rho \, d\Omega \leq \bar{v} \quad \text{(volume constraint)} \\
    & 0 < \rho_{min} \leq \rho \leq 1 \quad \text{(bounds)}
    \end{aligned}

**SIMP Material Interpolation:**

.. math::
    E(\rho) = E_{min} + \rho^p (E_{max} - E_{min})

where :math:`p = 3` is the penalty factor that promotes 0-1 solutions.

Sensitivity Analysis
^^^^^^^^^^^^^^^^^^^^

Using the adjoint method, the compliance gradient is:

.. math::
    \frac{\partial C}{\partial \rho_e} = -\mathbf{u}_e^T \frac{\partial \mathbf{K}^e}{\partial \rho_e} \mathbf{u}_e

For SIMP interpolation:

.. math::
    \frac{\partial C}{\partial \rho_e} = -p \rho_e^{p-1} (E_{max} - E_{min}) \, \mathbf{u}_e^T \mathbf{K}_0^e \mathbf{u}_e

With TensorMesh, you don't need to implement this manually - PyTorch's autograd handles it automatically!

TensorMesh Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from tensormesh import mesh, MeshGen
    from tensormesh.assemble import ElementAssembler
    from tensormesh.optimizer import OCOptimizer

    # 1. Create mesh
    msh = MeshGen.rectangle(nx=60, ny=30, order=1)
    
    # 2. Define SIMP stiffness assembler
    class SIMPStiffnessAssembler(ElementAssembler):
        def forward(self, gradu, gradv, rho):
            E = E_min + (rho ** penal) * (E_max - E_min)
            nu = 0.3
            D11 = E / (1.0 - nu * nu)
            D12 = nu * E / (1.0 - nu * nu)
            D33 = E / (2.0 * (1.0 + nu))
            
            # Plane stress stiffness
            gux, guy = gradu[..., 0], gradu[..., 1]
            gvx, gvy = gradv[..., 0], gradv[..., 1]
            
            K00 = D11 * gux * gvx + D33 * guy * gvy
            K01 = D12 * gux * gvy + D33 * guy * gvx
            K10 = D12 * guy * gvx + D33 * gux * gvy
            K11 = D11 * guy * gvy + D33 * gux * gvx
            
            return torch.stack([K00, K01, K10, K11], dim=-1)

    # 3. Initialize density
    rho = torch.full((n_elements,), 0.5, requires_grad=True)
    optimizer = OCOptimizer([rho], vf=0.5)

    # 4. Optimization loop
    for epoch in range(100):
        # Assemble stiffness matrix
        K = assembler(msh, rho)
        
        # Solve forward problem
        u = solver(K, f)
        
        # Compute compliance (objective)
        compliance = (f * u).sum()
        
        # Compute gradients via autograd
        optimizer.zero_grad()
        compliance.backward()
        
        # Update design using OC method
        optimizer.step()

Optimization Results
^^^^^^^^^^^^^^^^^^^^

**Optimization Progress:**

.. list-table::
   :widths: 16 42 42

   * - Iter
     - JAX-FEM
     - TensorMesh
   * - 0
     - .. image:: ../_static/inverse/jaxfem_frame_0.png
          :width: 100%
     - .. image:: ../_static/inverse/tensormesh_frame_0.png
          :width: 100%
   * - 10
     - .. image:: ../_static/inverse/jaxfem_frame_10.png
          :width: 100%
     - .. image:: ../_static/inverse/tensormesh_frame_10.png
          :width: 100%
   * - 25
     - .. image:: ../_static/inverse/jaxfem_frame_25.png
          :width: 100%
     - .. image:: ../_static/inverse/tensormesh_frame_25.png
          :width: 100%
   * - 50
     - .. image:: ../_static/inverse/jaxfem_frame_50.png
          :width: 100%
     - .. image:: ../_static/inverse/tensormesh_frame_50.png
          :width: 100%

**Final Result:**

.. image:: ../_static/inverse/final_comparison.png
   :width: 100%
   :align: center

**Convergence Comparison:**

.. image:: ../_static/inverse/convergence_comparison.png
   :width: 80%
   :align: center

Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Metric
     - JAX-FEM
     - TensorMesh
     - Speedup
   * - Setup time
     - 2.62 s
     - 0.58 s
     - **4.5×**
   * - Optimization (51 iters)
     - 28.51 s
     - 7.77 s
     - **3.7×**
   * - **Total time**
     - **31.13 s**
     - **8.35 s**
     - **3.7× 🚀**

.. list-table::
   :header-rows: 1
   :widths: 35 25 25 15

   * - Accuracy
     - JAX-FEM
     - TensorMesh
     - Status
   * - Initial Compliance
     - 53.71
     - 53.71
     - ✅ Match
   * - Final Compliance
     - 84.03
     - 83.75
     - ✅ 0.33% diff
   * - Volume Fraction
     - 0.500
     - 0.500
     - ✅ Match

API Reference
-------------

.. autoclass:: tensormesh.optimizer.OCOptimizer
    :members:
