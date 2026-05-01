Nonlinear Poisson Equation
==========================

This example solves a nonlinear Poisson equation with a cubic reaction term using the Newton-Raphson method and implicit differentiation.

Problem Statement
-----------------

We solve the following nonlinear PDE on a rectangular domain :math:`\Omega`:

.. math::

    -\Delta u + u^3 = f \quad \text{in } \Omega

with zero Dirichlet boundary conditions :math:`u = 0` on :math:`\partial \Omega`.

The problem is discretized using finite elements. The nonlinear system of algebraic equations :math:`F(\mathbf{u}) = 0` is solved using the Newton-Raphson method. The gradients of the solution with respect to parameters (e.g., the source term :math:`f`) are computed using implicit differentiation, avoiding the need to backpropagate through the iterative solver.

Implementation
--------------

.. code-block:: python

    import torch
    import numpy as np
    import os
    from tensormesh import Mesh, LaplaceElementAssembler, ElementAssembler, NodeAssembler, Condenser
    from tensormesh.sparse import nonlinear_solve

    # 1. Define Assemblers
    class CubicTermAssembler(NodeAssembler):
        """Assembles the nonlinear term ∫ u^3 v dx"""
        def forward(self, u, v):
            return u**3 * v

    class WeightedMassAssembler(ElementAssembler):
        """Assembles the mass matrix weighted by w: ∫ w u v dx"""
        def forward(self, u, v, w):
            return w * u * v

    class SourceAssembler(NodeAssembler):
        """Assembles the source term ∫ f v dx"""
        def forward(self, v):
            return 1.0 * v

    def run():
        # 2. Generate Mesh
        mesh = Mesh.gen_rectangle(chara_length=0.05, element_type="tri")
        print(f"Mesh generated with {mesh.n_points} points.")
        
        # 3. Initialize Assemblers
        laplace_asm = LaplaceElementAssembler.from_mesh(mesh)
        cubic_asm = CubicTermAssembler.from_mesh(mesh)
        weighted_mass_asm = WeightedMassAssembler.from_mesh(mesh)
        source_asm = SourceAssembler.from_mesh(mesh)
        
        # 4. Setup Problem
        # Boundary Conditions
        condenser = Condenser(mesh.boundary_mask, torch.zeros(mesh.boundary_mask.sum()))
        
        # Pre-assemble constant parts
        K = laplace_asm(mesh.points)
        f_load = source_asm(mesh.points)
        
        # Initialize condenser layout
        condenser._compute_layout(K)
        
        # Initial guess (inner DOFs)
        n_inner = condenser.n_inner_dof
        u0_inner = torch.zeros(n_inner, device=mesh.device, dtype=mesh.dtype)
        
        # Parameter (source term) that we might want to differentiate with respect to
        f_param = f_load.clone().requires_grad_(True)
        
        # 5. Define Residual and Jacobian Functions
        
        def func_F(u_inner, f_vec):
            """
            Compute residual: F(u) = K u + M(u^3) - f
            """
            # Recover full solution vector
            u_full = condenser.recover(u_inner)
            
            # Linear term: K u
            term1 = K @ u_full
            
            # Nonlinear term: ∫ u^3 v
            term2 = cubic_asm(point_data={"u": u_full})
            
            # Combine
            res_full = term1 + term2 - f_vec
            
            # Return residual for inner DOFs
            return res_full[condenser.is_inner_dof]

        def func_J(u_inner, f_vec):
            """
            Compute Jacobian: J = K + M(3u^2)
            """
            u_full = condenser.recover(u_inner)
            
            # Jacobian of cubic term is mass matrix weighted by 3u^2
            w = 3 * u_full**2
            M_w = weighted_mass_asm(point_data={"w": w})
            
            J_full = K + M_w
            
            # Condense Jacobian
            J_inner, _ = condenser(J_full)
            return J_inner

        # 6. Solve using Newton-Raphson with Implicit Differentiation
        print("Starting non-linear solve...")
        u_inner = nonlinear_solve(func_F, func_J, u0_inner, (f_param,), tol=1e-6, verbose=True)
        
        # Recover full solution
        u_final = condenser.recover(u_inner)
        
        # 7. Compute Gradient
        # We can compute gradients of any scalar loss w.r.t parameters
        loss = u_inner.sum()
        loss.backward()
        print("Gradient norm w.r.t source:", f_param.grad.norm().item())

    if __name__ == "__main__":
        run()

Results
-------

The solver converges quadratically, and gradients can be computed efficiently without unrolling the optimization loop.

.. image:: ../_static/nonlinear_poisson.png
   :width: 80%
   :align: center
   :alt: Nonlinear Poisson Solution

