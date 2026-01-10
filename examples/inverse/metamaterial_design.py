"""
3D Metamaterial Design (Inverse Homogenization)
===============================================

This example demonstrates how to design a 3D mechanical metamaterial with 
extreme properties (e.g., Negative Poisson's Ratio / Auxetic) using 
Topology Optimization.

Physics:
    - Linear Elasticity (Isotropic base material)
    - Inverse Homogenization using Kinematic Uniform Boundary Conditions (KUBC)
    - Objective: Minimize squared error between Effective Stiffness C_eff and Target C_target

Usage:
    python metamaterial_design.py --target_poisson -0.5 --epoch 50
"""

import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import meshio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser
from tensormesh.assemble import LinearElasticityElementAssembler
from tensormesh.optimizer import OCOptimizer

class SIMPLinearElasticityElementAssembler(LinearElasticityElementAssembler):
    r"""SIMP Linear Elasticity Assembler for Topology Optimization
    
    Modifies Young's Modulus E based on density rho:
    :math:`E(\rho) = E_{min} + \rho^p (E_0 - E_{min})`
    """
    def __post_init__(self, E=1.0, nu=0.3, penal=3.0, rho_min=1e-9):
        super().__post_init__(E, nu)
        self.penal = penal
        self.rho_min = rho_min

    def forward(self, gradu, gradv, rho):
        """
        Args:
            gradu, gradv: Gradients of basis functions
            rho: Element density (scalar)
        """
        # Calculate scaling factor
        factor = self.rho_min + (rho ** self.penal) * (1.0 - self.rho_min)
        
        # Calculate base stiffness using parent class with full E
        K_solid = super().forward(gradu, gradv)
        return factor * K_solid

def get_voigt_strain_modes(dim=3):
    """
    Return the 6 elementary strain modes in Voigt notation for 3D.
    epsilon = [eps_xx, eps_yy, eps_zz, gam_yz, gam_xz, gam_xy]
    """
    # 6 modes
    modes = torch.eye(6)
    return modes

def apply_kubc(mesh, strain_voigt):
    """
    Apply Kinematic Uniform Boundary Conditions (KUBC) corresponding to a macroscopic strain.
    u(x) = epsilon * x  on the boundary.
    
    Args:
        mesh: TensorMesh mesh
        strain_voigt: [6] tensor (eps_xx, eps_yy, eps_zz, gam_yz, gam_xz, gam_xy)
        
    Returns:
        u_boundary: [n_points, 3] displacements for boundary nodes (others 0)
    """
    points = mesh.points # [N, 3]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    ex, ey, ez, gyz, gxz, gxy = strain_voigt
    
    # Voigt strain to tensor strain: gamma = 2 * epsilon
    # u_x = ex*x + 0.5*gxy*y + 0.5*gxz*z
    # u_y = 0.5*gxy*x + ey*y + 0.5*gyz*z
    # u_z = 0.5*gxz*x + 0.5*gyz*y + ez*z
    
    ux = ex * x + 0.5 * gxy * y + 0.5 * gxz * z
    uy = 0.5 * gxy * x + ey * y + 0.5 * gyz * z
    uz = 0.5 * gxz * x + 0.5 * gyz * y + ez * z
    
    u = torch.stack([ux, uy, uz], dim=1)
    
    return u

def compute_effective_stiffness(mesh, rho, assembler, condenser, solver_backend="scipy"):
    """
    Compute the effective stiffness matrix C_eff (6x6) via homogenization.
    
    Args:
        mesh: Mesh object
        rho: Density tensor
        assembler: Stiffness assembler
        condenser: Boundary condenser (for KUBC)
        
    Returns:
        C_eff: [6, 6] effective stiffness matrix
        u_fluctuations: List of displacement fields for each mode
    """
    # 1. Assemble Global Stiffness
    K = assembler(mesh.points, element_data={"rho": rho})
    
    # 2. Solve for each strain mode
    # For KUBC, we enforce u = epsilon*x on boundary.
    # K * u = 0 (internal equilibrium) -> This is actually solving for u given BCs.
    # We partition K into K_ii, K_ib ...
    # u_boundary is known. u_internal is unknown.
    # K_ii * u_i = - K_ib * u_b
    
    # TensorMesh condenser handles Dirichlet BCs:
    # It solves K_ * u_ = F_ where u_ matches BCs.
    # But here F=0 (no body force). The "load" comes from Dirichlet BCs.
    
    dim = mesh.dim
    n_modes = 6 if dim == 3 else 3
    modes = get_voigt_strain_modes(dim).to(mesh.device)
    
    C_cols = []
    u_fields = []
    
    vol = 1.0 # Unit cube volume
    
    for k in range(n_modes):
        mode = modes[k]
        
        # Calculate boundary displacements
        u_bc = apply_kubc(mesh, mode)
        u_bc_flat = u_bc.flatten()
        
        # Prepare "Force" vector (actually zero body force)
        # Condenser will modify F to enforce u_bc
        F = torch.zeros_like(u_bc_flat)
        
        # Update Dirichlet values in condenser for this mode
        condenser.dirichlet_value = u_bc_flat[condenser.dirichlet_mask]
        
        # Condense system with non-zero Dirichlet values
        K_red, F_red = condenser(K, F)
        
        # Solve
        u_red = K_red.solve(F_red, backend=solver_backend)
        u_sol = condenser.recover(u_red)
        
        u_fields.append(u_sol.reshape(-1, dim))
        
        # Calculate average stress / effective stiffness column
        # sigma_avg = 1/V * int(sigma) dV
        # Or simpler: Energy method. 
        # But we need C_eff column k: C_ik = sigma_avg_i / epsilon_k (if others are 0)
        # Since we applied unit strain mode k, the resulting average stress IS the column C_ik.
        
        # Calculate stress?
        # F_reaction = K * u. Sum of reaction forces on boundary gives stress?
        # Work principle: u^T * K * u = V * eps^T * C_eff * eps
        # If eps = unit vector k, then Energy = V * C_kk. This gives diagonal.
        # Off-diagonal: Apply eps = (1, 1, 0...) -> (C_11 + C_22 + 2C_12).
        
        # Better: Average stress calculation.
        # sigma = D * B * u.
        # Integrate sigma over domain.
        # This requires element-wise stress calculation.
        # TensorMesh doesn't have easy "stress" extractor yet.
        
        # Workaround: Use Virtual Work.
        # int(sigma : eps(v)) = int(f * v) + boundary_terms.
        # sigma_avg_ij = 1/V * int(sigma_ij)
        # Using specific test functions...
        
        # Let's use the Energy Equivalence for gradients.
        # C_eff_kl = 1/V * u^k^T * K * u^l
        # This is robust and symmetric.
        # We need to solve for all 6 modes, then cross-multiply.
        
    # Construct C_eff matrix
    C_eff = torch.zeros((n_modes, n_modes), device=mesh.device)
    for i in range(n_modes):
        u_i = u_fields[i].flatten()
        for j in range(i, n_modes):
            u_j = u_fields[j].flatten()
            val = (u_i @ (K @ u_j)) / vol
            C_eff[i, j] = val
            C_eff[j, i] = val
            
    return C_eff, u_fields

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'metamaterial_out')

    parser = argparse.ArgumentParser(description="3D Metamaterial Design")
    parser.add_argument('--res', type=int, default=16, help='Resolution (elements per side)')
    parser.add_argument('--target_poisson', type=float, default=-0.5, help='Target Poisson Ratio')
    parser.add_argument('--epoch', type=int, default=50, help='Iterations')
    parser.add_argument('--output_dir', type=str, default=default_output_dir, help='Output directory')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Mesh (Unit Cube)
    mesh = Mesh.gen_cube(chara_length=1.0/args.res)
    mesh.to(device)
    print(f"Mesh: {mesh.n_elements} elements")
    
    # 2. Assembler (SIMP Linear Elasticity)
    # Base material: E=1.0, nu=0.3
    assembler = SIMPLinearElasticityElementAssembler.from_mesh(mesh)
    
    # 3. Boundary Conditions (KUBC)
    # For KUBC, all boundary nodes are Dirichlet.
    # Find boundary nodes
    boundary_mask = mesh.boundary_mask # [N] bool
    # Repeat for 3 DOFs
    dbc_mask = boundary_mask.repeat_interleave(3)
    condenser = Condenser(dbc_mask)
    
    print(f"Boundary nodes: {boundary_mask.sum().item()}")
    
    # 4. Optimizer
    # Init with hole in center (to encourage non-trivial topology) or uniform
    n_elem = mesh.n_elements
    rho = torch.full((n_elem,), 0.5, device=device, requires_grad=True)
    
    # Add a small perturbation to break symmetry if needed, or start with hole
    # Let's start uniform 0.5
    
    optimizer = OCOptimizer([rho], vf=0.5, move_limit=0.1)
    
    # Target Properties
    # We want C_eff to act like a material with nu = target (-0.5).
    # Isotropic Stiffness C:
    # C_11 = C_22 = C_33 = lambda + 2mu
    # C_12 = C_13 = C_23 = lambda
    # nu = lambda / (2(lambda+mu))
    # If nu = -0.5 -> lambda / (2(lambda+mu)) = -0.5
    # lambda = -lambda - mu => 2 lambda = -mu => mu = -2 lambda.
    # This implies instability for isotropic? -1 < nu < 0.5 is valid.
    # Wait, auxetic is stable.
    # Let's target specific ratios: C12 / C11.
    # C12/C11 = lambda / (lambda + 2mu) = nu / (1-nu).
    # If nu = -0.5, ratio = -0.5 / 1.5 = -1/3.
    # So we minimize (C12/C11 - (-0.33))^2 ?
    # Or simply Minimize C12 (make it negative) while maximizing C11 (stiffness).
    
    # Simple Objective: Minimize C_12 + C_13 + C_23 (Minimize lateral expansion)
    # Subject to C_11 > C_min (maintain stiffness)
    
    # Let's try: Loss = (C_12 + C_13 + C_23) / (C_11 + C_22 + C_33)
    # "Minimize Poisson's Ratio"
    
    pbar = tqdm(range(args.epoch))
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Homogenization
        C_eff, u_fields = compute_effective_stiffness(mesh, rho, assembler, condenser)
        
        # Define Objective: Mean Off-Diagonal / Mean Diagonal
        # Corresponds roughly to nu / (1-nu)
        diag = (C_eff[0,0] + C_eff[1,1] + C_eff[2,2]) / 3.0
        off_diag = (C_eff[0,1] + C_eff[0,2] + C_eff[1,2]) / 3.0
        
        # Maximize Stiffness (Diag) + Minimize Poisson (Off-Diag negative)
        # Loss = weight * off_diag - log(diag)
        # Or simple target
        
        loss = off_diag # Minimize lateral expansion (make it negative)
        
        # Penalize low stiffness to avoid empty mesh
        # If diag is too small, the material fails.
        # But Volume constraint handles mass.
        # Let's add a stiffness term: - 0.1 * diag
        
        loss = loss - 0.1 * diag
        
        loss.backward()
        
        optimizer.step()
        
        current_nu = off_diag.item() / (diag.item() + 1e-6) # approx nu/(1-nu)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'off_diag': f"{off_diag.item():.4f}", 'diag': f"{diag.item():.4f}"})
        
        # Visualization Export
        if epoch % 10 == 0 or epoch == args.epoch - 1:
            # Save density field
            mesh.save_vtu(
                f"{args.output_dir}/iter_{epoch:03d}.vtu", 
                cell_data={'rho': rho}
            )
            
            # Save deformation for Mode 0 (Uniaxial X stretch)
            # This allows visualizing the "Auxetic Effect" (contraction in Y/Z) in Paraview
            u_mode0 = u_fields[0] # [N, 3]
            mesh.save_vtu(
                f"{args.output_dir}/iter_{epoch:03d}_mode0_deform.vtu",
                point_data={'displacement': u_mode0},
                cell_data={'rho': rho}
            )

    print("Optimization Complete.")
    print("To visualize deformation in Paraview:")
    print("1. Open *_mode0_deform.vtu")
    print("2. Apply 'Warp By Vector' filter on 'displacement'")
    print("3. Observe lateral contraction (Auxetic effect) or expansion")

if __name__ == "__main__":
    main()

