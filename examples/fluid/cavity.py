import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser, ElementAssembler, NodeAssembler
from tensormesh.visualization import StreamPlotter
import matplotlib.pyplot as plt

class NavierStokesAssembler(ElementAssembler):
    def __post_init__(self, rho=1.0, mu=0.01, tau=0.1):
        self.rho = rho
        self.mu = mu
        self.tau = tau

    def forward(self, u, v, gradu, gradv, w_prev):
        """
        Weak form for Navier-Stokes using Picard iteration and PSPG/SUPG stabilization.
        Convention: u=phi_i (test/row), v=phi_j (trial/col), gradu=grad phi_i, gradv=grad phi_j.
        """
        dim = gradu.shape[0]

        # --- Standard Galerkin terms ---
        convection = self.rho * torch.dot(w_prev, gradv) * u
        diffusion = self.mu * torch.dot(gradu, gradv)
        k_diag = convection + diffusion

        # --- Stabilization terms ---
        # SUPG test: tau * (w_prev . grad phi_i)
        supg_weight = self.tau * torch.dot(w_prev, gradu)
        # SUPG applied to convection residual: rho * (w . grad phi_j) * tau * (w . grad phi_i)
        supg_convection = self.rho * torch.dot(w_prev, gradv) * supg_weight
        
        # Build the matrix row by row to avoid in-place operations
        rows = []
        for d_test in range(dim):
            row = []
            for d_trial in range(dim):
                if d_test == d_trial:
                    # Diffusion + Convection + SUPG Convection
                    entry = k_diag + supg_convection
                else:
                    entry = torch.tensor(0.0, device=gradu.device, dtype=gradu.dtype)
                row.append(entry)
            
            # Pressure term in momentum: -p * div v_test + SUPG pressure
            # -phi_j * (grad phi_i)_{d_test} + tau * (grad phi_j)_{d_test} * (w_prev . grad phi_i)
            pressure_entry = -v * gradu[d_test] + self.tau * gradv[d_test] * torch.dot(w_prev, gradu)
            row.append(pressure_entry)
            rows.append(torch.stack(row))
            
        # Continuity equation row (test function phi_i is for pressure)
        cont_row = []
        for d_trial in range(dim):
            # div w * phi_i + PSPG convection
            # (grad phi_j)_{d_trial} * phi_i + tau * rho * (w_prev . grad phi_j) * (grad phi_i)_{d_trial}
            cont_entry = gradv[d_trial] * u + self.tau * self.rho * torch.dot(w_prev, gradv) * gradu[d_trial]
            cont_row.append(cont_entry)
            
        # PSPG pressure: tau * grad p . grad phi_i
        cont_row.append(self.tau * torch.dot(gradv, gradu))
        rows.append(torch.stack(cont_row))
        
        return torch.stack(rows)

def solve_cavity(re=100, n_grid=20, max_iter=20, tol=1e-4):
    # 1. Mesh setup
    print(f"Solving Lid-Driven Cavity at Re={re} with {n_grid}x{n_grid} grid...")
    mesh = Mesh.gen_rectangle(chara_length=1.0/n_grid).double()
    points = mesh.points
    n_points = points.shape[0]
    
    # 2. Physical parameters
    rho = 1.0
    mu = 1.0 / re
    h = 1.0 / n_grid
    tau = 0.5 * h # Simple stabilization parameter
    
    # 3. Boundary conditions
    # Velocity BCs
    is_top = points[:, 1] > 1.0 - 1e-6
    is_boundary = mesh.boundary_mask
    
    # DOFs: (u, v, p) -> flatten to (n_points * 3)
    # Mapping: node i -> [3*i, 3*i+1, 3*i+2]
    u_mask = torch.zeros(n_points * 3, dtype=torch.bool)
    u_val = torch.zeros(n_points * 3, dtype=torch.float64)
    
    # Wall boundaries (all sides): u=0, v=0
    for d in range(2):
        u_mask[torch.arange(n_points) * 3 + d] = is_boundary
    
    # Top lid boundary: u=1
    u_val[torch.where(is_top)[0] * 3] = 1.0
    
    # Pressure BC: pin one node to zero
    u_mask[2] = True # p=0 at node 0
    u_val[2] = 0.0
    
    # 4. Iteration state
    # u_full stores [u, v, p] for all nodes
    u_full = torch.zeros(n_points * 3, dtype=torch.float64)
    # Apply initial BCs
    u_full[u_mask] = u_val[u_mask]
    
    # 5. Assembler
    assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
    condenser = Condenser(u_mask, u_val)
    
    # 6. Picard Iteration
    for i in range(max_iter):
        # Current velocity for linearization
        w_prev = u_full.reshape(-1, 3)[:, :2]
        
        # Assemble matrix
        # forward(u, v, gradu, gradv, w_prev, gradw_prev)
        # point_data expects keys that match forward arguments or 'grad' + key
        # 'u' and 'v' are reserved for shape functions in ElementAssembler.__call__
        K_sparse = assembler(points, point_data={"w_prev": w_prev})
        
        # RHS is zero for this problem (no body forces)
        f = torch.zeros(n_points * 3, dtype=torch.float64)
        
        # Condense and solve
        K_cond, f_cond = condenser(K_sparse, f)
        u_new_cond = K_cond.solve(f_cond)
        u_new = condenser.recover(u_new_cond)
        
        # Check convergence
        diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
        print(f"Iteration {i}: relative diff = {diff:.6e}")
        
        u_full = u_new
        if diff < tol:
            print("Converged!")
            break
            
    # 7. Post-processing and Visualization
    u_res = u_full.reshape(-1, 3)
    vel = u_res[:, :2]
    pres = u_res[:, 2]
    
    speed = torch.norm(vel, dim=1)
    
    print("Saving visualization...")
    # Use StreamPlotter for static visualization (it also has draw_mesh_2d_static)
    from tensormesh.visualization.stream_plotter import draw_mesh_2d_static
    
    # Prepare elements dict for plotter
    elements = mesh.elements()
    if isinstance(elements, torch.Tensor):
        elements = {mesh.default_element_type: elements}
        
    draw_mesh_2d_static(
        points, 
        elements, 
        {"speed": speed, "pressure": pres},
        filename="cavity_results.png",
        show_mesh=False,
        cmap="jet"
    )
    print("Done! Results saved to cavity_results.png")

if __name__ == "__main__":
    solve_cavity(re=100, n_grid=30)

