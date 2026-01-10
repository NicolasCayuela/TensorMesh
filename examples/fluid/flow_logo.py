import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser, ElementAssembler, NodeAssembler, MeshGen
from tensormesh.visualization import draw_mesh_2d_static
import matplotlib.pyplot as plt

# Reuse the NavierStokesAssembler logic from cavity.py but more robustly
class NavierStokesAssembler(ElementAssembler):
    def __post_init__(self, rho=1.0, mu=0.01, tau=0.1):
        self.rho = rho
        self.mu = mu
        self.tau = tau

    def forward(self, u, v, gradu, gradv, w_prev):
        dim = gradu.shape[0]
        
        convection = self.rho * torch.dot(w_prev, gradv) * u
        diffusion = self.mu * torch.dot(gradu, gradv)
        k_diag = convection + diffusion
        
        rows = []
        for d_test in range(dim):
            row = []
            for d_trial in range(dim):
                if d_test == d_trial:
                    entry = k_diag + self.rho * torch.dot(w_prev, gradv) * (self.tau * torch.dot(w_prev, gradu))
                else:
                    entry = torch.tensor(0.0, device=gradu.device, dtype=gradu.dtype)
                row.append(entry)
            
            # Pressure term + SUPG pressure
            pressure_entry = -v * gradu[d_test] + self.tau * gradv[d_test] * torch.dot(w_prev, gradu)
            row.append(pressure_entry)
            rows.append(torch.stack(row))
            
        cont_row = []
        for d_trial in range(dim):
            cont_entry = gradv[d_trial] * u + self.tau * self.rho * torch.dot(w_prev, gradv) * gradu[d_trial]
            cont_row.append(cont_entry)
            
        cont_row.append(self.tau * torch.dot(gradv, gradu))
        rows.append(torch.stack(cont_row))
        
        return torch.stack(rows)

def solve_flow_obstacles(re=200, n_grid=25, max_iter=20):
    print(f"Generating mesh with multiple obstacles...")
    gen = MeshGen(chara_length=1.0/n_grid)
    gen.add_rectangle(0, 0, 3.0, 1.0)
    
    # Add obstacles (e.g., "T" and "M" like arrangement or just circles)
    obstacles = [
        (0.5, 0.5, 0.1),
        (1.0, 0.3, 0.08),
        (1.0, 0.7, 0.08),
        (1.5, 0.5, 0.12),
        (2.0, 0.4, 0.07),
        (2.0, 0.6, 0.07)
    ]
    for ox, oy, orad in obstacles:
        gen.remove_circle(ox, oy, orad)
        
    mesh = gen.gen().double()
    points = mesh.points
    n_points = points.shape[0]
    
    rho = 1.0
    mu = 1.0 / re
    tau = 0.5 / n_grid
    
    is_inlet = points[:, 0] < 1e-6
    is_outlet = points[:, 0] > 3.0 - 1e-6
    is_wall = (points[:, 1] < 1e-6) | (points[:, 1] > 1.0 - 1e-6)
    
    # Identify obstacle boundaries
    is_obstacle = torch.zeros(n_points, dtype=torch.bool)
    for ox, oy, orad in obstacles:
        dist = torch.sqrt((points[:, 0] - ox)**2 + (points[:, 1] - oy)**2)
        is_obstacle |= (dist < orad + 1e-3)
        
    u_mask = torch.zeros(n_points * 3, dtype=torch.bool)
    u_val = torch.zeros(n_points * 3, dtype=torch.float64)
    
    # Inlet: Parabolic
    inlet_indices = torch.where(is_inlet)[0]
    y_in = points[inlet_indices, 1]
    u_inlet = 4.0 * 1.0 * y_in * (1.0 - y_in) / (1.0**2)
    u_mask[inlet_indices * 3] = True
    u_val[inlet_indices * 3] = u_inlet
    u_mask[inlet_indices * 3 + 1] = True # v=0
    
    # Walls and Obstacles: No-slip
    no_slip = is_wall | is_obstacle
    u_mask[torch.where(no_slip)[0] * 3] = True
    u_mask[torch.where(no_slip)[0] * 3 + 1] = True
    
    # Outlet: p=0
    outlet_indices = torch.where(is_outlet)[0]
    u_mask[outlet_indices * 3 + 2] = True
    u_val[outlet_indices * 3 + 2] = 0.0
    
    u_full = torch.zeros(n_points * 3, dtype=torch.float64)
    u_full[u_mask] = u_val[u_mask]
    
    assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
    condenser = Condenser(u_mask, u_val)
    
    print(f"Solving Flow Past Obstacles at Re={re}...")
    for i in range(max_iter):
        w_prev = u_full.reshape(-1, 3)[:, :2]
        K_sparse = assembler(points, point_data={"w_prev": w_prev})
        f = torch.zeros(n_points * 3, dtype=torch.float64)
        
        K_cond, f_cond = condenser(K_sparse, f)
        u_new_cond = K_cond.solve(f_cond)
        u_new = condenser.recover(u_new_cond)
        
        diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
        print(f"Iteration {i}: relative diff = {diff:.6e}")
        u_full = u_new
        if diff < 5e-4:
            print("Converged!")
            break
            
    # Visualization
    print("Saving visualization...")
    u_res = u_full.reshape(-1, 3)
    speed = torch.norm(u_res[:, :2], dim=1)
    pressure = u_res[:, 2]
    
    elements = mesh.elements()
    if isinstance(elements, torch.Tensor):
        elements = {mesh.default_element_type: elements}
        
    draw_mesh_2d_static(
        points,
        elements,
        {"Speed": speed, "Pressure": pressure},
        filename="flow_obstacles.png",
        show_mesh=False,
        cmap="jet"
    )
    print("Done! Results saved to flow_obstacles.png")

if __name__ == "__main__":
    solve_flow_obstacles(re=150, n_grid=40)

