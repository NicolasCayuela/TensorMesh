import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Mesh, Condenser, ElementAssembler


class RayleighBenardAssembler(ElementAssembler):
    def __post_init__(self, rho=1.0, mu=0.01, kappa=0.01, g=9.81, beta=0.1, tau=0.1):
        self.rho = rho
        self.mu = mu
        self.kappa = kappa
        self.g = g
        self.beta = beta # Thermal expansion coefficient
        self.tau = tau

    def forward(self, u, v, gradu, gradv, w_prev, T_prev):
        """
        Weak form for Boussinesq Navier-Stokes (Rayleigh-Bénard).
        DOFs: (u, v, p, T) -> dim + 2 = 4
        """
        dim = gradu.shape[0]
        
        # --- Standard Galerkin terms ---
        convection = self.rho * torch.dot(w_prev, gradv) * u
        diffusion = self.mu * torch.dot(gradu, gradv)
        k_ns_diag = convection + diffusion
        
        t_convection = self.rho * torch.dot(w_prev, gradv) * u
        t_diffusion = self.kappa * torch.dot(gradu, gradv)
        k_t_diag = t_convection + t_diffusion
        
        # Build the matrix entries out-of-place
        rows = []
        
        # Row 0: Momentum X
        # Row 1: Momentum Y
        # Row 2: Continuity
        # Row 3: Energy (Temperature)
        
        # Row 0 (u)
        rows.append(torch.stack([
            k_ns_diag,                                # K[0,0]
            torch.tensor(0.0, device=u.device, dtype=u.dtype),
            -v * gradu[0],                            # K[0,2] (pressure gradient x)
            torch.tensor(0.0, device=u.device, dtype=u.dtype)
        ]))
        
        # Row 1 (v)
        rows.append(torch.stack([
            torch.tensor(0.0, device=u.device, dtype=u.dtype),
            k_ns_diag,                                # K[1,1]
            -v * gradu[1],                            # K[1,2] (pressure gradient y)
            -self.rho * self.g * self.beta * v * u    # K[1,3] (buoyancy)
        ]))
        
        # Row 2 (p)
        rows.append(torch.stack([
            gradv[0] * u,                             # K[2,0] (divergence x)
            gradv[1] * u,                             # K[2,1] (divergence y)
            self.tau * torch.dot(gradv, gradu),       # K[2,2] (PSPG)
            torch.tensor(0.0, device=u.device, dtype=u.dtype)
        ]))
        
        # Row 3 (T)
        rows.append(torch.stack([
            torch.tensor(0.0, device=u.device, dtype=u.dtype),
            torch.tensor(0.0, device=u.device, dtype=u.dtype),
            torch.tensor(0.0, device=u.device, dtype=u.dtype),
            k_t_diag                                  # K[3,3]
        ]))
        
        return torch.stack(rows)

def solve_rayleigh_benard(ra=1e4, aspect_ratio=2, n_grid=20, max_iter=30):
    # Rayleigh number Ra = (g * beta * deltaT * L^3) / (nu * alpha)
    # deltaT = 1, L = 1, nu = mu/rho, alpha = kappa/rho
    # Let rho=1, mu=0.1, kappa=0.1, beta = Ra * mu * kappa / (g * deltaT * L^3)
    
    print(f"Solving Rayleigh-Bénard Convection at Ra={ra:.1e}...")
    mesh = Mesh.gen_rectangle(left=0, right=aspect_ratio, bottom=0, top=1.0, chara_length=1.0/n_grid, element_type="tri").double()
    points = mesh.points
    n_points = points.shape[0]
    
    rho = 1.0
    mu = 0.1
    kappa = 0.1
    g = 10.0
    beta = ra * mu * kappa / (g * 1.0 * 1.0**3)
    tau = 0.05 / n_grid
    
    # DOFs: (u, v, p, T)
    u_mask = torch.zeros(n_points * 4, dtype=torch.bool)
    u_val = torch.zeros(n_points * 4, dtype=torch.float64)
    
    is_boundary = mesh.boundary_mask
    is_bottom = points[:, 1] < 1e-6
    is_top = points[:, 1] > 1.0 - 1e-6
    
    # Velocity BCs: no-slip on all boundaries
    for d in range(2):
        u_mask[torch.arange(n_points) * 4 + d] = is_boundary
        
    # Temperature BCs: T=1 at bottom, T=0 at top
    u_mask[torch.where(is_bottom)[0] * 4 + 3] = True
    u_val[torch.where(is_bottom)[0] * 4 + 3] = 1.0
    
    u_mask[torch.where(is_top)[0] * 4 + 3] = True
    u_val[torch.where(is_top)[0] * 4 + 3] = 0.0
    
    # Pressure pin
    u_mask[2] = True
    u_val[2] = 0.0
    
    # Initial state: linear temp profile + small perturbation to trigger convection
    u_full = torch.zeros(n_points * 4, dtype=torch.float64)
    u_full[torch.arange(n_points) * 4 + 3] = 1.0 - points[:, 1] # Linear profile
    # Perturbation
    u_full[torch.arange(n_points) * 4 + 3] += 0.01 * torch.sin(np.pi * points[:, 0] / aspect_ratio) * torch.sin(np.pi * points[:, 1])
    u_full[u_mask] = u_val[u_mask]
    
    assembler = RayleighBenardAssembler.from_mesh(mesh, rho=rho, mu=mu, kappa=kappa, g=g, beta=beta, tau=tau)
    condenser = Condenser(u_mask, u_val)
    
    for i in range(max_iter):
        sol = u_full.reshape(-1, 4)
        w_prev = sol[:, :2]
        T_prev = sol[:, 3]
        
        K_sparse = assembler(points, point_data={"w_prev": w_prev, "T_prev": T_prev})
        f = torch.zeros(n_points * 4, dtype=torch.float64)

        K_cond, f_cond = condenser(K_sparse, f)
        u_new_cond = K_cond.solve(f_cond)
        u_new = condenser.recover(u_new_cond)
        
        diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
        print(f"Iteration {i}: relative diff = {diff:.6e}")
        u_full = u_new
        if diff < 1e-4:
            print("Converged!")
            break
            
    # Visualization
    sol = u_full.reshape(-1, 4)
    T = sol[:, 3]
    V = torch.norm(sol[:, :2], dim=1)

    mesh.plot(
        {"Temperature": T, "Velocity": V},
        save_path="rayleigh_benard.png",
        show_mesh=False,
        cmap="inferno",
    )
    print("Done! Results saved to rayleigh_benard.png")

if __name__ == "__main__":
    solve_rayleigh_benard(ra=2e4, aspect_ratio=2, n_grid=30)
