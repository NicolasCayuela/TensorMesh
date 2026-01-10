import time
import torch
import numpy as np
from tensormesh import Mesh, Condenser, ElementAssembler, NodeAssembler, MeshGen

def benchmark_speed():
    re = 200
    dt = 0.02
    n_steps = 10 # Only 10 steps for benchmark
    
    print("Generating mesh...")
    gen = MeshGen(chara_length=0.025)
    gen.add_rectangle(0, 0, 2.2, 0.41).remove_circle(0.2, 0.2, 0.05)
    mesh = gen.gen().double()
    points, n_pts = mesh.points, mesh.points.shape[0]
    print(f"Mesh size: {n_pts} points")
    
    # Simple setup
    u_mask, u_val = torch.zeros(n_pts * 3, dtype=torch.bool), torch.zeros(n_pts * 3, dtype=torch.float64)
    u_full = torch.zeros(n_pts * 3, dtype=torch.float64)
    
    class NavierStokesTransientAssembler(ElementAssembler):
        def __post_init__(self, rho=1.0, mu=0.01, tau=0.1, dt=0.01):
            self.rho, self.mu, self.tau, self.dt = rho, mu, tau, dt
        def forward(self, u, v, gradu, gradv, w_prev):
            dim = gradu.shape[0]
            k_diag = self.rho * (v * u) / self.dt + self.rho * torch.dot(w_prev, gradv) * u + self.mu * torch.dot(gradu, gradv)
            supg_w = self.tau * torch.dot(w_prev, gradu)
            supg_res = (self.rho * v / self.dt + self.rho * torch.dot(w_prev, gradv)) * supg_w
            rows = []
            for d in range(dim):
                row = []
                for trial_d in range(dim):
                    entry = k_diag + supg_res if d == trial_d else torch.tensor(0.0, device=u.device, dtype=u.dtype)
                    row.append(entry)
                row.append(-v * gradu[d] + self.tau * gradv[d] * supg_w)
                rows.append(torch.stack(row))
            cont = []
            for d in range(dim):
                cont.append(gradv[d] * u + self.tau * (self.rho * v / self.dt + self.rho * torch.dot(w_prev, gradv)) * gradu[d])
            cont.append(self.tau * torch.dot(gradv, gradu))
            rows.append(torch.stack(cont))
            return torch.stack(rows)

    assembler = NavierStokesTransientAssembler.from_mesh(mesh, rho=1.0, mu=1.0/re, tau=0.0005, dt=dt)
    condenser = Condenser(u_mask, u_val)
    
    class MomentumRHS(NodeAssembler):
        def forward(self, v, gradv, u_prev):
            r0, r1 = (u_prev[0]/dt)*v, (u_prev[1]/dt)*v
            r2 = 0.0005 * (u_prev[0]/dt)*gradv[0] + 0.0005 * (u_prev[1]/dt)*gradv[1]
            return torch.stack([r0, r1, r2])
    rhs_asm = MomentumRHS.from_mesh(mesh)

    print("Starting simulation benchmark (10 steps)...")
    t0 = time.time()
    
    for step in range(n_steps):
        t_step_start = time.time()
        
        # Matrix Assembly
        K = assembler(points, point_data={"w_prev": u_full.reshape(-1, 3)[:, :2]})
        t_asm = time.time()
        
        # RHS Assembly
        f = rhs_asm(points, point_data={"u_prev": u_full.reshape(-1, 3)})
        t_rhs = time.time()
        
        # Solve
        K_c, f_c = condenser(K, f)
        u_full = condenser.recover(K_c.solve(f_c))
        t_solve = time.time()
        
        print(f"Step {step}: Total={t_solve-t_step_start:.3f}s (Asm={t_asm-t_step_start:.3f}s, RHS={t_rhs-t_asm:.3f}s, Solve={t_solve-t_rhs:.3f}s)")

    total_time = time.time() - t0
    print(f"Total time for {n_steps} steps: {total_time:.2f}s")
    print(f"Average time per step: {total_time/n_steps:.3f}s")
    print(f"Estimated time for 200 steps: {total_time/n_steps*200/60:.1f} minutes")

if __name__ == "__main__":
    benchmark_speed()

