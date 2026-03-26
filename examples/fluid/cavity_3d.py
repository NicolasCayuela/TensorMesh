import os
import sys

import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.visualization import setup_headless


class NavierStokesAssembler(ElementAssembler):
    """Steady-state Navier-Stokes with SUPG/PSPG stabilization (dimension-generic)."""

    def __post_init__(self, rho=1.0, mu=0.01, tau=0.1):
        self.rho = rho
        self.mu = mu
        self.tau = tau

    def forward(self, u, v, gradu, gradv, w_prev):
        dim = gradu.shape[0]

        convection = self.rho * torch.dot(w_prev, gradv) * u
        diffusion = self.mu * torch.dot(gradu, gradv)
        k_diag = convection + diffusion

        supg_weight = self.tau * torch.dot(w_prev, gradu)
        supg_convection = self.rho * torch.dot(w_prev, gradv) * supg_weight

        rows = []
        for d_test in range(dim):
            row = []
            for d_trial in range(dim):
                if d_test == d_trial:
                    row.append(k_diag + supg_convection)
                else:
                    row.append(torch.tensor(0.0, device=gradu.device, dtype=gradu.dtype))
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


def solve_cavity_3d(re=100, chara_length=0.1, max_iter=30, tol=1e-4):
    setup_headless()

    print(f"Solving 3D Lid-Driven Cavity at Re={re}, chara_length={chara_length}...")
    mesh = Mesh.gen_cube(chara_length=chara_length).double()
    points = mesh.points
    n_points = points.shape[0]
    n_dof = 4  # (u, v, w, p)

    print(f"  Mesh: {n_points} nodes, {n_points * n_dof} DOFs")

    rho = 1.0
    mu = 1.0 / re
    h = chara_length
    tau = 0.5 * h

    # --- Boundary conditions ---
    is_boundary = mesh.point_data["is_boundary"]
    is_top = mesh.point_data["is_top_boundary"]  # y = 1

    bc_mask = torch.zeros(n_points * n_dof, dtype=torch.bool)
    bc_val = torch.zeros(n_points * n_dof, dtype=torch.float64)

    # No-slip on all boundary nodes (u=v=w=0)
    for d in range(3):
        bc_mask[torch.arange(n_points) * n_dof + d] = is_boundary

    # Top lid: u=1, v=0, w=0 (override u on top face)
    bc_val[torch.where(is_top)[0] * n_dof] = 1.0

    # Pressure pin at node 0
    bc_mask[n_dof - 1] = True  # p=0 at node 0
    bc_val[n_dof - 1] = 0.0

    # --- Initial state ---
    u_full = torch.zeros(n_points * n_dof, dtype=torch.float64)
    u_full[bc_mask] = bc_val[bc_mask]

    # --- Assembler ---
    assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
    condenser = Condenser(bc_mask, bc_val)

    # --- Picard iteration ---
    for i in tqdm(range(max_iter), desc="Picard iteration"):
        w_prev = u_full.reshape(-1, n_dof)[:, :3]

        K = assembler(points, point_data={"w_prev": w_prev})
        f = torch.zeros(n_points * n_dof, dtype=torch.float64)

        K_cond, f_cond = condenser(K, f)
        u_new_cond = K_cond.solve(f_cond)
        u_new = condenser.recover(u_new_cond)

        diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
        tqdm.write(f"  Iteration {i}: relative diff = {diff:.6e}")

        u_full = u_new
        if diff < tol:
            print("Converged!")
            break

    # --- Post-processing ---
    sol = u_full.reshape(-1, n_dof)
    velocity = sol[:, :3]
    pressure = sol[:, 3]
    speed = torch.norm(velocity, dim=1)

    print(f"  Max speed: {speed.max().item():.4f}")
    print(f"  Pressure range: [{pressure.min().item():.4f}, {pressure.max().item():.4f}]")

    # Export VTU
    mesh.register_point_data("speed", speed)
    mesh.register_point_data("pressure", pressure)
    mesh.register_point_data("velocity", velocity)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    vtu_path = os.path.join(out_dir, "cavity_3d.vtu")
    mesh.save(vtu_path)
    print(f"Saved: {vtu_path}")

    # PyVista visualization: mid-plane cross-section
    try:
        import pyvista as pv

        grid = pv.read(vtu_path)

        p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 700))

        # Slice at z=0.5 (mid-depth) to see the primary x-y recirculation
        slice_z = grid.slice(normal="z", origin=(0.5, 0.5, 0.5))

        # Speed on z=0.5 slice
        p.subplot(0, 0)
        p.add_mesh(slice_z, scalars="speed", cmap="jet", show_scalar_bar=True)
        p.add_text("Speed (z=0.5 slice)", font_size=10, position="upper_edge")
        p.view_xy()

        # Pressure on z=0.5 slice
        p.subplot(0, 1)
        p.add_mesh(slice_z, scalars="pressure", cmap="coolwarm", show_scalar_bar=True)
        p.add_text("Pressure (z=0.5 slice)", font_size=10, position="upper_edge")
        p.view_xy()

        png_path = os.path.join(out_dir, "cavity_3d.png")
        p.screenshot(png_path)
        p.close()
        print(f"Saved: {png_path}")
    except Exception as ex:
        print(f"Skip PyVista visualization: {type(ex).__name__}: {ex}")

    print("Done!")


if __name__ == "__main__":
    solve_cavity_3d(re=100, chara_length=0.05, max_iter=30)
