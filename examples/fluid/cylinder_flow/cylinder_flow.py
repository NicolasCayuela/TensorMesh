import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, ElementAssembler, MeshGen, NodeAssembler
from tensormesh.assemble import MassElementAssembler


class NavierStokesTransientAssembler(ElementAssembler):
    """Transient incompressible Navier-Stokes with SUPG/PSPG stabilization."""

    def __post_init__(self, rho: float = 1.0, mu: float = 0.01, dt: float = 1e-3, tau: float = 1e-3):
        self.rho = rho
        self.mu = mu
        self.dt = dt
        self.tau = tau

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def forward(self, u, v, gradu, gradv, w_prev):
        dim = gradu.shape[0]

        mass = self.rho * (u * v) / self.dt
        convection = self.rho * torch.dot(w_prev, gradv) * u
        diffusion = self.mu * torch.dot(gradu, gradv)

        # SUPG term in momentum equations
        supg_test = self.tau * torch.dot(w_prev, gradu)
        supg_residual = (self.rho / self.dt * v + self.rho * torch.dot(w_prev, gradv)) * supg_test

        momentum_diag = mass + convection + diffusion + supg_residual

        rows = []
        for d_test in range(dim):
            row = []
            for d_trial in range(dim):
                if d_test == d_trial:
                    row.append(momentum_diag)
                else:
                    row.append(torch.tensor(0.0, dtype=u.dtype, device=u.device))

            # Pressure gradient + SUPG pressure coupling
            # supg_test already contains tau, so no extra tau here
            row.append(-v * gradu[d_test] + gradv[d_test] * supg_test)
            rows.append(torch.stack(row))

        # Continuity equation with PSPG stabilization
        continuity = []
        for d_trial in range(dim):
            continuity.append(
                gradv[d_trial] * u
                + self.tau * (self.rho / self.dt * v + self.rho * torch.dot(w_prev, gradv)) * gradu[d_trial]
            )
        continuity.append(self.tau * torch.dot(gradv, gradu))
        rows.append(torch.stack(continuity))

        return torch.stack(rows)


class MomentumRHSAssembler(NodeAssembler):
    """RHS for implicit Euler and SUPG counterpart."""

    def __post_init__(self, rho: float = 1.0, dt: float = 1e-3, tau: float = 1e-3):
        self.rho = rho
        self.dt = dt
        self.tau = tau

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def forward(self, v, gradv, u_prev, w_prev):
        # SUPG modified test function: N_i + tau * (w · grad N_i)
        supg_v = self.tau * (w_prev[0] * gradv[0] + w_prev[1] * gradv[1])
        r0 = self.rho * u_prev[0] * (v + supg_v) / self.dt
        r1 = self.rho * u_prev[1] * (v + supg_v) / self.dt
        # PSPG consistent RHS contribution for pressure row
        r2 = self.tau * self.rho * (u_prev[0] * gradv[0] + u_prev[1] * gradv[1]) / self.dt
        return torch.stack([r0, r1, r2])


class VorticityProjectionAssembler(NodeAssembler):
    """Assemble RHS for L2 projection of scalar vorticity."""

    def forward(self, gradv, velocity):
        # Weak vorticity via integration by parts: ∫ ω N_i dΩ = ∫ (v_x ∂N_i/∂y - v_y ∂N_i/∂x) dΩ
        omega = velocity[0] * gradv[1] - velocity[1] * gradv[0]
        return omega


def build_cylinder_channel_mesh(chara_length: float = 0.02):
    length = 2.2
    height = 0.41
    cx, cy, radius = 0.2, 0.2, 0.05

    gen = MeshGen(chara_length=chara_length)
    gen.add_rectangle(0.0, 0.0, length, height)
    gen.remove_circle(cx, cy, radius)
    mesh = gen.gen().double()
    return mesh, (length, height, cx, cy, radius)


def build_boundary_conditions(
    points: torch.Tensor,
    geometry: Tuple[float, float, float, float, float],
    inflow_umax: float,
    chara_length: float,
):
    length, height, cx, cy, radius = geometry
    x = points[:, 0]
    y = points[:, 1]

    eps = 2.0e-3
    is_inlet = x <= eps
    is_outlet = x >= length - eps
    is_wall = (y <= eps) | (y >= height - eps)

    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # MeshGen generated meshes may not carry mesh.point_data['is_boundary'].
    # Use geometric identification for cylinder boundary nodes.
    is_cylinder = torch.abs(dist - radius) <= 0.6 * chara_length

    n_points = points.shape[0]
    bc_mask = torch.zeros(n_points * 3, dtype=torch.bool, device=points.device)
    bc_val = torch.zeros(n_points * 3, dtype=points.dtype, device=points.device)

    inlet_idx = torch.where(is_inlet)[0]
    y_in = y[inlet_idx]
    u_in = 4.0 * inflow_umax * y_in * (height - y_in) / (height * height)
    bc_mask[inlet_idx * 3] = True
    bc_val[inlet_idx * 3] = u_in
    bc_mask[inlet_idx * 3 + 1] = True

    no_slip = is_wall | is_cylinder
    no_slip_idx = torch.where(no_slip)[0]
    bc_mask[no_slip_idx * 3] = True
    bc_mask[no_slip_idx * 3 + 1] = True

    # Pressure gauge to remove null space while keeping do-nothing outlet velocity.
    outlet_idx = torch.where(is_outlet)[0]
    if outlet_idx.numel() > 0:
        center = torch.argmin(torch.abs(y[outlet_idx] - 0.5 * height))
        pin_idx = outlet_idx[center]
    else:
        pin_idx = torch.tensor(0, device=points.device)
    bc_mask[pin_idx * 3 + 2] = True
    bc_val[pin_idx * 3 + 2] = 0.0

    return bc_mask, bc_val


def compute_tau(chara_length: float, mu: float, rho: float, velocity: torch.Tensor) -> float:
    speed = torch.norm(velocity, dim=1)
    speed_ref = float(torch.quantile(speed.detach().cpu(), 0.95).item()) + 1e-8
    return (chara_length * chara_length) / (4.0 * mu + 2.0 * rho * speed_ref * chara_length)


def solve_cylinder_flow(
    re: float = 100.0,
    chara_length: float = 0.02,
    dt: float = 1.0e-3,
    n_steps: int = 700,
    picard_iter: int = 2,
    picard_tol: float = 1.0e-5,
    save_every: int = 10,
):
    torch.random.manual_seed(0)

    rho = 1.0
    height = 0.41
    diameter = 0.1
    u_mean = 1.0
    u_max = 1.5
    mu = rho * u_mean * diameter / re

    mesh, geometry = build_cylinder_channel_mesh(chara_length=chara_length)
    points = mesh.points
    n_points = points.shape[0]

    bc_mask, bc_val = build_boundary_conditions(
        points=points,
        geometry=geometry,
        inflow_umax=u_max,
        chara_length=chara_length,
    )

    condenser = Condenser(bc_mask, bc_val)
    u_full = torch.zeros(n_points * 3, dtype=points.dtype, device=points.device)
    u_full[bc_mask] = bc_val[bc_mask]

    # Small asymmetric perturbation on v_y to trigger vortex shedding faster
    u_full[1::3] += 1e-3 * torch.sin(2.0 * torch.pi * points[:, 0] / 2.2)

    tau0 = (chara_length * chara_length) / (4.0 * mu + 2.0 * rho * u_max * chara_length)

    ns_asm = NavierStokesTransientAssembler.from_mesh(mesh, rho=rho, mu=mu, dt=dt, tau=tau0)
    rhs_asm = MomentumRHSAssembler.from_mesh(mesh, rho=rho, dt=dt, tau=tau0)
    mass_asm = MassElementAssembler.from_mesh(mesh)
    vort_rhs_asm = VorticityProjectionAssembler.from_mesh(mesh)
    m_mat = mass_asm()

    speed_frames: List[torch.Tensor] = []
    pressure_frames: List[torch.Tensor] = []
    vorticity_frames: List[torch.Tensor] = []

    print(f"Start cylinder flow simulation: Re={re}, points={n_points}, dt={dt}, steps={n_steps}")
    for step in tqdm(range(n_steps), desc="Time marching"):
        u_prev = u_full.clone()
        u_iter = u_prev.clone()

        for _ in range(picard_iter):
            vel_prev = u_iter.reshape(-1, 3)[:, :2]
            tau = compute_tau(chara_length=chara_length, mu=mu, rho=rho, velocity=vel_prev)
            ns_asm.set_tau(tau)
            rhs_asm.set_tau(tau)

            k_mat = ns_asm(points, point_data={"w_prev": vel_prev})
            f_vec = rhs_asm(points, point_data={"u_prev": u_prev.reshape(-1, 3)[:, :2], "w_prev": vel_prev})

            k_cond, f_cond = condenser(k_mat, f_vec)
            u_cond = k_cond.solve(f_cond)
            u_new = condenser.recover(u_cond)

            picard_err = torch.norm(u_new - u_iter) / (torch.norm(u_new) + 1e-12)
            u_iter = u_new
            if float(picard_err) < picard_tol:
                break

        u_full = u_iter

        if step % save_every == 0 or step == n_steps - 1:
            sol = u_full.reshape(-1, 3)
            velocity = sol[:, :2]
            pressure = sol[:, 2]
            speed = torch.norm(velocity, dim=1)

            omega_rhs = vort_rhs_asm(point_data={"velocity": velocity})
            omega = m_mat.solve(omega_rhs)

            speed_frames.append(speed.detach().cpu())
            pressure_frames.append(pressure.detach().cpu())
            vorticity_frames.append(omega.detach().cpu())

    os.makedirs("frames", exist_ok=True)
    for i, (vort, spd, pres) in enumerate(zip(vorticity_frames, speed_frames, pressure_frames)):
        mesh.plot(
            {"vorticity": vort, "speed": spd, "pressure": pres},
            save_path=f"frames/frame_{i:04d}.png",
            show_mesh=False,
        )
    print(f"Saved {len(vorticity_frames)} frames to frames/")

    mesh.plot(
        {"vorticity": vorticity_frames[-1], "speed": speed_frames[-1], "pressure": pressure_frames[-1]},
        save_path="cylinder_flow_final.png",
        show_mesh=False,
    )
    print("Saved: cylinder_flow_final.png")


if __name__ == "__main__":
    solve_cylinder_flow(
        re=100.0,
        chara_length=0.02,
        dt=1.0e-3,
        n_steps=5000,
        picard_iter=2,
        picard_tol=1.0e-5,
        save_every=50,
    )
