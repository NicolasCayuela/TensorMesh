"""3D lid-driven cavity — steady incompressible Navier-Stokes.

The 3D extension of ``cavity.py``. The ``NavierStokesAssembler`` below is
dimension-generic (it reads ``dim`` from ``gradu.shape[0]`` and stamps a
``(dim+1) x (dim+1)`` block), so the only real change from 2D is the mesh,
the per-node DOF layout ``[u, v, w, p]``, and the volumetric output.
"""
import os
import sys

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.visualization import setup_headless


class NavierStokesAssembler(ElementAssembler):
    r"""Steady Navier-Stokes weak form with SUPG/PSPG stabilization.

    ``forward`` returns the ``(dim+1) x (dim+1)`` block coupling one test
    node to one trial node, with velocity components and pressure laid
    out as ``[u, v, (w,) p]``::

        [ A_uu   B_up ]   A_uu : velocity-velocity (convection + diffusion + SUPG)
        [ B_pu   C_pp ]   B_up : pressure gradient in the momentum eqn (+ PSPG)
                          B_pu : divergence in the continuity eqn (+ PSPG)
                          C_pp : PSPG pressure Laplacian

    Equal-order P1-P1 violates the inf-sup (LBB) condition, so the bare
    Galerkin form admits spurious pressure modes; the SUPG/PSPG terms
    scaled by ``tau`` restore stability. The velocity block is kept
    diagonal (components decoupled) -- the standard simplification for a
    stabilized equal-order formulation. Identical to the assembler in
    ``cavity.py``; the same class handles 2D and 3D.
    """

    def __post_init__(self, rho=1.0, mu=0.01, tau=0.1):
        self.rho = rho
        self.mu = mu
        self.tau = tau

    def forward(self, u, v, gradu, gradv, w_prev):
        dim = gradu.shape[0]
        eye = torch.eye(dim, dtype=gradu.dtype, device=gradu.device)

        # velocity-velocity: convection + diffusion + SUPG, diagonal in components
        convection = self.rho * torch.dot(w_prev, gradv) * u
        diffusion = self.mu * torch.dot(gradu, gradv)
        supg = self.rho * torch.dot(w_prev, gradv) * self.tau * torch.dot(w_prev, gradu)
        A_uu = (convection + diffusion + supg) * eye                          # [dim, dim]

        # pressure gradient in the momentum equation (+ PSPG consistency term)
        B_up = -v * gradu + self.tau * torch.dot(w_prev, gradu) * gradv       # [dim]

        # divergence in the continuity equation (+ PSPG)
        B_pu = u * gradv + self.tau * self.rho * torch.dot(w_prev, gradv) * gradu  # [dim]

        # PSPG pressure Laplacian
        C_pp = self.tau * torch.dot(gradv, gradu)                             # scalar

        top = torch.cat([A_uu, B_up.unsqueeze(1)], dim=1)                     # [dim, dim+1]
        bottom = torch.cat([B_pu, C_pp.reshape(1)]).unsqueeze(0)             # [1, dim+1]
        return torch.cat([top, bottom], dim=0)                               # [dim+1, dim+1]


def component_dofs(n_points, n_dof, comp):
    """Global DOF indices of component ``comp`` under the node-major
    ``[u, v, w, p]`` layout (``comp`` 0..dim-1 = velocity, last = pressure)."""
    return torch.arange(n_points) * n_dof + comp


def solve_cavity_3d(re=100, chara_length=0.05, max_iter=30, tol=1e-4):
    setup_headless()
    print(f"Solving 3D lid-driven cavity at Re={re}, chara_length={chara_length}...")

    # --- Mesh and physical parameters ---
    mesh = Mesh.gen_cube(chara_length=chara_length).double()
    points = mesh.points
    n_points = points.shape[0]
    n_dof = 4  # (u, v, w, p) per node
    print(f"  Mesh: {n_points} nodes, {n_points * n_dof} DOFs")

    rho = 1.0
    mu = 1.0 / re
    tau = 0.5 * chara_length  # mesh-size-scaled stabilization parameter

    # --- Boundary conditions ---
    is_boundary = mesh.boundary_mask
    is_top = points[:, 1] > 1.0 - 1e-6

    bc_mask = torch.zeros(n_points * n_dof, dtype=torch.bool)
    bc_val = torch.zeros(n_points * n_dof, dtype=torch.float64)

    for d in range(3):  # no-slip (u = v = w = 0) on every boundary node
        bc_mask[component_dofs(n_points, n_dof, d)] = is_boundary
    bc_val[component_dofs(n_points, n_dof, 0)[is_top]] = 1.0  # moving lid: u = 1 on top
    bc_mask[n_dof - 1] = True  # pin pressure at node 0 to fix the constant null space

    # --- Picard iteration ---
    assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
    condenser = Condenser(bc_mask, bc_val)

    u_full = torch.zeros(n_points * n_dof, dtype=torch.float64)
    u_full[bc_mask] = bc_val[bc_mask]

    for i in range(max_iter):
        w_prev = u_full.reshape(-1, n_dof)[:, :3]  # previous-iterate velocity (3D)
        K = assembler(points, point_data={"w_prev": w_prev})
        f = torch.zeros(n_points * n_dof, dtype=torch.float64)

        K_, f_ = condenser(K, f)
        u_new = condenser.recover(K_.solve(f_))

        diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
        print(f"  Picard {i:2d}: relative diff = {diff:.6e}")
        u_full = u_new
        if diff < tol:
            print("Converged!")
            break

    # --- Post-processing ---
    sol = u_full.reshape(-1, n_dof)
    velocity = sol[:, :3]
    pressure = sol[:, 3]
    speed = torch.norm(velocity, dim=1)
    print(f"  Max speed: {speed.max().item():.4f}, "
          f"pressure range: [{pressure.min().item():.4f}, {pressure.max().item():.4f}]")

    # Volumetric output: VTU for ParaView (full field) + a quick PyVista slice.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mesh.register_point_data("speed", speed)
    mesh.register_point_data("pressure", pressure)
    mesh.register_point_data("velocity", velocity)
    vtu_path = os.path.join(out_dir, "cavity_3d.vtu")
    mesh.save(vtu_path)
    print(f"Saved: {vtu_path}")

    try:
        import pyvista as pv

        grid = pv.read(vtu_path)
        slice_z = grid.slice(normal="z", origin=(0.5, 0.5, 0.5))  # mid-depth x-y plane

        p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 700))
        p.subplot(0, 0)
        p.add_mesh(slice_z, scalars="speed", cmap="jet", show_scalar_bar=True)
        p.add_text("Speed (z=0.5 slice)", font_size=10, position="upper_edge")
        p.view_xy()
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


if __name__ == "__main__":
    solve_cavity_3d(re=100, chara_length=0.05, max_iter=30)
