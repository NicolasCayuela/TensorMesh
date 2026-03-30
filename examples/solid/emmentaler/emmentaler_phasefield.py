"""
Emmentaler Phase-Field Fracture (AT1 Brittle Fracture)
=======================================================

Phase 3: Brittle phase-field fracture with Amor energy split on the Emmentaler
geometry, reproducing solidmechanics_datagen using TensorMesh.

Two coupled fields:
  - Displacement u [n_nodes, 3]
  - Damage alpha [n_nodes] (0 = intact, 1 = fully fractured)

Staggered solver (alternating minimization):
  - u sub-problem: LBFGS energy minimization with alpha frozen
  - alpha sub-problem: projected LBFGS with irreversibility alpha >= alpha_old

Boundary conditions (matching solidmechanics_datagen):
  - Bottom (z=0): u=0, alpha=0
  - Top (z=T): prescribed displacement (tension+bending+torsion), alpha=0

Usage:
    python emmentaler_phasefield.py                              # quick test
    python emmentaler_phasefield.py --h 0.08 --load_steps 20     # finer mesh
    python emmentaler_phasefield.py --mesh_file B0001.msh         # existing mesh
"""

import sys
import os
import argparse
import csv
import math
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Mesh
from tensormesh.assemble import ElementAssembler

from emmentaler_elasticity import (
    generate_emmentaler_mesh,
    identify_boundaries,
    compute_top_displacement,
)


# ---------------------------------------------------------------------------
# Phase-field fracture model (AT1 with Amor split)
# ---------------------------------------------------------------------------
class BrittlePhaseFieldModel(ElementAssembler):
    """AT1 brittle phase-field fracture model with Amor energy split.

    Energy density:
        psi = g(alpha) * psi_D(eps) + psi_R(eps) + psi_frac(alpha, grad_alpha)

    Amor split:
        psi_0(eps) = lam/2 * tr(eps)^2 + mu * eps:eps
        psi_R(eps) = kappa/2 * H(-tr(eps)) * tr(eps)^2   (compressive)
        psi_D(eps) = psi_0 - psi_R                        (degradable)

    Degradation: g(alpha) = (1 - alpha)^2 + g0
    Dissipation: w(alpha) = alpha  (AT1 model)
    Fracture energy: Gc/cw * [alpha/ell + ell * |grad_alpha|^2]

    Reference: solidmechanics_datagen/fem/model.py
    """

    def __post_init__(self, E=12000.0, nu=0.3, Gc=0.0014, ell=0.075, g0=2e-5):
        self.E_mod = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = E / (3 * (1 - 2 * nu))  # bulk modulus (3D)
        self.Gc = Gc
        self.ell = ell
        self.g0 = g0
        self.cw = 8.0 / 3.0  # AT1 normalization constant

    def element_energy(self, graddisplacement, alpha, gradalpha):
        """Total energy density (elastic + fracture) at one quadrature point.

        Parameters
        ----------
        graddisplacement : [dim, dim]  displacement gradient
        alpha : scalar                 damage field value
        gradalpha : [dim]              damage field gradient
        """
        grad_u = graddisplacement
        dim = grad_u.shape[-1]

        # Strain: eps = sym(grad_u)
        eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))
        tr_eps = eps.diagonal(dim1=-2, dim2=-1).sum(-1)

        # Standard elastic energy
        psi_0 = 0.5 * self.lam * tr_eps ** 2 + self.mu * (eps * eps).sum()

        # Amor split: compressive (residual) part
        psi_R = torch.where(
            tr_eps < 0,
            0.5 * self.kappa * tr_eps ** 2,
            torch.zeros_like(tr_eps),
        )

        # Degradable part
        psi_D = psi_0 - psi_R

        # Degradation function
        g_alpha = (1 - alpha) ** 2 + self.g0

        # Elastic energy with damage
        psi_el = g_alpha * psi_D + psi_R

        # Fracture energy: Gc/cw * [w(alpha)/ell + ell * |grad_alpha|^2]
        psi_frac = (self.Gc / self.cw) * (
            alpha / self.ell + self.ell * (gradalpha * gradalpha).sum()
        )

        return psi_el + psi_frac


# ---------------------------------------------------------------------------
# Sub-problem solvers
# ---------------------------------------------------------------------------
def solve_u_subproblem(model, u_data, alpha_frozen, u_prescribed,
                       free_mask_u_float, max_iter=50):
    """Minimize E_tot w.r.t. u with alpha frozen (LBFGS)."""
    u_opt = u_data.detach().clone().requires_grad_(True)
    alpha_det = alpha_frozen.detach()

    optimizer = optim.LBFGS(
        [u_opt], lr=1.0, max_iter=max_iter, max_eval=int(max_iter * 1.2),
        tolerance_grad=1e-8, tolerance_change=1e-12,
        history_size=50, line_search_fn="strong_wolfe",
    )

    final_energy = [0.0]

    def closure():
        optimizer.zero_grad()
        u_active = u_opt * free_mask_u_float + u_prescribed
        E = model.energy(point_data={"displacement": u_active, "alpha": alpha_det})
        if E.requires_grad:
            E.backward()
        final_energy[0] = E.item()
        return E

    optimizer.step(closure)
    return u_opt, final_energy[0]


def solve_alpha_subproblem(model, alpha_data, u_frozen, alpha_lower,
                           free_mask_alpha_float, max_iter=50):
    """Minimize E_tot w.r.t. alpha with u frozen (projected LBFGS).

    After each LBFGS step, projects alpha into [alpha_lower, 1.0]
    and enforces Dirichlet BCs (alpha=0 at boundaries).
    """
    u_det = u_frozen.detach()
    alpha_opt = alpha_data.detach().clone().requires_grad_(True)

    optimizer = optim.LBFGS(
        [alpha_opt], lr=1.0, max_iter=max_iter, max_eval=int(max_iter * 1.2),
        tolerance_grad=1e-8, tolerance_change=1e-12,
        history_size=50, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        # Apply BC and bounds
        alpha_active = alpha_opt * free_mask_alpha_float
        alpha_clamped = torch.max(alpha_active, alpha_lower)
        alpha_clamped = torch.clamp(alpha_clamped, max=1.0)
        E = model.energy(point_data={"displacement": u_det, "alpha": alpha_clamped})
        if E.requires_grad:
            E.backward()
        return E

    optimizer.step(closure)

    # Final projection: enforce [alpha_lower, 1] and BCs
    with torch.no_grad():
        alpha_opt.data.mul_(free_mask_alpha_float)
        alpha_opt.data = torch.max(alpha_opt.data, alpha_lower)
        alpha_opt.data.clamp_(max=1.0)
        alpha_opt.data.mul_(free_mask_alpha_float)

    return alpha_opt


def compute_u_residual_norm(model, u, alpha, u_prescribed, free_mask_u,
                            free_mask_u_float):
    """Compute ||dE/du||_free — staggered convergence criterion."""
    u_eval = (u.detach() * free_mask_u_float + u_prescribed).requires_grad_(True)
    alpha_det = alpha.detach()
    E = model.energy(point_data={"displacement": u_eval, "alpha": alpha_det})
    grad_u = torch.autograd.grad(E, u_eval)[0]  # [n_nodes, 3]
    return grad_u[free_mask_u].norm().item()


def compute_energies(model, u, alpha, u_prescribed, free_mask_u_float,
                     free_mask_alpha_float):
    """Compute E_pot and E_frac separately for monitoring."""
    with torch.no_grad():
        u_active = u.detach() * free_mask_u_float + u_prescribed
        alpha_active = (alpha.detach() * free_mask_alpha_float).clamp(min=0.0)

        # Total energy
        E_tot = model.energy(
            point_data={"displacement": u_active, "alpha": alpha_active}
        ).item()

        # Elastic energy only (set alpha=0 → no fracture energy, no degradation)
        # To separate E_pot and E_frac, compute E with alpha=0 (elastic) separately
        alpha_zero = torch.zeros_like(alpha_active)
        E_elastic_undamaged = model.energy(
            point_data={"displacement": u_active, "alpha": alpha_zero}
        ).item()

        # E_frac = E_tot - E_pot where E_pot = integral of g(alpha)*psi_D + psi_R
        # But simpler: E_frac from fracture terms only
        # We can compute E_pot = E_tot - E_frac
        # For monitoring, E_tot is the most important

    return E_tot


# ---------------------------------------------------------------------------
# Stress computation with damage
# ---------------------------------------------------------------------------
def compute_nodal_fields(model, mesh, u_vec, alpha_vec, E, nu, g0=2e-5):
    """Compute effective (damaged) stress, strain, and von Mises stress."""
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    kappa = E / (3 * (1 - 2 * nu))

    n_nodes = mesh.points.shape[0]
    dim = mesh.points.shape[1]

    nodal_stress_sum = torch.zeros(n_nodes, 6, dtype=u_vec.dtype)
    nodal_strain_sum = torch.zeros(n_nodes, 6, dtype=u_vec.dtype)
    nodal_count = torch.zeros(n_nodes, dtype=u_vec.dtype)

    for element_type in model.element_types:
        trans = model.transformation[element_type]
        elements = model.elements[element_type]
        n_elem, n_basis = elements.shape

        shape_grad = trans.shape_grad  # [n_elem, n_quad, n_basis, dim]
        shape_val = trans.shape_val    # [n_quad, n_basis]

        elem_u = u_vec[elements]       # [n_elem, n_basis, dim]
        elem_alpha = alpha_vec[elements]  # [n_elem, n_basis]

        # grad_u at quad points: [n_elem, n_quad, dim, dim]
        grad_u = torch.einsum("eqbd,ebc->eqdc", shape_grad, elem_u)

        # alpha at quad points: [n_elem, n_quad]
        alpha_q = torch.einsum("qb,eb->eq", shape_val, elem_alpha)

        # Strain
        eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))
        tr_eps = eps.diagonal(dim1=-2, dim2=-1).sum(-1)  # [n_elem, n_quad]

        # Amor split stress
        eye = torch.eye(dim, dtype=eps.dtype, device=eps.device)

        # Undamaged stress: sigma_0 = lam*tr(eps)*I + 2*mu*eps
        sigma_0 = lam * tr_eps.unsqueeze(-1).unsqueeze(-1) * eye + 2 * mu * eps

        # Compressive stress: sigma_R = kappa * H(-tr_eps) * tr_eps * I
        tr_eps_neg = torch.where(tr_eps < 0, tr_eps, torch.zeros_like(tr_eps))
        sigma_R = kappa * tr_eps_neg.unsqueeze(-1).unsqueeze(-1) * eye

        # Degradable stress: sigma_D = sigma_0 - sigma_R
        sigma_D = sigma_0 - sigma_R

        # Effective (damaged) stress: sigma = g(alpha)*sigma_D + sigma_R
        g_alpha = ((1 - alpha_q) ** 2 + g0).unsqueeze(-1).unsqueeze(-1)
        sigma = g_alpha * sigma_D + sigma_R

        # Average over quadrature points
        sigma_avg = sigma.mean(dim=1)
        eps_avg = eps.mean(dim=1)

        # To Voigt: [xx, yy, zz, yz, xz, xy]
        sigma_voigt = torch.stack([
            sigma_avg[:, 0, 0], sigma_avg[:, 1, 1], sigma_avg[:, 2, 2],
            sigma_avg[:, 1, 2], sigma_avg[:, 0, 2], sigma_avg[:, 0, 1],
        ], dim=1)
        eps_voigt = torch.stack([
            eps_avg[:, 0, 0], eps_avg[:, 1, 1], eps_avg[:, 2, 2],
            eps_avg[:, 1, 2], eps_avg[:, 0, 2], eps_avg[:, 0, 1],
        ], dim=1)

        for b in range(n_basis):
            node_ids = elements[:, b]
            nodal_stress_sum.index_add_(0, node_ids, sigma_voigt)
            nodal_strain_sum.index_add_(0, node_ids, eps_voigt)
            nodal_count.index_add_(0, node_ids, torch.ones(n_elem, dtype=u_vec.dtype))

    nodal_count = nodal_count.clamp(min=1).unsqueeze(1)
    nodal_stress = nodal_stress_sum / nodal_count
    nodal_strain = nodal_strain_sum / nodal_count

    s = nodal_stress
    von_mises = torch.sqrt(0.5 * (
        (s[:, 0] - s[:, 1]) ** 2 +
        (s[:, 1] - s[:, 2]) ** 2 +
        (s[:, 2] - s[:, 0]) ** 2 +
        6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    ))

    return nodal_strain, nodal_stress, von_mises


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve_emmentaler_phasefield(
    mesh_file=None, h=0.15, E=12000.0, nu=0.3,
    Gc=0.0014, ell=0.075, g0=2e-5,
    normal_mag=1.0, bending_mag=1.0, torsion_mag=1.0,
    T=1.5, load_steps=10, max_stag_iter=100, stag_tol=1e-4,
    lbfgs_iter_u=50, lbfgs_iter_alpha=50,
    output_dir=".",
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Mesh ----
    if mesh_file is None:
        mesh_path = os.path.join(output_dir, "emmentaler.msh")
        generate_emmentaler_mesh(T=T, h=h, out_file=mesh_path)
        mesh = Mesh.read(mesh_path, reorder=True)
    else:
        mesh = Mesh.read(mesh_file, reorder=True)

    n_nodes = mesh.points.shape[0]
    n_cells = sum(v.shape[0] for v in mesh.cells.values())
    print(f"Mesh: {n_nodes} nodes, {n_cells} elements, types: {list(mesh.cells.keys())}")

    # ---- 2. Model ----
    model = BrittlePhaseFieldModel.from_mesh(mesh, E=E, nu=nu, Gc=Gc, ell=ell, g0=g0)
    print(f"Material: E={E}, nu={nu}, Gc={Gc}, ell={ell}, g0={g0}")

    # ---- 3. Boundaries ----
    bottom_mask, top_mask = identify_boundaries(mesh.points, T)
    print(f"Boundary nodes: {bottom_mask.sum().item()} bottom, {top_mask.sum().item()} top")

    # Displacement BC mask: [n_nodes] bool → [n_nodes, 1] float for broadcasting
    free_mask_u = ~(bottom_mask | top_mask)
    free_mask_u_float = free_mask_u.unsqueeze(1).to(mesh.points.dtype)

    # Alpha BC mask: alpha=0 at top and bottom
    free_mask_alpha = ~(bottom_mask | top_mask)
    free_mask_alpha_float = free_mask_alpha.to(mesh.points.dtype)

    points_top = mesh.points[top_mask]

    # ---- 4. Initialize fields ----
    u = torch.zeros(n_nodes, 3, dtype=mesh.points.dtype)
    alpha = torch.zeros(n_nodes, dtype=mesh.points.dtype)
    alpha_old = torch.zeros(n_nodes, dtype=mesh.points.dtype)

    # ---- 5. CSV monitoring ----
    csv_path = os.path.join(output_dir, "monitoring.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter="\t")
    csv_writer.writerow(["step", "lambda", "u_z", "kappa", "theta",
                         "E_tot", "u_max", "alpha_max", "R_u_norm", "stag_iters"])

    # ---- 6. Load stepping ----
    print(f"Solving with {load_steps} load steps, max {max_stag_iter} staggered iters...")
    for step in range(load_steps + 1):
        lf = step / load_steps

        # Update displacement BC
        u_top_target, u_z_val, kappa_val, theta_val = compute_top_displacement(
            points_top, lf,
            normal_mag=normal_mag, bending_mag=bending_mag, torsion_mag=torsion_mag,
        )
        u_prescribed = torch.zeros_like(mesh.points)
        u_prescribed[top_mask] = u_top_target

        # Irreversibility lower bound
        alpha_lower = alpha_old.detach().clone()
        # Ensure alpha_lower respects BC (alpha=0 at boundaries)
        alpha_lower[bottom_mask] = 0.0
        alpha_lower[top_mask] = 0.0

        # Staggered iteration
        R_u_norm = float("inf")
        stag_iter = 0
        for stag_iter in range(max_stag_iter):
            # (a) u sub-problem
            u, e_val = solve_u_subproblem(
                model, u, alpha, u_prescribed, free_mask_u_float,
                max_iter=lbfgs_iter_u,
            )

            # (b) alpha sub-problem
            alpha = solve_alpha_subproblem(
                model, alpha, u, alpha_lower, free_mask_alpha_float,
                max_iter=lbfgs_iter_alpha,
            )

            # (c) Convergence check
            R_u_norm = compute_u_residual_norm(
                model, u, alpha, u_prescribed, free_mask_u, free_mask_u_float,
            )

            if R_u_norm < stag_tol:
                stag_iter += 1  # count from 1
                break

        stag_iters = stag_iter + 1

        # Update irreversibility
        with torch.no_grad():
            alpha_old = torch.max(alpha.detach(), alpha_old)

        # Monitoring
        with torch.no_grad():
            u_active = u.detach() * free_mask_u_float + u_prescribed
            u_max = u_active.norm(dim=1).max().item()
            alpha_max = alpha.detach().max().item()
            E_tot = compute_energies(
                model, u, alpha, u_prescribed, free_mask_u_float,
                free_mask_alpha_float,
            )

        csv_writer.writerow([
            step, f"{lf:.6f}",
            f"{u_z_val:.6e}", f"{kappa_val:.6e}", f"{theta_val:.6e}",
            f"{E_tot:.6e}", f"{u_max:.6e}", f"{alpha_max:.6e}",
            f"{R_u_norm:.6e}", stag_iters,
        ])

        print(f"  step {step:3d}/{load_steps}  λ={lf:.4f}  "
              f"E={E_tot:.4e}  u_max={u_max:.4e}  α_max={alpha_max:.4f}  "
              f"‖R_u‖={R_u_norm:.2e}  stag={stag_iters}")

    csv_file.close()
    print(f"Monitoring data saved to {csv_path}")

    # ---- 7. Post-processing ----
    with torch.no_grad():
        u_final = (u.detach() * free_mask_u_float + u_prescribed)
        alpha_final = alpha.detach().clone()

    print("Computing stress and strain fields...")
    nodal_strain, nodal_stress, von_mises = compute_nodal_fields(
        model, mesh, u_final, alpha_final, E, nu, g0,
    )
    print(f"Max von Mises stress: {von_mises.max().item():.6e}")

    # ---- 8. Save VTK ----
    mesh.register_point_data("displacement", u_final)
    mesh.register_point_data("displacement_magnitude", u_final.norm(dim=1))
    mesh.register_point_data("alpha", alpha_final)
    mesh.register_point_data("strain", nodal_strain)
    mesh.register_point_data("stress", nodal_stress)
    mesh.register_point_data("von_mises_stress", von_mises)

    out_path = os.path.join(output_dir, "emmentaler_phasefield.vtk")
    mesh.save(out_path)
    print(f"Result saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emmentaler brittle phase-field fracture (TensorMesh)")
    parser.add_argument("--mesh_file", type=str, default=None)
    parser.add_argument("--h", type=float, default=0.15,
                        help="Mesh element size (default: 0.15)")
    parser.add_argument("--E", type=float, default=12000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--Gc", type=float, default=0.0014,
                        help="Fracture toughness (default: 0.0014)")
    parser.add_argument("--ell", type=float, default=0.075,
                        help="Length scale (default: 0.075)")
    parser.add_argument("--normal_mag", type=float, default=1.0)
    parser.add_argument("--bending_mag", type=float, default=1.0)
    parser.add_argument("--torsion_mag", type=float, default=1.0)
    parser.add_argument("--load_steps", type=int, default=10,
                        help="Number of load steps (default: 10)")
    parser.add_argument("--max_stag_iter", type=int, default=100,
                        help="Max staggered iterations (default: 100)")
    parser.add_argument("--stag_tol", type=float, default=1e-4,
                        help="Staggered convergence tolerance (default: 1e-4)")
    parser.add_argument("--output_dir", type=str, default="output_emmentaler_pf")
    args = parser.parse_args()

    solve_emmentaler_phasefield(
        mesh_file=args.mesh_file, h=args.h, E=args.E, nu=args.nu,
        Gc=args.Gc, ell=args.ell,
        normal_mag=args.normal_mag, bending_mag=args.bending_mag,
        torsion_mag=args.torsion_mag, load_steps=args.load_steps,
        max_stag_iter=args.max_stag_iter, stag_tol=args.stag_tol,
        output_dir=args.output_dir,
    )
