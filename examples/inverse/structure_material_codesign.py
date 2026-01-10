"""
Topology Optimization - Minimum Compliance (SIMP Method)

Optimizes structural topology using the SIMP (Solid Isotropic Material with 
Penalization) method with the Optimality Criteria (OC) update scheme.

Key formulation:
- Design variable: ρ ∈ [0, 1] (element density)
- Material model: E(ρ) = E_min + ρ^p * (E_max - E_min)
- Volume constraint: mean(ρ) ≤ vf

Usage:
    python structure_material_codesign.py --epoch 100 --vf 0.5

Reference: 
  - 88-line MATLAB topology optimization code
  - JAX-FEM topology optimization examples
"""

import sys
sys.path.append("../..")

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.tri as mtri
import scipy.spatial

from tensormesh import ElementAssembler, Mesh, Condenser


# ============================================================================
# Optimality Criteria (OC) Optimizer
# ============================================================================

class OCOptimizer:
    """
    Optimality Criteria (OC) optimizer for topology optimization.
    
    Similar interface to torch.optim.Optimizer, but specifically designed
    for density-based topology optimization with volume constraints.
    
    The OC update rule is:
        ρ_new = ρ * Be^η
        
    where:
        Be = sqrt(-dC/dρ / (λ * dV/dρ))
        λ: Lagrange multiplier (found by bisection to satisfy volume constraint)
        η = 0.5: damping exponent
    
    Args:
        params: Tensor or list of tensors (design variables)
        vf: Target volume fraction
        move_limit: Maximum density change per iteration (default: 0.2)
        rho_min: Minimum density (default: 1e-3)
        rho_max: Maximum density (default: 1.0)
        eta: Damping exponent (default: 0.5)
        bisection_tol: Tolerance for bisection (default: 1e-4)
        bisection_max_iter: Maximum bisection iterations (default: 50)
    """
    
    def __init__(
        self,
        params,
        vf: float,
        move_limit: float = 0.2,
        rho_min: float = 1e-3,
        rho_max: float = 1.0,
        eta: float = 0.5,
        bisection_tol: float = 1e-4,
        bisection_max_iter: int = 50,
    ):
        if isinstance(params, torch.Tensor):
            self.params = [params]
        else:
            self.params = list(params)
        
        self.vf = vf
        self.move_limit = move_limit
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.eta = eta
        self.bisection_tol = bisection_tol
        self.bisection_max_iter = bisection_max_iter
        
        self.state = {'step': 0, 'lambda': 1.0}
    
    def zero_grad(self):
        """Clear gradients of all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    @torch.no_grad()
    def step(self, dc=None, dv=None):
        """
        Perform OC update step.
        
        Args:
            dc: Compliance sensitivity (dC/dρ). If None, uses param.grad.
            dv: Volume sensitivity (dV/dρ). If None, uses uniform 1/n_elem.
        
        Returns:
            dict: Step info including 'lambda' and 'volume'
        """
        self.state['step'] += 1
        
        for rho in self.params:
            if dc is None:
                if rho.grad is None:
                    raise RuntimeError("No gradient found. Call backward() first.")
                dc_local = rho.grad.clone()
            else:
                dc_local = dc
            
            if dv is None:
                dv_local = torch.ones_like(rho) / rho.numel()
            else:
                dv_local = dv
            
            lam_min, lam_max = 1e-10, 1e10
            
            for _ in range(self.bisection_max_iter):
                lam_mid = 0.5 * (lam_min + lam_max)
                
                dc_positive = (-dc_local).clamp(min=1e-10)
                Be = (dc_positive / (lam_mid * dv_local)).clamp(min=1e-10) ** self.eta
                
                rho_new = rho * Be
                rho_new = torch.maximum(rho_new, rho - self.move_limit)
                rho_new = torch.minimum(rho_new, rho + self.move_limit)
                rho_new = rho_new.clamp(self.rho_min, self.rho_max)
                
                if rho_new.mean() > self.vf:
                    lam_min = lam_mid
                else:
                    lam_max = lam_mid
                
                if (lam_max - lam_min) / (lam_min + lam_max) < self.bisection_tol:
                    break
            
            rho.copy_(rho_new)
            self.state['lambda'] = lam_mid
        
        return {'lambda': self.state['lambda'], 'volume': self.params[0].mean().item()}
    
    def get_stats(self):
        """Get optimizer statistics."""
        rho = self.params[0]
        return {
            'step': self.state['step'],
            'lambda': self.state['lambda'],
            'volume': rho.mean().item(),
            'n_void': (rho < 0.1).sum().item(),
            'n_solid': (rho > 0.9).sum().item(),
        }


# ============================================================================
# SIMP Stiffness Assembler
# ============================================================================

class SIMPStiffnessAssembler(ElementAssembler):
    """
    SIMP stiffness matrix assembler for topology optimization.
    
    Material model:
        E(ρ) = E_min + ρ^p * (E_max - E_min)
        
    where:
        E_max = 1.0 (solid material)
        E_min = 1e-9 (void, for numerical stability)
        p = 3 (penalization power)
        
    Constitutive relation (linear elasticity):
        σ = λ tr(ε) I + 2μ ε
        
    where:
        μ = E / (2(1+ν))
        λ = Eν / ((1+ν)(1-2ν))
    """
    
    def __post_init__(self):
        self.Emax = 1.0        # Solid material Young's modulus
        self.Emin = 1e-9       # Void stiffness (numerical stability)
        self.nu = 0.3          # Poisson's ratio
        self.penal = 3.0       # SIMP penalization power
    
    def forward(self, gradu, gradv, rho):
        """
        Parameters (after vmap):
            gradu, gradv: [dim] - gradient of single basis function
            rho: [] - scalar element density
        
        Returns: [dim, dim] - contribution to stiffness matrix
        """
        dim = gradu.shape[0]
        
        # SIMP interpolation
        E = self.Emin + (rho ** self.penal) * (self.Emax - self.Emin)
        
        # Lamé parameters
        mu = E / (2.0 * (1.0 + self.nu))
        lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        
        # Stiffness contribution
        K = lam * (gradu @ gradv) * torch.eye(dim, dtype=gradu.dtype, device=gradu.device) \
            + mu * (torch.outer(gradu, gradv) + torch.outer(gradv, gradu))
        
        return K


def create_neumann_load(mesh, load_mask, traction, dim=2):
    """Create Neumann load vector."""
    n_points = mesh.n_points
    F = torch.zeros(n_points, dim, dtype=mesh.dtype, device=mesh.device)
    
    traction = torch.tensor(traction, dtype=mesh.dtype, device=mesh.device)
    n_load_nodes = load_mask.sum().item()
    
    if n_load_nodes > 0:
        F[load_mask] = traction / n_load_nodes
    
    return F.flatten()


# ============================================================================
# Density Filter (not sensitivity filter)
# ============================================================================

def build_density_filter(mesh, rmin):
    """
    Build density filter (convolution-based).
    
    H[i,j] = max(0, rmin - dist(i,j))
    rho_filtered[i] = sum_j(H[i,j] * rho[j]) / sum_j(H[i,j])
    """
    elements = mesh.elements()
    points = mesh.points.detach().cpu().numpy()
    centroids = points[elements.cpu().numpy()].mean(axis=1)
    n_elements = len(elements)
    
    print(f"Building density filter with rmin = {rmin:.4f}")
    
    # Build filter matrix using KD-tree
    kd_tree = scipy.spatial.KDTree(centroids)
    
    I, J, V = [], [], []
    
    for i in range(n_elements):
        # Query all neighbors within rmin
        neighbors = kd_tree.query_ball_point(centroids[i], rmin)
        for j in neighbors:
            d = np.linalg.norm(centroids[i] - centroids[j])
            val = max(0.0, rmin - d)
            if val > 0:
                I.append(i)
                J.append(j)
                V.append(val)
    
    H = scipy.sparse.csc_array((V, (I, J)), shape=(n_elements, n_elements))
    H_np = np.array(H.todense())
    Hs_np = H_np.sum(axis=1, keepdims=True)
    
    # Normalize to create convolution filter
    H_normalized = H_np / (Hs_np + 1e-10)
    
    H_torch = torch.tensor(H_normalized, dtype=mesh.dtype, device=mesh.device)
    
    return H_torch


def apply_density_filter(H, rho):
    """Apply density filter."""
    return H @ rho


def heaviside_projection(rho, beta, eta=0.5):
    """
    Heaviside projection for sharper 0/1 designs.
    As beta increases, the projection becomes sharper.
    """
    if beta <= 0:
        return rho
    
    beta_tensor = torch.tensor(beta, dtype=rho.dtype, device=rho.device)
    eta_tensor = torch.tensor(eta, dtype=rho.dtype, device=rho.device)
    one_tensor = torch.tensor(1.0, dtype=rho.dtype, device=rho.device)
    
    numerator = torch.tanh(beta_tensor * eta_tensor) + torch.tanh(beta_tensor * (rho - eta_tensor))
    denominator = torch.tanh(beta_tensor * eta_tensor) + torch.tanh(beta_tensor * (one_tensor - eta_tensor))
    return numerator / denominator


# ============================================================================
# Visualization
# ============================================================================

def plot_result(mesh, rho, u, save_path="codesign_result.png"):
    """Plot optimization results."""
    fig = plt.figure(figsize=(12, 4))
    
    points = mesh.points.detach().cpu().numpy()
    elements = mesh.elements().cpu().numpy()
    verts = [points[elem] for elem in elements]
    
    rho_np = rho.detach().cpu().numpy()
    
    # Displacement magnitude
    u_np = u.detach().cpu().numpy()
    u_x, u_y = u_np[0::2], u_np[1::2]
    u_mag = np.sqrt(u_x**2 + u_y**2)
    
    # Create triangulation for displacement
    if mesh.default_element_type == 'quad':
        tri_elements = []
        for elem in elements:
            tri_elements.append([elem[0], elem[1], elem[2]])
            tri_elements.append([elem[0], elem[2], elem[3]])
        tri_elements = np.array(tri_elements)
    else:
        tri_elements = elements
    tri = mtri.Triangulation(points[:, 0], points[:, 1], tri_elements)
    
    n_void = (rho_np < 0.1).sum()
    n_solid = (rho_np > 0.9).sum()
    
    # Panel 1: Density (structure)
    ax1 = fig.add_subplot(1, 2, 1)
    coll1 = PolyCollection(verts, array=rho_np, cmap='gray_r', 
                           edgecolors='none', linewidths=0.0)
    coll1.set_clim(0, 1)
    ax1.add_collection(coll1)
    ax1.autoscale()
    ax1.set_aspect('equal')
    ax1.set_title(f'Density ρ\n(void={n_void}, solid={n_solid})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(coll1, ax=ax1, shrink=0.6)
    
    # Panel 2: Displacement
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.tripcolor(tri, u_mag, cmap='coolwarm', shading='gouraud')
    ax2.set_aspect('equal')
    ax2.set_title(f'Displacement |u|\n(max={u_mag.max():.4f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im, ax=ax2, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_convergence(history, vf, save_path="codesign_convergence.png"):
    """Plot convergence history."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    epochs = np.arange(len(history['compliance']))
    
    ax = axes[0]
    ax.semilogy(epochs, history['compliance'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Compliance (log)')
    ax.set_title('Objective Function')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(epochs, history['volume'], 'r-', linewidth=2, label='Volume')
    ax.axhline(y=vf, color='k', linestyle='--', linewidth=1, label=f'Target vf={vf}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Volume Fraction')
    ax.set_title('Volume Constraint')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def create_animation(mesh, frames, save_path="codesign_animation.mp4", fps=10,
                     bc_info=None):
    """
    Create animation of optimization process with boundary conditions.
    
    Args:
        mesh: TensorMesh mesh object
        frames: List of frame dictionaries with 'epoch', 'theta1'/'rho', 'compliance'
        save_path: Output file path
        fps: Frames per second
        bc_info: Dictionary with boundary condition info:
            - 'fixed_pts': indices of fixed nodes
            - 'load_pt': index of load node
            - 'load_dir': load direction vector [fx, fy]
    """
    if len(frames) == 0:
        print("  No frames to animate")
        return
    
    points = mesh.points.detach().cpu().numpy()
    elements = mesh.elements().cpu().numpy()
    verts = [points[elem] for elem in elements]
    
    # Get domain bounds for scaling
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    domain_size = max(x_max - x_min, y_max - y_min)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Density field name (support both 'theta1' and 'rho')
    density_key = 'theta1' if 'theta1' in frames[0] else 'rho'
    
    # Density plot
    coll = PolyCollection(verts, array=frames[0][density_key], cmap='gray_r',
                         edgecolors='none', linewidths=0.0)
    coll.set_clim(0, 1)
    ax1.add_collection(coll)
    ax1.autoscale()
    ax1.set_aspect('equal')
    title1 = ax1.set_title(f'Iteration 0', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(coll, ax=ax1, shrink=0.6, label='Density')
    
    # Draw boundary conditions
    if bc_info is not None:
        # Fixed boundary (triangular markers)
        if 'fixed_pts' in bc_info and len(bc_info['fixed_pts']) > 0:
            fixed_pts = bc_info['fixed_pts']
            fixed_coords = points[fixed_pts]
            # Draw triangular fixed supports
            marker_size = domain_size * 0.02
            for pt in fixed_coords[::max(1, len(fixed_coords)//10)]:  # Sample points
                # Triangle pointing left (fixed in x)
                triangle = plt.Polygon([
                    [pt[0] - marker_size, pt[1]],
                    [pt[0] - marker_size * 2, pt[1] + marker_size * 0.5],
                    [pt[0] - marker_size * 2, pt[1] - marker_size * 0.5]
                ], color='blue', alpha=0.8)
                ax1.add_patch(triangle)
            # Draw fixed boundary line
            ax1.plot([x_min, x_min], [y_min, y_max], 'b-', linewidth=3, label='Fixed')
        
        # Load point and direction (arrow)
        if 'load_pt' in bc_info and 'load_dir' in bc_info:
            load_pt = bc_info['load_pt']
            load_dir = bc_info['load_dir']
            load_coord = points[load_pt]
            arrow_scale = domain_size * 0.15
            # Normalize direction
            load_mag = np.sqrt(load_dir[0]**2 + load_dir[1]**2)
            if load_mag > 0:
                load_dir_norm = [load_dir[0]/load_mag, load_dir[1]/load_mag]
                ax1.annotate('', 
                    xy=(load_coord[0] + load_dir_norm[0] * arrow_scale * 0.3, 
                        load_coord[1] + load_dir_norm[1] * arrow_scale * 0.3),
                    xytext=(load_coord[0] - load_dir_norm[0] * arrow_scale, 
                            load_coord[1] - load_dir_norm[1] * arrow_scale),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3),
                )
                ax1.plot(load_coord[0], load_coord[1], 'ro', markersize=8)
                ax1.annotate('F', (load_coord[0] - load_dir_norm[0] * arrow_scale * 1.2,
                                   load_coord[1] - load_dir_norm[1] * arrow_scale * 1.2),
                            fontsize=12, color='red', fontweight='bold')
    
    # Compliance history
    compliances = [f['compliance'] for f in frames]
    epochs = [f['epoch'] for f in frames]
    line, = ax2.semilogy([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, max(epochs) + 1)
    ax2.set_ylim(min(compliances) * 0.9, max(compliances) * 1.1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Compliance')
    ax2.set_title('Convergence', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    def update(frame_idx):
        frame = frames[frame_idx]
        coll.set_array(frame[density_key])
        title1.set_text(f'Iteration {frame["epoch"]}, C={frame["compliance"]:.2e}')
        
        # Update compliance line
        current_epochs = epochs[:frame_idx+1]
        current_compliances = compliances[:frame_idx+1]
        line.set_data(current_epochs, current_compliances)
        
        return coll, title1, line
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    
    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Topology Optimization'})
        anim.save(save_path, writer=writer, dpi=150)
        print(f"  Saved: {save_path}")
    except Exception as e:
        # Fallback to GIF if ffmpeg not available
        gif_path = save_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
            print(f"  Saved: {gif_path} (ffmpeg not available)")
        except Exception as e2:
            print(f"  Warning: Could not save animation ({e2})")
    
    plt.close()


# ============================================================================
# Volume Constraint
# ============================================================================

def compute_volume_constraint(rho, vf):
    """
    Compute volume constraint.
    g = mean(ρ) / vf - 1
    
    g = 0 means we're at the target volume fraction.
    g < 0 means below target (feasible for inequality constraint).
    g > 0 means above target (violated).
    """
    return rho.mean() / vf - 1.0


# ============================================================================
# Main Optimization
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SIMP Topology Optimization")
    
    # Geometry
    parser.add_argument('--length', type=float, default=2.0, help='Domain length')
    parser.add_argument('--height', type=float, default=1.0, help='Domain height')
    parser.add_argument('--n_elem_x', type=int, default=80, help='Elements in x')
    parser.add_argument('--n_elem_y', type=int, default=40, help='Elements in y')
    
    # Optimization
    parser.add_argument('--epoch', type=int, default=100, help='Number of iterations')
    parser.add_argument('--vf', type=float, default=0.5, help='Volume fraction')
    parser.add_argument('--filter_radius', type=float, default=0.03, help='Filter radius')
    parser.add_argument('--move_limit', type=float, default=0.2, help='Move limit per iteration')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  SIMP TOPOLOGY OPTIMIZATION")
    print("=" * 70)
    print(f"  Domain: {args.length} × {args.height}")
    print(f"  Mesh:   {args.n_elem_x} × {args.n_elem_y} quads")
    print(f"  vf:     {args.vf}")
    print(f"  Filter: {args.filter_radius}")
    print("=" * 70)
    
    # Create structured mesh
    chara_length = min(args.length / args.n_elem_x, args.height / args.n_elem_y)
    mesh = Mesh.gen_rectangle(
        left=0.0, right=args.length,
        bottom=0.0, top=args.height,
        chara_length=chara_length,
        element_type="quad"
    ).double()
    
    n_points = mesh.n_points
    n_elements = mesh.n_elements
    dim = 2
    
    print(f"\nMesh: {n_points} nodes, {n_elements} elements")
    
    # Setup assembler
    K_asm = SIMPStiffnessAssembler.from_mesh(mesh)
    
    # Load location: single point at (length, height/2) - like SIMP example
    points_np = mesh.points.detach().cpu().numpy()
    right_boundary = mesh.point_data['is_right_boundary'].flatten().cpu().numpy()
    y_center = args.height / 2.0
    
    # Find closest point to (length, height/2) on right boundary
    right_pts = np.where(right_boundary)[0]
    target = np.array([args.length, y_center])
    dists = np.linalg.norm(points_np[right_pts] - target, axis=1)
    load_idx = right_pts[np.argmin(dists)]
    
    # Create point load vector
    F = torch.zeros(n_points * dim, dtype=mesh.dtype, device=mesh.device)
    F[load_idx * dim + 1] = -1.0  # Downward force
    
    # Boundary conditions
    dbc_mask = mesh.point_data['is_left_boundary'].flatten().repeat_interleave(dim)
    condenser = Condenser(dbc_mask)
    
    print(f"Dirichlet DOFs: {dbc_mask.sum().item()} / {n_points * dim}")
    print(f"Load at node {load_idx}, position {points_np[load_idx]}")
    
    # Build density filter
    H_filter = build_density_filter(mesh, args.filter_radius)
    
    # Optimizer parameters for OC method
    move_limit = args.move_limit
    
    # History
    history = {
        'compliance': [],
        'volume': [],
    }
    
    # For animation: store frames
    animation_frames = []
    save_interval = max(1, args.epoch // 50)  # Save ~50 frames for animation
    
    # For paper figures: store snapshots
    middle_epoch = 5  # Early iteration to show topology evolution
    rho_middle = None
    
    # Initialize density with uniform vf
    rho = torch.full((n_elements,), args.vf, dtype=mesh.dtype, device=mesh.device)
    rho_initial = rho.clone()
    
    print(f"\nInitial density: {args.vf}")
    print("Starting optimization with OC method...")
    pbar = tqdm(range(args.epoch), desc="Optimizing", ncols=100)
    
    for epoch in pbar:
        # Apply density filter
        rho_filt = apply_density_filter(H_filter, rho).clamp(1e-3, 1.0)
        
        # Need gradients for sensitivity analysis
        rho_var = rho.clone().requires_grad_(True)
        rho_p = apply_density_filter(H_filter, rho_var).clamp(1e-3, 1.0)
        
        # Assemble and solve
        K = K_asm(mesh.points, element_data={"rho": rho_p})
        K_, F_ = condenser(K, F)
        u_ = K_.solve(F_, backend="scipy")
        u = condenser.recover(u_)
        
        # Compliance (objective)
        compliance = u @ F
        
        # Compute sensitivities via autograd
        compliance.backward()
        dc = rho_var.grad.clone()
        
        # Volume sensitivity
        dv = torch.ones_like(rho) / n_elements
        
        # OC update
        with torch.no_grad():
            # Find Lagrange multiplier using bisection
            lam_min, lam_max = 1e-10, 1e10
            for _ in range(50):
                lam_mid = 0.5 * (lam_min + lam_max)
                
                # OC update formula: Be = sqrt(-dc / (lambda * dv))
                # Note: dc is negative, so -dc is positive
                dc_positive = (-dc).clamp(min=1e-10)
                Be = (dc_positive / (lam_mid * dv)).clamp(min=1e-10) ** 0.5
                
                # Compute new density
                rho_new = rho * Be
                
                # Apply move limits and bounds
                rho_new = torch.maximum(rho_new, rho - move_limit)
                rho_new = torch.minimum(rho_new, rho + move_limit)
                rho_new = rho_new.clamp(1e-3, 1.0)
                
                # Check constraint
                if rho_new.mean() > args.vf:
                    lam_min = lam_mid
                else:
                    lam_max = lam_mid
            
            # Update density
            rho = rho_new
        
        # Record history
        history['compliance'].append(compliance.item())
        history['volume'].append(rho.mean().item())
        
        # Save frame for animation
        if epoch % save_interval == 0 or epoch == args.epoch - 1:
            animation_frames.append({
                'epoch': epoch,
                'rho': rho.clone().cpu().numpy(),
                'compliance': compliance.item()
            })
        
        # Save middle snapshot for paper figures
        if epoch == middle_epoch:
            rho_middle = rho.clone()
        
        # Show topology info in progress bar
        n_void = (rho < 0.1).sum().item()
        n_solid = (rho > 0.9).sum().item()
        pbar.set_postfix({
            'C': f"{compliance.item():.2e}",
            'void': n_void,
            'solid': n_solid
        })
    
    # Final results
    n_void = (rho < 0.1).sum().item()
    n_solid = (rho > 0.9).sum().item()
    n_gray = n_elements - n_void - n_solid
    
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    with torch.no_grad():
        rho_filt = apply_density_filter(H_filter, rho).clamp(1e-3, 1.0)
        K_final = K_asm(mesh.points, element_data={"rho": rho_filt})
        K_f, F_f = condenser(K_final, F)
        u_final = condenser.recover(K_f.solve(F_f, backend="scipy"))
        final_compliance = (u_final @ F).item()
    
    print(f"  Final compliance: {final_compliance:.4e}")
    print(f"  Density range: [{rho.min().item():.3f}, {rho.max().item():.3f}]")
    print(f"  Elements: {n_void} void, {n_solid} solid, {n_gray} gray")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    out_prefix = os.path.join(args.output_dir, "topo_opt")
    plot_result(mesh, rho, u_final, save_path=f"{out_prefix}_result.png")
    plot_convergence(history, args.vf, save_path=f"{out_prefix}_convergence.png")
    
    # Generate animation with boundary condition visualization
    print("  Generating optimization animation...")
    
    # Get fixed boundary node indices
    left_boundary = mesh.point_data['is_left_boundary'].flatten().cpu().numpy()
    fixed_pts = np.where(left_boundary)[0]
    
    bc_info = {
        'fixed_pts': fixed_pts,
        'load_pt': load_idx,
        'load_dir': [0.0, -1.0]  # Downward force
    }
    create_animation(mesh, animation_frames, save_path=f"{out_prefix}_animation.mp4",
                    bc_info=bc_info)
    
    # =========================================================================
    # Generate paper figures (4 individual plots)
    # =========================================================================
    print("\nGenerating individual paper figures...")
    
    rho_final = rho.clone()
    if rho_middle is None:
        rho_middle = rho_final  # fallback
    
    points = mesh.points.detach().cpu().numpy()
    elements = mesh.elements().cpu().numpy()
    verts = [points[elem] for elem in elements]
    
    def plot_density_paper(rho_data, title, save_name, verts, points, bc_info=None):
        """Plot a single density figure for paper."""
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        rho_np = rho_data.detach().cpu().numpy()
        coll = PolyCollection(verts, array=rho_np, cmap='gray_r',
                             edgecolors='none', linewidths=0.0)
        coll.set_clim(0, 1)
        ax.add_collection(coll)
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        plt.colorbar(coll, ax=ax, shrink=0.8, label='Density ρ')
        
        # Draw boundary conditions
        if bc_info is not None:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            domain_size = max(x_max - x_min, y_max - y_min)
            
            # Fixed boundary
            if 'fixed_pts' in bc_info and len(bc_info['fixed_pts']) > 0:
                ax.plot([x_min, x_min], [y_min, y_max], 'b-', linewidth=3)
                fixed_pts_arr = bc_info['fixed_pts']
                fixed_coords = points[fixed_pts_arr]
                marker_size = domain_size * 0.02
                for pt in fixed_coords[::max(1, len(fixed_coords)//8)]:
                    triangle = plt.Polygon([
                        [pt[0] - marker_size, pt[1]],
                        [pt[0] - marker_size * 2, pt[1] + marker_size * 0.5],
                        [pt[0] - marker_size * 2, pt[1] - marker_size * 0.5]
                    ], color='blue', alpha=0.8)
                    ax.add_patch(triangle)
            
            # Load
            if 'load_pt' in bc_info and 'load_dir' in bc_info:
                load_pt_idx = bc_info['load_pt']
                load_dir = bc_info['load_dir']
                load_coord = points[load_pt_idx]
                arrow_scale = domain_size * 0.12
                load_mag = np.sqrt(load_dir[0]**2 + load_dir[1]**2)
                if load_mag > 0:
                    load_dir_norm = [load_dir[0]/load_mag, load_dir[1]/load_mag]
                    ax.annotate('', 
                        xy=(load_coord[0] + load_dir_norm[0] * arrow_scale * 0.3, 
                            load_coord[1] + load_dir_norm[1] * arrow_scale * 0.3),
                        xytext=(load_coord[0] - load_dir_norm[0] * arrow_scale, 
                                load_coord[1] - load_dir_norm[1] * arrow_scale),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                    )
                    ax.plot(load_coord[0], load_coord[1], 'ro', markersize=6)
                    ax.annotate('F', (load_coord[0] - load_dir_norm[0] * arrow_scale * 1.3,
                                       load_coord[1] - load_dir_norm[1] * arrow_scale * 1.3),
                                fontsize=11, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_name + '.pdf', bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.png/pdf")
    
    # 1. Initial density
    plot_density_paper(rho_initial, 'Initial Design (ρ = vf)', 
                       f'{out_prefix}_paper_initial', verts, points, bc_info)
    
    # 2. Middle iteration
    plot_density_paper(rho_middle, f'Intermediate (Iter {middle_epoch})', 
                       f'{out_prefix}_paper_middle', verts, points, bc_info)
    
    # 3. Final result
    plot_density_paper(rho_final, f'Final Design (Iter {args.epoch})', 
                       f'{out_prefix}_paper_final', verts, points, bc_info)
    
    # 4. Compliance curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(history['compliance'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Compliance C', fontsize=12)
    ax.set_title('Convergence History', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Mark key points
    ax.axvline(x=middle_epoch, color='orange', linestyle='--', alpha=0.7, label=f'Iter {middle_epoch}')
    ax.scatter([0, middle_epoch, args.epoch-1], 
               [history['compliance'][0], history['compliance'][middle_epoch], history['compliance'][-1]],
               c=['green', 'orange', 'red'], s=60, zorder=5)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_paper_compliance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_prefix}_paper_compliance.pdf', bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out_prefix}_paper_compliance.png/pdf")
    
    print("\n  All paper figures saved!")
    
    print("\n" + "=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    print(f"  {out_prefix}_result.png/pdf")
    print(f"  {out_prefix}_convergence.png/pdf")
    print(f"  {out_prefix}_animation.mp4")
    print(f"  {out_prefix}_paper_initial.png/pdf")
    print(f"  {out_prefix}_paper_middle.png/pdf")
    print(f"  {out_prefix}_paper_final.png/pdf")
    print(f"  {out_prefix}_paper_compliance.png/pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()
