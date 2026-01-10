"""
SIMP Topology Optimization (Minimum Compliance)

Classic 88-line MATLAB style topology optimization implemented with TensorMesh.
This is a baseline implementation to verify the solver works correctly.

Problem:
    min  C = F^T u = u^T K u   (compliance)
    s.t. K u = F
         V(x) <= vf * V0        (volume constraint)
         0 < x_min <= x <= 1    (design bounds)

where:
    x: element densities
    K(x) = sum_e x_e^p K_e     (SIMP interpolation)
    p: penalization power (usually 3)

Usage:
    python simp_topology_optimization.py --epoch 100 --vf 0.5
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
import scipy.spatial

from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.optimizer import OCOptimizer


# ============================================================================
# Optimality Criteria (OC) Optimizer
# ============================================================================
# Moved to tensormesh.optimizer.OCOptimizer


class SIMPStiffnessAssembler(ElementAssembler):
    """
    SIMP stiffness matrix assembler.
    K_e = x_e^p * K_0
    """
    
    def __post_init__(self):
        self.E0 = 1.0          # Base Young's modulus
        self.Emin = 1e-9       # Minimum stiffness (for numerical stability)
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
        E = self.Emin + (rho ** self.penal) * (self.E0 - self.Emin)
        
        # Lamé parameters
        mu = E / (2.0 * (1.0 + self.nu))
        lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        
        # Stiffness contribution
        K = lam * (gradu @ gradv) * torch.eye(dim, dtype=gradu.dtype, device=gradu.device) \
            + mu * (torch.outer(gradu, gradv) + torch.outer(gradv, gradu))
        
        return K


def build_filter(mesh, rmin):
    """Build density filter matrix."""
    elements = mesh.elements()
    points = mesh.points.detach().cpu().numpy()
    centroids = points[elements.cpu().numpy()].mean(axis=1)
    n_elements = len(elements)
    
    print(f"Building filter with rmin = {rmin:.4f}")
    
    kd_tree = scipy.spatial.KDTree(centroids)
    
    I, J, V = [], [], []
    for i in range(n_elements):
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
    H_normalized = H_np / (Hs_np + 1e-10)
    
    return torch.tensor(H_normalized, dtype=mesh.dtype, device=mesh.device)


def create_animation(mesh, frames, save_path="simp_animation.mp4", fps=10, bc_info=None):
    """
    Create animation of optimization process with boundary conditions.
    
    Args:
        mesh: TensorMesh mesh object
        frames: List of frame dictionaries with 'epoch', 'rho', 'compliance'
        save_path: Output file path
        fps: Frames per second
        bc_info: Dictionary with boundary condition info
    """
    if len(frames) == 0:
        print("  No frames to animate")
        return
    
    points = mesh.points.detach().cpu().numpy()
    elements = mesh.elements().cpu().numpy()
    verts = [points[elem] for elem in elements]
    
    # Get domain bounds
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    domain_size = max(x_max - x_min, y_max - y_min)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Density plot
    coll = PolyCollection(verts, array=frames[0]['rho'], cmap='gray_r',
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
        # Fixed boundary
        if 'fixed_pts' in bc_info and len(bc_info['fixed_pts']) > 0:
            fixed_pts = bc_info['fixed_pts']
            fixed_coords = points[fixed_pts]
            marker_size = domain_size * 0.02
            for pt in fixed_coords[::max(1, len(fixed_coords)//10)]:
                triangle = plt.Polygon([
                    [pt[0] - marker_size, pt[1]],
                    [pt[0] - marker_size * 2, pt[1] + marker_size * 0.5],
                    [pt[0] - marker_size * 2, pt[1] - marker_size * 0.5]
                ], color='blue', alpha=0.8)
                ax1.add_patch(triangle)
            ax1.plot([x_min, x_min], [y_min, y_max], 'b-', linewidth=3, label='Fixed')
        
        # Load point and direction
        if 'load_pt' in bc_info and 'load_dir' in bc_info:
            load_pt = bc_info['load_pt']
            load_dir = bc_info['load_dir']
            load_coord = points[load_pt]
            arrow_scale = domain_size * 0.15
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
        coll.set_array(frame['rho'])
        title1.set_text(f'Iteration {frame["epoch"]}, C={frame["compliance"]:.2e}')
        current_epochs = epochs[:frame_idx+1]
        current_compliances = compliances[:frame_idx+1]
        line.set_data(current_epochs, current_compliances)
        return coll, title1, line
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    
    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'SIMP Topology Optimization'})
        anim.save(save_path, writer=writer, dpi=150)
        print(f"  Saved: {save_path}")
    except Exception as e:
        gif_path = save_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
            print(f"  Saved: {gif_path} (ffmpeg not available)")
        except Exception as e2:
            print(f"  Warning: Could not save animation ({e2})")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SIMP Topology Optimization")
    
    parser.add_argument('--length', type=float, default=2.0, help='Domain length')
    parser.add_argument('--height', type=float, default=1.0, help='Domain height')
    parser.add_argument('--n_elem_x', type=int, default=80, help='Elements in x')
    parser.add_argument('--n_elem_y', type=int, default=40, help='Elements in y')
    
    parser.add_argument('--epoch', type=int, default=100, help='Number of iterations')
    parser.add_argument('--vf', type=float, default=0.5, help='Volume fraction')
    parser.add_argument('--filter_radius', type=float, default=0.03, help='Filter radius')
    parser.add_argument('--move_limit', type=float, default=0.2, help='Move limit')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("  SIMP TOPOLOGY OPTIMIZATION")
    print("=" * 70)
    print(f"  Domain: {args.length} × {args.height}")
    print(f"  Mesh:   {args.n_elem_x} × {args.n_elem_y}")
    print(f"  vf:     {args.vf}")
    print("=" * 70)
    
    # Create mesh
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
    
    # Load: point load at mid-right boundary, downward
    points_np = mesh.points.detach().cpu().numpy()
    right_boundary = mesh.point_data['is_right_boundary'].flatten().cpu().numpy()
    y_mid = args.height / 2.0
    
    # Find closest point to (length, height/2)
    right_pts = np.where(right_boundary)[0]
    target = np.array([args.length, y_mid])
    dists = np.linalg.norm(points_np[right_pts] - target, axis=1)
    load_idx = right_pts[np.argmin(dists)]
    
    F = torch.zeros(n_points * dim, dtype=mesh.dtype, device=mesh.device)
    F[load_idx * dim + 1] = -1.0  # Downward force
    
    # Boundary conditions: fix left boundary
    dbc_mask = mesh.point_data['is_left_boundary'].flatten().repeat_interleave(dim)
    condenser = Condenser(dbc_mask)
    
    print(f"Dirichlet DOFs: {dbc_mask.sum().item()} / {n_points * dim}")
    print(f"Load at node {load_idx}, position {points_np[load_idx]}")
    
    # Filter
    H_filter = build_filter(mesh, args.filter_radius)
    
    # Initialize densities
    rho = torch.full((n_elements,), args.vf, dtype=mesh.dtype, device=mesh.device)
    
    # Create OC optimizer
    optimizer = OCOptimizer(
        rho,
        vf=args.vf,
        move_limit=args.move_limit,
        rho_min=1e-3,
        rho_max=1.0,
    )
    
    print(f"\nStarting OC optimization...")
    
    history = {'compliance': [], 'volume': []}
    
    # For animation and paper figures
    animation_frames = []
    save_interval = max(1, args.epoch // 50)
    
    # Store snapshots for paper figures
    rho_initial = rho.clone()
    rho_middle = None
    middle_epoch = 5  # Early iteration to show topology evolution
    
    for epoch in tqdm(range(args.epoch), desc="Optimizing"):
        # Apply filter
        rho_filt = H_filter @ rho
        rho_filt = rho_filt.clamp(1e-3, 1.0)
        
        # Need gradients for sensitivity analysis
        rho_var = rho_filt.clone().requires_grad_(True)
        
        # Assemble and solve
        K = K_asm(mesh.points, element_data={"rho": rho_var})
        K_, F_ = condenser(K, F)
        u_ = K_.solve(F_, backend="scipy")
        u = condenser.recover(u_)
        
        # Compliance
        compliance = u @ F
        
        # Compute sensitivity
        compliance.backward()
        dc = rho_var.grad.clone()
        
        # Apply filter to sensitivity (chain rule)
        dc = H_filter.T @ dc
        
        # Volume sensitivity
        dv = torch.ones_like(rho) / n_elements
        
        # OC update step
        step_info = optimizer.step(dc=dc, dv=dv)
        
        history['compliance'].append(compliance.item())
        history['volume'].append(step_info['volume'])
        
        # Save frame for animation
        if epoch % save_interval == 0 or epoch == args.epoch - 1:
            animation_frames.append({
                'epoch': epoch,
                'rho': rho.clone().cpu().numpy(),
                'compliance': compliance.item()
            })
        
        # Save middle snapshot
        if epoch == middle_epoch:
            rho_middle = rho.clone()
        
        if epoch % 20 == 0:
            n_void = (rho < 0.1).sum().item()
            n_solid = (rho > 0.9).sum().item()
            print(f"\n  Epoch {epoch}: C={compliance.item():.4e}, void={n_void}, solid={n_solid}")
    
    # Final results
    n_void = (rho < 0.1).sum().item()
    n_solid = (rho > 0.9).sum().item()
    n_gray = n_elements - n_void - n_solid
    
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"  Final compliance: {history['compliance'][-1]:.4e}")
    print(f"  Density range: [{rho.min().item():.3f}, {rho.max().item():.3f}]")
    print(f"  Elements: {n_void} void, {n_solid} solid, {n_gray} gray")
    
    # Plot result
    print("\nGenerating visualizations...")
    
    points = mesh.points.detach().cpu().numpy()
    elements = mesh.elements().cpu().numpy()
    verts = [points[elem] for elem in elements]
    rho_np = rho.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Density plot
    ax = axes[0]
    coll = PolyCollection(verts, array=rho_np, cmap='gray_r', 
                         edgecolors='none', linewidths=0.0)
    coll.set_clim(0, 1)
    ax.add_collection(coll)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'Density (void={n_void}, solid={n_solid})')
    plt.colorbar(coll, ax=ax, shrink=0.6)
    
    # Compliance history
    ax = axes[1]
    ax.semilogy(history['compliance'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Compliance')
    ax.set_title('Objective')
    ax.grid(True, alpha=0.3)
    
    # Volume history
    ax = axes[2]
    ax.plot(history['volume'], 'r-', linewidth=2)
    ax.axhline(y=args.vf, color='k', linestyle='--')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Volume Fraction')
    ax.set_title('Volume Constraint')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('simp_result.png', dpi=300, bbox_inches='tight')
    plt.savefig('simp_result.pdf', bbox_inches='tight')
    print("  Saved: simp_result.png/pdf")
    plt.close()
    
    # Create animation with boundary conditions
    print("  Generating optimization animation...")
    
    # Get fixed boundary node indices
    left_boundary = mesh.point_data['is_left_boundary'].flatten().cpu().numpy()
    fixed_pts = np.where(left_boundary)[0]
    
    bc_info = {
        'fixed_pts': fixed_pts,
        'load_pt': load_idx,
        'load_dir': [0.0, -1.0]  # Downward force
    }
    create_animation(mesh, animation_frames, 'simp_animation.mp4', bc_info=bc_info)
    
    # =========================================================================
    # Generate paper figures (4 individual plots)
    # =========================================================================
    print("\nGenerating individual paper figures...")
    
    rho_final = rho.clone()
    if rho_middle is None:
        rho_middle = rho_final  # fallback
    
    def plot_density_single(rho_data, title, save_name, verts, points, bc_info=None):
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
                load_pt = bc_info['load_pt']
                load_dir = bc_info['load_dir']
                load_coord = points[load_pt]
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
    plot_density_single(rho_initial, 'Initial Design (ρ = vf)', 
                       'simp_paper_initial', verts, points, bc_info)
    
    # 2. Middle iteration
    plot_density_single(rho_middle, f'Intermediate (Iter {middle_epoch})', 
                       'simp_paper_middle', verts, points, bc_info)
    
    # 3. Final result
    plot_density_single(rho_final, f'Final Design (Iter {args.epoch})', 
                       'simp_paper_final', verts, points, bc_info)
    
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
    plt.savefig('simp_paper_compliance.png', dpi=300, bbox_inches='tight')
    plt.savefig('simp_paper_compliance.pdf', bbox_inches='tight')
    plt.close()
    print("    Saved: simp_paper_compliance.png/pdf")
    
    print("\n  All paper figures saved!")


if __name__ == "__main__":
    main()

