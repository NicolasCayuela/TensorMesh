"""
Hertzian Contact Problem (Circle on Flat Block)
===============================================

Geometry:
- Indenter (Left): Full Circle (Tri6), Radius=1.0, centered at (-1.0, 0.0).
- Block (Right): Rectangle (Quad9), fixed at the right boundary.
- Contact interface at x=0.

Method:
- Penalty method with Point-to-Segment contact detection.
- Accurate Von Mises stress calculation from strain.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import pyvista as pv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh
from tensormesh.dataset.mesh import gen_rectangle
from tensormesh.assemble import LinearElasticityElementAssembler, ContactAssembler
from tensormesh.visualization import mesh_to_pyvista, setup_headless

# Define custom Contact Assembler using TensorMesh interface
class HertzianContact(ContactAssembler):
    def __post_init__(self, penalty=1e6, dist_thresh=1e-4):
        self.penalty = penalty
        self.dist_thresh = dist_thresh
    
    def element_energy(self, displacement, x, master_segments, master_normals):
        # x: Reference coordinate [2]
        # displacement: [2]
        # master_segments: [M, 2, 2] - Start/End points of M segments
        # master_normals: [M, 2] - Normal vectors of M segments
        
        curr_pos = x + displacement
        
        # Broadcast current position against all master segments
        # P0: [M, 2]
        P0 = master_segments[:, 0, :]
        P1 = master_segments[:, 1, :]
        V = P1 - P0
        
        len_sq = (V**2).sum(dim=1)
        
        # Vector from P0 to point
        W = curr_pos.unsqueeze(0) - P0 # [M, 2]
        
        # Projection parameter t
        dot = (W * V).sum(dim=1)
        t = dot / (len_sq + 1e-12)
        t = torch.clamp(t, 0.0, 1.0)
        
        # Closest point on each segment
        closest = P0 + t.unsqueeze(1) * V
        
        # Vector from closest point to slave point
        vec = curr_pos.unsqueeze(0) - closest
        dist_sq = (vec**2).sum(dim=1)
        
        # Find the single closest segment
        min_idx = torch.argmin(dist_sq)
        
        # Calculate gap for ALL segments first
        # vec: [M, 2], master_normals: [M, 2]
        # all_gaps > 0 means separation relative to that segment's plane
        all_gaps = (vec * master_normals).sum(dim=1)
        
        # Select the gap corresponding to the closest segment
        # Use gather to be vmap-safe
        # all_gaps: [M], min_idx: scalar
        gap = torch.gather(all_gaps, 0, min_idx.unsqueeze(0)).squeeze(0)
        
        # Penalty
        penetration = self.dist_thresh - gap
        active = torch.nn.functional.relu(penetration)
        
        return 0.5 * self.penalty * active**2

def gen_circle_indenter(r=1.0, chara_length=0.1, order=2):
    """Generate a full circle mesh on the left side, touching x=0."""
    import gmsh
    
    # Unique cache name for circle
    cache_path = f".gmsh_cache/circle_indenter_{r}_{chara_length}_{order}.msh"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if not os.path.exists(cache_path):
        gmsh.initialize()
        gmsh.model.add("circle")
        
        # Disk centered at (-r, 0), radius r. 
        # Rightmost point is at (0, 0). Leftmost at (-2r, 0).
        gmsh.model.occ.addDisk(-r, 0, 0, r, r)
        
        gmsh.model.occ.synchronize()
        
        # Add Physical Group to ensure only 2D elements are exported
        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
        
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", chara_length)
        gmsh.model.mesh.generate(2)
        gmsh.write(cache_path)
        gmsh.finalize()
    
    mesh = Mesh.from_file(cache_path, reorder=True)
    
    # Register boundary data
    pts = mesh.points
    
    # "Back" surface is the left side of the circle (x approx -2r)
    # Let's pick an arc on the left side, e.g., x < -1.8*r
    is_back = pts[:, 0] < -1.8 * r
    
    # Contact surface is the front part near x=0
    # Let's pick the right hemisphere or just the front arc
    is_contact = pts[:, 0] > -0.5 * r
    
    mesh.register_point_data("is_back", is_back)
    mesh.register_point_data("is_contact_front", is_contact)
    
    return mesh

def main():
    print("=" * 60)
    print("Hertzian Contact - Full Circle Indenter")
    print("=" * 60)
    
    # Parameters
    E = 1000.0
    nu = 0.3
    R = 1.0
    
    # 1. Generate Meshes
    print("Generating meshes...")
    # Indenter: Full Circle - Refine mesh for smoother contact
    indenter = gen_circle_indenter(r=R, chara_length=0.06, order=2)
    
    # Block: Right side - Refine mesh
    # Use generic gen_rectangle but we will implement custom transfinite logic there if needed
    # Or just write custom generation here to ensure symmetry
    
    def gen_symmetric_block(left, right, bottom, top, chara_length, order):
        import gmsh
        cache_path = f".gmsh_cache/rect_sym_{left}_{right}_{bottom}_{top}_{chara_length}_{order}.msh"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if not os.path.exists(cache_path):
            gmsh.initialize()
            gmsh.model.add("rectangle_sym")
            
            # Create rectangle
            rect = gmsh.model.occ.addRectangle(left, bottom, 0, right-left, top-bottom)
            gmsh.model.occ.synchronize()
            
            # Physical group
            gmsh.model.addPhysicalGroup(2, [rect], 1)
            
            # Transfinite Meshing for perfect symmetry
            # Get lines
            lines = gmsh.model.getEntities(1)
            
            # We want specific number of elements.
            width = right - left
            height = top - bottom
            n_x = int(width / chara_length)
            n_y = int(height / chara_length)
            
            # Ensure even number for symmetry if crossing 0
            if n_y % 2 != 0: n_y += 1
            
            for dim, tag in lines:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                dx = abs(bbox[3] - bbox[0])
                dy = abs(bbox[4] - bbox[1])
                
                # Check if horizontal or vertical
                if dx > dy: # Horizontal
                    gmsh.model.mesh.setTransfiniteCurve(tag, n_x + 1)
                else: # Vertical
                    gmsh.model.mesh.setTransfiniteCurve(tag, n_y + 1)
            
            surfs = gmsh.model.getEntities(2)
            for dim, tag in surfs:
                gmsh.model.mesh.setTransfiniteSurface(tag)
                gmsh.model.mesh.setRecombine(dim, tag)
            
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            gmsh.model.mesh.generate(2)
            gmsh.write(cache_path)
            gmsh.finalize()
            
        mesh = Mesh.from_file(cache_path, reorder=True)
        
        # Register boundary data manually since we bypassed the standard gen_rectangle
        pts = mesh.points
        tol = 1e-4
        is_left = torch.abs(pts[:, 0] - left) < tol
        is_right = torch.abs(pts[:, 0] - right) < tol
        is_bottom = torch.abs(pts[:, 1] - bottom) < tol
        is_top = torch.abs(pts[:, 1] - top) < tol
        
        mesh.register_point_data("is_left_boundary", is_left)
        mesh.register_point_data("is_right_boundary", is_right)
        mesh.register_point_data("is_bottom_boundary", is_bottom)
        mesh.register_point_data("is_top_boundary", is_top)
        
        return mesh

    block = gen_symmetric_block(left=0.0, right=1.0, bottom=-1.0, top=1.0,
                          chara_length=0.06, order=2)
    
    print(f"Indenter (Circle): {indenter.points.shape[0]} nodes")
    print(f"Block (Rectangle): {block.points.shape[0]} nodes")
    
    # 2. Physics Models
    ind_model = LinearElasticityElementAssembler.from_mesh(indenter, E=E, nu=nu)
    blk_model = LinearElasticityElementAssembler.from_mesh(block, E=E, nu=nu)
    
    ind_pts = indenter.points
    blk_pts = block.points
    
    # 3. Boundary Conditions
    # Modified Strategy for better visualization:
    # 1. Fix the BACK (left) of the circle (u=0)
    # 2. Push the RIGHT of the block to the LEFT
    
    ind_back_mask = indenter.point_data["is_back"]
    ind_fix_nodes = torch.nonzero(ind_back_mask, as_tuple=True)[0]
    
    blk_right_mask = block.point_data["is_right_boundary"]
    blk_push_nodes = torch.nonzero(blk_right_mask, as_tuple=True)[0]
    
    # 4. Identification of Contact Surfaces (Segments)
    # We need to construct master segments for the assembler
    
    # --- Block as Master (for Circle Slave) ---
    # Surface 2: Block Left (Flat)
    is_surface_block = torch.abs(block.points[:, 0] - 0.0) < 1e-3
    surf2_indices = torch.nonzero(is_surface_block, as_tuple=True)[0]
    
    # Sort by Y to form continuous segments
    surf2_y = blk_pts[surf2_indices, 1]
    sorted_idx = torch.argsort(surf2_y)
    surf2_sorted = surf2_indices[sorted_idx]
    
    # --- Circle as Master (for Block Slave) ---
    # Surface 1: Indenter Front
    dist_from_center = torch.sqrt((ind_pts[:, 0] + 1.0)**2 + ind_pts[:, 1]**2)
    is_surface_circle = torch.abs(dist_from_center - 1.0) < 1e-3
    is_front = ind_pts[:, 0] > -0.5
    surf1_mask = is_surface_circle & is_front
    surf1_indices = torch.nonzero(surf1_mask, as_tuple=True)[0]
    
    surf1_y = ind_pts[surf1_indices, 1]
    sorted_idx_circ = torch.argsort(surf1_y)
    surf1_sorted = surf1_indices[sorted_idx_circ]
    
    # 5. Initialize Assemblers
    # We use boundary masks to restrict the contact assembler to specific surfaces.
    # HertzianContact.from_mesh(..., boundary_mask=...) will automatically
    # select facets (edges) where all nodes are in the mask.
    
    ind_node_mask = torch.zeros(indenter.n_points, dtype=torch.bool, device=indenter.device)
    ind_node_mask[surf1_indices] = True
    
    blk_node_mask = torch.zeros(block.n_points, dtype=torch.bool, device=block.device)
    blk_node_mask[surf2_indices] = True
    
    displacement = -0.15  # Moderate push
    penalty = 2e6         # Very stiff penalty

    # 1. Circle Slave (contacting Block)
    contact_ind = HertzianContact.from_mesh(indenter, boundary_mask=ind_node_mask, penalty=penalty, dist_thresh=1e-4)
    
    # 2. Block Slave (contacting Circle)
    contact_blk = HertzianContact.from_mesh(block, boundary_mask=blk_node_mask, penalty=penalty, dist_thresh=1e-4)
    
    # 5. Optimization
    u_ind = torch.zeros_like(ind_pts, requires_grad=True)
    u_blk = torch.zeros_like(blk_pts, requires_grad=True)
    
    optimizer = optim.LBFGS([u_ind, u_blk], lr=1.0, max_iter=200,
                            line_search_fn="strong_wolfe")
    
    def closure():
        optimizer.zero_grad()
        
        # BCs
        u_ind_bc = u_ind.clone()
        u_ind_bc[ind_fix_nodes] = 0  # Fix Circle
        
        u_blk_bc = u_blk.clone()
        u_blk_bc[blk_push_nodes, 0] = displacement  # Push Block Left
        u_blk_bc[blk_push_nodes, 1] = 0             # No vertical slip at push boundary
        
        E1 = ind_model.energy(point_data={'displacement': u_ind_bc})
        E2 = blk_model.energy(point_data={'displacement': u_blk_bc})
        
        # Two-way Contact using TensorMesh Assemblers
        
        # 1. Circle Slave vs Block Master
        # Update Master Geometry based on current displacement!
        u_blk_master = u_blk_bc[surf2_sorted]
        curr_blk_pts = blk_pts[surf2_sorted] + u_blk_master
        
        # Reconstruct segments on the fly
        P_j = curr_blk_pts[:-1]
        P_jp1 = curr_blk_pts[1:]
        curr_blk_segments = torch.stack([P_j, P_jp1], dim=1)
        
        # Recompute normals
        V = P_jp1 - P_j
        curr_blk_normals = torch.stack([-V[:, 1], V[:, 0]], dim=1)
        curr_blk_normals = curr_blk_normals / (torch.norm(curr_blk_normals, dim=1, keepdim=True) + 1e-12)
        
        E_c1 = contact_ind.energy(
            point_data={'displacement': u_ind_bc},
            scalar_data={'master_segments': curr_blk_segments, 'master_normals': curr_blk_normals}
        )
        
        # 2. Block Slave vs Circle Master
        u_ind_master = u_ind_bc[surf1_sorted]
        curr_ind_pts = ind_pts[surf1_sorted] + u_ind_master
        
        P_j_c = curr_ind_pts[:-1]
        P_jp1_c = curr_ind_pts[1:]
        curr_circ_segments = torch.stack([P_j_c, P_jp1_c], dim=1)
        
        V_c = P_jp1_c - P_j_c
        curr_circ_normals = torch.stack([V_c[:, 1], -V_c[:, 0]], dim=1)
        curr_circ_normals = curr_circ_normals / (torch.norm(curr_circ_normals, dim=1, keepdim=True) + 1e-12)

        E_c2 = contact_blk.energy(
            point_data={'displacement': u_blk_bc},
            scalar_data={'master_segments': curr_circ_segments, 'master_normals': curr_circ_normals}
        )
        
        loss = E1 + E2 + E_c1 + E_c2
        
        if loss.requires_grad:
            loss.backward()
            
        with torch.no_grad():
            if u_ind.grad is not None:
                u_ind.grad[ind_fix_nodes] = 0
            if u_blk.grad is not None:
                u_blk.grad[blk_push_nodes] = 0
                
        return loss

    print("Solving...")
    for i in range(25):
        loss = optimizer.step(closure)
        if i % 5 == 0:
            print(f"Iter {i}: Loss = {loss.item():.4e}")

    # Final visualization
    with torch.no_grad():
        u_ind[ind_fix_nodes] = 0
        u_blk[blk_push_nodes, 0] = displacement
        u_blk[blk_push_nodes, 1] = 0
        
        def compute_element_stress(mesh, u, E, nu):
            # Manual implementation of shape function derivatives for stress
            def get_dNdxi_tri6(xi_val):
                xi, eta = xi_val[0], xi_val[1]
                dNdxi = torch.zeros((6, 2), dtype=torch.float64)
                dNdxi[0, 0] = 4*xi + 4*eta - 3; dNdxi[0, 1] = 4*xi + 4*eta - 3
                dNdxi[1, 0] = 4*xi - 1;         dNdxi[1, 1] = 0
                dNdxi[2, 0] = 0;                dNdxi[2, 1] = 4*eta - 1
                dNdxi[3, 0] = 4 - 8*xi - 4*eta; dNdxi[3, 1] = -4*xi
                dNdxi[4, 0] = 4*eta;            dNdxi[4, 1] = 4*xi
                dNdxi[5, 0] = -4*eta;           dNdxi[5, 1] = 4 - 4*xi - 8*eta
                return dNdxi

            def get_dNdxi_quad9(xi_val):
                xi, eta = xi_val[0], xi_val[1]
                dNdxi = torch.zeros((9, 2), dtype=torch.float64)
                def phi(i, x):
                    return 0.5*x*(x-1) if i==0 else (1-x**2 if i==1 else 0.5*x*(x+1))
                def dphi(i, x):
                    return x-0.5 if i==0 else (-2*x if i==1 else x+0.5)
                idx_map = {0:(0,0), 1:(2,0), 2:(2,2), 3:(0,2), 4:(1,0), 5:(2,1), 6:(1,2), 7:(0,1), 8:(1,1)}
                for k in range(9):
                    i, j = idx_map[k]
                    dNdxi[k, 0] = dphi(i, xi) * phi(j, eta)
                    dNdxi[k, 1] = phi(i, xi) * dphi(j, eta)
                return dNdxi

            if "tri" in mesh.default_element_type:
                dNdxi = get_dNdxi_tri6(torch.tensor([1/3, 1/3], dtype=torch.float64))
            else:
                dNdxi = get_dNdxi_quad9(torch.tensor([0.0, 0.0], dtype=torch.float64))
            
            elems = mesh.elements(mesh.default_element_type)
            elem_nodes = mesh.points[elems]
            elem_u = u[elems]
            
            J = torch.einsum('eki,kj->eij', elem_nodes.double(), dNdxi.double())
            try:
                invJ = torch.inverse(J)
            except:
                invJ = torch.eye(2, dtype=torch.float64).expand(elems.shape[0], -1, -1)
                
            dNdx = torch.einsum('kj,eji->eki', dNdxi.double(), invJ)
            grad_u = torch.einsum('eki,ekj->eij', elem_u.double(), dNdx)
            eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
            
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            trace_eps = eps[:, 0, 0] + eps[:, 1, 1]
            sigma = 2 * mu * eps
            sigma[:, 0, 0] += lam * trace_eps
            sigma[:, 1, 1] += lam * trace_eps
            
            s11, s22, s12 = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 0, 1]
            s33 = nu * (s11 + s22)
            von_mises = torch.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 6 * s12**2))
            
            node_stress = torch.zeros(mesh.points.shape[0], dtype=torch.float64)
            node_count = torch.zeros(mesh.points.shape[0], dtype=torch.float64)
            elems_flat = elems.flatten()
            vm_rep = von_mises.repeat_interleave(elems.shape[1])
            node_stress.scatter_add_(0, elems_flat, vm_rep)
            node_count.scatter_add_(0, elems_flat, torch.ones_like(vm_rep))
            return (node_stress / (node_count + 1e-8)).float()

        s_ind = compute_element_stress(indenter, u_ind, E, nu)
        s_blk = compute_element_stress(block, u_blk, E, nu)
        
        print("Visualizing...")
        setup_headless()
        
        pv_ind = mesh_to_pyvista(indenter)
        pv_blk = mesh_to_pyvista(block)
        
        # Add displacement for warping
        u_ind_3d = np.pad(u_ind.numpy(), ((0,0),(0,1)))
        u_blk_3d = np.pad(u_blk.numpy(), ((0,0),(0,1)))
        pv_ind.point_data['Displacement'] = u_ind_3d
        pv_blk.point_data['Displacement'] = u_blk_3d
        
        # Warp meshes
        pv_ind = pv_ind.warp_by_vector('Displacement', factor=1.0)
        pv_blk = pv_blk.warp_by_vector('Displacement', factor=1.0)
        
        pv_ind.point_data['VonMises'] = s_ind.numpy()
        pv_blk.point_data['VonMises'] = s_blk.numpy()
        
        # --- Smart Color Scaling ---
        # 1. Identify "Singularity Nodes" (Fixed boundary)
        # These nodes have unphysically high stress. We exclude them from clim calculation.
        mask_ind_valid = np.ones(s_ind.shape[0], dtype=bool)
        mask_ind_valid[ind_fix_nodes.detach().numpy()] = False
        
        # Also exclude neighbors of fixed nodes to be safe (heuristic: high stress outliers)
        # Or just take a percentile of the *rest*
        s_ind_valid = s_ind.numpy()[mask_ind_valid]
        s_blk_valid = s_blk.numpy() # Block has no singularity usually
        
        all_valid_stress = np.concatenate([s_ind_valid, s_blk_valid])
        
        # Use 99th percentile of the VALID data as max
        # This allows the contact stress (which is high but not infinite) to be red/yellow
        clim_max = np.percentile(all_valid_stress, 99.5)
        
        # Setup Plotter
        pv.set_plot_theme("document")
        pl = pv.Plotter(off_screen=True, window_size=[1600, 1000]) # Larger res
        pl.enable_anti_aliasing("ssaa")
        
        # Use 'turbo' for high contrast rainbow-like mapping, easier to see gradients
        cmap = "turbo"
        
        pl.add_mesh(pv_ind, scalars='VonMises', cmap=cmap, clim=[0, clim_max], 
                   show_edges=False, label="Indenter", show_scalar_bar=False)
        pl.add_mesh(pv_blk, scalars='VonMises', cmap=cmap, clim=[0, clim_max], 
                   show_edges=False, label="Block", show_scalar_bar=False)
        
        # Wireframe (more visible)
        pl.add_mesh(pv_ind.extract_all_edges(), color="black", opacity=0.3, line_width=1.0)
        pl.add_mesh(pv_blk.extract_all_edges(), color="black", opacity=0.3, line_width=1.0)
        
        # --- BC Visualization (Uniform Spatial Sampling) ---
        
        def get_spatially_uniform_indices(points, axis=1, n_markers=15):
            """Select indices uniformly spaced along a given axis."""
            vals = points[:, axis]
            v_min, v_max = vals.min(), vals.max()
            # Shrink range slightly to avoid edges
            margin = (v_max - v_min) * 0.05
            targets = np.linspace(v_min + margin, v_max - margin, n_markers)
            
            selected = []
            for t in targets:
                # Find closest node
                dist = np.abs(vals - t)
                idx = np.argmin(dist)
                selected.append(idx)
            return np.unique(selected)

        # 1. Block Push Arrows (Right side)
        blk_push_indices_all = blk_push_nodes.detach().numpy()
        push_pts_all = pv_blk.points[blk_push_indices_all]
        # Filter for uniform Y distribution
        subset_idx = get_spatially_uniform_indices(push_pts_all, axis=1, n_markers=15)
        push_pts = push_pts_all[subset_idx]
        
        pd_push = pv.PolyData(push_pts)
        pd_push["vectors"] = np.tile([-1.0, 0.0, 0.0], (len(push_pts), 1))
        # Thinner, longer arrows
        arrows_push = pd_push.glyph(orient="vectors", scale=False, factor=0.15, geom=pv.Arrow(tip_radius=0.2, shaft_radius=0.05))
        
        # 2. Circle Fixed Cones (Left side) -> Replaced with Translucent Mask
        # Instead of using mesh points which are irregular, we draw a geometric "Clamp" 
        # that represents the fixed region (x < -1.8).
        # We'll draw a Box from x=-2.2 to x=-1.8, spanning Y=[-1, 1] (approx radius)
        
        # Create a geometric box
        clamp_box = pv.Box(bounds=(-2.2, -1.8, -1.1, 1.1, -0.1, 0.1))
        
        # Colors
        bc_push_color = "#F1C40F" # Yellow
        bc_fix_color = "#E74C3C"  # Red
        
        pl.add_mesh(arrows_push, color=bc_push_color, label="Displacement")
        # Add translucent clamp
        pl.add_mesh(clamp_box, color=bc_fix_color, opacity=0.3, show_edges=True, label="Fixed Support")
        
        pl.view_xy()
        pl.camera.zoom(1.3)
        pl.add_title("Hertzian Contact Stress (Von Mises)", font_size=16, color="black")
        
        pl.add_scalar_bar(title="Von Mises Stress", 
                         title_font_size=14, 
                         label_font_size=12, 
                         color="black", 
                         position_x=0.3, position_y=0.05, width=0.4, height=0.06,
                         fmt="%.2f")
        
        pl.add_legend(bcolor=(0.95, 0.95, 0.95), size=(0.15, 0.15), loc='upper right')
        
        out_path = os.path.join(os.path.dirname(__file__), "hertzian_contact.png")
        pl.screenshot(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
