import torch
import numpy as np
import os
from tensormesh import Mesh, LaplaceElementAssembler, ElementAssembler, NodeAssembler, Condenser
from tensormesh.sparse import nonlinear_solve

class CubicTermAssembler(NodeAssembler):
    def forward(self, u, v):
        # u: interpolated u at quad points
        # v: shape functions at quad points
        return u**3 * v

class WeightedMassAssembler(ElementAssembler):
    def forward(self, u, v, w):
        # w: interpolated weight at quad points
        # u, v: basis functions
        return w * u * v

class SourceAssembler(NodeAssembler):
    def forward(self, v):
        return 1.0 * v

def run():
    # 1. Mesh
    mesh = Mesh.gen_rectangle(chara_length=0.05, element_type="tri")
    print(f"Mesh generated with {mesh.n_points} points.")
    
    # 2. Assemblers
    laplace_asm = LaplaceElementAssembler.from_mesh(mesh)
    cubic_asm = CubicTermAssembler.from_mesh(mesh)
    weighted_mass_asm = WeightedMassAssembler.from_mesh(mesh)
    source_asm = SourceAssembler.from_mesh(mesh)
    
    # 3. Problem Setup
    # - div (grad u) + u^3 = f
    # f = 1
    # u = 0 on boundary
    
    # Boundary Conditions
    condenser = Condenser(mesh.boundary_mask, torch.zeros(mesh.boundary_mask.sum()))
    
    # Pre-assemble constant parts
    K = laplace_asm(mesh.points)
    f_load = source_asm(mesh.points)
    
    # Initialize condenser
    condenser._compute_layout(K)
    
    # Initial guess (inner DOFs)
    n_inner = condenser.n_inner_dof
    u0_inner = torch.zeros(n_inner, device=mesh.device, dtype=mesh.dtype)
    
    # Parameter (can be optimized)
    f_param = f_load.clone().requires_grad_(True)
    
    def func_F(u_inner, f_vec):
        # Recover full u
        u_full = condenser.recover(u_inner)
        
        # 1. Laplace term: K u
        term1 = K @ u_full
        
        # 2. Cubic term: int u^3 v
        term2 = cubic_asm(point_data={"u": u_full})
        
        # 3. Combine
        res_full = term1 + term2 - f_vec
        
        # Return inner residual
        return res_full[condenser.is_inner_dof]

    def func_J(u_inner, f_vec):
        u_full = condenser.recover(u_inner)
        
        # J = K + M(3u^2)
        # Weight w = 3 * u^2
        w = 3 * u_full**2
        M_w = weighted_mass_asm(point_data={"w": w})
        
        J_full = K + M_w
        
        # Condense J
        # We only need the matrix part
        J_inner, _ = condenser(J_full)
        return J_inner

    # Solve
    print("Starting non-linear solve...")
    u_inner = nonlinear_solve(func_F, func_J, u0_inner, (f_param,), tol=1e-6, verbose=True)
    
    # Recover
    u_final = condenser.recover(u_inner)
    
    # Plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "nonlinear_poisson.png")
    
    # Use matplotlib directly to avoid potential visualization issues
    if mesh.dim == 2:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.tri as mtri
            
            points = mesh.points.detach().cpu().numpy()
            # Get triangle elements
            if hasattr(mesh, 'cells') and isinstance(mesh.cells, dict):
                 # Handle BufferDict or dict
                 cell_keys = list(mesh.cells.keys())
            else:
                 # Fallback access
                 cell_keys = []

            # Try to find a triangular element type
            tri_key = None
            if "triangle" in cell_keys: tri_key = "triangle"
            elif "tri" in cell_keys: tri_key = "tri"
            elif "triangle3" in cell_keys: tri_key = "triangle3"
            elif "tri3" in cell_keys: tri_key = "tri3"
            
            if tri_key:
                triangles = mesh.cells[tri_key].detach().cpu().numpy()
            else:
                # Try getting any 2D element
                try:
                    elems = mesh.elements(2)
                    if isinstance(elems, dict):
                        triangles = next(iter(elems.values())).detach().cpu().numpy()
                    else:
                        triangles = elems.detach().cpu().numpy()
                except:
                    triangles = None
                    print("Could not find 2D elements for plotting.")
                
            if triangles is not None:
                u_np = u_final.detach().cpu().numpy()
                triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
                
                plt.figure(figsize=(8, 6))
                plt.tricontourf(triang, u_np, levels=20)
                plt.colorbar(label='u')
                plt.title("Non-linear Poisson Solution")
                plt.savefig(save_path)
                print(f"Solution saved to {save_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")
            
    # Test gradients
    loss = u_inner.sum()
    loss.backward()
    print("Gradient norm w.r.t source:", f_param.grad.norm().item())

if __name__ == "__main__":
    run()

