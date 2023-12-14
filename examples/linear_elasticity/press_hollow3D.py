import sys
sys.path.append("../..")
import torch
from torch_fem import ElementAssembler, Mesh, Condenser
from torch_fem import matrix, matmul, dot

from torch_fem import Mesh, ElementAssembler, NodeAssembler



class KAssembler(ElementAssembler):
    def __post_init__(self):
        E = 1
        nu = 0.3
        lambd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        C = torch.tensor([
            [lambd + 2 * mu, lambd, lambd, 0, 0, 0],
            [lambd, lambd + 2 * mu, lambd, 0, 0, 0],
            [lambd, lambd, lambd + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])
        self.C = C

    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
            Returns:
            --------
                K: torch.Tensor[n_basis, n_basis, n_dim, n_dim]
        """

        

        zeros = torch.zeros_like(gradu[0, 0]) # [n_basis]
        
        B = matrix([
            [gradu[0, 0], zeros, zeros, gradu[1, 0], zeros, zeros, gradu[2, 0], zeros, zeros, gradu[3, 0], zeros, zeros],
            [zeros, gradu[0, 1], zeros, zeros, gradu[1,1], zeros, zeros, gradu[2, 1], zeros, zeros, gradu[3, 1], zeros],
            [zeros, zeros, gradu[0,2], zeros, zeros, gradu[1, 2], zeros, zeros, gradu[2, 2], zeros, zeros, gradu[3, 2]],
            [gradu[0,1], gradu[0, 0], zeros, gradu[1, 1], gradu[1, 0], zeros, gradu[2, 1], gradu[2, 0], zeros, gradu[3, 1], gradu[3, 0], zeros],
            [zeros, gradu[0,2], gradu[0, 1], zeros, gradu[1,2], gradu[1, 1], zeros, gradu[2, 2], gradu[2, 1], zeros, gradu[3, 2], gradu[3, 1]],
            [gradu[0,2], zeros, gradu[0, 0], gradu[1, 2], zeros, gradu[1, 0], gradu[2, 2], zeros, gradu[2, 0], gradu[3, 2], zeros, gradu[3, 0]]
        ]) # [6, n_basis*n_dim]

        K  = B.T @ self.C.type(gradu.dtype).to(gradu.device) @ B # [n_basis*n_dim, n_basis*n_dim]
        K = K.view(4, 3, 4, 3) # [n_basis, n_dim, n_basis, n_dim]
        K = K.permute(0, 2, 1, 3)

        return K
    
class FAssemble(NodeAssembler):
    def __post_init__(self):
        pass

    def forward(self, u):
        f = torch.tensor([0, 0, -1]).type(u.dtype).to(u.device)
        zeros = torch.zeros_like(u[0])

        N = matrix([
            [u[0], zeros, zeros, u[1], zeros, zeros, u[2], zeros, zeros, u[3], zeros, zeros],
            [zeros, u[0], zeros, zeros, u[1], zeros, zeros, u[2], zeros, zeros, u[3], zeros],
            [zeros, zeros, u[0], zeros, zeros, u[1], zeros, zeros, u[2], zeros, zeros, u[3]]
            ]) # [3, n_basis*n_dim]
        return (N.T @ f).view(4, 3) # [n_basis, n_dim]


mesh = Mesh.gen_cube(chara_length=0.1).double()

K_asm = KAssembler.from_mesh(mesh, quadrature_order=1)
F_asm = FAssemble.from_mesh(mesh, quadrature_order=1)


K = K_asm(mesh.points)
# F = F_asm(mesh.points)
F = torch.zeros_like(mesh.points)
F[mesh.point_data['is_top_boundary'], 1] = 0.05
F = F.flatten()
dirichlet_mask = torch.zeros_like(mesh.points, dtype=torch.bool)
dirichlet_mask[mesh.point_data['is_bottom_boundary'], :] = True
condenser = Condenser(dirichlet_mask.flatten())

K_, f_ = condenser(K, F)

u_ = K_.solve(f_)

u = condenser.recover(u_)
mesh.register_point_data("u", u.view(-1, 3))
mesh.save("press_hollow_cube.vtk")

# mesh.plot(
#     {"ux":u[0::2],
#      "uy":u[1::2]},
#      backend="matplotlib",
#     save_path="press_hollow_rectangle.png",
#     show_mesh=True
# )