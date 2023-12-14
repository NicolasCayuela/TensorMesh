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

        C = E/(1 - nu*nu) * torch.tensor([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, 0.5*(1-nu)]
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
            [gradu[0, 0], zeros, gradu[1, 0], zeros, gradu[2, 0], zeros],
            [zeros, gradu[0, 1], zeros, gradu[1,1], zeros, gradu[2, 1]],
            [gradu[0,1], gradu[0, 0], gradu[1, 1], gradu[1, 0],gradu[2, 1],gradu[2, 0]]
        ]) # [3, n_basis*n_dim]

        K  = B.T @ self.C.type(gradu.dtype).to(gradu.device) @ B # [n_basis*n_dim, n_basis*n_dim]
        K = K.view(3, 2, 3, 2) # [n_basis, n_dim, n_basis, n_dim]
        K = K.permute(0, 2, 1, 3)

        return K
    
class FAssemble(NodeAssembler):
    def __post_init__(self):
        pass

    def forward(self, u):
        f = torch.tensor([0, -1]).type(u.dtype).to(u.device)
        zeros = torch.zeros_like(u[0])

        N = matrix([
            [u[0], zeros, u[1], zeros, u[2], zeros],
            [zeros, u[0], zeros, u[1], zeros, u[2]]
        ]) # [2, n_basis*n_dim]
        return (N.T @ f).view(3, 2) # [n_basis, n_dim]


mesh = Mesh.gen_hollow_rectangle(chara_length=0.02, element_type="tri").double()

K_asm = KAssembler.from_mesh(mesh, quadrature_order=1)
F_asm = FAssemble.from_mesh(mesh, quadrature_order=1)


K = K_asm(mesh.points)
F = F_asm(mesh.points)


dirichlet_mask = torch.zeros_like(mesh.points, dtype=torch.bool)
dirichlet_mask[mesh.point_data['is_outer_bottom_boundary'], :] = True
condenser = Condenser(dirichlet_mask.flatten())

K_, f_ = condenser(K, F)

u_ = K_.solve(f_)

u = condenser.recover(u_)

mesh.plot(
    {"ux":u[0::2],
     "uy":u[1::2]},
     backend="matplotlib",
    save_path="press_hollow_rectangle.png",
    show_mesh=True
)