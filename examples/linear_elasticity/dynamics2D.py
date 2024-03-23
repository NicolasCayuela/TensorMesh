import sys
sys.path.append("../..")
import torch
from torch_fem import ElementAssembler, Mesh, Condenser
from torch_fem import matrix, matmul, dot, mul

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

class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return mul(u, v)


if __name__ == '__main__':
    # Simulation parameters
    dt = 0.001  # Time step
    n_steps = 100  # Number of time steps

    # Mesh and material properties setup
    mesh = Mesh.gen_hollow_rectangle(chara_length=0.02, element_type="tri").double()

    # Assemblers
    K_asm = KAssembler.from_mesh(mesh, quadrature_order=1)
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=1)
    
    F_asm = FAssemble.from_mesh(mesh, quadrature_order=1)

    # Initial conditions
    u = torch.zeros_like(mesh.points)
    v = torch.zeros_like(mesh.points)  # Initial velocity

    for step in range(n_steps):
        # Update external forces if necessary
        F = F_asm(mesh.points)

        # Assemble matrices
        K = K_asm(mesh.points)
        M = M_asm(mesh.points)

        # Dynamic equation: M * u_ddot + K * u = F
        # Using central difference or Newmark-beta for time integration
        # Implement the time integration scheme here
        acc = K.solve(F - M @ u)
        # Update u and v (displacement and velocity)
        v = v + dt * acc
        u = u + dt * v

        # Post-processing (visualization, data collection, etc.)
        
    # End of time-stepping loop
