import sys
sys.path.append("../..")
import torch
from tqdm import tqdm
from torch_fem import ElementAssembler, Mesh, Condenser
from torch_fem import matrix, matmul, dot

from torch_fem import Mesh, ElementAssembler, NodeAssembler, FacetAssembler
from torch_fem.functional import ddot, strain, isotropic_stress
from torch_fem.functional import voigt_B, voigt_C, voigt_N
from torch_fem.visualization import draw_element_value_2d



import matplotlib.pyplot as plt 

class KAssembler(ElementAssembler):

    def __post_init__(self):
        E = 1
        nu = 0.3
    
        self.Emax = E 
        self.Emin = E * 0.001
        self.nu = nu

    def forward(self, gradu, gradv, theta):
        # gradu: [n_basis, 2]
        # gradv: [n_basis, 2]
        assert gradu.shape[-1] == 2
        n_basis, dim = gradu.shape

        E = self.Emin + (self.Emax - self.Emin) * theta # [n_basis]
       
        C = voigt_C(E, self.nu, dim=2)  # [3, 3]
        Bu = voigt_B(gradu) # [3, n_basis*2]
        Bv = voigt_B(gradv) # [3, n_basis*2]

        displacement = Bu.T @ C @ Bv # [n_basis*2, n_basis*2]
        displacement = displacement.view(n_basis, dim, n_basis, dim) # [n_basis, n_dim, n_basis, n_dim]
        displacement = displacement.permute(0, 2, 1, 3) # [n_basis, n_basis, n_dim, n_dim]
        return displacement
    
    
class FAssemble(FacetAssembler):
    def __post_init__(self):
        pass

    def forward(self, u):
        n_basis = u.shape[0]
        f = torch.tensor([0, -1]).type(u.dtype).to(u.device)
        N = voigt_N(u, 2) # [2, n_basis*n_dim]
        return (f @ N).view(n_basis, 2) # [n_basis, n_dim]
    

mesh = Mesh.gen_rectangle(chara_length=0.1, element_type="tri").double()

K_asm = KAssembler.from_mesh(mesh)
F_asm = FAssemble.from_mesh(mesh,  boundary_mask=mesh.point_data['is_right_boundary'])

# parameters
theta = torch.ones((mesh.n_elements,),
                   dtype = mesh.dtype,
                   device =  mesh.device
                   ).requires_grad_(True)

optimizer = torch.optim.Adam([theta],lr=1e-2)
epoch = 100000
condenser = Condenser(mesh.point_data['is_left_boundary'].flatten().repeat_interleave(2)) # since dim=2
F = F_asm()

pbar = tqdm(range(epoch))
for _ in pbar:

    optimizer.zero_grad()

    K = K_asm(element_data={"theta":theta})
    
    K_, f_ = condenser(K, F)

    u_ = K_.solve(f_)

    u = condenser.recover(u_)

    loss = u @ F

    pbar.set_postfix({"loss":loss.item()})

    loss.backward()

    optimizer.step()

fig ,ax = plt.subplots(figsize=(8,8))
collections, ax = draw_element_value_2d(
    mesh.points, 
    {mesh.default_element_type:mesh.elements()},
    {mesh.default_element_type:theta.detach().cpu().numpy()}, ax = ax)
umax = theta.max().item()
umin = theta.min().item()
cb = fig.colorbar(collections[mesh.default_element_type], ax=ax)
fig.savefig("topo_optim.png")
