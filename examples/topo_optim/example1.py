import torch 
import sys 
sys.path.append("../..")

from torch_fem import ElementAssembler,FacetAssembler, Mesh,Condenser

from torch_fem.functional import sym,eye,trace,ddot

class KAssembler(ElementAssembler):
    def __post_init__(self):
        self.d = 0.1
        self.E = 200e9
        self.nu = 0.3
    def forward(self, u, v, gradu, gradv):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
            Returns:
            --------
                K: torch.Tensor[n_basis, n_basis, n_dim, n_dim]
        """
       
        n_basis, n_dim = gradu.shape
        strain_u = sym(gradu) # [n_basis, n_dim, n_dim]
        stress_u = self.E / (1 + self.nu) * (strain_u + self.nu / (1 - nu) * eye(trace(strain_u), n_dim))   # [n_basis, n_dim, n_dim]
        strain_v = sym(gradv) # [n_basis, n_dim, n_dim]

        return d**3 / 12.0 * ddot(stress_u, strain_v)