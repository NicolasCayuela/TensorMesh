
import torch 
from .assemble_operator import sym, eye, trace, \
                            sqrt, ddot, matrix, \
                            full,\
                            diagonal, coupled_tri_diagonal

def strain(gradu:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    gradu: torch.Tensor
        ND Tensor of shape [..., dim]
        gradient of displacement 
    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., dim, dim]
    """
    return sym(gradu)


def isotropic_stress(strain:torch.Tensor, 
           E:float|torch.Tensor=70.0, 
           nu:float|torch.Tensor = 0.3)->torch.Tensor:
    """
    Parameters
    ----------
    strain: torch.Tensor
        ND Tensor of shape [..., dim, dim]
        strain tensor

    E: float|torch.Tensor
        if torch.Tensor, ND Tensor of shape [...]
        Young's modulus

    nu: float
        if torch.Tensor, ND Tensor of shape [...]
        Poisson's ratio

    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., dim, dim]
    """
    # assertion
    if isinstance(E, torch.Tensor):
        assert E.shape == strain.shape[:-2]
    if isinstance(nu, torch.Tensor):
        assert nu.shape == strain.shape[:-2]

    dim = strain.shape[-1]
    mu = E/(2.*(1. + nu))
    _lambda = E*nu/((1+nu)*(1-2*nu))
    return eye(_lambda * trace(strain), dim) + 2*mu*strain
 

def deviatoric_stress(stress:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    stress: torch.Tensor
        ND Tensor of shape [..., dim, dim]
        stress tensor
    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., dim, dim]
    """
    dim = stress.shape[-1]
    return stress - 1./dim * eye(trace(stress), dim)

def deviatoric_stress_norm(stress:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    stress: torch.Tensor
        ND Tensor of shape [..., dim, dim]
        stress tensor
    Returns
    -------
    torch.Tensor
        ND Tensor of shape [...]
    """
    stress = deviatoric_stress(stress)
    return sqrt(1.5*ddot(stress, stress))


def voigt_shape_grad(gradu:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    gradu: torch.Tensor
        2D Tensor of shape [n_basis, dim]
        shape gradient 
    Returns
    -------
    torch.Tensor
        2D Tensor of shape [3, n_basis*2] or [6, n_basis*3]
    """
    assert gradu.dim() == 2
    dim = gradu.shape[-1]
    assert dim in (2, 3),f"dimension is only supported for 2, 3 for voigt shape grad, but got {dim}"
    if dim == 2:
        a = torch.vmap(torch.diag)(gradu) # [n_basis, dim, dim]
        b = torch.flip(gradu, dims=[-1]) # [n_basis, dim]
       
        B = torch.cat([a, b[:, None, :]], 1) # [n_basis, dim+1, dim]
        B = B.permute(1, 0, 2) # [dim+1, n_basis, dim]
        B = B.reshape(dim+1, -1)
    elif dim == 3:
        a = torch.vmap(torch.diag)(gradu) # [n_basis, dim, dim]
        b = coupled_tri_diagonal(gradu) # [n_basis, dim, dim]
        B = torch.cat([a, b], 1) # [n_basis,  dim+dim, dim]
        B = B.permute(1, 0, 2)
        B = B.reshape(dim+dim, -1)
    else:
        raise Exception(f"dim must be 2 or 3, but got {dim}")
    return B 

def voigt_stiffness(E:float|torch.Tensor, 
            nu:float|torch.Tensor,
            dim:int = 2)->torch.Tensor:
    """
    Parameters
    ----------
    E: float|torch.Tensor
        if torch.Tensor, ND Tensor of shape [...]
        Young's modulus

    nu: float
        if torch.Tensor, ND Tensor of shape [...]
        Poisson's ratio

    dim: int 
        dimension of the problem
        default is 2 , can only be 2 or 3

    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., 3, 3] or [3, 3] when dim == 2
        ND Tensor of shape [..., 6, 6] or [6, 6] when dim == 3

    """
    # assertion
    assert dim in (2, 3)

    mu = E/(2.*(1. + nu))
    _lambda = E*nu/((1+nu)*(1-2*nu))

    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu)
    if not isinstance(_lambda, torch.Tensor):
        _lambda = torch.tensor(_lambda)

    if dim == 2:
        C = eye(mu, 3)
        C[..., :2, :2] = _lambda + C[..., :2, :2]
        C[..., :2, :2] = eye(mu, 2) + C[..., :2, :2]
    elif dim == 3:
        C = eye(mu, 6)
        C[..., :3, :3] = _lambda + C[..., :3, :3]
        C[..., :3, :3] = eye(mu, 3) + C[..., :3, :3]
    else:
        raise Exception(f"dim must be 2 or 3, but got {dim}")
    
    return C
    
def voigt_shape_val(u:torch.Tensor, dim:int)->torch.Tensor:
    """
    Parameters
    ----------
    u: torch.Tensor
        2D Tensor of shape [n_basis]
        shape value
    dim: int 

    Returns
    -------
    torch.Tensor
        2D Tensor of shape [n_dim, n_basis*n_dim]
    """
    assert dim in (2, 3),f"dimension is only supported for 2, 3 for voigt shape val, but got {dim}"

    u = eye(u, dim = dim) # [n_basis, dim, dim]
    u = u.permute(1, 0, 2) # [dim, n_basis, dim]
    u = u.reshape(dim, -1) # [dim, n_basis*dim]
    return u

voigt_C = voigt_stiffness   
voigt_B = voigt_shape_grad
voigt_N = voigt_shape_val

