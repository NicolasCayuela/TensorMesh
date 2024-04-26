import torch
from typing import Sequence

#################
# Basi Operation
#################
"""
Since the tensor operation is inside the torch.vmap
some tensor operation cannot be used directly, we strong suggest to use the einsum operation
"""

def trace(x:torch.Tensor)->torch.Tensor:
    """

    .. math::
    
            \\text{trace}(A)_{\\cdots} = \\sum_{i=1}^n A_{\\cdots ii}

    Parameters
    ----------
    x : torch.Tensor 
        :math:`[..., D, D]`, where :math:`D` is the dimension of the matrix

    Returns
    -------
    torch.Tensor
        :math:`[...]` 
    """
    return torch.einsum(f"...ii->...", x)

def dot(a:torch.Tensor, b:torch.Tensor, reduce_dim:int=-1)->torch.Tensor:
    """

    .. math::

        \\text{dot}(A, B)_{\\cdots ab} = \\sum_{i=1}^n A_{\\cdots ai} B_{\\cdots bi}

    Parameters
    ----------
    a : torch.Tensor 
        :math:`[..., B, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix
    b : torch.Tensor
        :math:`[..., B, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix
    Returns
    -------
    torch.Tensor
        :math:`[..., B, B]`, where :math:`B` is the number of basis
    """
    if reduce_dim == -1:
        return torch.einsum("...ik,...jk->...ij", a, b)
    elif reduce_dim == -2:
        return torch.einsum("...ika,...jkb->...ijab", a, b)
    else:
        raise ValueError(f"reduce_dim must be -1 or -2, but got {reduce_dim}")
    
def ddot(a:torch.Tensor, b:torch.Tensor, is_basis_different:bool=True)->torch.Tensor:
    """

    .. math::
    
            \\text{ddot}(A, B)_{\\cdots ab} = \\sum_{i=1}^n A_{\\cdots aij} B_{\\cdots bij}

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., B, D, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix
    b : torch.Tensor   
        :math:`[..., B, D, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix     
    is_basis_different : bool
        whether the basis is different, default is True
    Returns
    --------
    torch.Tensor
        :math:`[..., B, B]`, where :math:`B` is the number of basis
        or 
        :math:`[...]` if is_basis_different = False
    """
    if is_basis_different:
        return torch.einsum("...aij,...bij->...ab", a, b)
    else:
        return torch.einsum("...ij,...ij->...", a, b)
  
def mul(a:torch.Tensor, b:torch.Tensor, is_basis_different:bool=True)->torch.Tensor:
    """

    .. math::

        \\text{mul}(A, B)_{\\cdots ij} = \\sum_{i=1}^n A_{\\cdots i} B_{\\cdots j} 

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., B]`, where :math:`B` is the number of basis
    b : torch.Tensor
        :math:`[..., B]`, where :math:`B` is the number of basis
    Returns
    -------
    torch.Tensor
        [..., B, B] if is_basis_different = True
        [..., B] if is_basis_different = False
    """
    if is_basis_different:  
        return torch.einsum("...i,...j->...ij", a, b)
    else:
        return a * b

def eye(value:torch.Tensor, dim:int)->torch.Tensor:
    """

    .. math::

        \\text{eye}(v, n)_{\\cdots ij} = \\begin{cases} v_{\\cdots}, & i=j \\\\ 0, & i \\neq j \\end{cases}

    Parameters
    ----------
    value : torch.Tensor
        :math:`[...]`, the filled value of the eye
    dim : int
        :math:`D`, the dimension of the eye

    Returns 
    -------
    torch.Tensor
        :math:`[..., D, D]`
    """
    indices = torch.arange(dim)
    mask    = indices.unsqueeze(0) == indices.unsqueeze(1)
    mask    = mask.type(value.dtype).to(value.device)
    return torch.einsum('...,ij->...ij', value, mask)

def full(value:torch.Tensor, dim:int)->torch.Tensor:
    """

    .. math::

        \\text{full}(v, n)_{\\cdots ij} = v_{\\cdots}

    Parameters
    ----------
    value : torch.Tensor
        :math:`[...]`, the filled value of the eye
    dim : int
        :math:`D`, the dimension of the eye

    Returns 
    -------
    torch.Tensor
        :math:`[..., D, D]`
    """
    ones = torch.ones((dim,dim), device=value.device, dtype=value.dtype)
    return torch.einsum("...,ij->...ij", value, ones)

def sym(a:torch.Tensor)->torch.Tensor:
    """

    .. math::

        \\text{sym}(A)_{\\cdots ij} = \\frac{1}{2} (A_{\\cdots i} + A_{\\cdots j})

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., D]`, where :math:`D` is the dimension of the matrix
    Returns
    -------
    torch.Tensor
        :math:`[..., D, D]`, where :math:`D` is the dimension of the matrix
    """
    return 0.5 * (a[..., None] + a[..., None, :])

def vector(x:Sequence[torch.Tensor])->torch.Tensor:
    """

    .. math::

        \\text{vector}(A) = \\begin{bmatrix}A_{\\cdots}^0\\ \\vdots \\ A_{\\cdots}^{n_{\\text{row}}-1\end{bmatrix}

    Parameters
    ----------
    x: : List[torch.Tensor]
        tensor list of shape [...]
    Returns
    -------
    torch.Tensor
        :math:`[..., n_{\\text{row}}]`
    """
    for i in range(1,len(x)):
        assert x[i].shape == x[0].shape
    return torch.stack(x, -1)

def matrix(x:Sequence[Sequence[torch.Tensor]])->torch.Tensor:
    """

    .. math::

        \\text{matrix}(A) = 
        \\begin{bmatrix}
        A_{\\cdots}^{0,0} & \\cdots & A_{\\cdots}^{n_{\\text{col}}-1} \\\\
        \\vdots & \\ddots & \\vdots \\\\
        A_{\\cdots}^{0,n_{\\text{row}}-1} & \\cdots & A_{\\cdots}^{n_{\\text{col}}-1,n_{\\text{row}}-1}
        \\end{bmatrix}


    Parameters
    ----------
        x : List[List[torch.Tensor]]
            tensor list of list of shape [...]
    Returns
    -------
    torch.Tensor
            :math:`[..., n_{\\text{col}}, n_{\\text{row}}]`
    """
    for i in range(1, len(x)):
        for j in range(1, len(x[i])):
            assert x[i][j].shape == x[0][0].shape
    return torch.stack([torch.stack(row, -1) for row in x], -2)

def transpose(x:torch.Tensor)->torch.Tensor:
    """

    .. math::
    
            \\text{transpose}(A)_{\\cdots ij} = A_{\\cdots ji}  


    Parameters
    ----------
    x : torch.Tensor
        :math:`[..., a, b]`
    Returns
    -------
    torch.Tensor
        :math:`[..., b, a]`
    """
    return torch.einsum("...ij->...ji", x)

def matmul(a:torch.Tensor,  b:torch.Tensor)->torch.Tensor:
    """

    .. math::
    
            \\text{matmul}(A, B)_{\\cdots ij} = \\sum_{k=1}^n A_{\\cdots ik} B_{\\cdots kj} 

    Parameters:
    -----------
    a : torch.Tensor
        :math:`[..., a, b]`
    b : torch.Tensor
        :math:`[..., b, c]`
    Returns:
    --------
    torch.Tensor
        :math:`[..., a, c]`
    """
    return torch.einsum("...ij,...jk->...ik", a, b)

def diagonal(x:torch.Tensor)->torch.Tensor:
    """

    .. math::
        
                \\text{diagonal}(A)_{\\cdots i} = A_{\\cdots ii}    

    Parameters
    ----------
    x:torch.Tensor
        ND Tensor of shape [..., D]
    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., D, D]
    """
    mask = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.einsum("...i,ij->...ij",x,mask)

def coupled_tri_diagonal(x:torch.Tensor)->torch.Tensor:
    r"""
    
    .. math::
        \begin{bmatrix}a\\b\\c\end{bmatrix} = \begin{bmatrix}b&a&0\\c&0&a\\0&c&b\end{bmatrix}
    
    Parameters
    ----------
    x:torch.Tensor
        ND Tensor of shape [..., D] D = 3
    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., D, D]
    """
    assert x.shape[-1] == 3
    y = torch.zeros(*x.shape,3, device=x.device, dtype=x.dtype)
    y[..., [1,1],[0,2]] = x[...,0]
    y[..., [0,2],[0,2]] = x[...,1]
    y[..., [0,2],[1,1]] = x[...,2]
    return y


#################
# Clamp Min Operation
#################

def sqrt(x:torch.Tensor)->torch.Tensor:  
    """it will return 0 if x < 0
    Parameters
    ----------
    x : torch.Tensor
        :math:`[...]`
    Returns
    -------
    torch.Tensor
        :math:`[...]`
    """
    x = torch.clamp_min(x, 0.)
    return torch.sqrt(x)

def divide(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
    """it will return 0 if y = 0
    Parameters
    ----------
    x : torch.Tensor
        :math:`[...]`
    y : torch.Tensor
        :math:`[...]`
    Returns
    -------
    torch.Tensor
        :math:`[...]`
    """
    return torch.where(y == 0., 0., x/y)