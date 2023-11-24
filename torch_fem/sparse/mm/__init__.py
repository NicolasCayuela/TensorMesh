from .cupy_mm  import SparseMMCupy, SparseMVCupy
from .scipy_mm   import SparseMMScipy, SparseMVScipy

def spmv(edata, row, col, shape, B):
    """
    Parameters
    ----------
    edata : torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data
    row  : torch.Tensor 
        1D tensor of shape [n_edge]
        the row indices
    col  : torch.Tensor 
        1D tensor of shape [n_edge]
        the column indices
    shape: Tuple[int,  int]
        the shape of the sparse matrix
    B    : torch.Tensor 
        1D tensor of shape [n_node]
        the dense vector
    Returns
    -------
    torch.Tensor 
        1D tensor of shape [n_node]
        the output vector
    """
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    assert B.dim() == 1
    if edata.device.type == 'cpu':
        return SparseMVScipy.apply(edata, row, col, shape, B)
    elif edata.device.type == 'cuda':
        return SparseMVCupy.apply(edata, row, col, shape, B)
    else:
        raise NotImplementedError(f"device {edata.device.type} not supported")

def spmm(edata, row, col, shape, B):
    """
    Parameters
    ----------
    edata: torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data
    row  : torch.Tensor 
        1D tensor of shape [n_edge]
        the row indices
    col  : torch.Tensor 
        1D tensor of shape [n_edge]
        the column indices
    shape: Tuple[int int]
        the shape of the sparse matrix
    B    : torch.Tensor 
        2D or 1D torch.Tensor of shape [n_node, n_feature] or [n_node]
        the dense matrix/vector
    Returns:
    --------
    torch.Tensor 
        2D or 1D torch.Tensor of shape [n_node, n_feature] or [n_node]
        the output feature matrix
    """
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    if B.dim() == 1:
        return spmv(edata, row, col, shape, B)
    assert B.dim() == 2
    if edata.device.type == 'cpu':
        return SparseMMScipy.apply(edata, row, col, shape, B)
    elif edata.device.type == 'cuda':
        return SparseMMCupy.apply(edata, row, col, shape, B)
    else:
        raise NotImplementedError(f"device {edata.device.type} not supported")
