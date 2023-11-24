import torch
import warnings
from .scipy_solve import SparseSolveScipy, SparseLUSolveScipy
from .petsc_solve import SparseSolvePETSc, SparseLUSolvePETSc
from .cupy_solve import SparseSolveCupy, SparseLUSolveCupy
from ..utils import is_petsc_available
def spsolve(edata, row, col, shape, b, backend=None, verbose=True):
    """solve the sparse linear system Ax = b

    if the b of shape [n_node, n_batch], then a superLU will be used 
    else it will use spsolve 

    Parameters
    ----------
    edata: torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data of the sparse matrix A
    row: torch.Tensor 
        1D tensor of shape [n_edge]
        the row index of the sparse matrix A
    col: torch.Tensor 
        1D tensor of shape [n_edge]
        the col index of the sparse matrix A
    shape: Tuple[int, int]
        the shape of the sparse matrix A
    b: torch.Tensor 
        1D or 2D tensor of shape [n_node] or [n_node,batch]
        the right hand side vector b
    backend: str, optional
        the backend to use, by default None
        if None, 
            if edata.device == "cuda", then it will use cupy backend
            else if petsc is available, then it will use petsc backend
            else it will use scipy backend
        if "scipy", then it will use scipy backend
        if "petsc", then it will use petsc backend
        if "cupy", then it will use cupy backend
    Returns
    -------
    torch.Tensor 
        1D or 2D tensor  of shape [n_node] or [n_node,batch]
        the solution of the linear system
    """
    assert edata.device == row.device == col.device == b.device, f"edata, row, col b should be on the same device, but got {edata.device}, {row.device}, {col.device}, {b.device}"
    assert backend  in [None, "scipy", "petsc", "cupy"], f"backend should be None, scipy, petsc or cupy, but got {backend}"
    if edata.dtype != torch.float64:
        warnings.warn("Accuracy insufficient, float64 is recommended for better accuracy in spsolve")
    assert len(b.shape) <= 2, f"b should be of shape [n_node] or [n_node,batch], but got {b.shape}"
    if len(b.shape) == 2:
        if verbose:
            print(f"Use SuperLU to solve the batched linear system")
        if edata.device.type == "cpu":
            if is_petsc_available and (backend == "petsc" or backend is None):
                return SparseLUSolvePETSc.apply(edata, row, col, shape, b)
            else:
                assert edata.device.type == "cpu", f"get edata type {edata.device.type}, but backend is {backend}, should be None or scipy {'or petsc' if is_petsc_available else ''}"
                return SparseLUSolveScipy.apply(edata, row, col, shape, b)
        elif edata.device.type == "cuda":
            assert edata.device.type == "cuda", f"get edata type {edata.device.type}, but backend is {backend}, should be None or cupy"
            return SparseLUSolveScipy.apply(edata, row, col, shape, b)
        else:
            raise NotImplementedError("Only CPU and CUDA are supported")
    else:
        if edata.device.type == "cpu":
            if is_petsc_available and (backend == "petsc" or backend is None):
                return SparseSolvePETSc.apply(edata, row, col, shape, b)
            else:
                assert edata.device.type == "cpu", f"get edata type {edata.device.type}, but backend is {backend}, should be None or scipy {'or petsc' if is_petsc_available else ''}"
                return SparseSolveScipy.apply(edata, row, col, shape, b)
        elif edata.device.type == "cuda":
            assert edata.device.type == "cuda", f"get edata type {edata.device.type}, but backend is {backend}, should be None or cupy"
            return SparseSolveCupy.apply(edata, row, col, shape, b)
        else:
            raise NotImplementedError("Only CPU and CUDA are supported")