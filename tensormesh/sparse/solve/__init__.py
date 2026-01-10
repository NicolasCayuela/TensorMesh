import torch
import warnings
from .scipy_solve import SparseSolveScipy, SparseLUSolveScipy
from .cupy_solve import SparseSolveCupy, SparseLUSolveCupy
from .torch_solve import SparseSolveTorch, is_cpp_backend_available
from ..utils import is_petsc_available, is_cupy_available

# Lazy import PETSc solvers only when needed
SparseSolvePETSc = None
SparseLUSolvePETSc = None

def _get_petsc_solvers():
    global SparseSolvePETSc, SparseLUSolvePETSc
    if SparseSolvePETSc is None:
        from .petsc_solve import SparseSolvePETSc as _SparseSolvePETSc
        from .petsc_solve import SparseLUSolvePETSc as _SparseLUSolvePETSc
        SparseSolvePETSc = _SparseSolvePETSc
        SparseLUSolvePETSc = _SparseLUSolvePETSc
    return SparseSolvePETSc, SparseLUSolvePETSc





def spsolve(edata, row, col, shape, b, backend=None, verbose=True, x0=None, tol=1e-5, max_iter=10000):
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
            if edata.device == "cuda", then it will use torch backend (or cupy if available)
            else it will use torch backend
        if "scipy", then it will use scipy backend
        if "petsc", then it will use petsc backend (requires petsc4py)
        if "cupy", then it will use cupy backend
        if "torch", then it will use torch backend
    tol: float, optional
        tolerance for iterative solvers
    max_iter: int, optional
        maximum number of iterations for iterative solvers
    Returns
    -------
    torch.Tensor 
        1D or 2D tensor  of shape [n_node] or [n_node,batch]
        the solution of the linear system
    """
    assert edata.device == row.device == col.device == b.device, f"edata, row, col b should be on the same device, but got {edata.device}, {row.device}, {col.device}, {b.device}"
    assert backend  in [None, "scipy", "petsc", "cupy", "torch"], f"backend should be None, scipy, petsc or cupy, but got {backend}"
    if edata.dtype != torch.float64:
        warnings.warn("Accuracy insufficient, float64 is recommended for better accuracy in spsolve")
    assert len(b.shape) <= 2, f"b should be of shape [n_node] or [n_node,batch], but got {b.shape}"
    if len(b.shape) == 2:
        if verbose:
            print(f"Use SuperLU to solve the batched linear system")
        if edata.device.type == "cpu":
            if backend is None:
                # Default to scipy for batched solve on CPU (torch doesn't support batched yet)
                return SparseLUSolveScipy.apply(edata, row, col, shape, b)
            elif backend == "scipy":
                return SparseLUSolveScipy.apply(edata, row, col, shape, b)
            elif backend == "petsc":
                assert is_petsc_available, f"petsc is not available, please install petsc4py"
                _SparseSolvePETSc, _SparseLUSolvePETSc = _get_petsc_solvers()
                return _SparseLUSolvePETSc.apply(edata, row, col, shape, b)
            else:
                raise NotImplementedError(f"backend {backend} not supported for CPU batched solve")
        elif edata.device.type == "cuda":
            assert is_cupy_available, f"cupy is not available, please install cupy"
            return SparseLUSolveCupy.apply(edata, row, col, shape, b)
        else:
            raise NotImplementedError("Only CPU and CUDA are supported")
    else:
        if edata.device.type == "cpu":
            if backend is None:
                # Default to torch backend
                return SparseSolveTorch.apply(edata, row, col, shape, b, x0, tol, max_iter)
            elif backend == "torch":
                return SparseSolveTorch.apply(edata, row, col, shape, b, x0, tol, max_iter)
            elif backend == "scipy":
                return SparseSolveScipy.apply(edata, row, col, shape, b)
            elif backend == "petsc":
                assert is_petsc_available, f"petsc is not available, please install petsc4py"
                _SparseSolvePETSc, _ = _get_petsc_solvers()
                return _SparseSolvePETSc.apply(edata, row, col, shape, b)
            else:
                raise NotImplementedError(f"backend {backend} not supported for CPU")
        elif edata.device.type == "cuda":
            if backend is None:
                # Default to torch backend on CUDA
                return SparseSolveTorch.apply(edata, row, col, shape, b, x0, tol, max_iter)
            elif backend == "torch":
                return SparseSolveTorch.apply(edata, row, col, shape, b, x0, tol, max_iter)
            elif backend == "cupy":
                assert is_cupy_available, f"cupy is not available, please install cupy"
                return SparseSolveCupy.apply(edata, row, col, shape, b)
            else:
                raise NotImplementedError(f"backend {backend} not supported for CUDA")
        else:
            raise NotImplementedError("Only CPU and CUDA are supported")