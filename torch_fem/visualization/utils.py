from re import T
from matplotlib.pyplot import isinteractive
import torch 
import numpy as np
from typing import Sequence, Union, TypeVar,Generic
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, dia_matrix, dok_matrix, lil_matrix, issparse
from ..sparse import SparseMatrix
ScipySparseMatrix = Union[coo_matrix, csr_matrix, csc_matrix, dia_matrix, dok_matrix, lil_matrix]
def as_sparse_matrix(x:SparseMatrix|ScipySparseMatrix)->SparseMatrix:
    if issparse(x):
        x = x.tocoo()
        x = SparseMatrix.from_scipy_coo(x)
    elif isinstance(x, SparseMatrix):
        x = x.detach().cpu()
    else:
        raise TypeError(f"{type(x)} is not acceptable for SparseMatrix|ScipySparseMatrix")
    return x

def as_tensor(x:torch.Tensor|np.ndarray)->torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        raise ValueError(f"unsupported type {type(x)}")


def as_ndarray(x:torch.Tensor|np.ndarray)->np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"unsupported type {type(x)}")

def dim(x:torch.Tensor|np.ndarray)->int:
    if isinstance(x, torch.Tensor):
        return x.dim()
    elif isinstance(x, np.ndarray):
        return len(x.shape)
    else:
        raise ValueError(f"unsupported type {type(x)}")
    
