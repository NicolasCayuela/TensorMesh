"""Sparse matrix operations for FEM computations.

Built on top of ``torch-sla`` for differentiable sparse linear algebra.
``torch-sla`` is a hard dependency: import of this module fails if it is
not installed (see ``tensormesh.sparse.matrix``).
"""

from torch_sla import SparseTensor

from .matrix import SparseMatrix
from .solve import spsolve
from .mm import spmm
from .nonlinear_solve import nonlinear_solve
from .utils import is_petsc_available, is_cupy_available

__all__ = [
    'SparseMatrix',
    'SparseTensor',
    'spsolve',
    'spmm',
    'nonlinear_solve',
    'is_petsc_available',
    'is_cupy_available',
]
