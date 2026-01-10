from .matrix import SparseMatrix
from .nonlinear_solve import nonlinear_solve

from .utils import is_petsc_available, is_cupy_available
from .solve import is_cpp_backend_available as is_solve_cpp_backend_available