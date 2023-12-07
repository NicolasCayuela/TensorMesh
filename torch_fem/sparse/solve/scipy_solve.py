from typing import Any
import torch 
from torch.autograd import Function
import scipy.sparse.linalg
import warnings
from ..utils import tensor2cupy, cupy2tensor, shapeT


class SparseSolveScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsr()
        u = scipy.sparse.linalg.spsolve(A_scipy, b)
        u = torch.tensor(u, dtype=b.dtype)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        A_T             = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=shapeT(ctx.A_shape)).tocsr()
        b_grad          = scipy.sparse.linalg.spsolve(A_T, grad_output.numpy())
        b_grad          = torch.tensor(b_grad, dtype=u.dtype)

        edata_grad      = - b_grad[row] * u[col] 
        return edata_grad, None, None, None, b_grad
    

class SparseLUSolveScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsc()
        lu = scipy.sparse.linalg.splu(A_scipy)
        u = lu.solve(b.numpy())
        u = torch.tensor(u, dtype=b.dtype)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        
        A_T             = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]]).tocsc()
        b_grad          = scipy.sparse.linalg.splu(A_T).solve(grad_output.numpy())
        b_grad          = torch.tensor(b_grad)

        edata_grad      = - (b_grad[row] * u[col]).sum(-1)
        return edata_grad, None, None, None, b_grad
    

