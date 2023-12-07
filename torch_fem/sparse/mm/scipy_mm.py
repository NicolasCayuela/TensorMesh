
import torch
from torch.autograd import Function
import scipy.sparse
from ..utils import shapeT


class SparseMMScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        """
        Parameters
        ---------
        edata: torch.Tensor 
            1D tensor of shape [n_edge]
            the edge data
        row  : torch.Tensor 
            1D tensor of shape [n_edge]
            the row indices
        col  : torch.Tensor 
            1D tensor of shape [n_edge]
            the column indices
        shape: Tuple[int, int]
            bi-tuple [M, N]
            the shape of the sparse matrix
        B    : torch.Tensor 
            2D tensor of shape [N, K]
            the dense matrix
        Returns:
        --------
        torch.Tensor 
            2D tensor of shape [M, K]
            the output feature matrix
        """
        ctx.save_for_backward(edata, row, col, B)
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape)
        B_numpy = B.numpy()
        C_scipy = A_scipy.dot(B_numpy)
        C = torch.tensor(C_scipy, dtype=B.dtype)
        ctx.A_shape = shape
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Parameters
        -----------
        grad_outputs : torch.Tensor 
            torch.Tensor of shape [M, K]
            the gradient of the output feature matrix
        Returns
        -------
        edata_grad : torch.Tensor 
            1D tensor of shape [n_edge]
            the gradient of the edge data
        row_grad  : None
        col_grad  : None
        shape_grad: None
        B_grad    : torch.Tensor of shape [N, K]
            the gradient of the feature matrix
        """

        edata, row, col, B = ctx.saved_tensors
        edata_grad  = (grad_outputs[row] * B[col]).sum(dim=1)
       
        A_T    = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]])
        grad_B = A_T.dot(grad_outputs.numpy())
        grad_B = torch.tensor(grad_B, dtype=B.dtype)
        return edata_grad, None, None, None, grad_B


class SparseMVScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        ctx.save_for_backward(edata, row, col, B)
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape)
        B_numpy = B.numpy()
        C_scipy = A_scipy.dot(B_numpy)
        C = torch.tensor(C_scipy, dtype=B.dtype)
        ctx.A_shape = shape
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):

        edata, row, col, B = ctx.saved_tensors  
        edata_grad  = grad_outputs[row] * B[col]
        A_T    = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]])
        grad_B = A_T.dot(grad_outputs.numpy())
        grad_B = torch.tensor(grad_B, dtype=B.dtype)
        return edata_grad, None, None, None, grad_B