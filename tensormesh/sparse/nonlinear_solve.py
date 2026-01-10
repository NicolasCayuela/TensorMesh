import torch
from torch.autograd import Function
from typing import Callable, Optional, Tuple, Any, Dict
from ..sparse.matrix import SparseMatrix

def newton_solve(
    f: Callable[..., torch.Tensor],
    j: Callable[..., SparseMatrix],
    u0: torch.Tensor,
    params: Tuple[torch.Tensor, ...],
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> torch.Tensor:
    """
    Newton-Raphson solver for F(u, params) = 0
    
    Parameters
    ----------
    f : Callable
        Function F(u, *params) -> residual (Tensor)
    j : Callable
        Function J(u, *params) -> Jacobian (SparseMatrix)
    u0 : torch.Tensor
        Initial guess
    params : Tuple
        Parameters for f and j
    max_iter : int
        Maximum iterations
    tol : float
        Tolerance (residual norm)
    verbose : bool
        Print progress
        
    Returns
    -------
    torch.Tensor
        Solution u
    """
    u = u0.clone()
    
    for i in range(max_iter):
        res = f(u, *params)
        res_norm = torch.norm(res)
        
        if verbose:
            print(f"Iter {i}: |F(u)| = {res_norm:.6e}")
            
        if res_norm < tol:
            return u
            
        J = j(u, *params)
        # Newton step: u_{n+1} = u_n - J^{-1} F(u_n)
        # J du = res => du = J^{-1} res
        # Solve linear system with tighter tolerance than Newton tolerance
        linear_tol = max(tol * 0.01, 1e-12)
        du = J.solve(res, tol=linear_tol)
        u = u - du
        
    if verbose:
        print(f"Newton solver reached max_iter ({max_iter}) with residual {res_norm:.6e}")
        
    return u

class NonLinearSolveFunction(Function):
    @staticmethod
    def forward(ctx, f, j, u0, solver_config, *params):
        # 1. Newton Solve (detached)
        with torch.no_grad():
            u = newton_solve(f, j, u0, params, **solver_config)
            
        # 2. Save for backward
        ctx.save_for_backward(u, *params)
        ctx.f = f
        ctx.j = j
        
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u = ctx.saved_tensors[0]
        params = ctx.saved_tensors[1:]
        f = ctx.f
        j = ctx.j
        
        # 1. Recompute Jacobian at solution
        with torch.no_grad():
            J = j(u, *params)
        
        # 2. Solve adjoint system: J^T * lambda = grad_output
        # This implies lambda = J^{-T} * grad_output
        # Check if J is SparseMatrix
        if not isinstance(J, SparseMatrix):
            raise TypeError(f"Jacobian function j must return a SparseMatrix, got {type(J)}")
            
        # Solve J.T @ lam = grad_output
        # We use J.T.solve(grad_output)
        lam = J.T.solve(grad_output)
        
        # 3. Compute VJP: dL/dparams = - lambda^T * dF/dparams
        # We use autograd to compute (dF/dparams)^T * (-lambda)
        # which is equivalent to vector-Jacobian product with vector = -lam
        
        with torch.enable_grad():
            # We need to trace F(u, params) w.r.t params
            # u is treated as constant (detached)
            u_detached = u.detach()
            
            # We rely on params being captured from forward and having requires_grad status preserved
            # But saved_tensors might not automatically track if we are in a no_grad block in backward?
            # Actually, backward runs in no_grad by default? No.
            # But we want to differentiate F w.r.t params.
            
            # We must explicitly enable grad if we are in a no_grad context, which backward is NOT by default,
            # but usually it's safer to use torch.enable_grad() explicitly if we are re-running forward code.
            
            # We need to extract params that require grad to avoid errors?
            # autograd.grad can handle it if we pass allow_unused=True
            
            # We assume params in saved_tensors are the same tensor objects as input
            # We need to ensure they are connected to the graph?
            # saved_tensors returns the tensors. If they required grad in forward, they still do (unless detached).
            # But in forward we didn't use them in a graph (we used no_grad).
            # So the tensors themselves are fine, but F needs to build a graph.
            
            # Critical: We must enable grad to build the graph for F
            res = f(u_detached, *params)
            
            grads = torch.autograd.grad(
                outputs=res,
                inputs=params,
                grad_outputs=-lam,
                allow_unused=True
            )
            
        # Return gradients for (f, j, u0, solver_config, *params)
        # None for f, j, u0, solver_config
        return (None, None, None, None) + grads

def nonlinear_solve(
    f: Callable[..., torch.Tensor],
    j: Callable[..., SparseMatrix],
    u0: torch.Tensor,
    params: Tuple[torch.Tensor, ...],
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> torch.Tensor:
    """
    Solve F(u, params) = 0 for u using Newton-Raphson method with implicit differentiation support.
    
    Parameters
    ----------
    f : Callable
        Function F(u, *params) -> residual (Tensor).
        Should support autograd for params if gradients are needed.
    j : Callable
        Function J(u, *params) -> Jacobian (SparseMatrix).
        The Jacobian dF/du at u.
    u0 : torch.Tensor
        Initial guess for u.
    params : Tuple[torch.Tensor, ...]
        Parameters passed to f and j. These can be optimized.
    max_iter : int, optional
        Maximum number of Newton iterations. Default 100.
    tol : float, optional
        Convergence tolerance for residual norm. Default 1e-6.
    verbose : bool, optional
        Whether to print solver progress. Default False.
        
    Returns
    -------
    torch.Tensor
        The solution u such that F(u, params) approx 0.
        Gradients can be backpropagated through u to params.
    """
    solver_config = {
        'max_iter': max_iter,
        'tol': tol,
        'verbose': verbose
    }
    return NonLinearSolveFunction.apply(f, j, u0, solver_config, *params)

