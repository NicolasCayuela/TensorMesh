from numpy import where
from sympy import N
import torch
import torch.nn as nn
from functools import lru_cache, reduce
import math 
import torch
import re 
from typing import List, Tuple, Optional, Type, Sequence, Union,Self



class Polynomial(nn.Module):
    r"""
    
    Polynomail for multi dimension
    
    Usage:
    ------
    >>> exp = torch.tensor([[1, 2, 3], [2, 1, 0]]) # [n_vars(2), n_terms(3)]
    >>> coef = torch.tensor([1, 1, 1])             # [n_terms(3)]
    >>> poly = Polynomial(exp, coef)
    >>> print(poly)
    xy^2 + x^2y + x^3
    >>> x = torch.tensor([2, 3]) # [n_vars]
    >>> print(poly(x)) # 2*3^2 + 2^2*3 + 2^3 = 38
    torch.Float(38)
    >>> print(poly.deriv(0)) 
    y^2 + 2xy + 3x^2 
    >>> print(poly.deriv(1))
    2xy + x^2
    """

    _coef:torch.Tensor # [n_terms]
    _exp:torch.Tensor  # [n_vars, n_terms]
    n_vars:int
    n_terms:int

    def __init__(self, 
                 exp:torch.Tensor, 
                 coef:Optional[torch.Tensor] = None):
        """
        coef: torch.Tensor [n_terms]
        exp: torch.Tensor [n_vars, n_terms]
        """
        super().__init__()

        assert exp.dim() == 2, f"exp should be of shape [n_vars, n_terms], but got exp of shape {exp.shape}"
        if coef is None:
            coef = torch.ones(exp.shape[1],dtype=exp.dtype, device=exp.device)
        assert (coef.dim() == 1 and 
                coef.shape[0] == exp.shape[1]), (f"coef should be of shape [n_terms], exp should be of shape [n_vars, n_terms], "
                                               f"but got coef of shape {coef.shape}, and exp of shape {exp.shape}")
        assert exp.dtype == coef.dtype, f"exp and coef should have the same dtype, but got {exp.dtype} and {coef.dtype}"
        assert exp.device == coef.device, f"exp and coef should have the same device, but got {exp.device} and {coef.device}"
        self.register_buffer('_coef', coef)
        self.register_buffer('_exp', exp)
        self.n_vars  = exp.shape[0]
        self.n_terms = exp.shape[1]

    @property 
    def device(self):
        return self._coef.device
    
    @property
    def dtype(self):
        return self._coef.dtype

    def __len__(self):
        return self.n_terms

    def __getitem__(self, index:int|slice|torch.Tensor
                    )->Tuple[torch.Tensor, torch.Tensor]|\
                        'Polynomial'|\
                        'Polynomials':
        _coef = self._coef[index]
        _exp  = self._exp[:, index]
        if _coef.dim() == 0:
            return _coef, _exp
        elif _coef.dim() == 1:
            return Polynomial(_exp, _coef)
        elif _coef.dim() > 1:
            return Polynomials(_exp, _coef)
        else:
            raise NotImplementedError(f"Invalid input shape {index}")

    def __str__(self, max_show_col:int=2):
        assert self.n_vars <= 26, "The number of variables should be less than 26"
        if self.n_vars <= 26:
            VAR_NAME = lambda x: list("xyzabcdefghijklmnopqrstuvw")[x]
        else:
            VAR_NAME = lambda x: f"(x{x})"
        string = []
        for i,c in enumerate(self._coef):
            if c != 0:
                c = c.item()
                _vars = []
                for j, e in enumerate(self._exp[:,i]):
                    if e > 0:
                        _vars.append(f"{VAR_NAME(j)}^{e}")
                if len(_vars) > 0 and c == 1: 
                    string.append(''.join(_vars))
                else:
                    string.append(f"{c}{''.join(_vars)}")
        if max_show_col is not None and len(string) > max_show_col * 2:
            string = string[:max_show_col] + ["..."] + string[-max_show_col:]
              
        return " + ".join(string)
    
    def __repr__(self):
        return str(self)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
            x : torch.Tensor [n_batch, n_vars] or [n_vars]

        Returns
        -------
            torch.Tensor [n_batch] or torch.Float
        """
        assert x.dtype == self.dtype, f"x and polynomial should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.device == self.device, f"x and polynomial should have the same device, but got {x.device} and {self.device}"
        assert x.dim() == 1 or x.dim() == 2, f"x should be of shape [n_batch, n_vars] or [n_vars], but got x of shape {x.shape}"
        assert x.shape[-1] == self.n_vars, f"x should have the same number of variables as the polynomial, but got {x.shape[-1]} and {self.n_vars}"
        x = self.get_exp_terms(x) # [n_batch, n_terms] or [n_terms]
        x = x @ self._coef
        
        return x

    def get_exp_terms(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
            x : torch.Tensor [n_batch, n_vars] or [n_vars]

        Returns
        -------
            torch.Tensor [n_batch, n_terms] or [n_terms]
        """
        assert x.dtype  == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        if x.dim() == 1:
            x = torch.pow(x[:, None], self._exp) # [n_vars, n_terms]
            x = torch.prod(x, dim=0)    # [n_terms]
        else:
            assert x.dim() == 2, f"x should be of shape [n_batch, n_vars], but got x of shape {x.shape}"
            # breakpoint()
            x = torch.pow(x[:, :, None], self._exp[None, :, :]) # [n_batch, n_vars, n_terms]
            x = torch.prod(x, dim=1)    # [n_batch, n_terms]
        return x
        
    def deriv(self, var_ind:int=0)->'Polynomial':
        """
        Parameters
        ----------
            var_ind: int 
                    the index of the variable to be differentiated
        
        Returns
        -------
            deriv_poly: Polynomial
                        the derivative of the polynomial
        """
        assert var_ind < self.n_vars, f"var_ind should be less than {self.n_vars}, but got {var_ind}"
        coef = self._coef.clone()
        exp  = self._exp.clone()
        where_constant = exp[var_ind] == 0
        exp[var_ind] = exp[var_ind] - 1
        exp[var_ind] = torch.clamp_min(exp[var_ind], 0.0)
        mask         = torch.ones_like(coef)
        mask[where_constant] = 0
        coef     = coef * (exp[var_ind] + 1) * mask        
        return self.__class__(exp, coef)
    
    def grad(self)->'Polynomials':
        """
        Returns
        -------
            grad_poly: Polynomials
                        the gradient of the polynomial
        """
        # TODO: Could be more efficient
        return Polynomials.stack([self.deriv(i) for i in range(self.n_vars)])

    def reset_coef(self, coef:torch.Tensor)->Self:
        """
        Reset the coefficients of the polynomial
        """

        assert coef.shape == self._coef.shape, f"reset coef error: expected {self._coef.shape} but got {coef.shape}"
        assert coef.device== self.device, f"reset coef error: expected {self.device} but got {coef.device}"
        assert coef.dtype == self.dtype, f"reset coef error: expected {self.dtype} but got {coef.dtype}"

        self._coef = coef

        return self

    def repeat(self, *args)->'Polynomials':
        """
        Parameters
        ----------
        n: int
            the number of times to repeat the polynomial
        Returns
        -------
        polynomials: Polynomials
        """
        
        exps = self._exp[None, :, :].repeat(math.prod(args), 1, 1, 1)
        coef = self._coef[None, :].repeat(math.prod(args), 1)
        exps = exps.reshape(*args, self.n_vars, self.n_terms)
        coef = coef.reshape(*args, self.n_terms)
        return Polynomials(exps, coef)
       
    @classmethod 
    def lin_exp(cls, 
                n_vars:int, 
                dtype:torch.dtype=torch.float32, 
                device:torch.device=torch.device('cpu')
                )->'Polynomial':
        """
        Linear Polynomial exponential

        Parameters
        ----------
        n_vars: int 
            number of vairables
        dtype: torch.dtype
            the data type of the polynomial
        device: torch.device
            the device of the polynomial
        Returns
        -------
        polynomial: Polynomial n_vars=n_vars n_terms=n_vars+1
        """
        exp = torch.cat([
            torch.zeros(n_vars, 1),
            torch.eye(n_vars)
        ], -1) # [n_vars, n_vars+1]
        exp = exp.type(dtype).to(device)
        return Polynomial(exp)
                
    @classmethod 
    def poly_exp(cls, 
                 n_vars:int, 
                 dim:int,
                 dtype:torch.dtype=torch.float32,
                 device:torch.device=torch.device('cpu')
                 )->'Polynomial':
        """
        Triangle / Tetra Polynomial exponential

        Parameters
        ----------
        n_vars: int 
            number of vairables
        dim: int
            number of maximum dimension 
        dtype: torch.dtype
            the data type of the polynomial
        device: torch.device
            the device of the polynomial
        Returns
        -------
        polynomial: Polynomial n_vars=n_vars 
        """
        if n_vars == 1:
            exp = torch.arange(dim+1)
        else:
            axes = torch.meshgrid(*[torch.arange(dim+1) for _ in range(n_vars)], indexing='xy')
            axes = list(map(lambda x: x.flatten(), axes))
            exp  = torch.stack(axes, 0)
            exp  = exp[:, exp.sum(0) <= dim]
            exp  = exp[:, exp.sum(0).argsort()]

        exp = exp.type(dtype).to(device)
        return Polynomial(exp)

    @classmethod 
    def tens_exp(cls, 
                 n_vars:int, 
                 dim:int,
                 dtype:torch.dtype=torch.float32,  
                 device:torch.device=torch.device('cpu')
                 )->'Polynomial':
        """
        Tensor product exponential

        Parameters
        ----------
        n_vars: int 
            number of vairables
        dim: int
            number of maximum dimension 
        dtype: torch.dtype
            the data type of the polynomial
        device: torch.device
            the device of the polynomial
        Returns
        -------
        polynomial: Polynomial n_vars=n_vars n_terms=(dim+1)**n_vars
        """
        if n_vars == 1:
            exp = torch.arange(dim+1)[None, :] # [1, dim+1]
        else:
            axes = torch.meshgrid(*[torch.arange(dim+1) for _ in range(n_vars)], indexing='xy') 
            axes = list(map(lambda x: x.flatten(), axes))
            exp  = torch.stack(axes, 0) # [n_vars, (dim+1)**n_vars]
            exp  = exp[:, exp.sum(0).argsort()]

        exp = exp.type(dtype).to(device)

        return Polynomial(exp)

    @classmethod 
    def pyr_exp(cls, 
                order:int,
                dtype:torch.dtype=torch.float32,
                device:torch.device=torch.device('cpu')
                )->'Polynomial':
        axes = torch.meshgrid(torch.arange(order+1), 
                              torch.arange(order+1), 
                              torch.arange(order+1), 
                              indexing='xy')
        axes = list(map(lambda x: x.flatten(), axes))
        exp  = torch.stack(axes, 0)
        exp  = exp[:, (exp[0]+exp[2] < order+1) & (exp[1]+exp[2] < order+1)]
        exp  = exp[:, exp.sum(0).argsort()]
        exp  = exp.type(dtype).to(device)
        return Polynomial(exp)
            
    @classmethod 
    def pri_exp(cls, 
                order:int,
                dtype:torch.dtype=torch.float32,
                device:torch.device=torch.device('cpu')
                )->'Polynomial':
        axes = torch.meshgrid(torch.arange(order+1), 
                              torch.arange(order+1), 
                              torch.arange(order+1), 
                              indexing='xy')
        axes = list(map(lambda x: x.flatten(), axes))
        exp  = torch.stack(axes, 0)
        exp  = exp[:, exp[0]+exp[1] < order+1]
        exp  = exp[:, exp.sum(0).argsort()]
        exp  = exp.type(dtype).to(device)
        return Polynomial(exp)

class Polynomials(nn.Module):
    
    _coef:torch.Tensor # [n_poly1, ..., n_terms]
    _exp:torch.Tensor  # [n_poly1, ..., n_vars, n_terms]
    n_polys:Tuple[int, ...]
    n_vars:int
    n_terms:int

    def __init__(self, 
                 exp:torch.Tensor, 
                 coef:Optional[torch.Tensor]=None):
        """
        coef: torch.Tensor [n_poly1, .... , n_terms]
        exp: torch.Tensor [n_poly1, ..., n_vars, n_terms]
        """
        super().__init__()
        *n_polys, n_vars, n_terms = exp.shape
        if coef is None:
            coef = torch.ones(*n_polys, n_terms, dtype=exp.dtype, device=exp.device)
        for i in range(len(n_polys)):
            assert coef.shape[i] == n_polys[i]
        assert coef.shape[-1] == n_terms
        assert exp.dtype == coef.dtype, f"exp and coef should have the same dtype, but got {exp.dtype} and {coef.dtype}"
        assert exp.device == coef.device, f"exp and coef should have the same device, but got {exp.device} and {coef.device}"
        # self._coef  = coef
        # self._exp   = exp
        self.register_buffer('_coef', coef)
        self.register_buffer('_exp', exp)
        self.n_polys = n_polys 
        self.n_vars  = n_vars 
        self.n_terms = n_terms

    @property 
    def device(self):
        return self._coef.device
    
    @property
    def dtype(self):
        return self._coef.dtype 
    
    def __len__(self):
        return self.n_polys[0]
    
    @property 
    def shape(self):
        return tuple(self.n_polys) 
    
    def dim(self):
        return len(self.n_polys)
    
    def __getitem__(self, indices)->Tuple[torch.Tensor,torch.Tensor]|\
                                    'Polynomial'|\
                                    'Polynomials':
        _coef = self._coef[indices]
        _exp  = self._exp[indices]
        if _coef.dim() == 0:
            return _coef, _exp
        elif _coef.dim() == 1:
            return Polynomial(_exp, _coef)
        elif _coef.dim() > 1:
            return Polynomials(_exp, _coef)
        else:
            raise Exception(f"Invalid input shape {indices}")
        
    def __str__(self, max_show_item=2):

        def vector2str(max_show_item=2):
            string = []
            for i in range(len(self)):
                string.append(self[i].__str__(max_show_item))

            if max_show_item is None or len(self) <= max_show_item*2:
                return  "[" + ",\n".join(string) + "]"
            else:
                return "[" + ",\n".join(string[:max_show_item]) + ",\n...\n" + ",\n".join(string[-max_show_item:]) + "]" 

        def matrix2str(max_show_row=2, max_show_col=2):
            matrix = []
            for i in range(self.shape[0]):
                row = []
                for j in range(self.shape[1]):
                    row.append(self[i,j].__str__(max_show_col))
                matrix.append(row)

            if max_show_row is None or self.shape[0] <= max_show_row*2:
                if max_show_col is None or self.shape[1] <= max_show_col*2:
                    return "["+"\n".join(["["+", ".join(row)+"]" for row in matrix])+"]"
                else:
                    return "[" + "\n".join(["["+", ".join(row[:max_show_col])+", ..., "+", ".join(row[-max_show_col:])+"]" for row in matrix]) + "]"

            else:
                if max_show_col is None or self.shape[1] <= max_show_col*2:
                    return "["+"\n".join(["["+", ".join(row)+"]" for row in matrix[:max_show_row]])+"\n...\n"+"\n".join(["["+", ".join(row)+"]" for row in matrix[-max_show_row:]])+"]"
                else:
                    return ("[" +
                            "\n".join(["["+", ".join(row[:max_show_col])+", ..., "+", ".join(row[-max_show_col:])+"]" for row in matrix[:max_show_row]]) + 
                            "\n...\n" + 
                            "\n".join(["["+", ".join(row[:max_show_col])+", ..., "+", ".join(row[-max_show_col:])+"]" for row in matrix[-max_show_row:]]) + 
                            "]")
        
        if self.dim() == 1:
            return vector2str(max_show_item)
        elif self.dim() == 2:
            return matrix2str(max_show_item, max_show_item)
        else:
            return f"PolynomialTensor[{self.shape}] n_terms={self.n_terms} n_vars={self.n_vars}"
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
            x: torch.Tensor [n_batch, n_poly1, ..., n_vars] or [n_poly1, ..., n_vars]
        Returns
        -------
            torch.Tensor [n_batch, n_poly1, ...] or [n_poly1, ...]
        """

        x = self.get_exp_terms(x)             # [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        x = self.apply_coefficient(x)         # [n_batch, n_poly1, ...] or [n_poly1, ...]
        return x
    
    def get_exp_terms(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
            x: torch.Tensor [n_batch, n_poly1, ..., n_vars] or [n_poly1, ..., n_vars]
        Returns
        -------
            torch.Tensor  [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        """
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        assert x.dtype == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        if x.dim() == self.dim() + 2:
            # shape check
            n_batch, *n_polys, n_vars = x.shape
            assert n_polys == self.n_polys, f"Expected n_polys: {self.n_polys} but got {n_polys}"
            assert n_vars  == self.n_vars, f"Expected n_vars: {self.n_vars} but got {n_vars}"
            x    = torch.pow(x[..., None], self._exp[None, ...])  # [n_batch, n_poly1, ..., n_vars, n_terms]
            x    = torch.prod(x, dim=-2) # [n_batch, n_poly1, ..., n_terms]
        elif x.dim() == self.dim() + 1:
            # shape check 
            *n_polys, n_vars = x.shape 
            assert n_polys == self.n_polys, f"Expected n_polys: {self.n_polys} but got {n_polys}"
            assert n_vars  == self.n_vars, f"Expected n_vars: {self.n_vars} but got {n_vars}"
            x    = torch.pow(x[..., None], self._exp)  # [n_poly1, ..., n_vars, n_terms]
            x    = torch.prod(x, dim=-2) # [n_poly1, ..., n_terms]
        else:
            raise Exception(f"Should be shape of [n_batch, {self.shape}, {self.n_vars}] or [{self.shape}, {self.n_vars}], but got Invalid input shape {x.shape}")
        return x
    
    def apply_coefficient(self,  x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        x:torch.Tensor
            ND Tensor of shape [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        Returns
        -------
        torch.Tensor
        """
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        assert x.dtype == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.shape[-1] == self.n_terms
        for i in range(self.dim()):
            assert x.shape[-2 - i] == self.n_polys[-1-i], f"Expected {self.n_polys} but got {x.shape[-self.dim()-1:-1]}"


        if x.dim() == self.dim() + 2:
            n_batch, *n_polys, n_terms = x.shape
            x = self._coef.reshape(1, *n_polys, self.n_terms) * x # [n_batch, n_poly1, ..., n_terms]
            x = torch.sum(x, dim=-1)           # [n_batch, n_poly1, ..., n_poly2]
        elif x.dim() == self.dim()+1:
            *n_polys, n_terms = x.shape
            x = self._coef.reshape(*self.shape, self.n_terms) * x # [n_poly1, ..., n_terms]
            x = torch.sum(x, dim=-1) # [n_poly1, ..., n_poly2]
        else:
            raise ValueError(f"Expected input of shape [n_batch, {self.shape}, {self.n_vars}] or [{self.shape}, {self.n_vars}], but got {x.shape}")
        return x
    
    def map(self, x:torch.Tensor)->torch.Tensor:
        """broadcasting x to all dimension
        Parameters
        ----------
            x: torch.Tensor [n_batch, n_vars] or [n_vars]
        Returns
        -------
            torch.Tensor  [n_batch, n_poly1, ...,] or [n_poly1, ...,]
        """
        x = self.map_exp_terms(x) # [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        x = self.apply_coefficient(x) # [n_batch, n_poly1, ...,] or [n_poly1, ...,]
        return x

    def map_exp_terms(self, x:torch.Tensor)->torch.Tensor:
        """broadcasting x to all dimension
        Parameters
        ----------
            x: torch.Tensor [n_batch, n_vars] or [n_vars]
        Returns
        -------
            torch.Tensor  [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        """
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        assert x.dtype == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.shape[-1] == self.n_vars
        if x.dim() == 1:
            x = x.repeat(math.prod(self.n_polys), self.n_vars)
            x = x.reshape(*self.n_polys, self.n_vars)
        elif x.dim() == 2:
            n_batch = x.shape[0]
            x = x[:, None, :].repeat(1, math.prod(self.n_polys), 1)
            x = x.reshape(n_batch, *self.n_polys, self.n_vars)
        else:
            raise Exception(f"Should be shape of [n_batch, {self.n_vars}] or [{self.n_vars}], but got Invalid input shape {x.shape}")
           
        return self.get_exp_terms(x)

    def reshape(self, *args:Sequence[int])->'Polynomials':
        exps = self._exp.reshape(*args, self.n_vars, self.n_terms)
        coef = self._coef.reshape(*args, self.n_terms)
        return Polynomials(exps, coef)
    
    def transpose(self, *args:Sequence[int])->'Polynomials':
        assert len(args) == self.dim() 
        exps = self._exp.transpose(*args, self.dim(), self.dim() + 1)
        coef = self._coef.transpose(*args, self.dim())
        return Polynomials(exps, coef)
        
    def deriv(self, var_ind:int=0)->'Polynomials':
        """
        Parameters
        ----------
            var_ind: int 
                    the index of the variable to be differentiated
        
        Returns
        -------
            deriv_poly: Polynomial
                        the derivative of the polynomial
        """
        assert var_ind < self.n_vars, f"var_ind should be less than {self.n_vars}, but got {var_ind}"
        coef = self._coef.clone() # [n_poly1, ..., n_terms]
        exp  = self._exp.clone()  # [n_poly1, ..., n_vars, n_terms]
        where_constant = exp[..., var_ind, :] == 0 # [n_poly1, ..., n_terms]
        exp[..., var_ind, :] = exp[..., var_ind, :] - 1 # [n_poly1, ..., n_vars, n_terms]
        exp[..., var_ind, :] = torch.clamp_min(exp[..., var_ind, :], 0.0) # [n_poly1, ..., n_vars, n_terms]
        mask     = torch.ones_like(coef) # [n_poly1, ..., n_terms]
        mask[where_constant] = 0         # [n_poly1, ..., n_terms]
        coef     = coef * (exp[..., var_ind, :] + 1) * mask # [n_poly1, ..., n_terms]
        return self.__class__(exp, coef)
    
    def grad(self)->'Polynomials':
        """
        Returns
        -------
        grad_poly: PolynomialTensor [n_vars, n_poly1, ..., n_polyn] n_vars=n_vars n_terms = n_terms
                    the gradient of the polynomial
        """
        # TODO: could be more efficient
        return Polynomials.stack([self.deriv(i) for i in range(self.n_vars)])

    def reset_coef(self, coef:torch.Tensor):
        """
        Reset the coefficients of the polynomial
        Parameters
        ----------
        coef: torch.Tensor [n_poly, n_terms]
        """
        assert coef.shape == self._coef.shape, f"reset coef error: expected {self._coef.shape} but got {coef.shape}"
        assert coef.device== self.device, f"reset coef error: expected {self.device} but got {coef.device}"
        assert coef.dtype == self.dtype, f"reset coef error: expected {self.dtype} but got {coef.dtype}"
        self._coef = coef
        return self

    def repeat(self, *args):
        """
        Parameters
        ----------
        n: int
            number of times to repeat the PolynomialVector
        Return
        -------
            PolynomialTensor
        """
        assert len(args) == self.dim()
        exps = self._exp.repeat(*args, 1, 1)
        coef = self._coef.repeat(*args, 1)
        return PolynomialTensor(exps, coef)

    @classmethod 
    def stack(cls, polys:Sequence[Union[Polynomial,'Polynomials']])->'Polynomials':
        # check shape
        for i in range(1, len(polys)):
            assert polys[i].__class__ == polys[0].__class__, "All polynomials must have the same class."
            assert (polys[i].n_vars == polys[0].n_vars and 
                    polys[i].n_terms == polys[0].n_terms), \
                "All polynomials must have the same number of variables and terms."
            assert polys[i].dtype == polys[0].dtype, "All polynomials must have the same dtype."
            assert polys[i].device== polys[0].device, "All polynomials must have the same device."
            if isinstance(polys[0], Polynomials):
                assert polys[i].n_polys == polys[0].n_polys, "All polynomials must have the same number of polynomials."
        
        exps = torch.stack([p._exp for p in polys])
        coef = torch.stack([p._coef for p in polys])
        return cls(exps, coef)



   


