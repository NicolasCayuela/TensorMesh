
from ast import Not
from functools import lru_cache, reduce
from logging import warning
import math 
from scipy.datasets import face
from sympy import factor
import torch
import re 
from typing import List, Tuple, Optional, Type, Sequence, Required, Union
from abc import ABC, abstractmethod

from zmq import device

from torch_fem.quadrature import quad
from .polynomial import Polynomial, \
                        Polynomials
from .basis import  lin_basis,\
                    tri_basis,\
                    quad_basis,\
                    tet_basis,\
                    hex_basis,\
                    pyr_basis,\
                    pri_basis,\
                    facet_basis_index_2d, \
                    tet_facet_basis_index, \
                    hex_facet_basis_index,\
                    mix_facet_basis_index
from .quadrature import lin_quadrature,\
                        tri_quadrature,\
                        tet_quadrature,\
                        quad_quadrature,\
                        hex_quadrature,\
                        pyr_quadrature,\
                        pri_quadrature,\
                        facet_quadrature_2d,\
                        tet_facet_quadrature,\
                        hex_facet_quadrature,\
                        mix_facet_quadrature_3d
from .normal import outwards_normal_2d,\
                    outwards_normal_3d
from .plot import plot_1d,\
                  plot_2d,\
                  plot_3d 
from .types import Tensorx1, Tensorx2, Tensorx3, Tensorx4, Tensorx5






class Element:
    points: torch.Tensor # [n_vertex, dim]
    vertex: torch.Tensor # [n_vertex, 1]
    edge: torch.Tensor # [n_face, 2]
    face: Optional[Tuple[Tuple[int,...],...]]   # [n_face, n_vertex_per_face]
    cell: Optional[torch.Tensor]  # [n_cell, n_vertex_per_cell]
    dim: int
    n_vertex: int
    n_edge: int
    n_face: int 
    n_cell: int 

    is_mix_facet:bool = False

    # @abstractmethod
    @classmethod
    def get_facet_type(cls)->Type['Element']|Tuple[Type['Element'],Type['Element']]:
        raise NotImplementedError()

    # @abstractmethod
    @classmethod
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->torch.Tensor:
        """
        Parameters
        ----------
            order: int 
                the order of the basis
            dtype: torch.dtype
                the float data type of the polynomial
            device: torch.device
                the device of the polynomial
        Returns
        -------
            basis: torch.Tensor [n_basis, n_dim]
        """
        raise NotImplementedError()
    
    # @abstractmethod
    @classmethod 
    def get_facet(cls, order:int=1)->Tensorx1|Tensorx2:
        """
        Parameters
        ----------
            order: int 
                the order of the basis

        Returns
        -------
            if cls.is_mix_facet:
                tri_facet: torch.Tensor [n_tri_facet, n_basis_per_tri_facet]
                quad_facet: torch.Tensor [n_quad_facet, n_basis_per_quad_facet]
            else:
                facet: torch.Tensor [n_facet, n_basis_per_facet]
        """
        raise NotImplementedError()
    
    # @abstractmethod
    @classmethod
    def get_polynomial(cls, 
                       order:int=1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Polynomial:
        """
        get the polynomial form for the element
        Parameters
        ----------
            order: int 
                the order of the basis
            dtype: torch.dtype
                the float data type of the polynomial
            device: torch.device
                the device of the polynomial
        Returns
        -------
            polynomial: Polynomial n_vars=dim 
        """
        raise NotImplementedError()

    # @abstractmethod
    @classmethod
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        """
        Parameters
        ----------
            order: int 
                quadrature order
            dtype: torch.dtype
                the float data type of the polynomial
            device: torch.device
                the device of the polynomial
        Returns
        -------
            quadrature_weights:torch.Tensor [n_quadrature]
            quadrature_points: torch.Tensor [n_quadrature, n_dim]
        """
        raise NotImplementedError()

    # @abstractmethod
    @classmethod
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )-> Tensorx2|Tensorx4:
        """
        Parameters
        ----------
            order: int 
                quadrature order                
                default is 1
            transform: bool
                whether return the transformed facet quadrature
                if true will return of shape [n_facet, n_quadrature_per_facet, dim] (if all facet are of the same element)
                otherwise will return of shape [n_quadrature_per_facet, dim - 1]
            dtype: torch.dtype
                the float data type of the polynomial
            device: torch.device
                the device of the polynomial
        Returns
        -------
            quadrature_weights: torch.Tensor 
                2D tensor of shape [n_facet, n_quadrature_per_facet] or 1D tensor of shape [n_quadrature_per_facet]
            quadrature_points: torch.Tensor 
                3D tensor of shape [n_facet, n_quadrature_per_facet, n_dim] or 2D tensor of shape [n_quadrature_per_facet, n_dim-1]
            or 
            tri_quadrature_weights: torch.Tensor
                2D tensor of shape [n_tri_facet, n_quadrature_per_tri_facet]
            tri_quadrature_points : torch.Tensor 
                3D tensor of shape [n_tri_facet, n_quadrature_per_tri_facet, n_dim]
            quad_quadrature_weights: torch.Tensor
                3D tensor of shape [n_quad_facet, n_quadrature_per_quad_facet]
            quad_quadrature_points: torch.Tensor
                3D tensor of shape [n_quad_facet, n_quadrature_per_quad_facet, n_dim]
            or 
            tri_quadrature_weights: torch.Tensor
                1D tensor of shape [n_quadrature_per_tri_facet]
            tri_quadrature_points : torch.Tensor
                2D tensor of shape [n_quadrature_per_tri_facet, n_dim-1]
            quad_quadrature_weights: torch.Tensor
                1D tensor of shape [n_quadrature_per_quad_facet]
            quad_quadrature_points: torch.Tensor
                2D tensor of shape [n_quadrature_per_quad_facet, n_dim-1]

        """
        raise NotImplementedError()

    @classmethod
    @lru_cache(1)
    def get_basis_fns(cls, 
                      order:int = 1, 
                      dtype:torch.dtype = torch.float32,
                      device:torch.device = torch.device('cpu')
                      )->Polynomials:
        """get the basis functions
        Parameters
        ----------
            order: int 
                the order of the basis 
            dtype: torch.dtype
                the float data type of the polynomial
            device: torch.device
                the device of the polynomial
        Returns
        -------
            polynomials: Polynomials [n_basis] n_vars=dim n_exp=n_basis
                the basis function of the element
        """
        def adaptive_inv(x:torch.Tensor)->torch.Tensor:
            eps = 1e-7
            assert x.dim() == 2
            assert x.shape[0] == x.shape[1]
            if order >= 3:
                result = torch.linalg.pinv(x)
            else:
                result = torch.linalg.inv(x)
            if torch.dist(x @ result, torch.eye(x.shape[0], dtype=x.dtype, device=x.device)) > eps:
                warning(f"basis functions are not accurate for {cls} with order {order}")
            return result
        
        basis = cls.get_basis(order, dtype, device) # [n_basis, n_dim] n_dim ~ n_vars
        
        n_basis = basis.shape[0]
        # basis [order + 1, 1]
        
        polys = cls.get_polynomial(order, dtype, device).repeat(n_basis) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        
        V = polys.get_exp_terms(basis) # [n_basis, n_basis]
        coef = adaptive_inv(V).T # [n_basis, n_basis] 
        polys.reset_coef(coef)
        polys = polys.to(device)
        return polys

    @classmethod
    @lru_cache(1)
    def get_basis_grad_fns(cls, 
                                 order:int = 1, 
                                 dtype:torch.dtype = torch.float32,
                                 device:torch.device = torch.device('cpu')
                                 )->Polynomials:
        """
        Parameters
        ----------
            order: int 
                the order of the basis
            dtype: torch.dtype
                the float data type of the polynomial
            device: torch.device
                the device of the polynomial
        Returns
        -------
            Polynomials [dim, n_basis] n_vars = dim n_terms = n_basis
        """
        basis_fns = cls.get_basis_fns(order, dtype, device) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        return basis_fns.grad() # Polynomials [dim, n_basis] n_vars=dim n_terms = n_basis

    @classmethod 
    def eval_cell_jacobian(cls, 
                        quadrature:torch.Tensor, 
                        element_coords:torch.Tensor, 
                        basis_order:int=1)->torch.Tensor:
        """
        Parameters
        ----------
            quadrature: torch.Tensor [n_quadrature, dim] 
                        or torch.Tensor [n_element, n_quadrature_per_element, dim]
                quadrature for each element
            element_coords: torch.Tensor [n_element, n_basis, dim]
            basis_order:int 
                default is 1
        Returns
        -------
            cell_jacobian: torch.Tensor [n_element, n_quadrature, dim, dim]
        """
        assert quadrature.dtype == element_coords.dtype, f"quadrature dtype {quadrature.dtype} != element_coords dtype {element_coords.dtype}"
        assert quadrature.device == element_coords.device, f"quadrature device {quadrature.device} != element_coords device {element_coords.device}"
        dtype  = quadrature.dtype
        device = quadrature.device

        shape_grad_fns = cls.get_basis_grad_fns(basis_order, dtype, device) # Polynomials  [dim, n_basis] n_vars=dim n_terms = n_basis
        
        if quadrature.dim() == 2:
            n_basis        = shape_grad_fns.shape[1]
            shape_grad     = shape_grad_fns.map(quadrature) # [n_quadrature, dim, n_basis]
            cell_jacobian  = torch.einsum("ebj,qib->eqij", element_coords, shape_grad)
        else:
            n_element, n_quadrature_per_element, dim = quadrature.shape
            assert n_element == element_coords.shape[0]
            n_basis        = element_coords.shape[1]
            shape_grad     = shape_grad_fns.map(quadrature.reshape(-1, dim)) # [n_element,n_quadrature, dim, n_basis]
            shape_grad     = shape_grad.reshape(n_element, n_quadrature_per_element, dim, n_basis)
            cell_jacobian  = torch.einsum("ebj,eqib->eqij", element_coords, shape_grad)

        return cell_jacobian

    @classmethod 
    def eval_shape_val(cls, 
                       quadrature_points:Optional[torch.Tensor] = None, 
                       order:int = 1, 
                       quadrature_order:int=1,
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->torch.Tensor:
        """
        Parameters
        ----------
            quadrature_points : Optional[torch.Tensor] [n_quadrature, n_dim]
            order : int
            quadrature_order : int 
                the order for quadrature, if quadrature is not None, quadrature_order will be ignored
            dtype: torch.dtype
                the float data type of the shape value, if quadrature_points is not None, the dtype will be ignored
                default is torch.float32
            device: torch.device
                the device of the shape value, if quadrature_points is not None, the device will be ignored
                default is torch.device('cpu')
        Returns 
        -------
            torch.Tensor [n_quadrature, n_basis]
        """
        dtype = quadrature_points.dtype if quadrature_points is not None else dtype
        device= quadrature_points.device if quadrature_points is not None else device
        if quadrature_points is None:
            _, quadrature_points = cls.get_quadrature(quadrature_order, dtype, device)
        basis_fns = cls.get_basis_fns(order, dtype, device) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        shape_val = basis_fns.map(quadrature_points) # [n_quadrature, n_basis]
        return shape_val 
    
    @classmethod 
    def eval_shape_grad(cls, 
                        element_coords:torch.Tensor, 
                        quadrature:Optional[torch.Tensor] = None, 
                        basis_order:int=1, 
                        quadrature_order:int=1
                        )->Tensorx2:
        """
        Parameters
        ----------
            element_coords: torch.Tensor [n_element, n_basis, dim]
            quadrature: torch.Tensor [n_quadrature, n_dim]
                quadrature points for each element
            basis_order:int 
                order for basis
            quadrature_order:int 
                order for quadrature, if element_coords is not None, the quadrature_order will be ignored,
                default is 1
        Returns
        -------
            shape_grad:    torch.Tensor [n_element, n_quadrature, n_basis, n_dim]
            cell_jacobian: torch.Tensor [n_element, n_quadrature, dim, dim]
        """
        dtype = element_coords.dtype
        device= element_coords.device
        if quadrature is None:
            _, quadrature = cls.get_quadrature(quadrature_order, dtype, device) # [n_quadrature, dim]

        shape_grad_fns = cls.get_basis_grad_fns(basis_order, dtype, device) # Polynomials  [dim, n_basis]
    
        shape_grad     = shape_grad_fns.map(quadrature) # [n_qudrature, dim, n_basis]

        cell_jacobian  = torch.einsum("ebj,qib->eqij", element_coords, shape_grad) # TODO: check the order
        
        inv_cell_jacobian = torch.inverse(cell_jacobian)

        shape_grad     = torch.einsum("qib, eqji->eqbj", shape_grad, inv_cell_jacobian) 

        return shape_grad, cell_jacobian

    @classmethod
    @lru_cache(1)
    def get_local_facet_mapping_fns(cls, 
                                    dtype:torch.dtype=torch.float32,
                                    device:torch.device=torch.device('cpu')
                                    )->Polynomials:
        r"""

        :math: `\gamma_f (r)`

        Parameters
        ----------
        dtype: torch.dtype
            the float data type of the polynomials
            default is torch.float32
        device: torch.device
            the device of the polynomials
            default is torch.device('cpu')
        Returns
        -------
        local_facet_fns:Polynomials [n_facet, dim] n_vars=dim-1 n_terms = dim, because the transform looks like :math:`1+\alpha \xi + \beta \epsilon`
        """
        facet = {
            2 : cls.edge, 
            3 : cls.face 
        }[cls.dim]
    
        facet_dim = cls.dim - 1
    
        if cls.dim == 2:
            origin_facet_coords = Line.points[Line.vertex] # [2(points), 1, 1(dim-1)]
            origin_facet_coords = origin_facet_coords.permute(1, 0, 2) # [1, 2(points), 1(dim-1)]
            transf_facet_coords = cls.points[facet]    # [n_facet, n_edge, 1(dim-1)]
        elif cls.dim == 3:
            origin_facet_coords = Triangle.points[Triangle.vertex] # [3(points), n_triangle, 2(dim-1)]
            origin_facet_coords = origin_facet_coords.permute(1, 0, 2) # [n_triangle, 3(points), 2(dim-1)]
            transf_facet_coords = cls.points[torch.tensor([x[:3] for x in facet])] # [n_triangle, 3(points), 3(dim)]
        else:
            raise Exception(f"Invalid dim {cls.dim}")
        
        transf_facet_coords = transf_facet_coords.type(dtype).to(device) # [n_facet, n_vertex_per_facet, dim]   
       
        polys = Polynomial.lin_exp(facet_dim, dtype, device) # n_vars=dim-1(n_vars), n_terms=dim
        polys = polys.repeat(len(facet), cls.dim)         # [n_face, dim] n_vars=dim-1 n_terms=dim
       
        origin_facet_coords = origin_facet_coords.repeat(len(facet), 1, 1) 
        origin_facet_coords = origin_facet_coords.type(dtype).to(device)    # [n_face, n_vertex_per_facet, dim-1]

        result= torch.linalg.lstsq(polys.get_exp_terms(origin_facet_coords), # [n_vertex_per_facet, n_facet, dim] 
                                transf_facet_coords)
        x     = result.solution # [n_facet, dim, dim]
        
        assert x.shape == (len(facet), cls.dim, cls.dim)
        polys.reset_coef(x)   
        return polys     
    
    @classmethod 
    @lru_cache(1)
    def get_local_facet_mapping_grad_fns(cls, 
                                         dtype:torch.dtype=torch.float32,
                                         device:torch.device=torch.device('cpu')
                                         )->Polynomials:
        r"""

        :math:`\frac{\partial \gamma_f (r)}{\partial r}`

        Parameters
        ----------
        dtype: torch.dtype
            the float data type of the polynomials
            default is torch.float32
        device: torch.device
            the device of the polynomials
            default is torch.device('cpu')
        Returns
        -------
            PolynomialTensor [dim-1(gradient), n_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim
        """
        local_facet_fns = cls.get_local_facet_mapping_fns(dtype, device) # Polynomials [n_facet, dim] n_vars=dim-1 n_terms = dim
        return local_facet_fns.grad() # [dim-1(gradient), n_facet, dim(global_coords)]

    @classmethod 
    @lru_cache(1)
    def get_outwards_facet_normal(cls, 
                                  dtype:torch.dtype=torch.float32,
                                  device:torch.device=torch.device('cpu')
                                  )->Tensorx1:
        """
        only facet normal without nanson scale
        Returns
        -------
            outward_facet_normal: torch.Tensor [n_facet, dim]
        """
        outwards_normal_fns = {
            2: outwards_normal_2d,
            3: outwards_normal_3d
        }
        facet = {
            2: cls.edge,
            3: cls.face
        }
        return outwards_normal_fns[cls.dim](cls.points.type(dtype).to(device), facet[cls.dim])

    @classmethod
    @lru_cache(1)
    def get_n_facet(cls)->int:
        dim2nfacet = {
            2 : len(cls.edge), 
            3 : len(cls.face)
        }
        return dim2nfacet[cls.dim]

    @classmethod 
    @lru_cache(1)
    def get_n_basis(cls, order:int = 1)->int:
        return cls.get_basis(order).shape[0]

    @classmethod
    def eval_facet_cell_jacobian(cls, 
                                 element_coords:torch.Tensor,
                                 basis_order:int=1,
                                 quadrature_order:int=1
                                 )->Tensorx1|Tensorx2:
        """
        Parameters
        ----------
        element_coords: torch.Tensor 
            3D Tensor of shape [n_element, n_basis, dim]
        basis_order: int
        quadrature_order: int
        Returns
        -------
        facet_cell_jacobian: torch.Tensor 
            5D Tensor of shape [n_element, n_facet, n_quadrature_per_facet, dim, dim]
        or 
        tri_facet_cell_jacobian: torch.Tensor 
            5D Tensor of shape [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim, dim]
        quad_facet_cell_jacobian: torch.Tensor 
            5D Tensor of shape [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim, dim]
        """
        device = element_coords.device
        dtype  = element_coords.dtype
        if cls.is_mix_facet:
            tri_w, tri_q, quad_w, quad_q = cls.get_facet_quadrature(
                                            quadrature_order, 
                                            dtype=dtype, device=device)
            n_tri_facet, n_quadrature_per_tri_facet, _ = tri_q.shape
            n_quad_facet, n_quadrature_per_quad_facet, _ = quad_q.shape
            q = torch.cat([tri_q.reshape(-1,  3), quad_q.reshape(-1, 3)], 0)
            j = cls.eval_cell_jacobian(q, element_coords, basis_order)
            tri_j = j[:, :n_tri_facet * n_quadrature_per_tri_facet]
            quad_j= j[:, n_tri_facet * n_quadrature_per_tri_facet:]
            tri_j = tri_j.reshape(-1, n_tri_facet, n_quadrature_per_tri_facet, cls.dim, cls.dim)
            quad_j= quad_j.reshape(-1, n_quad_facet, n_quadrature_per_quad_facet, cls.dim, cls.dim)
            return tri_j, quad_j
        else:
            w, q = cls.get_facet_quadrature(quadrature_order, dtype=dtype, device=device) 
            n_facet, n_quadrature_per_facet, _ = q.shape
            q    = q.reshape(-1, cls.dim)
            j = cls.eval_cell_jacobian(q, element_coords, basis_order) # [n_element, n_facet*n_quadrature_per_facet, dim, dim]
            return j.reshape(-1, n_facet, n_quadrature_per_facet, cls.dim, cls.dim)

    @classmethod 
    def eval_facet_jacobian(cls,
                            element_coords:torch.Tensor, 
                            basis_order:int=1,
                            quadrature_order:int=1,
                            )->Tensorx1|Tensorx2:
        """
        Parameters
        ----------
            element_coords: torch.Tensor [n_element, n_basis, dim] 
                boundary element coordinates
            basis_order: int 
                the order of the basis
            quadrature_order : int 
           
        Returns
        -------
            facet_jacobian : torch.Tensor 
                5D tensor of shape[n_element, n_facet, n_quadrature_per_face, dim-1(gradient), dim(global_coords)]
            or 
            tri_facet_jacobian, quad_facet_jacobian, tri_mask : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                5D tensor of shape [n_element, n_tri_facet, n_quadrature_per_tri_face, dim-1(gradient), dim(global_coords)]
                5D tensor of shape [n_element, n_quad_facet, n_quadrature_per_quad_face, dim-1(gradient), dim(global_coords)]
        """

        dtype = element_coords.dtype
        device= element_coords.device
  
        facet_quadrature_orig     = cls.get_facet_quadrature(order=quadrature_order, 
                                                             transform=False, 
                                                             dtype=dtype, 
                                                             device=device) 

        if cls.is_mix_facet:
            assert len(facet_quadrature_orig) == 4

            tri_j, quad_j = cls.eval_facet_cell_jacobian(element_coords, basis_order, quadrature_order)

            facet_mapping_grad_fns = cls.get_local_facet_mapping_grad_fns(dtype, device) # Polynomials [dim-1(gradient), n_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim

            tri_mask  = cls.get_tri_mask() # [n_facet]

            tri_facet_mapping_grad_fns = facet_mapping_grad_fns[:, tri_mask]   # Polynomials [dim-1(gradient), n_tri_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim
            quad_facet_mapping_grad_fns= facet_mapping_grad_fns[:, ~tri_mask]  # Polynomials [dim-1(gradient), n_quad_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim

            n_tri_facet = tri_facet_mapping_grad_fns.shape[1]
            n_quad_facet= quad_facet_mapping_grad_fns.shape[1]

            assert isinstance(tri_facet_mapping_grad_fns, Polynomials) and tri_facet_mapping_grad_fns.shape == (cls.dim-1, n_tri_facet, cls.dim) 
            assert isinstance(quad_facet_mapping_grad_fns, Polynomials) and quad_facet_mapping_grad_fns.shape == (cls.dim-1, n_quad_facet, cls.dim)

            tri_w, tri_q,quad_w, quad_q = facet_quadrature_orig 
            # triw, triq [n_quadrature_per_tri_facet] [n_quadrature_per_tri_facet, dim-1]

            tri_facet_mapping_grad = tri_facet_mapping_grad_fns.map(tri_q)  # [n_quadrature_per_tri_face,  dim-1(gradient), n_tri_facet, dim(global_coords)]
            quad_facet_mapping_grad= quad_facet_mapping_grad_fns.map(quad_q)# [n_quadrature_per_quad_face, dim-1(gradient), n_quad_facet, dim(global_coords)]

            tri_facet_mapping_grad = torch.einsum("efqij, qgfi-> efqgj",tri_j, tri_facet_mapping_grad)   #[n_element, n_tri_facet, n_quadrature_per_tri_face, dim-1(gradient), dim(global_coords)]
            quad_facet_mapping_grad= torch.einsum("efqij, qgfi-> efqgj",quad_j, quad_facet_mapping_grad) #[n_element, n_quad_facet, n_quadrature_per_quad_face, dim-1(gradient), dim(global_coords)]

            return tri_facet_mapping_grad, quad_facet_mapping_grad

        else:
            assert len(facet_quadrature_orig) == 2
            w, q                   = facet_quadrature_orig
    
            j                      = cls.eval_facet_cell_jacobian(element_coords, basis_order, quadrature_order)

            facet_mapping_grad_fns = cls.get_local_facet_mapping_grad_fns(dtype, device)        # PolynomialTensor [dim-1(gradient), n_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim
            facet_mapping_grad     = facet_mapping_grad_fns.map(q) # [n_quadrature_per_face, dim-1(gradient), n_facet, dim(global_coords)]
            facet_mapping_grad     = torch.einsum("efqij, qkfi-> efqkj",j, facet_mapping_grad) #[n_element, n_facet, n_quadrature_per_face, dim-1(gradient), dim(global_coords)]

            return facet_mapping_grad

    @classmethod
    def element_to_facet(cls, 
                         elements:torch.Tensor,
                         order:int,
                         )->Tensorx1|Tensorx2:
        """
        Parameters
        ----------
            elements: torch.Tensor 
                    2D tensor of shape [n_element, n_basis]
                    or 
                    1D tensor of shape [n_basis]
            order: int 
                    the order of basis
        Returns
        -------
            facet: torch.Tensor
                3D tensor of shape [n_element, n_facet, n_basis_per_facet]
                or 
                2D tensor of shape [n_facet, n_basis_per_facet]
            or 
            tri_facet, quad_facet, tri_mask: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                3D tensor of shape [n_element, n_tri_facet, n_basis_per_tri_facet]
                3D tensor of shape [n_element, n_quad_facet, n_basis_per_quad_facet]
                or 
                2D tensor of shape [n_tri_facet, n_basis_per_tri_facet]
                2D tensor of shape [n_quad_facet, n_basis_per_quad_facet]
        """
        assert elements.dim() in (1, 2)
        assert elements.shape[-1] == cls.get_n_basis(order)
        

        if cls.is_mix_facet:
            tri_facet, quad_facet = cls.get_facet(order)
            if elements.dim() == 1:
                tri_facet = elements[tri_facet] # [n_tri_facet, n_basis_per_tri_facet]
                quad_facet= elements[quad_facet]# [n_quad_facet, n_basis_per_quad_facet]
            else:
                tri_facet = elements[:, tri_facet] # [n_element, n_tri_facet, n_basis_per_tri_facet]
                quad_facet= elements[:, quad_facet]# [n_element, n_quad_facet, n_basis_per_quad_facet]
            return tri_facet, quad_facet
        
        else: # facet of the same shape 
            facet = cls.get_facet(order) # [n_facet, n_basis_per_facet]
            if elements.dim() == 1:
                facet = elements[facet] # [n_facet, n_basis_per_facet]
            elif elements.dim() == 2:
                facet = elements[:, facet] # [n_elements, n_facet, n_basis_per_facet]
            else:
                raise Exception(f"elemets dim should be 1 or 2, but got {elements.dim()}")
            return facet

    @classmethod 
    @lru_cache(1)
    def get_tri_mask(cls)->torch.Tensor:
        """
        Returns
        -------
            tri_mask: torch.Tensor 
                1D boolean Tensor of shape [n_facet]
        """
        assert cls.is_mix_facet 
        return torch.tensor([len(face) == 3 for face in cls.face])

    @classmethod
    @lru_cache(1)
    def get_quad_mask(cls)->torch.Tensor:
        """
        Returns
        -------
            quad_mask: torch.Tensor 
                1D boolean Tensor of shape [n_facet]
        """
        assert cls.is_mix_facet 
        return torch.tensor([len(face) == 4 for face in cls.face])

    @classmethod 
    @lru_cache(1)
    def get_contour(cls, order:int)->torch.Tensor:
        assert order >= 1
        assert cls.dim == 2, f"Only 2d elements has contour, but got {cls.dim}d element"
        raise NotImplementedError()
        

class Line(Element):
    points = torch.tensor([[0.0],[1.0]]) # 2x1
    vertex = torch.tensor([[0], [1]]) # 2x1
    edge   = torch.tensor([[0, 1]]) # 1x2
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = 0
    n_cell   = 0

    @classmethod
    @lru_cache(1)
    def get_basis(cls, order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                    )->Tensorx1:
        return lin_basis(
                cls.points.type(dtype).to(device), 
                cls.vertex, cls.edge, order)

    @classmethod 
    @lru_cache(1)
    def get_polynomial(cls, 
                       order:int=1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Polynomial:
        return Polynomial.tens_exp(1, order, dtype, device)

    @classmethod 
    @lru_cache(1)
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        return lin_quadrature(order, dtype, device)

class Triangle(Element):
    points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0]]) # 3x2
    vertex = torch.tensor([[0], [1], [2]]) # 3x1
    edge   = torch.tensor([[1, 2], [0, 2], [0, 1]]) # 3x2
    face   = ((0, 1, 2),) # 1x3
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = 0

    @classmethod
    @lru_cache(1)
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return tri_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         torch.tensor(cls.face), 
                         order
                         )

    @classmethod
    @lru_cache(1)
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.poly_exp(2, order, dtype, device)
    
    @classmethod
    @lru_cache(1)
    def get_facet(cls, order:int=1)->Tensorx1:
        facet = facet_basis_index_2d(cls.vertex, cls.edge, order)
        return facet
    
    @classmethod
    @lru_cache(1) 
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return tri_quadrature(order, dtype, device)

    @classmethod
    @lru_cache(1)
    def get_facet_quadrature(cls, 
                             order:int=1, 
                             transform=True, 
                             dtype:torch.dtype=torch.float32,
                             device:torch.device=torch.device('cpu')
                             )->Tensorx2:
        return facet_quadrature_2d(
                    cls.get_local_facet_mapping_fns(dtype, device), 
                    order, transform)
        
    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Line
    
    @classmethod 
    @lru_cache(1)
    def get_contour(cls, order:int)->torch.Tensor:
        if order == 1:
            return torch.tensor([0, 1, 2])
        elif order > 1:
            facet = cls.get_facet(order) # [n_facet, n_basis_per_facet]
            facet = facet[:, 2:] # facet without boundaries
            return torch.cat([
                torch.tensor([0]),
                facet[2],
                torch.tensor([1]),
                facet[0],
                torch.tensor([2]),
                facet[1].flip([0])
            ])
        else:
            raise Exception(f"Invalid order {order}")

class Quadrilateral(Element):
    points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]]) # 4x2
    vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
    edge   = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # 4x2
    face   = ((0, 1, 2, 3),) # 1x4
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = 0

   
    @classmethod
    @lru_cache(1)
    def get_basis(cls, 
                  order:int = 1, 
                  dtype:torch.dtype = torch.float32,
                  device:torch.device = torch.device('cpu')
                  )->Tensorx1:
        return quad_basis(
                    cls.points.type(dtype).to(device), 
                    cls.vertex, 
                    cls.edge, 
                    torch.tensor(cls.face), 
                    order)
    
    @classmethod
    @lru_cache(1)
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Polynomial:
        return Polynomial.tens_exp(2, order, dtype, device)

    @classmethod
    @lru_cache(1)
    def get_facet(cls, order:int=1):
        facet = facet_basis_index_2d(cls.vertex, cls.edge, order)
        return facet

    @classmethod 
    @lru_cache(1)
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype=torch.float32, 
                       device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return quad_quadrature(order, dtype, device)
       
    @classmethod
    @lru_cache(1)
    def get_facet_quadrature(cls,  
                             order:int = 1, 
                             transform:bool = True, 
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx2:
        return facet_quadrature_2d(
                cls.get_local_facet_mapping_fns(dtype, device), 
                order, transform)

    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Line
    
    @classmethod
    @lru_cache(1)
    def get_coutour(cls, order:int)->torch.Tensor:
        if order == 1:
            return torch.tensor([0, 1, 3, 2])
        elif order > 1:
            facet = cls.get_facet(order) # [n_facet, n_basis_per_facet]
            facet = facet[:, 2:] # facet without boundaries
            return torch.cat([
                torch.tensor([0]),
                facet[0],
                torch.tensor([1]),
                facet[2],
                torch.tensor([3]),
                facet[3].flip([0]),
                torch.tensor([2]),
                facet[1].flip([0])
            ])
        else:
            raise Exception(f"Invalid order {order}")

class Tetrahedron(Element):
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 4x3
    vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
    edge   = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]) # 6x2
    face   = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)) # 4x3
    cell   = torch.tensor([[0, 1, 2, 3]]) # 1x4
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    @classmethod
    @lru_cache(1)
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return tet_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         torch.tensor(cls.face), 
                         cls.cell,  # type: ignore
                         order)

    @classmethod
    @lru_cache(1) 
    def get_polynomial(cls, 
                        order:int = 1, 
                        dtype:torch.dtype = torch.float32,
                        device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.poly_exp(3, order, dtype, device)
    
    @classmethod
    @lru_cache(1) 
    def get_quadrature(cls, 
                        order:int = 1, 
                        dtype:torch.dtype=torch.float32,
                        device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return tet_quadrature(order, dtype, device)
       
    @classmethod 
    @lru_cache(1)
    def get_facet(cls, order: int = 1) -> Tensorx1:
        index = tet_facet_basis_index(
            cls.vertex, 
            cls.edge, 
            torch.tensor(cls.face), 
            order)
        return index
    
    @classmethod
    @lru_cache(1)
    def get_facet_quadrature(cls, 
                                order:int = 1, 
                                transform:bool = True, 
                                dtype:torch.dtype = torch.float32,  
                                device:torch.device = torch.device('cpu')
                             )->Tensorx2:
        return tet_facet_quadrature(
                    cls.get_local_facet_mapping_fns(dtype, device), 
                    order, 
                    transform)
    
    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Triangle

class Hexahedron(Element):
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0]]) # 8x3
    vertex = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]]) # 8x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]) # 12x2
    face   = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), (1, 3, 5, 7), (2, 3, 6, 7), (4, 5, 6, 7)) # 6x4
    cell   = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]) # 1x8
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    @classmethod
    @lru_cache(1)
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return hex_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge,
                         torch.tensor(cls.face), 
                         cls.cell,  # type: ignore
                         order)
    
    @classmethod
    @lru_cache(1) 
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.tens_exp(3, order, dtype, device)
        
    @classmethod 
    @lru_cache(1)
    def get_quadrature(cls, 
                        order:int = 1, 
                        dtype:torch.dtype = torch.float32,
                        device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        return hex_quadrature(order, dtype, device) 
       
    @classmethod 
    @lru_cache(1)
    def get_facet(cls, order:int=1)->Tensorx1:
        index = hex_facet_basis_index(
                    cls.vertex, 
                    cls.edge, 
                    torch.tensor(cls.face), 
                    order)
        return index

    @classmethod
    @lru_cache(1)
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx2:
        return hex_facet_quadrature(
            cls.get_local_facet_mapping_fns(dtype, device), 
            order, transform)
        
    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Quadrilateral


class Pyramid(Element):
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 5x3
    vertex = torch.tensor([[0], [1], [2], [3], [4]]) # 5x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 8x2
    face   = ((0, 1, 2, 3), (0, 1, 4), (0, 2, 4), (1, 3, 4), (2, 3, 4)) # 5x4
    cell   = torch.tensor([[0, 1, 2, 3, 4]]) # 1x5
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    is_mix_facet = True
   
    @classmethod
    @lru_cache(1)
    def get_basis(cls, 
                  order:int = 1, 
                  dtype:torch.dtype = torch.float32,
                  device:torch.device = torch.device('cpu')
                  )->Tensorx1:
        return pyr_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         cls.face, # type: ignore
                         cls.cell, # type: ignore
                         order) 

    @classmethod 
    @lru_cache(1)
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,  
                       device:torch.device= torch.device('cpu')
                       )->Polynomial:
        return Polynomial.pyr_exp(order, dtype, device)

    @classmethod
    @lru_cache(1)
    def get_quadrature(cls, 
                        order:int=1, 
                        dtype:torch.dtype=torch.float32, 
                        device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return pyr_quadrature(order, dtype, device)
       
    @classmethod
    @lru_cache(1) 
    def get_facet(cls, order:int=1)->Tensorx2:
        assert isinstance(cls.face, tuple)
        tri_facet, quad_facet = mix_facet_basis_index(
                    cls.vertex, 
                    cls.edge, 
                    cls.face, 
                    order) # type: ignore
        return tri_facet, quad_facet

    @classmethod
    @lru_cache(1)
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx4:
        tri_w, tri_q,\
        quad_w,quad_q = mix_facet_quadrature_3d(
                    cls.get_local_facet_mapping_fns(dtype, device), 
                    cls.face, order, transform)
        return (tri_w.type(dtype), 
                tri_q.type(dtype), 
                quad_w.type(dtype), 
                quad_q.type(dtype))

    @classmethod 
    def get_facet_type(cls)->Tuple[Type['Element'],Type['Element']]:
        return Triangle, Quadrilateral

class Prism(Element):
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]]) # 5x3
    vertex = torch.tensor([[0], [1], [2], [3], [4], [5]]) # 6x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 9x2
    face   = ((0, 1, 2), (0, 1, 3, 4), (0, 2, 3, 5), (1, 2, 4, 5),(3,4,5)) # 4x4
    cell   = torch.tensor([[0, 1, 2, 3, 4, 5]]) # 1x5
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    is_mix_facet = True
 
    @classmethod
    @lru_cache(1)
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32, 
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return pri_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         cls.face,  # type: ignore
                         cls.cell,  # type: ignore
                         order)

    @classmethod
    @lru_cache(1) 
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,  
                       device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.pri_exp(order, dtype, device)
   
    @classmethod 
    @lru_cache(1)
    def get_quadrature(cls, 
                       order:int = 1,
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        return pri_quadrature(order, dtype, device)
       
    @classmethod
    @lru_cache(1)
    def get_facet(cls, order:int=1)->Tensorx2:
        assert isinstance(cls.face, tuple)
        tri_facet, quad_facet = mix_facet_basis_index(
                    cls.vertex, 
                    cls.edge, 
                    cls.face, 
                    order)
        return tri_facet, quad_facet

    @classmethod 
    @lru_cache(1)
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx4:
        tri_w, tri_q,\
        quad_w,quad_q = mix_facet_quadrature_3d(
                            cls.get_local_facet_mapping_fns(dtype, device), 
                            cls.face, order, transform)
        return (tri_w.type(dtype), 
                tri_q.type(dtype), 
                quad_w.type(dtype), 
                quad_q.type(dtype))
    
    @classmethod 
    def get_facet_type(cls)->Tuple[Type['Element'],Type['Element']]:
        return Triangle, Quadrilateral

if __name__ == '__main__':


    # basis_fns = Quadrilateral.get_basis_fns(2)

    # plot_2d(basis_fns)

    # basis_fns = Line.get_basis_fns(4)
    # plot_1d(basis_fns)
    # print(binomial_exponents(2))

    basis_fns = Pyramid.get_basis_fns(2)

    plot_3d(Pyramid, 
            Pyramid.get_basis(), 
            Pyramid.get_basis_fns(2))