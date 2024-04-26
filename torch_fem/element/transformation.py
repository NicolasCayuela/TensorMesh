import toml
import torch
import torch.nn as nn 
from typing import Optional, Tuple, Type
from .types import Tensorx1, \
                    Tensorx2, \
                    Tensorx3, \
                    Tensorx4
from .element import Element
from .element_type import element_type2element, element_type2order

class Transformation(nn.Module):

    elements:torch.Tensor # [n_element, n_basis]
    points:torch.Tensor   # [n_points, dim]
    element:Type[Element]
    basis_order:int
    quadrature_order:int


    def __init__(self, 
                 points:torch.Tensor,
                 elements:torch.Tensor,
                 element_type:str, 
                 quadrature_order:int=2):
        """
        Parameters
        ----------
        points: torch.Tensor
            2D tensor of shape [n_points, n_dim]
            the coordinates of the points
        elements: torch.Tensor 
            2D tensor of shape [n_element, n_basis]
        element_type : str
            the type of the element
        quadrature_order : int
            the order of the quadrature
            default :obj:`2`
        """
        super().__init__()
        self.register_buffer("elements", elements)
        self.element:Type[Element] = element_type2element(element_type)
        self.basis_order:int = element_type2order[element_type]
        self.quadrature_order= quadrature_order
        
        self.update_points(points)

    def update_points(self, points:torch.Tensor):
        """
        Parameters
        ----------
        points: torch.Tensor
            2D tensor of shape [n_points, n_dim]
            the coordinates of the points

        """
        dtype, device            = points.dtype, points.device
        self.register_buffer("points", points)
        self.n_points            = points.shape[0] 
        # self.element_coords      = self.points[self.elements]


    @property
    def dim(self)->int:
        return self.points.shape[-1] # type:ignore

    @property 
    def device(self)->torch.device:
        return self.points.device
    
    @property
    def dtype(self)->torch.dtype:
        return self.points.dtype
    
    @property
    def element_coords(self)->torch.Tensor:
        return self.points[self.elements] # type:ignore [n_element, n_basis, dim]
    
    @property
    def n_elements(self)->int:
        return self.elements.shape[0]

    @property
    def basis(self)->Tensorx1:
        return self.element.get_basis(self.basis_order).type(self.dtype).to(self.device) # [n_basis, dim]
    
    @property
    def n_basis(self)->int:
        return self.basis.shape[0]

    @property 
    def quadrature(self)->Tensorx2:
        if "_quadrature_w" not in self._buffers: # type: ignore
            w, q =  self.element.get_quadrature(self.quadrature_order, self.dtype, self.device) # [n_quadrature], [n_quadrature, dim]
            self.register_buffer("_quadrature_w", w)
            self.register_buffer("_quadrature_q", q)
        return self._quadrature_w, self._quadrature_q # type: ignore [n_quadrature], [n_quadrature, dim]
    
    @property
    def n_quadrature(self)->int:
        return self.quadrature[0].shape[0]

    @property 
    def shape_val(self)->Tensorx1:
        """
        Returns
        -------
        shape_val: torch.Tensor
            2D Tensor of shape [n_quadrature, n_basis]
        """
        if "_shape_val" not in self._buffers: # type: ignore
            _, p      = self.quadrature
            shape_val = self.element.eval_shape_val(p, self.basis_order, self.quadrature_order)
            self.register_buffer("_shape_val", shape_val)
        return self._shape_val  # type:ignore [n_quadrature, n_basis]

    @property 
    def shape_grad(self)->Tensorx1:
        """
        Returns
        -------
        shape_grad: torch.Tensor
            4D Tensor of shape [n_element, n_quadrature, n_basis, dim]
        """
        if "_shape_grad" not in self._buffers: # type: ignore
            _, q = self.quadrature
            shape_grad, jacobian = self.element.eval_shape_grad(
                self.element_coords, q, self.basis_order, self.quadrature_order)
            self.register_buffer("_shape_grad", shape_grad)
            self.register_buffer("_jacobian", jacobian)
        return shape_grad 
    
    @property
    def jacobian(self)->Tensorx1:
        """
        Returns
        -------
        jacobian: torch.Tensor
            4D Tensor of shape [n_element, n_quadrature, dim, dim]
        """
        if "_jacobian" not in self._buffers: # type: ignore
            self.shape_grad
        return self._jacobian # type:ignore [n_element, n_quadrature, dim, dim]
         
    @property 
    def facets(self)->Tensorx1|Tensorx2:
        """
        Returns
        -------
        facets: torch.Tensor
            2D Tensor of shape [n_facet, n_dim]
        or 
        tri_facets: torch.Tensor
            2D Tensor of shape [n_tri_facet, n_dim]
        quad_facets: torch.Tensor
            2D Tensor of shape [n_quad_facet, n_dim]
        """
        if self.element.is_mix_facet:
            tri_facets, quad_facets = self.element.element_to_facet(self.elements, self.basis_order) # type: ignore
            return (tri_facets, quad_facets)
        else:
            facets = self.element.element_to_facet(self.elements, self.basis_order) # type: ignore
            assert isinstance(facets, torch.Tensor)
            return facets

    @property
    def facet_quadrature(self)->Tensorx2|Tensorx4:
        """
        Returns
        -------
        weights: torch.Tensor
            2D Tensor of shape [n_facet, n_quadrature_per_facet]
        points: torch.Tensor
            3D Tensor of shape [n_facet, n_quadrature_per_facet, dim]

        or

        tri_weights: torch.Tensor
            2D Tensor of shape [n_tri_facet, n_quadrature_per_tri_facet]
        tri_points: torch.Tensor
            3D Tensor of shape [n_tri_facet, n_quadrature_per_tri_facet, dim]
        quad_weights: torch.Tensor
            2D Tensor of shape [n_quad_facet, n_quadrature_per_quad_facet] 
        quad_points: torch.Tensor
            3D Tensor of shape [n_quad_facet, n_quadrature_per_quad_facet, dim]
        """
        if self.element.is_mix_facet:
            tri_m, tri_q, quad_m, quad_q, tri_mask = self.element.get_facet_quadrature(self.quadrature_order, transform=True) # type:ignore
            tri_m = tri_m.type(self.dtype).to(self.device)
            tri_q = tri_q.type(self.dtype).to(self.device)
            quad_m= quad_m.type(self.dtype).to(self.device)
            quad_q= quad_q.type(self.dtype).to(self.device)
            return (tri_m, tri_q, quad_m, quad_q)
        else:
            m, q = self.element.get_facet_quadrature(self.quadrature_order, transform=True) # type:ignore [n_facet, n_quadrature_per_facet], [n_facet, n_quadrature_per_facet, dim]
            m = m.type(self.dtype).to(self.device)
            q = q.type(self.dtype).to(self.device)
            return m, q
        
    @property 
    def facet_shape_val(self)->Tensorx1|Tensorx2:
        """
        Returns
        -------
        shape_val: torch.Tensor
            3D Tensor of shape [n_facet, n_quadrature_per_facet, n_basis]
        or
        tri_shape_val: torch.Tensor
            3D Tensor of shape [n_tri_facet, n_quadrature_per_tri_facet, n_basis]
        quad_shape_val: torch.Tensor
            3D Tensor of shape [n_quad_facet, n_quadrature_per_quad_facet, n_basis]
        """
        

        if self.element.is_mix_facet:
            if "_facet_shape_val_tri" not in self._buffers: # type: ignore
                tri_m, tri_p, quad_m, quad_p, tri_mask = self.facet_quadrature # type:ignore
                n_tri_facet, n_quadrature_per_tri_facet, _   = tri_p.shape
                n_quad_facet, n_quadrature_per_quad_facet, _ = quad_p.shape
                tri_shape_val = self.element.eval_shape_val(tri_p, self.basis_order, self.quadrature_order)
                quad_shape_val= self.element.eval_shape_val(quad_p, self.basis_order, self.quadrature_order)
                tri_shape_val = tri_shape_val.reshape(n_tri_facet, n_quadrature_per_tri_facet, self.n_basis)
                quad_shape_val= quad_shape_val.reshape(n_quad_facet, n_quadrature_per_quad_facet, self.n_basis)

                self.register_buffer("_facet_shape_val_tri", tri_shape_val)
                self.register_buffer("_facet_shape_val_quad", quad_shape_val)
            
            return self._facet_shape_val_tri, self._facet_shape_val_quad # type:ignore [n_tri_facet, n_quadrature_per_tri_facet, n_basis], [n_quad_facet, n_quadrature_per_quad_facet, n_basis]

        else:
            if "_facet_shape_val" not in self._buffers: # type: ignore
                _, p = self.facet_quadrature # type:ignore [n_facet, n_quadrature_per_facet], [n_facet, n_quadrature_per_facet, dim]
                n_facet, n_quadrature_per_facet, dim = p.shape 
                p = p.reshape(-1, dim)
                shape_val = self.element.eval_shape_val(p, self.basis_order, self.quadrature_order)
                shape_val = shape_val.reshape(n_facet, n_quadrature_per_facet, self.n_basis)
                self.register_buffer("_facet_shape_val", shape_val)

            return self._facet_shape_val # type:ignore [n_facet, n_quadrature_per_facet, n_basis]
              
    @property 
    def facet_shape_grad(self)->Tensorx1|Tensorx2:
        """
        Returns
        -------
        shape_grad: torch.Tensor    
            4D Tensor of shape [n_facet, n_quadrature_per_facet, n_basis, dim]
        or
        tri_shape_grad: torch.Tensor
            4D Tensor of shape [n_tri_facet, n_quadrature_per_tri_facet, n_basis, dim]
        quad_shape_grad: torch.Tensor
            4D Tensor of shape [n_quad_facet, n_quadrature_per_quad_facet, n_basis, dim]
        """
        if self.element.is_mix_facet:

            if "_facet_shape_grad_tri" not in self._buffers: # type: ignore

                tri_m, tri_q, quad_m, quad_q, tri_mask = self.facet_quadrature # type:ignore
                n_tri_facet, n_quadrature_per_tri_facet, _   = tri_q.shape
                n_quad_facet, n_quadrature_per_quad_facet, _ = quad_q.shape

                tri_q = tri_q.reshape(-1, self.element.dim)
                quad_q= quad_q.reshape(-1, self.element.dim)
                tri_shape_grad, tri_cell_jacobian = self.element.eval_shape_grad(
                    self.element_coords, tri_q, self.basis_order
                ) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature, dim, dim]
                quad_shape_grad, quad_cell_jacobian = self.element.eval_shape_grad(
                    self.element_coords, quad_q, self.basis_order
                )
                tri_shape_grad    = tri_shape_grad.reshape(
                    self.n_elements, n_tri_facet, n_quadrature_per_tri_facet, self.n_basis, self.element.dim)
                quad_shape_grad   = quad_shape_grad.reshape(
                    self.n_elements, n_quad_facet, n_quadrature_per_quad_facet, self.n_basis, self.element.dim)


                self.register_buffer("_facet_shape_grad_tri", tri_shape_grad)
                self.register_buffer("_facet_shape_grad_quad", quad_shape_grad)
                self.register_buffer("_facet_cell_jacobian_tri", tri_cell_jacobian)
                self.register_buffer("_facet_cell_jacobian_quad", quad_cell_jacobian)

            return self._facet_shape_grad_tri, self._facet_shape_grad_quad # type:ignore [n_tri_facet, n_quadrature_per_tri_facet, n_basis, dim], [n_quad_facet, n_quadrature_per_quad_facet, n_basis, dim]
        
        else:

            if "_facet_shape_grad" not in self._buffers: # type:ignore
                w, q = self.facet_quadrature # type:ignore [n_facet, n_quadrature_per_facet], [n_facet, n_quadrature_per_facet, dim]
                n_facet, n_quadrature_per_facet, _ = q.shape
                
                q    = q.reshape(-1, self.element.dim)
                shape_grad, cell_jacobian = self.element.eval_shape_grad(
                    self.element_coords, q, self.basis_order
                ) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature, dim, dim]
                
                shape_grad    = shape_grad.reshape(
                    self.n_elements, n_facet, n_quadrature_per_facet, self.n_basis, self.element.dim)
                
                self.register_buffer("_facet_shape_grad", shape_grad)
                self.register_buffer("_facet_cell_jacobian", cell_jacobian)

            return self._facet_shape_grad # type:ignore

    @property
    def facet_jacobian(self)->Tensorx1|Tensorx2:
        """
        Returns
        -------
        facet_jacobian: torch.Tensor
            5D Tensor of shape [n_element, n_facet, n_quadrature_per_facet, dim-1, dim]
        or
        tri_facet_jacobian: torch.Tensor
            5D Tensor of shape [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim-1, dim]
        quad_facet_jacobian: torch.Tensor
            5D Tensor of shape [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim-1, dim]
        """

        if self.element.is_mix_facet:

            if "_facet_jacobian_tri" not in self._buffers: # type: ignore

                tri_facet_jacobian, quad_facet_jacobian = self.element.eval_facet_jacobian(
                    self.element_coords, self.basis_order, self.quadrature_order)
                
                self.register_buffer("_facet_jacobian_tri", tri_facet_jacobian)
                self.register_buffer("_facet_jacobian_quad", quad_facet_jacobian)

            return self._facet_jacobian_tri, self._facet_jacobian_quad # type:ignore [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim-1, dim], [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim-1, dim]


        else:

            if "_facet_jacobian" not in self._buffers: # type:ignore
                facet_jacobian = self.element.eval_facet_jacobian(
                    self.element_coords, self.basis_order, self.quadrature_order)
                
                assert isinstance(facet_jacobian, torch.Tensor)
                self.register_buffer("_facet_jacobian", facet_jacobian) 
     
            return self._facet_jacobian # type:ignore [n_element, n_facet, n_quadrature_per_facet, dim-1, dim]
                
    @property 
    def nanson_scale(self)->Tensorx1|Tensorx2:
        """
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Returns
        -------
        nanson_scale: torch.Tensor
            3D Tensor of shape [n_element, n_facet, n_quadrature_per_facet]
        or 
        tri_nanson_scale: torch.Tensor
            3D Tensor of shape [n_element, n_tri_facet, n_quadrature_per_tri_facet]
        quad_nanson_scale: torch.Tensor
            3D Tensor of shape [n_element, n_quad_facet, n_quadrature_per_quad_facet]
        """
      
       
        if self.element.is_mix_facet:
            if "_nanson_scale_tri" not in self._buffers:
                tri_w, tri_q, quad_w, quad_q = self.facet_quadrature # type:ignore

                # compute the facet cell jacobian
                tri_j, quad_j = self.element.eval_facet_cell_jacobian(
                    self.element_coords, 
                    self.basis_order, 
                    self.quadrature_order)
            
                tri_inv_j = torch.inverse(tri_j) # [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim, dim]
                quad_inv_j= torch.inverse(quad_j)# [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim, dim]
                # compute the nanson scale
                normals = self.element.get_outwards_facet_normal() # [n_facet_per_element, dim]
                tri_normals = normals[self.element.get_tri_mask()] # [n_tri_facet, dim]
                quad_normals= normals[self.element.get_quad_mask()]# [n_quad_facet, dim]

                det_tri_j = torch.linalg.det(tri_j) # [n_element, n_tri_facet, n_quadrature_per_tri_facet]
                det_quad_j= torch.linalg.det(quad_j)# [n_element, n_quad_facet, n_quadrature_per_quad_facet]

                tri_nanson  = torch.linalg.norm(
                    torch.einsum('fi,efqji->efqj', tri_normals, tri_inv_j)  # [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim]
                ) # [n_element, n_tri_facet, n_quadrature_per_tri_facet]
                quad_nanson = torch.linalg.norm(
                    torch.einsum('fi,efqji->efqj', quad_normals, quad_inv_j) # [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim]
                ) # [n_element, n_quad_facet, n_quadrature_per_quad_facet]

                # TODO: check if the absoulte value should be added to the determinant
                tri_nanson = torch.einsum("efq,efq,fq->efq",tri_nanson , det_tri_j , tri_w)
                quad_nanson= torch.einsum("efq,efq,fq->efq",quad_nanson, det_quad_j, quad_w)

                self.register_buffer("_nanson_scale_tri", tri_nanson)
                self.register_buffer("_nanson_scale_quad", quad_nanson)

            return self._nanson_scale_tri, self._nanson_scale_quad # type:ignore [n_element, n_tri_facet, n_quadrature_per_tri_facet], [n_element, n_quad_facet, n_quadrature_per_quad_facet]
        
        else: # single facet shape 

            if "_nanson_scale" not in self._buffers:

                j = self.element.eval_facet_cell_jacobian(
                        self.element_coords, 
                        self.basis_order, 
                        self.quadrature_order)
                
                inv_j = torch.inverse(j) # [n_element, n_facet, n_quadrature_per_facet, dim, dim]
                det_j = torch.linalg.det(j) # [n_element, n_facet, n_quadrature_per_facet]

                normals = self.element.get_outwards_facet_normal() # [n_facet, dim]

                nanson_scale = torch.linalg.norm(
                    torch.einsum('fi,efqji->efqj', normals, inv_j) # [n_element, n_facet, n_quadrature_per_facet, dim]
                ) # [n_element, n_facet, n_quadrature_per_facet]

                nanson_scale = torch.einsum("efq,efq,fq->efq", nanson_scale, det_j, self.facet_quadrature[0]) # [n_element, n_facet, n_quadrature_per_facet]

                self.register_buffer("_nanson_scale", nanson_scale)

            return self._nanson_scale # type:ignore [n_element, n_facet, n_quadrature_per_facet]


        
      



    ###############
    # Abbreviation
    ############### 


    # basis 
    @property
    def phi(self)->torch.Tensor:
        return self.shape_val # [n_quadrature, n_basis]
    
    @property
    def gradphi(self)->torch.Tensor:
        return self.shape_grad # [n_element, n_quadrature, n_basis, dim]

    # cell jacobian
    @property 
    def J(self)->Tensorx1:
        return self.jacobian 
    
    @property 
    def detJ(self)->Tensorx1:
        return torch.linalg.det(self.jacobian)
    
    @property 
    def JxW(self)->Tensorx1:
        """
        torch.Tensor: [n_element, n_quadrature]
        """
        if "_jxw" not in self._buffers: # type: ignore
            w, _       = self.quadrature
            jxw        = torch.einsum('q,eq->eq', w, self.detJ.abs())
            self.register_buffer("_jxw", jxw)
        return self._jxw # type:ignore [n_element, n_quadrature]

    @property
    def G(self)->Tensorx1:
        return self.J

    @property 
    def detG(self)->Tensorx1:
        return self.detJ

    @property
    def GxW(self)->Tensorx1:
        return self.JxW


    # facet jacobian
    @property 
    def F(self)->Tensorx1|Tensorx2: 
        return self.facet_jacobian # type:ignore
    
    @property
    def detF(self)->Tensorx1|Tensorx2:
        """
        Returns
        -------
        tri_jac_det: torch.Tensor
            3D Tensor of shape [n_element, n_tri_facet, n_quadrature_per_tri_facet]
        quad_jac_det: torch.Tensor
            3D Tensor of shape [n_element, n_quad_facet, n_quadrature_per_quad_facet]

        or 
        jac_det: torch.Tensor
            3D Tensor of shape [n_element, n_facet, n_quadrature_per_facet]
        """
        if self.element.is_mix_facet:
            tri_fj, quad_fj = self.facet_jacobian # type:ignore
            tri_jac_det = torch.sqrt(torch.linalg.det(
                torch.einsum('efqij,efqjk->efqik', tri_fj, tri_fj)))
            quad_jac_det= torch.sqrt(torch.linalg.det(
                torch.einsum('efqij,efqjk->efqik', quad_fj, quad_fj)))
            return tri_jac_det, quad_jac_det
        else:
            fj = self.facet_jacobian # type:ignore
           
            return torch.sqrt(torch.linalg.det(fj @ fj.mT))
   
    @property 
    def FxW(self)->Tensorx1|Tensorx2:
        if self.element.is_mix_facet:
            if "_jxf_tri" not in self._buffers: # type: ignore
                tri_w, tri_q, quad_w, quad_q = self.facet_quadrature # type:ignore
                tri_detF, quad_detF = self.detF
                tri_jxf = torch.einsum('fq,efq->efq', tri_w, tri_detF.abs())
                quad_jxf= torch.einsum('fq,efq->efq', quad_w, quad_detF.abs())
                self.register_buffer("_jxf_tri", tri_jxf)
                self.register_buffer("_jxf_quad", quad_jxf)
            return self._jxf_tri, self._jxf_quad # type:ignore [n_element, n_facet, n_quadrature_per_facet]
        else:
            if "_jxf" not in self._buffers: # type: ignore
                w, q = self.facet_quadrature # type:ignore
                detF = self.detF
                jxf  = torch.einsum('fq,efq->efq', w, detF.abs()) 
                self.register_buffer("_jxf", jxf)
            return self._jxf # type:ignore [n_element, n_facet, n_quadrature_per_facet]
         
    # facet normal
    @property
    def n(self)->Tensorx1|Tensorx2:
        return self.nanson_scale
    

    ######################
    # Efficient Methods
    ######################
    def batch_quadrature(self, start:int, batch:int)->Tensorx2:
        """a batch of quadrature points
        no efficiency improvement for this function, since it has nothing to do with elements
        Parameters
        ----------
        start : int
            the starting index of the batch
        batch : int
            the number of quadrature in the batch
            if -1 will return all the quadrature points starting from `start`
        
        Returns
        -------
        w : torch.Tensor
            1D Tensor of shape [batch]
        q : torch.Tensor
            1D Tensor of shape [batch, dim]
        """
        if start < 0:
            start += self.n_quadrature  
        assert start >= 0 and start < self.n_quadrature
        if start == 0 and batch == -1:
            return self.quadrature
        w, q = self.quadrature # [n_quadrature], [n_quadrature, dim]
        return w[start:start+batch], q[start:start+batch] # [batch], [batch, dim]
    
    def batch_elements_coords(self, start:int, batch:int)->Tensorx1:
        """a batch of element coordinates
        lazy load
        Parameters
        ----------
        start : int
            the starting index of the batch
        batch : int
            the number of elements in the batch
            if -1 will return all the elements starting from `start`
        
        Returns
        -------
        elements_coords : torch.Tensor
            3D Tensor of shape [batch, n_basis, dim]
        """
        if start < 0:
            start += self.n_elements
        if start == 0 and batch == -1:
            return self.element_coords
        else:
            elements:torch.Tensor        = self.elements[start:start+batch] # type:ignore [batch, n_basis]
            elements_coords:torch.Tensor = self.points[elements] # type:ignore [batch, n_basis, dim]
            return elements_coords

    def batch_shape_val(self, start:int, batch:int)->Tensorx1:
        """a batch of shape values
        no efficiency improvement for this function, since it has nothing to do with elements
        Parameters
        ----------
            start:int
                the starting quadrature index of the batch  
            batch:int
                the number of quadrature points in the batch
        Returns
        -------
            shape_val: torch.Tensor
                2D Tensor of shape [batch, n_basis]
        """
        return self.shape_val[start:start+batch] #  [batch, n_basis]

    def batch_shape_grad_jxw(self, 
                            element_start:int = 0,
                            element_batch:int = -1,
                            quadrature_start:int = 0, 
                            quadrature_batch:int = -1)->Tensorx2:
        """a batch of shape gradients and jacobian
        lazy load

        Parameters
        ----------
        element_start : int
            the starting index of the element batch
            default `0`
        element_batch : int
            the number of elements in the batch
            if -1 will return all the elements starting from `element_start`
            default `-1`
        quadrature_start : int
            the starting index of the quadrature batch
            default `0`
        quadrature_batch : int
            the number of quadrature points in the batch
            if -1 will return all the quadrature points starting from `quadrature_start`
            default `-1`

        Returns
        -------
        shape_grad: torch.Tensor
            4D Tensor of shape [element_batch, quadrature_batch, n_basis, dim]
        jxw: torch.Tensor
            2D Tensor of shape [element_batch, quadrature_batch]
        """
        if "_shape_grad" not in self._buffers or "_jxw" not in self._buffers: # type: ignore
            w, q = self.batch_quadrature(quadrature_start, quadrature_batch)
            e    = self.batch_elements_coords(element_start, element_batch)
            shape_grad, jacobian = self.element.eval_shape_grad(
                e, q, self.basis_order, self.quadrature_order)
            jxw  = torch.einsum('q,eq->eq', w, torch.linalg.det(jacobian))
            return shape_grad, jxw # [n_element, n_batch, n_basis, dim], [n_element, n_batch, dim, dim]
        else:
            return (self.shape_grad[:, quadrature_start:quadrature_start+quadrature_batch], 
                    self.JxW[:, quadrature_start:quadrature_start+quadrature_batch]) #type:ignore [n_element, n_batch, n_basis, dim], [n_element, n_batch]
        
    def mask_facet_jacobian(self, 
                            mask:Optional[torch.Tensor|Tuple[torch.Tensor,torch.Tensor]]=None):
        """
        Parameters
        ----------
        elements:torch.Tensor,
            2D tensor of shape [n_element, n_basis]
        mask: torch.Tensor or Tuple[torch.Tensor, torch.Tensor] or None
            2D tensor of shape [n_element, n_facet]
        
        Returns
        -------
        """
        if self.element.is_mix_facet:
            assert isinstance(mask, (tuple,list)) and len(mask) == 2, f"{self.element} element facet mask should be a tuple of two tensors"
            tri_mask, quad_mask = mask 
            self.facet
            tri_facet = self.element.element_to_facet(elements, self.basis_order)[0]
        else:
            assert isinstance(mask, torch.Tensor), f"{self.element} element facet mask should be a tensor"
            
