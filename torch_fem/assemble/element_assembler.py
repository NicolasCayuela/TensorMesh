
from abc import abstractmethod
from matplotlib.ticker import scale_range
import torch 
import torch.nn as nn
import numpy as np
import scipy.sparse
from functools import reduce, partial
import inspect
import operator
from typing import Tuple, Optional, Dict, Callable, List

from tests import quadrature
from .projector import Projector
from ..nn import BufferDict
from ..element import element_type2order, element_type2dimension, Transformation
from ..sparse import SparseMatrix
from ..mesh import Mesh


class ElementAssembler(nn.Module):
    r"""
    The :obj:`ElementAssembler` is inheritated from :class:`torch:torch.nn.Module`. Therefore, all the operation from :class:`torch:torch.nn.Module` is applicable to :obj:`ElementAssembler`

    You are not encouraged to build the ElementAssembler directly, instead, you should use :meth:`torch_fem.assemble.ElementAssembler.from_mesh` or :meth:`torch_fem.assemble.ElementAssembler.from_assembler` to build the ElementAssembler from a mesh

    The output when calling the ElementAssembler is a sparse matrix, which is the global galerkin matrix of shape :math:`\mathbb R_{\text{sparse}}^{|\mathcal V|, |\mathcal V|}` or :math:`\mathbb R_{\text{sparse}}^[|\mathcal V| \times  H, |\mathcal V| \times  H]`,
            where :math:`H` is the number of degree of freedom per point, :math:`|\mathcal V|` is the number of points.

    .. math::

        K \overset{\text{bsr matrix}}{\leftarrow}\hat K_\text{global}

        \hat K_{\text{global}}^{nkl} = \mathcal P_{\mathcal E}^{nhij} \hat K_{\text{local}}^{hklij}

        \hat K_{ij} = \int_\Omega  f(u, v) \text dv


    :math:`f` is `forward` function which is defined by inheritating this class

    * :math:`\hat K_{\text{global}}` : non zero value of the global galerkin matrix, :math:`K_{\text{global}}\in \mathbb R^{|\mathcal E|\times  d\times d}`
    * :math:`\hat K_{\text{local}}` : local galerkin matrix for each element , :math:`K_{\text{local}}\in \mathbb R^{|\mathcal C|\times h\times h\times d\times d}` 
    * :math:`\mathcal P_{\mathcal E}` : projection (assemble) tensor from :math:`\hat K_{\text{local}}` to :math:`\hat K_{\text{global}}, `\mathcal P_{\mathcal E} \in \mathbb R_{\text{sparse}}^{|\mathcal E|\times |\mathcal C|\times h\times h}`
    * :math:`\mathcal C` : elements/cells 
    * :math:`h` : number of basis for each element/cell
    * :math:`\mathcal E` : connections for nodes/vertices/points
    * :math:`\mathcal V` : nodes/vertices/points 


    Examples:
    ---------

    1. assemble the mass matrix

    .. math:: 

        M_{ij} = \int_\Omega u_i v_j \text dv

    .. code-block:: python

        import torch_fem
        import torch_fem.functional as F
        class MassAssembler(torch_fem.ElementAssembler):
            def forward(self, u, v):
                return F.dot(u, v)
        mesh = torch_fem.Mesh.gen_rectangle()
        assembler = MassAssembler.from_mesh(mesh)
        M = assembler(mesh.points)

    
    2. assemble the laplace matrix

    .. math::

        K_{ij} = \int_\Omega \nabla u_i \cdot \nabla v_j \text dv

    .. code-block:: python

        import torch_fem
        import torch_fem.functional as F
        class LaplaceAssembler(torch_fem.ElementAssembler):
            def forward(self, gradu, gradv):
                return F.dot(gradu, gradv)
        mesh = torch_fem.Mesh.gen_circle()
        assembler = LaplaceAssembler.from_mesh(mesh)
        K = assembler(mesh.points) 

    Attributes
    -----------
    quadrature_weights : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`torch_fem.shape.element_types`.
        Each :obj:`element_type` corresponds to a 1D tensor of shape :math:`[Q]`, where :math:`Q` is the number of quadrature points
        the quadrature weights of each element type, e.g. :obj:`{"triangle6": torch.tensor([0.5, 0.5])}`
    quadrature_points : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`torch_fem.shape.element_types`.
        Each :obj:`element_type` corresponds to a 2D tensor of shape :math:`[Q, D]`, where :math:`Q` is the number of quadrature points, :math:`D` is the dimension of the domain
        the quadrature points of each element type, e.g. :obj:`{"triangle6": torch.tensor([[0.5, 0.5], [0.5, 0.0]])}`
    shape_val : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`torch_fem.shape.element_types`.
        Each :obj:`element_type` corresponds to a 2D tensor of shape :math:`[Q, B]`, where :math:`Q` is the number of quadrature points, :math:`B` is the number of basis
        the shape value of each element type, e.g. :obj:`{"triangle6": torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])}`
    projector : BufferDict[str, Projector]
        The element type is the key, which should be one of :meth:`torch_fem.shape.element_types`.
        ach :obj:`element_type` corresponds to a projector from element to edge,
        each  projector is a :meth:`torch_fem.assemble.Projector` object, could be considered as a sparse matrix
        
        .. math::

            \mathcal P_e: \mathbb{R}_{\text{sparse}}^{|\mathcal C_e| \times B_e \times B_e} \rightarrow \mathbb{R}^{|\mathcal E|}

        where :math:`\mathcal C` is the set of elements, :math:`B` is the number of basis, :math:`\mathcal E` is the set of edges.

    elements : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`torch_fem.shape.element_types`.
        Each :obj:`element_type` corresponds to a 2D tensor of shape :math:`[|\mathcal C|, B]`, where :math:`\mathcal C` is the set of elements, :math:`B` is the number of basis
        the element connectivity of each element type, e.g. :obj:`{"triangle6": torch.tensor([[0, 1, 2], [1, 2, 3]])}`
    edges : torch.Tensor
        2D tensor of shape :math:`[2, |\mathcal E|]`, where :math:`\mathcal E` is the set of edges
        edge connectivity considering all element_types, e.g. :obj:`torch.tensor([[0, 1, 2], [1, 2, 3]])`
    n_points : int
        number of points
    dimension : int
        dimension of the mesh, either :math:`1` or :math:`2` or :math:`3`
    element_types : list[str]
        element types, e.g. :obj:`["triangle6", "quad9"]`
    """

    projector:nn.ModuleDict      # Dict[str, Projector]
    transformation:nn.ModuleDict # Dict[str, Transformation]
    elements:BufferDict          # Dict[str, torch.Tensor]
    edges:torch.Tensor           
    dimension:int
    element_types:List[str]
    n_points:int

    __autodoc__ = [
        '__call__',
        'forward',
        '__post_init__',
        'from_assembler',
        'from_mesh',
    ]
    def __init__(self,
                        projector:nn.ModuleDict, 
                        transformation:nn.ModuleDict,
                        elements:BufferDict,
                        edges:torch.Tensor,
                        *args,
                        **kwargs):
        super().__init__()
        element_types = list(projector.keys())
        dimension     = element_type2dimension[element_types[0]]

        self.projector          = projector
        self.transformation     = transformation
        self.elements           = elements
        self.register_buffer("edges", edges)
        self.dimension          = dimension
        self.element_types      = list(elements.keys())
        self.n_points           = next(iter(self.transformation.values())).n_points # type: ignore
        self.__post_init__(*args,**kwargs)

    @property
    def device(self)->torch.device:
        r"""
        Returns
        -------
        torch.device
            the device of the assembler, either :obj:`torch.device("cpu")` or :obj:`torch.device(f"cuda:{x}")`
        """
        return next(iter(self.transformation.values())).device # type: ignore

    @property
    def dtype(self)->torch.dtype:
        r"""
        Returns
        -------
        torch.dtype
            the data type of the assembler, either :obj:`torch.float32` or :obj:`torch.float64`
        """
        return next(iter(self.transformation.values())).device # type: ignore

    def type(self, dtype:torch.dtype):
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self

    def _integrate(self, batch_integral, jxw, n_element, n_basis, use_element_parallel):
        if use_element_parallel:
            error_msg = f"the shape returned by forward function is {[*batch_integral.shape]} which is not supported, should either be [{n_element}, batch_size, {n_basis},{n_basis}] or [{n_element}, batch_size,{n_basis},{n_basis}, dof_per_point, dof_per_point]"
            assert batch_integral.dim() == 4 or batch_integral.dim() == 6, error_msg
            assert batch_integral.shape[0] == n_element, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg 
            assert batch_integral.shape[3] == n_basis, error_msg
            if batch_integral.dim() == 6:
                assert batch_integral.shape[-1] == batch_integral.shape[-2], error_msg
            batch_integral = torch.einsum("eqij...,eq->eij...", batch_integral, jxw)
        else:
            error_msg = f"the shape returned by forward function is {[*batch_integral.shape]} which is not supported, should either be [batch_size, {n_basis},{n_basis}] or [batch_size,{n_basis},{n_basis}, dof_per_point, dof_per_point]"
            assert batch_integral.dim() == 3 or batch_integral.dim() == 5, error_msg
            assert batch_integral.shape[1] == n_basis, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg
            if batch_integral.dim() == 5:
                assert batch_integral.shape[-1] == batch_integral.shape[-2], error_msg
            batch_integral = torch.einsum("qij...,eq->eij...", batch_integral, jxw)

        return batch_integral
    
    def _build_output(self, integral):
        r"""
            Parameters:
            -----------
                integral: torch.Tensor of shape [n_edge, ...]
                    the integral of each edge
            Returns:
            --------
                SparseMatrix
        """
        if integral.dim() == 1:
            return SparseMatrix(integral, self.edges[0], self.edges[1], shape=(self.n_points, self.n_points))
        elif integral.dim() ==  3:
            return SparseMatrix.from_block_coo(integral, self.edges[0], self.edges[1], shape=(self.n_points, self.n_points))
        else:
            raise Exception(f"the shape of integral is supposed to be  1D or 3D, but got {integral.shape}")

    def __call__(self, points:Optional[torch.Tensor] = None, 
                       func:Optional[Callable] = None,
                       point_data:Optional[Dict[str, torch.Tensor]] = None, 
                       element_data:Optional[Dict[str, Dict[str,torch.Tensor]]|Dict[str,torch.Tensor]] = None, 
                       scalar_data:Optional[Dict[str, torch.Tensor]] = None,
                       batch_size:int = -1):
        r"""
        Parameters
        ----------
        points: torch.Tensor 
            2D tensor of shape :math:`[|\mathcal V|, D]`, where :math:`\mathcal V` is the set of nodes/vertices/points, :math:`D` is the dimension of the domain
            the coordinates of the points
        func: function or None, optional
            the bilinear function, when it's None the forward function will be used,
            if you want to reuse the same element assembler for different bilinear function, you can pass the bilinear function here
        point_data: Dict[str, torch.Tensor], optional
            tensor of shape :math:`[|\mathcal V|, ...]`, where :math:`\mathcal V` is the set of nodes/vertices/points
        element_data: Dict[str, torch.Tensor], optional 
            dict of :obj:`{key:{element_type:tensor}}`
            tensor of shape :math:`[|\mathcal C|, ...]` where :math:`\mathcal C` is the set of elements/cells
            if there is only one kinds of elements, you could simply passed as :obj:`{key:tensor}`  
        scalar_data: Dict[str, torch.Tensor|float|int], optional
            scalar data that should not be broadcasted, will be directly passed to forward
        batch_size: int or None, optional
            the batch size of quadrature points
            if :obj:`int` is given, the quadrature points will be divided into batches
            if :obj:`None` is given, the quadrature points will not be divided into batches
            default is :obj:`1`
        Returns
        -------
        SparseMatrix
            a torch.sparse_matrix of shape :math:`\mathbb R_{\text{sparse}}^{|\mathcal V|, |\mathcal V|}` or :math:`\mathbb R_{\text{sparse}}^[|\mathcal V| *  H, |\mathcal V| *  H]`,
            where :math:`H` is the number of degree of freedom per point, :math:`|\mathcal V|` is the number of points
        """
        assert isinstance(point_data, dict) or point_data is None, f"point_data should be a dict, but got {type(point_data)}. Please pass  in extra parameter using key-value pairs"
        # make sure point data is Dict[str, torch.Tensor]
        if point_data is None:
            point_data = {}

        # make sure element data is Dict[str,Dict[str,  torch.Tensor]]
        if element_data is None:
            element_data = {element_type:{} for element_type in self.element_types} # 
        else:
            if not isinstance(next(iter(element_data.keys())), dict):
                assert len(self.element_types) == 1
                element_type = self.element_types[0]
                element_data = {key:{element_type:value} for key, value in element_data.items()} # type:ignore
                for  key in element_data:
                    for element_type in self.element_types:
                        assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0], f"the shape of {key} should be [{self.elements[element_type].shape[0]}, ...], but got {element_data[key][element_type].shape[0]}"
            else:
                for key, value in element_data.items():
                    for element_type in self.element_types:
                        assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0]
        
        # make sure scalar data is Dict[str, torch.Tensor]
        if scalar_data is None:
            scalar_data = {}
        else:
            scalar_data = {k:torch.tensor(v) for k,v in scalar_data.items()}


        # make sure points is torch.Tensor
        if points is None:
            points = next(iter(self.transformation.values())).points # type: ignore
        else:
            for element_type in self.element_types:
                assert points.shape[1] == self.transformation[element_type].dim, f"the dimension of points should be {self.transformation[element_type].dimension}, but got {points.shape[1]}"
                self.transformation[element_type].update_points(points) # type: ignore
        

        point_data["x"] = points # type:ignore [n_point, n_dim]

        self = self.type(points.dtype).to(points.device) # type:ignore

        for key, value in point_data.items():
            assert value.shape[0] == points.shape[0], f"the shape of {key} should be [n_point, ...], but got {value.shape}"
        
        # parse signature

        fn = self.forward if func is None else func

        signature = inspect.signature(fn)

        element_dims = []
        quadrature_dims = []
        for key in signature.parameters:
            if key in ["u", "v"]:
                element_dims.append(None)
                quadrature_dims.append(0)
            elif key in element_data: # type:ignore
                element_dims.append(0)
                quadrature_dims.append(None)
            elif key in scalar_data: 
                element_dims.append(None)
                quadrature_dims.append(None)
            else:
                element_dims.append(0)
                quadrature_dims.append(0)
        
        element_dims = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)

        if all([x is None for x in element_dims]):
            # if all is shape_val
            parallel_fn = torch.vmap(fn, in_dims=quadrature_dims)
            use_element_parallel = False
        else:
            parallel_fn = torch.vmap(
                torch.vmap(
                    fn,
                    in_dims = quadrature_dims
                ),
                in_dims=element_dims
            )
            use_element_parallel = True

        integral = None
      
        for element_type in self.element_types:
            element_integral = None
            
            trans:Transformation  = self.transformation[element_type] # type:ignore 
            proj:Projector        = self.projector[element_type] # type:ignore
            n_quadrature          = trans.n_quadrature

            n_batch      = n_quadrature // batch_size if batch_size != -1 else 1
            n_batch_size = batch_size if batch_size != -1 else n_quadrature
            ele_point_data = {k:v[self.elements[element_type]] for k,v in point_data.items()}

            for i in range(n_batch):
                # prepare arguments
                shape_val            = trans.batch_shape_val(i*n_batch_size, n_batch_size) # [batch_size, n_basis]
                shape_grad, jxw      = trans.batch_shape_grad_jxw(
                                                quadrature_start = i*n_batch_size, 
                                                quadrature_batch = n_batch_size)
                
                args = []
                for key in signature.parameters:
                    if key in ["u", "v"]:
                        args.append(shape_val)
                    elif key in ["gradu", "gradv"]:
                        args.append(shape_grad)
                    elif key in ele_point_data:
                        args.append(torch.einsum("eb...,qb->eq...",ele_point_data[key], shape_val))
                    elif key.startswith("grad") and key[4:] in ele_point_data:
                        args.append(torch.einsum("eb...,eqbd->eq...d",ele_point_data[key[4:]], shape_grad))
                    elif key in element_data: # type:ignore
                        args.append(element_data[key][element_type]) # type:ignore
                    elif key in scalar_data: # type:ignore
                        args.append(scalar_data[key])
                    else:
                        raise NotImplementedError(f"key {key} is not implemented")

                # parallel dispatch 
                batch_integral = parallel_fn(*args) # [n_element, batch_size, n_basis, n_basis, ...] or [n_batch, batch_size, n_basis, ...]

                batch_integral = self._integrate(batch_integral, jxw, trans.n_elements, trans.n_basis, use_element_parallel)

                element_integral = batch_integral if element_integral is None else element_integral + batch_integral
    
            integral = proj(element_integral) if integral is None else integral + proj(element_integral) # type:ignore [n_edge, ...]

        return self._build_output(integral)

    @abstractmethod
    def forward(self, **kwargs):
        r"""The weak form of the operator, you should override this function.
        Similar to the :meth:`torch:torch.nn.Module.forward` function, you can use :meth: `torch_fem.assemble.ElementAssembler.__call__` to call this function

        Parameters
        ----------
        u : torch.Tensor, optional
            1D tensor shape :math:`[B]`, where :math:`B` is the number of basis
        v : torch.Tensor, optional
            1D tensor shape :math:`[B]`, where :math:`B` is the number of basis
        gradu : torch.Tensor, optional
            2D tensor shape :math:`[B,D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        gradv : torch.Tensor, optional
            2D tensor shape :math:`[B,D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        x : torch.Tensor, optional
            2D tensor shape :math:`[D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        gradx : torch.Tensor, optional
            3D tensor shape :math:`[D, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        **point_data : Dict[str, torch.Tensor], optional
            The point_data are passed by __call__
            if the point data :obj:`"example_key"` passed in is of shape :math:`[|\mathcal V|, ...]`, 
            then the point data :obj:`"example_key"` passed in will be of shape :math:`[...]`,
            and the point data :obj:`"gradexample_key"` passed in will be of shape :math:`[..., D]`,
            where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension

        Returns
        -------
        torch.Tensor
            2D tensor of shape :math:`[B,B]` or 4D tensor of shape :math:`[B, B, H, H]`, where :math:`B` is the number of basis, :math:`H` is the number of degree of freedom per point

        """
        raise NotImplementedError(f"forward is not implemented")
        
    def __post_init__(self):
        r"""Override this function to precompute some data after the initialization
        """
        pass

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    element_types: {self.element_types}\n"
            f"    n_element: {' '.join(f'{k}:{v.shape[0]}' for k, v in self.elements.items())}\n"
            f"    n_point: {self.n_points}\n"
            f"    n_basis: {' '.join(f'{k}:{v.shape[1]}' for k, v in self.elements.items())}\n"
            f"    n_dim: {self.dimension}\n"
            f"    n_quadrature: {' '.join(f'{k}:{v.n_quadrature}' for k, v in self.transformation.items())}\n"
            f"    forward: \n{inspect.getsource(self.forward)}"
            f")"
        )
    
    def __repr__(self):
        return str(self)

    @classmethod
    def from_assembler(cls, obj, *args, **kwargs):
        r"""Build an :meth:`torch_fem.assemble.ElementAssembler` from another :meth:`torch_fem.assemble.ElementAssembler`.
        It's much faster than :meth:`torch_fem.assemble.ElementAssembler.from_mesh`.
        When you already have an ElementAssembler, you can use this function to build another ElementAssembler sharig the same mesh

        Parameters
        ----------
        obj: torch_fem.assemble.ElementAssembler
            an meth:`torch_fem.assemble.ElementAssembler` object
        
        Returns
        -------
        torch_fem.assemble.ElementAssembler
            the new element assembler sharing the same mesh
        """
        assert isinstance(obj, ElementAssembler), f"obj must be an instance of ElementAssembler, but got {type(obj)}"
        return cls(
                   obj.projector, 
                   obj.transformation,
                   obj.elements,
                   obj.edges,
                   *args,**kwargs)

    @classmethod
    def from_mesh(cls, mesh:Mesh, 
                        quadrature_order:int = 2,
                        *args,
                        **kwargs):
        r"""Build an :meth:`torch_fem.assemble.ElementAssembler` from a mesh :meth:`torch_fem.mesh.Mesh`.
        It's much slower than :meth:`torch_fem.assemble.ElementAssembler.from_assembler`.
        Because it will precompute the projection matrix $\mathcal P_{\mathcal E}$

        Parameters
        ----------
        mesh: torch_fem.mesh.mesh.Mesh
            a meth:`torch_fem.mesh.Mesh` object
        quadrature_order: int
            the order should be poisitive integer, default is :obj:`2`
        
        Returns
        -------
        torch_fem.assemble.ElementAssembler
            the new element assembler use the topology of the mesh
        """
        points:torch.Tensor   = mesh.points # type: ignore
        elements              = mesh.elements() # type:ignore
        n_points:int          = points.shape[0]
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}

        projector          = {}
        transformations    = {}
        

        ######################
        # compute the edges
        ######################
        elem_u, elem_v = [], []
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            # quadrature_weights[element_type], quadrature_points[element_type] =\
            # get_quadrature(element_type, quadrature_order) # [n_quadrature], [n_quadrature, n_dim]
            # shape_val[element_type] = get_shape_val(element_type, quadrature_points[element_type]) # [n_quadrature, n_basis]
            for i in range(n_basis):
                for j in range(n_basis):
                    elem_u.append(value[:, i])
                    elem_v.append(value[:, j])

        elem_u, elem_v = torch.stack(elem_u, -1).flatten(), torch.stack(elem_v, -1).flatten() # [num_elements * num_basis * num_basis]
        elem_u, elem_v = elem_u.cpu().numpy().copy(), elem_v.cpu().numpy().copy()
        tmp = scipy.sparse.coo_matrix(( # used to remove duplicated edges
            np.ones_like(elem_u), # data
            (elem_u, elem_v), # (row, col)
        ), shape = (n_points,  n_points)).tocsr().tocoo()
        edge_u, edge_v = tmp.row, tmp.col
        num_edges  = len(edge_u)
        eids_csr = scipy.sparse.coo_matrix((
            np.arange(num_edges), (edge_u, edge_v)
        ), shape=(n_points, n_points)).tocsr()    
        elem_eids     = np.array(eids_csr[elem_u, elem_v].copy()).ravel()

        ptr = 0
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            elem_eids  = np.array(eids_csr[elem_u[ptr:ptr+n_element*n_basis*n_basis], elem_v[ptr:ptr+n_element*n_basis*n_basis]].copy()).ravel()
            ptr += n_element * n_basis * n_basis
            projector[element_type] = Projector( # [n_element, n_basis, n_basis] -> [num_edges]
                                        from_ = torch.arange(n_element * n_basis * n_basis), 
                                        to_    = torch.from_numpy(elem_eids),
                                        from_shape = (n_element, n_basis, n_basis), 
                                        to_shape = (num_edges,),
                                    )
            transformations[element_type] = Transformation(
                                        points  = points,
                                        elements = value,
                                        element_type   = element_type, 
                                        quadrature_order= quadrature_order
                                    )
         
        edges = torch.from_numpy(np.stack([edge_u, edge_v], 0))


        projector          = nn.ModuleDict(projector)
        transformations    = nn.ModuleDict(transformations)
        elements           = BufferDict({k:v.long() for k,v in elements.items()})
    

        assembler = cls(
                   projector, 
                   transformations,
                   elements,
                   edges,
                   *args,**kwargs)
        assembler = assembler.type(mesh.dtype).to(mesh.device)
        return assembler

      
ElementAssembler.type.__doc__ = nn.Module.type.__doc__