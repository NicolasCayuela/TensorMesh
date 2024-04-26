from typing import Optional, Union,Dict
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection, PathCollection
from matplotlib.image import AxesImage
import matplotlib.colors as mcolors
from matplotlib import tri
from scipy.interpolate import griddata
from .utils import as_ndarray, as_tensor, dim
from ..element import element_type2order,\
                      element_type2dimension,\
                      element_type2element

def draw_point_value_2d_tri_gouraud(points:torch.Tensor|np.ndarray,
                              point_values:torch.Tensor|np.ndarray,
                              elements:torch.Tensor|np.ndarray,
                              cmap:str = 'jet',
                              ax:Optional[plt.Axes] = None):
    """
    Parameters
    ----------
    points: torch.Tensor|np.ndarray
        2D tensor of shape [n_points, 2]
        the points of the mesh
    point_values: torch.Tensor|np.ndarray
        1D tensor of shape [n_points]
        the value of the points
    elements: torch.Tensor|np.ndarray
        2D tensor of shape [n_elements, 3]
        the elements of the mesh
    cmap: str, optional
        the colormap, default is 'jet'
    ax: matplotlib.axes.Axes, optional
        the axis, default is None

    Returns
    -------
    img: matplotlib.collections.PathCollection
    ax: matplotlib.axes.Axes
    """
    # assertion
    assert dim(points) == 2, f"points.dim() must be 2, but got {dim(points)}"
    assert points.shape[1] == 2, f"points.shape[1] must be 2, but got {points.shape[1]}"
    assert dim(point_values) == 1, f"point_values.dim() must be 1, but got {dim(point_values)}"
    assert dim(elements) ==2, f"elements.dim() must be 2, but got {dim(elements)}"
    assert elements.shape[1] == 3, f"elements.shape[1] must be 3, but got {elements.shape[1]}"
    assert point_values.shape[0] == points.shape[0], f"point_values.shape[0] must be equal to points.shape[0], but got {point_values.shape[0]} and {points.shape[0]}"
    assert elements.max() < points.shape[0], f"elements.max() must be less than points.shape[0], but got {elements.max()} and {points.shape[0]}"
    
    # input prepare
    points_np       = as_ndarray(points)
    point_values_np = as_ndarray(point_values)
    elements_np     = as_ndarray(elements)
    cmap = mpl.colormaps[cmap]
    ax   = plt.subplots(figsize=(10,10))[1] if ax is None else ax
    
    # draw elements
    img = ax.tripcolor(points_np[:,0], 
                 points_np[:,1], 
                 elements_np, 
                 facecolors=point_values_np, 
                 cmap=cmap, shading='gouraud')
    
    return img, ax

def update_point_value_2d_tri_gouraud(img:PolyCollection,
                                        point_values:torch.Tensor|np.ndarray):
    """
    Parameters
    ----------
    img: matplotlib.collections.PolyCollection
        the image
    point_values: torch.Tensor|np.ndarray
        the point values, 1D tensor of shape [n_points]
    """
    # assertion
    assert isinstance(img, PolyCollection), f"img must be an instance of matplotlib.collections.PolyCollection, but got {type(img)}"
    assert dim(point_values) == 1, f"point_values.dim() must be 1, but got {dim(point_values)}"

    point_values_np = as_ndarray(point_values)
    img.set_array(point_values_np)

def draw_point_value_2d_interpolation( points:torch.Tensor|np.ndarray,
                                    point_values:torch.Tensor|np.ndarray,
                                    density:int = 100,
                                    cmap:str = 'jet',
                                    use_scatter:bool = False,
                                    ax:Optional[plt.Axes] = None):
    """
    Parameters:
    -----------
    points: torch.Tensor|np.ndarray
        2D tensor of shape [n_points, 2]
        the points of the mesh
    point_values: torch.Tensor|np.ndarray
        1D tensor of shape [n_points]
        the value of the points
    density: int
        the density of the interpolation
        default is 100
    cmap: str
        the colormap
        default is 'jet'
    ax: Optional[matplotlib.axes.Axes]
        the axis
        default is None

    Returns
    -------
    img: matplotlib.collections.PathCollection
    ax: matplotlib.axes.Axes
    """
    # assertion
    assert dim(points) == 2, f"points.dim() must be 2, but got {dim(points)}"
    assert points.shape[1] == 2, f"points.shape[1] must be 2, but got {points.shape[1]}"
    assert dim(point_values) == 1, f"point_values.dim() must be 1, but got {dim(point_values)}"
    assert point_values.shape[0] == points.shape[0], f"point_values.shape[0] must be equal to points.shape[0], but got {point_values.shape[0]} and {points.shape[0]}"
    assert density > 0, f"density must be greater than 0, but got {density}"

    # input prepare
    points_np = as_ndarray(points)
    point_values_np = as_ndarray(point_values)
    cmap = mpl.colormaps[cmap]
    ax   = plt.subplots(figsize=(10,10))[1] if ax is None else ax
    xmin, xmax = points_np[:,0].min(), points_np[:,0].max()
    ymin, ymax = points_np[:,1].min(), points_np[:,1].max()
    grid_x, grid_y = np.mgrid[xmin:xmax:density*1j, ymin:ymax:density*1j]
    grid_z = griddata(points_np, point_values_np, (grid_x, grid_y), method='cubic')

    # draw elements
    if use_scatter:
        img = ax.scatter(grid_x.flatten(), grid_y.flatten(), c=grid_z.flatten(), cmap=cmap)
    else:
        img = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap=cmap, aspect='auto')
    
    return img, ax

def update_point_value_2d_interpolation(img:PathCollection|AxesImage,
                                        points:torch.Tensor|np.ndarray,
                                        point_values:torch.Tensor|np.ndarray):
    """
    Parameters
    ----------
    img: PathCollection|AxesImage
        the image
    points:torch.Tensor|np.ndarray
        the points, 2D tensor of shape [n_points, 2]
    point_values: torch.Tensor|np.ndarray
        the point values, 1D tensor of shape [n_points]
    """
    # assertion
    assert dim(point_values) == 1, f"point_values.dim() must be 1, but got {dim(point_values)}"
    assert point_values.shape[0] == points.shape[0], f"point_values.shape[0] must be equal to points.shape[0], but got {point_values.shape[0]} and {points.shape[0]}"
    assert dim(points) == 2, f"points.dim() must be 2, but got {dim(points)}"
    assert points.shape[1] == 2, f"points.shape[1] must be 2, but got {points.shape[1]}"

    points_np       = as_ndarray(points)
    point_values_np = as_ndarray(point_values)

    if isinstance(img, PathCollection):
        # scatter 
        grid_x, grid_y = img.get_offsets().T
        grid_z = griddata(points_np, point_values_np, (grid_x, grid_y), method='cubic')
        img.set_array(grid_z)
     
    elif isinstance(img, AxesImage):
        # imgshow
        image_data = img.get_array()
        height, width  = image_data.shape
        extend     = img.get_extent()
        xmin, xmax, ymin, ymax = extend
        grid_x, grid_y = np.mgrid[xmin:xmax:width*1j, ymin:ymax:height*1j]
        grid_z = griddata(points_np, point_values_np, (grid_x, grid_y), method='cubic')
        img.set_data(grid_z.T)
    else:
        raise NotImplementedError(f"img type {type(img)} is not supported")

def draw_point_value_2d(points:torch.Tensor|np.ndarray,
                        point_values:torch.Tensor|np.ndarray,
                        elements:Dict[str,torch.Tensor|np.ndarray],
                        density:int           = 100,
                        cmap:str              = 'jet',
                        use_scatter:bool      = False,
                        ax:Optional[plt.Axes] = None):
    """for first order triangle element draw using the tri_gouraud
    otherwise, draw with the 2d interpolation
    Parameters
    ----------
    points: torch.Tensor|np.ndarray
        2D tensor of shape [n_points, 2]
        the points of the mesh
    point_values: torch.Tensor|np.ndarray
        1D tensor of shape [n_points]
        the value of the points
    elements: Dict[str,torch.Tensor|np.ndarray]
        2D tensor of shape [n_elements, n_basis]
        the elements of the mesh
    density: int
        the density of the interpolation
        default is 100
    cmap: str
        the colormap
        default is 'jet'
    use_scatter: bool
    ax: Optional[matplotlib.axes.Axes]
        the axis
        default is None

    Returns
    -------
    img: matplotlib.collections.PathCollection
    ax: matplotlib.axes.Axes
        the axis
    """
    # assertion
    for key in elements.keys():
        assert element_type2dimension[key] == 2, f"element_type2dimension[{key}] must be 2, but got {element_type2dimension[key]}"
        assert dim(elements[key]) == 2, f"elements[{key}].dim() must be 2, but got {dim(elements[key])}"
    assert dim(points) == 2, f"points.dim() must be 2, but got {dim(points)}"
    assert points.shape[1] == 2, f"points.shape[1] must be 2, but got {points.shape[1]}"
    assert dim(point_values) == 1, f"point_values.dim() must be 1, but got {dim(point_values)}"
    assert point_values.shape[0] == points.shape[0], f"point_values.shape[0] must be equal to points.shape[0], but got {point_values.shape[0]} and {points.shape[0]}"
    assert density > 0, f"density must be greater than 0, but got {density}"
    
    for k, ele in elements.items():
        element = element_type2element(k)
        order   = element_type2order[k]
        if order == 1 and element == 'tri':
            img, _ = draw_point_value_2d_tri_gouraud(points, point_values, ele, cmap, ax=ax)
        else:
            img, _ = draw_point_value_2d_interpolation(points, point_values, density, cmap, use_scatter, ax=ax)
    
    return img, ax

def update_point_value_2d(img:PathCollection|PolyCollection|AxesImage,
                          points:torch.Tensor|np.ndarray,
                          point_values:torch.Tensor|np.ndarray):
    """
    Parameters
    ----------
    img: matplotlib.collections.PathCollection
        the image
    points:torch.Tensor|np.ndarray
        the points, 2D tensor of shape [n_points, 2]
    point_values: torch.Tensor|np.ndarray
        the point values, 1D tensor of shape [n_points]
    """
    # assertion
    assert dim(point_values) == 1, f"point_values.dim() must be 1, but got {dim(point_values)}"
    assert point_values.shape[0] == points.shape[0], f"point_values.shape[0] must be equal to points.shape[0], but got {point_values.shape[0]} and {points.shape[0]}"
    assert dim(points) == 2, f"points.dim() must be 2, but got {dim(points)}"
    assert points.shape[1] == 2, f"points.shape[1] must be 2, but got {points.shape[1]}"

    if isinstance(img, (PathCollection, AxesImage)):
        # scatter 
        update_point_value_2d_interpolation(img, points, point_values)
    elif isinstance(img, PolyCollection):

        update_point_value_2d_tri_gouraud(img, point_values)
    else:
        raise NotImplementedError(f"img type {type(img)} is not supported")


def draw_elements(mesh, 
                  value:Union[str, torch.Tensor, Dict[str, torch.Tensor]], 
                cmap="jet", 
                ax=None):
    r"""
    Parameters
    ----------
    mesh: torch_fem.mesh.Mesh
        the mesh
    value: str or torch.Tensor or Dict[str, torch.Tensor]
        the value to be drawn
    ax: matplotlib.axes.Axes, optional
        the axis, default is None

    Returns
    -------
    ax: matplotlib.axes.Axes
        the axis
    """
    assert mesh.points.shape[1] == 2, "only 2D mesh is currently supported"
    if isinstance(value, str):
        value = mesh.cell_data[value]
    elif isinstance(value, torch.Tensor):
        assert isinstance(mesh.element_type, str), f"torch.Tensor value corresponds to homogenous mesh, but mesh.element_type is {mesh.element_type}"
        value = {mesh.element_type: value}
    elif isinstance(value, dict):
        assert set(mesh.element_type.keys()).difference(set(value.keys())) == set(), f"mesh.element_type is {mesh.element_type}, but value.keys() is {value.keys()}"
    else:
        raise NotImplementedError(f"value type {type(value)} is not supported")
    
    for k,v in value.items():
        if isinstance(v, torch.Tensor):
            value[k] = v.cpu().numpy()

    min_value,  max_value = min([v.min().item() for k, v in value])
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    
    pos = mesh.points.numpy()
    cmap = getattr(plt.cm, cmap)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    points = mesh.points.numpy()
    for k,v in value.items():
    
        elements = mesh.elements(element_type=k)
        if value.shape[1] == 3 or value.shape[1] == 4: # tri or quad
            order = np.arra
            polygons = [patches.Polygon(pos[element], closed=True, fill=False, color=cmap(norm(v[i]))) for i,element in enumerate(elements)]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
        elif value.shape[1] == 6: # tri6
            order = np.array([0, 3, 1, 4, 2, 5])
            polygons = [patches.Polygon(pos[element[order]], closed=True, fill=False, color=cmap(norm(v[i]))) for i,element in enumerate(value)]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
        elif value.shape[1] == 9: # quad9 
            order = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            polygons = [patches.Polygon(pos[element[order]], closed=True, fill=False, color=cmap(norm(v[i]))) for element in value]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
        else:
            raise NotImplementedError(f"element type {value.shape[1]} is not supported")

    
    return ax