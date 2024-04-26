from optparse import Option
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
from matplotlib import tri
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, PolyCollection

from ..element import element_type2dimension, element_type2element, element_type2order, Triangle
from .utils import as_ndarray, dim

def draw_element_value_2d_tri(
                        points:torch.Tensor|np.ndarray,
                        elements:torch.Tensor|np.ndarray,
                        values:torch.Tensor|np.ndarray,
                        alpha:Optional[float|torch.Tensor|np.ndarray]=None,
                        cmap:str='viridis',
                        color:Optional[str]=None,
                        ax:Optional[plt.Axes]=None,
                        **kwargs)->Tuple[PolyCollection, plt.Axes]:
    """
    Parameters:
    -----------
        points: torch.Tensor or np.ndarray
            [n_points, 2]
        elements: torch.Tensor or np.ndarray
            [n_elements, 3]
        values: torch.Tensor or np.ndarray
            [n_elements]
        alpha: float or torch.Tensor or np.ndarray
            [n_elements]
            should be greater or equal 0
        cmap: str
            colormap, default is 'viridis'
        color: str
            color, if alpha is torch.Tensor or np.ndarray, the color will be used 
            default is None
        ax: plt.Axes
            default is None

    Returns:
    --------
        plt.Axes
    """
    # assertion
    assert dim(points) == 2
    assert dim(elements) == 2
    assert dim(values) == 1
    assert elements.shape[1] == 3
    assert elements.shape[0] == values.shape[0]
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        assert alpha.shape[0] == values.shape[0]
        assert (alpha >= 0).all()

    # to numpy 
    points   = as_ndarray(points)
    elements = as_ndarray(elements)
    values   = as_ndarray(values)
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        alpha = as_ndarray(alpha)

    
    if ax is None:
        fig, ax = plt.subplots()

    # draw the triangles
    triangles = tri.Triangulation(points[:, 0], points[:, 1], elements)
    if color is None: # use cmap
        img = ax.tripcolor(triangles, values, cmap=cmap, alpha=alpha, **kwargs)
    else: # use color
        img = ax.tripcolor(triangles, values, color=color, alpha=alpha, **kwargs)

    return img, ax


def draw_element_value_2d( points:torch.Tensor|np.ndarray,
                        elements:Dict[str,torch.Tensor|np.ndarray],
                        values:Dict[str,torch.Tensor|np.ndarray],
                        alpha:float|Dict[str,torch.Tensor|np.ndarray]=1.0,
                        cmap:str='viridis',
                        color:Optional[str]=None,
                        ax:Optional[plt.Axes]=None,
                        )->Tuple[Dict[str,PolyCollection], plt.Axes]:
    """
    Parameters
    ----------
        points: torch.Tensor or np.ndarray
            [n_points, 2]
        elements: Dict[str, torch.Tensor or np.ndarray]
            keys are 'tri' or 'quad'
            values are torch.Tensor or np.ndarray
            [n_elements, 3] or [n_elements, 4]
        values: Dict[str, torch.Tensor or np.ndarray]
            [n_elements]
        alpha: float or torch.Tensor or np.ndarray
            [n_elements]
            should be greater or equal 0 and less or equal than 1
        cmap: str
            colormap, default is 'viridis'
        color: str
            color, if alpha is torch.Tensor or np.ndarray, the color will be used 
            default is None
        ax: plt.Axes
            default is None
    """
    # assertion
    assert dim(points) == 2
    assert isinstance(elements, dict)
    for key in elements.keys():
        assert element_type2dimension[key] == 2, f"element type {key} is not 2D"
        assert values[key].shape[0] == elements[key].shape[0], f"values for {key} is not equal to elements"
        if isinstance(alpha, dict):
            assert alpha[key].shape[0] == values[key].shape[0], f"alpha for {key} is not equal to values"
            assert (alpha[key] >= 0).all(), f"alpha for {key} should be greater or equal 0"
            assert (alpha[key] <= 1).all(), f"alpha for {key} should be less or equal 1"
    if isinstance(alpha, float):
        assert 0 <= alpha <= 1, f"alpha should be between 0 and 1"

    # to numpy
    points = as_ndarray(points)
    elements = {key:as_ndarray(elements[key]) for key in elements.keys()}
    values = {key:as_ndarray(values[key]) for key in values.keys()}
    if isinstance(alpha, dict):
        alpha = {key:as_ndarray(alpha[key]) for key in alpha.keys()}
    else:
        alpha = {key:alpha for key in elements.keys()}

    if ax is None:
        fig, ax = plt.subplots()

    # draw the elements
    collections = {}
    for key in elements.keys():
        element = element_type2element(key)
        order   = element_type2order[key]
      
        if element is Triangle:
            img, ax = draw_element_value_2d_tri(points, elements[key], values[key], alpha[key], cmap, color, ax)
            collections[key] = img
            continue

        contour = element.get_contour(order)
        contour = elements[key][:, contour] # [n_elements, n_contour]
        contour = points[contour] # [n_elements, n_contour, 2]

        polygons= [Polygon(x,closed=True) for x in contour]
        collection = PolyCollection(polygons, cmap=cmap)
        
        if color is None: # use cmap 
            collection.set_facecolor(plt.cm.get_cmap(cmap)(values[key]))
        else: # use alpha
            collection.set_alpha(alpha[key])
            collection.set_color(color)

        collections[key] = ax.add_collection(collection)

    return collections,  ax


def update_element_value_2d_tri(
                        img:PolyCollection,
                        values:torch.Tensor|np.ndarray,
                        alpha:Optional[float|torch.Tensor|np.ndarray]=None,
                        )->PolyCollection:
    """
    Parameters:
    -----------
        img: PolyCollection
        values: torch.Tensor or np.ndarray
            [n_elements]
        alpha: float or torch.Tensor or np.ndarray
            [n_elements]
            should be greater or equal 0
    """
    # assertion
    assert dim(values) == 1
    assert img.get_array().shape[0] == values.shape[0]
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        assert alpha.shape[0] == values.shape[0]
        assert (alpha >= 0).all()

    # to numpy 
    values = as_ndarray(values)
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        alpha = as_ndarray(alpha)

    img.set_array(values)
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        img.set_alpha(alpha)

    return img

def update_element_value_2d(
                        collections:Dict[str,PolyCollection],
                        values:Dict[str,torch.Tensor|np.ndarray],
                        alpha:float|Dict[str,torch.Tensor|np.ndarray]=1.0,
                        )->List[PolyCollection]:
    """
    Parameters
    ----------
        collections: List[PolyCollection]
        values: Dict[str, torch.Tensor or np.ndarray]
            [n_elements]
        alpha: float or torch.Tensor or np.ndarray
            [n_elements]
            should be greater or equal 0 and less or equal than 1
    """
    # assertion
    for i in range(len(collections)):
        key = list(values.keys())[i]
        assert dim(values[key]) == 1
        assert collections[i].get_array().shape[0] == values[key].shape[0]
        if isinstance(alpha, dict):
            assert alpha[key].shape[0] == values[key].shape[0]
            assert (alpha[key] >= 0).all()
            assert (alpha[key] <= 1).all()
    if isinstance(alpha, float):
        assert 0 <= alpha <= 1

    # to numpy
    values = {key:as_ndarray(values[key]) for key in values.keys()}
    if isinstance(alpha, dict):
        alpha = {key:as_ndarray(alpha[key]) for key in alpha.keys()}
    else:
        alpha = {key:alpha for key in values.keys()}

    for i,key in enumerate(collections.keys()):
        collections[key].set_alpha(alpha[key])
        collections[key].set_array(values[key])
    return keys