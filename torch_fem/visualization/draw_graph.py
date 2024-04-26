
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon, Arc
from matplotlib import patches
from typing import Optional
from ..sparse import SparseMatrix
from .utils import ScipySparseMatrix, as_ndarray, as_sparse_matrix, dim

def draw_graph_2d(sparse_matrix:SparseMatrix|ScipySparseMatrix,
                  points:torch.Tensor|np.ndarray, 
                  draw_points:bool = True, 
                  point_color:str  = 'orange',
                  color:str = "blue",
                  linewidth:int = 3,
                  ax:Optional[plt.Axes] = None)->plt.Axes:
    """
    Parameters
    ----------
    sparse_matrix: torch_fem.sparse.SparseMatrix|ScipySparseMatrix
        the sparse matrix
    points: torch.Tensor|np.ndarray
        2D tensor of shape [n_points, 2]
        the points of the mesh
    color: str, optional
        the color of the edge, default is "blue"
    linewidth: int, optional
        the width of the edge, default is 3
    ax: matplotlib.axes.Axes, optional
        the axis, default is None

    Returns
    -------
    ax: matplotlib.axes.Axes
        the axis
    """

    # assertion
    assert dim(points) == 2, f"points.dim() must be 2, but got {dim(points)}"
    assert points.shape[1] == 2, f"points.shape[1] must be 2, but got {points.shape[1]}"
    n_points = points.shape[0]
    assert sparse_matrix.shape == (n_points, n_points), f"sparse_matrix.shape must be ({n_points}, {n_points}), but got {sparse_matrix.shape}"

    # input prepare
    points_np     = as_ndarray(points)
    sparse_matrix = as_sparse_matrix(sparse_matrix)
    ax            = plt.subplots(figsize=(10,10))[1] if ax is None else ax

    where_self_loop = sparse_matrix.row == sparse_matrix.col
    edges           = sparse_matrix.edges[:, ~where_self_loop]
    self_loops      = sparse_matrix.edges[:, where_self_loop]
    edges_np        = edges.detach().cpu().numpy()
    self_loops_np   = self_loops.detach().cpu().numpy()
    lines           = LineCollection([points_np[edges_np.T]], color=color, linewidth=linewidth)
    arcs            = []
    for loop in self_loops_np:
        arcs.append(Arc((points_np[loop,0], points_np[loop,1]), 0.02, 0.02, 0, 0, 360, color=color, linewidth=linewidth))
    arcs            = PatchCollection(arcs, match_original=True)

    ax.add_collection(lines)
    ax.add_collection(arcs)

    if draw_points:
        ax.scatter(points_np[:, 0], points[:, 1], c=point_color)

    return ax

def draw_graph(sparse_matrix, points, ax=None):
    r"""
    Parameters
    ----------
    sparse_matrix: torch_fem.sparse.SparseTensor
        the sparse matrix
    points: torch.Tensor
        2D tensor of shape :math:`[|\mathcal V|, 2]`, where  :math:`|\mathcal V|` is the number of vertices

    Returns
    -------
    ax: matplotlib.axes.Axes
        the axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    if isinstance(points, torch.Tensor):
        points = points.numpy()
    pos = points.numpy()
    row, col = sparse_matrix.row, sparse_matrix.col
    if isinstance(row, torch.Tensor):
        row, col = row.numpy(), col.numpy()
    fig, ax = plt.subplots(figsize=(10,10))
    lines = []
    selfloops = []
    diameter = 0.02
    for (u, v) in zip(row, col):
        if u == v:
            selfloops.append(Arc((pos[u,0],pos[u,1]+diameter/2), diameter, diameter, 0, 0, 360, color="black", linewidth=0.5))
        else:
            line = (pos[u], pos[v])
            lines.append(line)
    lc = LineCollection(lines, color="black", linewidth=0.5)
    loops = PatchCollection(selfloops, match_original=True)
    ax.add_collection(lc)
    ax.add_collection(loops)

    return ax