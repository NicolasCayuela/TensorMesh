import torch 
from typing import Tuple


def _vnorm(vec:torch.Tensor, dim=-1)->torch.Tensor:
    """vector normalize
    Parameters
    ----------
        vec : torch.Tensor
            2D Tensor of shape (n_points, 2)
    Returns
    -------
        normalized_vec : torch.Tensor
            2D Tensor of shape (n_points, 2)
    """
    norm = torch.norm(vec, dim=dim, keepdim=True)
    return vec / norm

def _vdot(a:torch.Tensor, b:torch.Tensor)->torch.Tensor:
    """vector dot product
    Parameters
    ----------
        a : torch.Tensor
            2D Tensor of shape (n_points, 2)
        b : torch.Tensor
            2D Tensor of shape (n_points, 2)
    Returns
    -------
        dot : torch.Tensor
            1D Tensor of shape (n_points, )
    """
    return torch.einsum("ij,ij->i", a, b)

def _vrot90(vec:torch.Tensor)->torch.Tensor:
    """for 2d vectors rotate 90 degree
    Parameters
    ----------
        vec : torch.Tensor
            2D Tensor of shape (n_points, 2)
    Returns
    -------
        rotated_vec : torch.Tensor
            2D Tensor of shape (n_points, 2)
    """
    assert vec.shape[-1] == 2
    return torch.stack([-vec[:, 1], vec[:, 0]], dim=1)

def outwards_normal_2d(points:torch.Tensor, edges:torch.Tensor)->torch.Tensor:
    """compute the outwards normal for 2d convex polygons
    Parameters
    ----------
        points : torch.Tensor
            2D Tensor of shape (n_points, 2)
        edges  : torch.Tensor
            2D Tensor of shape (n_edges, 2)
        is_right_hand: torch.Tensor
            1D Tensor of shape (n_edges, )
            if the outward normal is right-handed
    Returns
    -------
        normals : torch.Tensor
            2D Tensor of shape (n_edges, 2)
    """
    edge_coords = points[edges] # shape (n_edges, 2, 2)
    edge_vec    = edge_coords[:, 1] - edge_coords[:, 0] # shape (n_edges, 2)
    inner_point = points.mean(dim=0, keepdim=True) # shape (1, 2) 
    inner_vec   = inner_point - edge_coords[:,0] # shape (n_edges, 2)
    normals     = _vrot90(edge_vec) # shape (n_edges, 2)
    is_inwards  = _vdot(normals, inner_vec) < 0 # shape (n_edges, )
    normals[is_inwards] *= -1
    return _vnorm(normals)

def outwards_normal_3d(points:torch.Tensor, faces:Tuple[Tuple[int,...],...])->torch.Tensor:
    """compute the outwards normal for 3d convex polyhedrons
    Parameters
    ----------
        points: torch.Tensor
            2D Tensor of shape (n_points, dim)
        faces: Tuple[Tuple[int,...],...]
            Tuple of Tuple of int, each tuple is a face
    Returns
    -------
        normals: torch.Tensor
            2D Tensor of shape (n_faces, dim)
    """
    tri_faces    = torch.tensor([face[:3] for face in faces]) # [n_faces, 3]
    faces_coords = points[tri_faces, :] # [n_faces, 3, dim]
    vec1 = faces_coords[:, 1] - faces_coords[:, 0] # [n_faces, dim]
    vec2 = faces_coords[:, 2] - faces_coords[:, 0] # [n_faces, dim]
    normals = torch.cross(vec1, vec2) # [n_faces, dim]
    mid_point  = faces_coords.mean(dim=1) # [n_faces, dim]
    inner_point= points.mean(dim=0, keepdim=True) # [1, dim]
    inner_vec  = inner_point - mid_point # [n_faces, dim]
    is_inwards = _vdot(inner_vec, normals) < 0 # [n_faces]
    normals[is_inwards] *= -1
    return _vnorm(normals)


