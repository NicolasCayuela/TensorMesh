"""
Wave Extrapolation Comparison - Aligned with Graph-Galerkin-Learning

This script compares different loss functions for wave equation training:
- Galerkin: Finite Element weak form residual loss
- FDM: Graph-based finite difference Laplacian loss  
- Data: Supervised MSE loss
- PINN: Physics-Informed Neural Network loss

Key alignment with Graph-Galerkin-Learning:
- Uses MultiAnalytical initial conditions (square domain, sinusoidal modes)
- WaveEquation Galerkin loss uses proper weak form
- FDM uses gradient regression-based Laplacian
- Model architecture: FrequencyMLPEncoder + SAGE/SIGN + FrequencyMLPDecoder
- Hyperparameters from config: c=4.0, dt=0.0005, K=6, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
from matplotlib.lines import Line2D
import seaborn as sns
import os
from tqdm import tqdm
import torch_geometric.nn as gnn
import torch_geometric.utils as pyg_utils
import gmsh
import meshio
import scipy.sparse
import scipy.sparse.linalg

# Consistent plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# --- Mesh Generation (Gmsh) - Square Domain ---

class GmshGen:
    @staticmethod
    def gen_tri_square(side=1., center=(0.5, 0.5), chara_length=0.08):
        """Generate triangular mesh on unit square [0,1]x[0,1]"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Create square
        x0, y0 = 0., 0.
        rect = gmsh.model.occ.addRectangle(x0, y0, 0, side, side)
        gmsh.model.occ.synchronize()
        
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(node_coords).reshape(-1, 3)[:, :2]
        
        element_types, element_tags, node_tags_list = gmsh.model.mesh.getElements(dim=2)
        cells = None
        for i, t in enumerate(element_types):
            if t == 2:  # Triangle
                cells = np.array(node_tags_list[i]).reshape(-1, 3) - 1
                break
        
        # Boundary mask: nodes on the boundary of [0,1]x[0,1]
        tol = chara_length / 2
        boundary_mask = (
            (np.abs(points[:, 0] - 0) < tol) |
            (np.abs(points[:, 0] - side) < tol) |
            (np.abs(points[:, 1] - 0) < tol) |
            (np.abs(points[:, 1] - side) < tol)
        )
        
        mesh = meshio.Mesh(
            points=points.astype(np.float64), 
            cells={'triangle': cells}
        )
        mesh.point_data = {
            'boundary_mask': boundary_mask,
            'boundary_value': np.zeros(len(points), dtype=np.float64)
        }
        
        gmsh.finalize()
        return mesh

    @staticmethod
    def gen_tri_circle(radius=1., center=(0., 0.), chara_length=0.08):
        """Generate triangular mesh on circular domain"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        disk = gmsh.model.occ.addDisk(center[0], center[1], 0, radius, radius)
        gmsh.model.occ.synchronize()
        
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)
        gmsh.model.mesh.generate(2)
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.array(node_coords).reshape(-1, 3)[:, :2]
        
        element_types, element_tags, node_tags_list = gmsh.model.mesh.getElements(dim=2)
        cells = None
        for i, t in enumerate(element_types):
            if t == 2:
                cells = np.array(node_tags_list[i]).reshape(-1, 3) - 1
                break
                
        dist = np.linalg.norm(points - np.array(center), axis=1)
        boundary_mask = np.isclose(dist, radius, atol=chara_length/2)
        
        mesh = meshio.Mesh(
            points=points.astype(np.float64), 
            cells={'triangle': cells}
        )
        mesh.point_data = {
            'boundary_mask': boundary_mask,
            'boundary_value': np.zeros(len(points), dtype=np.float64)
        }
        
        gmsh.finalize()
        return mesh


# --- Initial Condition (MultiAnalytical - aligned with Graph-Galerkin-Learning) ---

class MultiAnalytical:
    """Exactly aligned with Graph-Galerkin-Learning/Dataset/wave_generator.py"""
    
    @staticmethod
    def initial_condition(points, a, r=0.5):
        """Generate wave initial condition using sinusoidal modes (vectorized, as original)
        
        Parameters:
            points: np.ndarray (n_points, 2)
            a: np.ndarray (K, K) or (N, K, K) - random coefficients
            r: float - decay exponent
            
        Returns:
            u0: np.ndarray (n_points,) or (N, n_points)
            v0: np.ndarray (n_points,) or (N, n_points)
        """
        K = a.shape[-1]
        j, i = np.meshgrid(np.arange(1, K+1), np.arange(1, K+1))  # (K, K)
        
        if len(a.shape) == 2:
            a = a[None, :, :]  # (1, K, K)
            i, j = i[None, :, :], j[None, :, :]  # (1, K, K)
            x, y = points[:, 0][:, None, None], points[:, 1][:, None, None]  # (n_points, 1, 1)
        else:
            a = a[:, None, :, :]  # (N, 1, K, K)
            i, j = i[None, None, :, :], j[None, None, :, :]  # (1, 1, K, K)
            x, y = points[:, 0][None, :, None, None], points[:, 1][None, :, None, None]
        
        u0 = np.pi / K / K * (a * (i*i + j*j)**(-r) * np.sin(np.pi * i * x) * np.sin(np.pi * j * y)).sum((-2, -1))
        
        # v0 is ones in original code (not zeros!)
        v0 = np.ones(u0.shape)
        
        # Squeeze if single sample
        if u0.ndim > 1 and u0.shape[0] == 1:
            u0 = u0.squeeze(0)
            v0 = v0.squeeze(0)
        
        return u0, v0
    
    @staticmethod
    def solution(points, a, r=0.5, c=1.0, t=0.1):
        """Analytical solution at time t (vectorized, as original)"""
        K = a.shape[-1]
        j, i = np.meshgrid(np.arange(1, K+1), np.arange(1, K+1))  # (K, K)
        
        if len(a.shape) == 2:
            a = a[None, :, :]  # (1, K, K)
            i, j = i[None, :, :], j[None, :, :]  # (1, K, K)
            x, y = points[:, 0][:, None, None], points[:, 1][:, None, None]  # (n_points, 1, 1)
        else:
            a = a[:, None, :, :]  # (N, 1, K, K)
            i, j = i[None, None, :, :], j[None, None, :, :]  # (1, 1, K, K)
            x, y = points[:, 0][None, :, None, None], points[:, 1][None, :, None, None]
        
        u0 = np.pi / K / K * (a * (i*i + j*j)**(-r) * np.sin(np.pi * i * x) * np.sin(np.pi * j * y) * np.cos(c * np.pi * t * np.sqrt(i*i + j*j))).sum((-2, -1))
        
        if u0.ndim > 1 and u0.shape[0] == 1:
            u0 = u0.squeeze(0)
        
        return u0
    
    @staticmethod
    def generate_trajectory(points, a, c=4.0, T=0.1, dt=0.0005, r=0.5):
        """Generate full trajectory using analytical solution"""
        n_steps = int(T / dt) + 1
        Us = []
        for step in range(n_steps):
            t = step * dt
            u = MultiAnalytical.solution(points, a, c=c, t=t, r=r)
            Us.append(u)
        return np.stack(Us, axis=0)


class GaussianInitial:
    """Gaussian initial condition generator for general domains (e.g., circle)"""
    
    @staticmethod
    def initial_condition(points, num_centers=(2, 6), seed=None):
        """Generate Gaussian mixture initial condition
        
        Parameters:
            points: np.ndarray (n_points, 2)
            num_centers: tuple (min, max) number of Gaussian centers
            seed: random seed
            
        Returns:
            u0: np.ndarray (n_points,)
            v0: np.ndarray (n_points,) - zero initial velocity
        """
        if seed is not None:
            np.random.seed(seed)
            
        u0 = np.zeros(len(points), dtype=np.float64)
        n = np.random.randint(num_centers[0], num_centers[1])
        
        # Get domain bounds
        x_min, y_min = points.min(axis=0)[:2]
        x_max, y_max = points.max(axis=0)[:2]
        
        centers = []
        for _ in range(n):
            while True:
                # Random center in interior
                xc = x_min + (x_max - x_min) * np.random.uniform(0.2, 0.8)
                yc = y_min + (y_max - y_min) * np.random.uniform(0.2, 0.8)
                s = np.random.uniform(0.039, 0.156)
                
                # Check distance from other centers
                valid = True
                for (xc_i, yc_i, s_i) in centers:
                    if np.sqrt((xc - xc_i)**2 + (yc - yc_i)**2) < 2 * s_i:
                        valid = False
                        break
                if valid:
                    centers.append((xc, yc, s))
                    break
        
        for xc, yc, s in centers:
            u0 += np.exp(-((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) / (2 * s**2))
        
        v0 = np.zeros_like(u0)
        return u0, v0


# --- FEM Discretization ---

def tri_gauss_points(ngp=1):
    """1-point Gauss quadrature on reference triangle"""
    if ngp == 1:
        return np.array([[0.5, 1/3, 1/3]])  # [weight, xi, eta]
    raise NotImplementedError

def tri3(xi, eta, coords):
    """Linear triangle shape functions
    
    Parameters:
        xi, eta: quadrature points in reference space
        coords: [n_elem, 3, 2] physical coordinates
        
    Returns:
        N: [n_quad, 3] shape function values
        dN_dx: [n_elem, n_quad, 3, 2] shape function gradients in physical space
        detJ: [n_elem, n_quad] Jacobian determinants
    """
    n_elem = coords.shape[0]
    n_quad = len(xi)
    
    # Shape function values at quadrature points
    N = np.stack([1 - xi - eta, xi, eta], axis=-1)  # [n_quad, 3]
    
    # Shape function gradients in reference space (constant for linear triangle)
    dN_dxi = np.array([[-1, -1], [1, 0], [0, 1]])  # [3, 2]
    
    # Jacobian: J[i,j] = d(x_i)/d(xi_j)
    # For linear triangle: J = [x2-x1, x3-x1; y2-y1, y3-y1]
    J = np.zeros((n_elem, 2, 2))
    J[:, 0, 0] = coords[:, 1, 0] - coords[:, 0, 0]  # dx/dxi
    J[:, 0, 1] = coords[:, 2, 0] - coords[:, 0, 0]  # dx/deta
    J[:, 1, 0] = coords[:, 1, 1] - coords[:, 0, 1]  # dy/dxi
    J[:, 1, 1] = coords[:, 2, 1] - coords[:, 0, 1]  # dy/deta
    
    detJ = np.abs(np.linalg.det(J))  # [n_elem]
    detJ = np.maximum(detJ, 1e-12)
    invJ = np.linalg.inv(J)  # [n_elem, 2, 2]
    
    # Physical gradients: dN/dx = invJ.T @ dN/dxi
    dN_dx = np.einsum('id,edj->eij', dN_dxi, invJ)  # [n_elem, 3, 2]
    dN_dx = dN_dx[:, None, :, :]  # [n_elem, 1, 3, 2] for n_quad=1
    detJ = detJ[:, None]  # [n_elem, 1]
    
    return N, dN_dx, detJ


# --- FEM Solver (Ground Truth) ---

class FEMSolver:
    """FEM wave equation solver aligned with Graph-Galerkin-Learning"""
    
    @staticmethod
    def solve_wave(mesh, u0, v0, c, T, dt, recording=True, verbose=False):
        """Solve wave equation using FEM time stepping
        
        Wave equation: u_tt = c^2 * laplacian(u)
        Time discretization: (u_{n+1} - 2*u_n + u_{n-1})/dt^2 = c^2 * laplacian(u_n)
        """
        points = mesh.points[:, :2] if mesh.points.shape[1] > 2 else mesh.points
        cells = mesh.cells_dict['triangle']
        n_points = len(points)
        
        boundary_mask = mesh.point_data.get('boundary_mask', np.zeros(n_points, dtype=bool))
        boundary_value = mesh.point_data.get('boundary_value', np.zeros(n_points, dtype=np.float64))
        
        # Quadrature
        qpoints = tri_gauss_points(ngp=1)
        w, xi, eta = qpoints[:, 0], qpoints[:, 1], qpoints[:, 2]
        N, dN_dx, detJ = tri3(xi, eta, points[cells])
        
        JxW = detJ * w  # [n_elem, n_quad]
        
        # Element matrices
        # Mass: M_ij = int(N_i * N_j) dA
        M_e = np.einsum('qi,qj,eq->eij', N, N, JxW)  # [n_elem, 3, 3]
        
        # Stiffness: K_ij = c^2 * int(grad(N_i) . grad(N_j)) dA
        K_e = c**2 * np.einsum('eqia,eqja,eq->eij', dN_dx, dN_dx, JxW)  # [n_elem, 3, 3]
        
        # Assembly
        n_elem, n_basis = cells.shape
        rows, cols = [], []
        for i in range(n_basis):
            for j in range(n_basis):
                rows.append(cells[:, i])
                cols.append(cells[:, j])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        
        M = scipy.sparse.coo_matrix((M_e.flatten(), (rows, cols)), shape=(n_points, n_points)).tocsr()
        K = scipy.sparse.coo_matrix((K_e.flatten(), (rows, cols)), shape=(n_points, n_points)).tocsr()
        
        # Boundary condensation
        inner_mask = ~boundary_mask
        inner_idx = np.where(inner_mask)[0]
        outer_idx = np.where(boundary_mask)[0]
        
        M_ii = M[np.ix_(inner_idx, inner_idx)]
        K_ii = K[np.ix_(inner_idx, inner_idx)]
        M_io = M[np.ix_(inner_idx, outer_idx)]
        K_io = K[np.ix_(inner_idx, outer_idx)]
        
        # Initialize
        u0 = u0.copy().astype(np.float64)
        v0 = v0.copy().astype(np.float64)
        u0[boundary_mask] = boundary_value[boundary_mask]
        
        Us = [u0.copy()]
        
        # First step: u1 = u0 + dt*v0 + 0.5*dt^2*a0
        # where a0 = c^2 * laplacian(u0) => M*a0 = -K*u0 => a0 = -M^{-1}*K*u0
        F = M @ (u0 + dt * v0) - 0.5 * dt**2 * K @ u0
        F_i = F[inner_mask] - M_io @ boundary_value[outer_idx]
        
        U = np.zeros(n_points, dtype=np.float64)
        U[inner_mask] = scipy.sparse.linalg.spsolve(M_ii, F_i)
        U[boundary_mask] = boundary_value[boundary_mask]
        Us.append(U.copy())
        
        U_prev = u0.copy()
        U_curr = U.copy()
        
        n_steps = int(T / dt)
        iterator = tqdm(range(2, n_steps + 1), desc="FEM Solver") if verbose else range(2, n_steps + 1)
        
        for _ in iterator:
            # Time stepping: M*u_{n+1} = 2*M*u_n - M*u_{n-1} - dt^2*K*u_n
            F = 2 * M @ U_curr - M @ U_prev - dt**2 * K @ U_curr
            F_i = F[inner_mask] - M_io @ boundary_value[outer_idx]
            
            U_next = np.zeros(n_points, dtype=np.float64)
            U_next[inner_mask] = scipy.sparse.linalg.spsolve(M_ii, F_i)
            U_next[boundary_mask] = boundary_value[boundary_mask]
            
            Us.append(U_next.copy())
            U_prev = U_curr.copy()
            U_curr = U_next.copy()
        
        return np.stack(Us, axis=0)


# --- Model Architecture (aligned with Graph-Galerkin-Learning) ---

class Activation(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        activation = activation.lower()
        if activation == 'relu':
            self.fn = nn.ReLU()
        elif activation == 'tanh':
            self.fn = nn.Tanh()
        elif activation == 'gelu':
            self.fn = nn.GELU()
        elif activation == 'swish':
            self.fn = nn.SiLU()
        else:
            self.fn = nn.ReLU()
    
    def forward(self, x):
        return self.fn(x)


class MLP(nn.Module):
    """MLP aligned with Graph-Galerkin-Learning"""
    def __init__(self, num_features, num_classes, num_hidden=64, num_layers=3, 
                 activation="relu", res=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(num_features, num_hidden)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_classes))
        self.activation = Activation(activation)
        self.linear = nn.Linear(num_features, num_classes) if res else None
        self.num_features = num_features
        self.num_classes = num_classes
    
    def forward(self, x):
        inp = x
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.linear is not None:
            x = x + self.linear(inp)
        return x


class FrequencyMLPEncoder(nn.Module):
    """Frequency encoding aligned with Graph-Galerkin-Learning"""
    def __init__(self, num_features, num_classes, L=4, num_hidden=64, num_layers=2,
                 activation="relu", res=False):
        super().__init__()
        self.L = L
        # Input dim: num_features * (1 + 2*(L-1) + 2*(L-1)) = num_features * (4*L - 1)
        input_dim = num_features * (4 * L - 1)
        self.mlp = MLP(input_dim, num_classes, num_hidden, num_layers, activation, res)
        self.num_features = num_features
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Parameters:
            x: [..., n_node, num_features]
        Returns:
            [..., n_node, num_classes]
        """
        features = [x]
        # Low frequencies: 1/L, 1/(L-1), ..., 1/1
        for i in range(self.L, 0, -1):
            omega = 1.0 / i
            features.append(torch.cos(omega * x))
            features.append(torch.sin(omega * x))
        # High frequencies: 2, 3, ..., L
        for i in range(2, self.L + 1):
            omega = float(i)
            features.append(torch.cos(omega * x))
            features.append(torch.sin(omega * x))
        x = torch.cat(features, dim=-1)
        return self.mlp(x)


class GSAGE(nn.Module):
    """GraphSAGE aligned with Graph-Galerkin-Learning"""
    def __init__(self, num_features, num_classes, num_hidden=64, num_layers=3, activation="relu"):
        super().__init__()
        self.layers = nn.ModuleList([gnn.SAGEConv(num_features, num_hidden, aggr='mean')])
        for _ in range(num_layers - 2):
            self.layers.append(gnn.SAGEConv(num_hidden, num_hidden, aggr='mean'))
        self.layers.append(gnn.SAGEConv(num_hidden, num_classes, aggr='mean'))
        self.activation = Activation(activation)
        self.num_features = num_features
        self.num_classes = num_classes
    
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, edge_index))
        return self.layers[-1](x, edge_index)


class SIGN(nn.Module):
    """SIGN GNN aligned with Graph-Galerkin-Learning"""
    def __init__(self, num_features, num_classes, num_hidden=64, num_layers=3, num_hops=8, activation="relu"):
        super().__init__()
        self.props = nn.ModuleList([gnn.SGConv(num_features, num_hidden, K=k) for k in range(1, num_hops + 1)])
        self.branches = nn.ModuleList(
            [MLP(num_features, num_hidden, num_hidden, num_layers, activation, res=True)] +
            [MLP(num_hidden, num_hidden, num_hidden, num_layers - 1, activation, res=True) for _ in range(num_hops)]
        )
        self.merger = MLP(num_hidden * (num_hops + 1), num_classes, num_hidden, num_layers, activation)
        self.num_features = num_features
        self.num_classes = num_classes
    
    def forward(self, x, edge_index):
        props = [x] + [prop(x, edge_index) for prop in self.props]
        branches = [branch(prop) for prop, branch in zip(props, self.branches)]
        x = torch.cat(branches, dim=-1)
        return self.merger(x)


class GNNPipeline(nn.Module):
    """GNN Pipeline: Encoder -> Processor -> Decoder (aligned with Graph-Galerkin-Learning)"""
    def __init__(self, encoder, processor, decoder, use_input_norm=True):
        super().__init__()
        num_features = encoder.num_features if hasattr(encoder, 'num_features') else processor.num_features
        self.input_norm = nn.BatchNorm1d(num_features) if use_input_norm else None
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
    
    def forward(self, x, edge_index):
        if self.input_norm is not None:
            shape = x.shape
            x = self.input_norm(x.reshape(-1, x.shape[-1]))
            x = x.view(*shape)
        x = self.encoder(x)
        x = self.processor(x, edge_index)
        x = self.decoder(x)
        return x


def build_wave_model(mesh, window_size=4, n_hidden=64, n_layers=3, num_hops=8, 
                     encoder_frequency=4, decoder_frequency=4, activation='relu'):
    """Build wave model aligned with Graph-Galerkin-Learning config"""
    # Input: [u_1, v_1, u_2, v_2, ..., u_ws, v_ws] = window_size*2
    # Output: window_size (acceleration at each sub-step)
    input_dim = window_size * 2
    output_dim = window_size
    
    encoder = FrequencyMLPEncoder(input_dim, n_hidden, L=encoder_frequency, 
                                   num_hidden=n_hidden, num_layers=2, activation=activation)
    processor = GSAGE(n_hidden, n_hidden, num_hidden=n_hidden, num_layers=n_layers, activation=activation)
    decoder = FrequencyMLPEncoder(n_hidden, output_dim, L=decoder_frequency,
                                   num_hidden=n_hidden, num_layers=2, activation=activation)
    
    return GNNPipeline(encoder, processor, decoder, use_input_norm=True)


# --- Wave Equation Loss ---

class WaveEquation(nn.Module):
    """Wave equation with Galerkin loss aligned with Graph-Galerkin-Learning"""
    
    def __init__(self, mesh, c=4.0, dt=0.0005, dtype=torch.float64):
        super().__init__()
        self.dt = dt
        self.c = c
        self.dtype = dtype
        
        points = mesh.points[:, :2] if mesh.points.shape[1] > 2 else mesh.points
        cells = mesh.cells_dict['triangle']
        n_points = len(points)
        n_elem = len(cells)
        
        # Quadrature
        qpoints = tri_gauss_points(ngp=1)
        w, xi, eta = qpoints[:, 0], qpoints[:, 1], qpoints[:, 2]
        N, dN_dx, detJ = tri3(xi, eta, points[cells])
        
        JxW = (np.abs(detJ) * w)[..., None]  # [n_elem, n_quad, 1]
        
        # Store for residual computation
        self.register_buffer('shape_val', torch.from_numpy(N).type(dtype))  # [n_quad, 3]
        self.register_buffer('shape_grad', torch.from_numpy(dN_dx).type(dtype))  # [n_elem, n_quad, 3, 2]
        self.register_buffer('JxW', torch.from_numpy(JxW).type(dtype))  # [n_elem, n_quad, 1]
        self.register_buffer('elements', torch.from_numpy(cells).long())
        self.register_buffer('points', torch.from_numpy(points).type(dtype))
        
        # Boundary
        boundary_mask = mesh.point_data.get('boundary_mask', np.zeros(n_points, dtype=bool))
        boundary_value = mesh.point_data.get('boundary_value', np.zeros(n_points, dtype=np.float64))
        self.register_buffer('boundary_mask', torch.from_numpy(boundary_mask).bool())
        self.register_buffer('boundary_value', torch.from_numpy(boundary_value).type(dtype))
        
        # Build element-to-mesh assembly matrix
        n_basis = 3
        elem_u, elem_v = [], []
        for i in range(n_basis):
            for j in range(n_basis):
                elem_u.append(cells[:, i])
                elem_v.append(cells[:, j])
        elem_u = np.concatenate(elem_u)
        elem_v = np.concatenate(elem_v)
        
        # Create sparse assembly for nodes
        self.n_points = n_points
        self.n_elements = n_elem
        self.n_basis = n_basis
        self.n_quadrature = len(w)
        
        ele2msh_node = scipy.sparse.coo_matrix((
            np.ones(n_elem * n_basis),
            (cells.flatten(), np.arange(n_elem * n_basis))
        ), shape=(n_points, n_elem * n_basis)).tocsr()
        
        # Convert to torch sparse
        row, col = ele2msh_node.tocoo().row, ele2msh_node.tocoo().col
        indices = torch.from_numpy(np.stack([row, col]))
        values = torch.from_numpy(ele2msh_node.tocoo().data).type(dtype)
        self.register_buffer('ele2msh_node_indices', indices)
        self.register_buffer('ele2msh_node_values', values)
        self.ele2msh_node_shape = (n_points, n_elem * n_basis)
    
    def apply_dirichlet_boundary(self, U):
        """Apply Dirichlet boundary conditions"""
        if U.dim() == 1:
            U = U.clone()
            U[self.boundary_mask] = self.boundary_value[self.boundary_mask]
        else:
            U = U.clone()
            U[..., self.boundary_mask] = self.boundary_value[self.boundary_mask]
        return U
    
    def apply_zero_boundary(self, R):
        """Zero out residual at boundary nodes"""
        if R.dim() == 1:
            R = R.clone()
            R[self.boundary_mask] = 0
        else:
            R = R.clone()
            R[..., self.boundary_mask] = 0
        return R
    
    def compute_residual(self, U_t1, U_t2, U_t3):
        """Compute Galerkin residual for wave equation
        
        Weak form: R^I = int((u_{n+1} - 2*u_n + u_{n-1})/dt^2 * N^I + c^2 * grad(u_n) . grad(N^I)) dx
        
        Parameters:
            U_t1: [..., n_points] solution at t_{n-1}
            U_t2: [..., n_points] solution at t_n
            U_t3: [..., n_points] solution at t_{n+1}
            
        Returns:
            R: [..., n_points] residual
        """
        # Apply boundary conditions
        U_t1 = self.apply_dirichlet_boundary(U_t1)
        U_t2 = self.apply_dirichlet_boundary(U_t2)
        U_t3 = self.apply_dirichlet_boundary(U_t3)
        
        # Get element values
        batch_shape = U_t1.shape[:-1]
        
        elemU_t1 = U_t1[..., self.elements]  # [..., n_elem, 3]
        elemU_t2 = U_t2[..., self.elements]
        elemU_t3 = U_t3[..., self.elements]
        
        # Interpolate to quadrature points
        # shape_val: [n_quad, 3], elemU: [..., n_elem, 3]
        U_t1_q = torch.einsum('qb,...eb->...eq', self.shape_val, elemU_t1)  # [..., n_elem, n_quad]
        U_t2_q = torch.einsum('qb,...eb->...eq', self.shape_val, elemU_t2)
        U_t3_q = torch.einsum('qb,...eb->...eq', self.shape_val, elemU_t3)
        
        # Compute gradients at quadrature points
        # shape_grad: [n_elem, n_quad, 3, 2], elemU: [..., n_elem, 3]
        gradU_t2 = torch.einsum('eqbd,...eb->...eqd', self.shape_grad, elemU_t2)  # [..., n_elem, n_quad, 2]
        
        # Second time derivative
        u_tt = (U_t3_q - 2 * U_t2_q + U_t1_q) / (self.dt ** 2)  # [..., n_elem, n_quad]
        
        # Weak form integral for each basis function
        # integral_i = (u_tt * N_i + c^2 * grad_u . grad_N_i) * JxW
        # shape_val: [n_quad, 3], shape_grad: [n_elem, n_quad, 3, 2]
        
        term1 = torch.einsum('...eq,qb,eq->...eb', u_tt, self.shape_val, self.JxW.squeeze(-1))  # [..., n_elem, 3]
        term2 = self.c**2 * torch.einsum('...eqd,eqbd,eq->...eb', gradU_t2, self.shape_grad, self.JxW.squeeze(-1))
        
        integral = term1 + term2  # [..., n_elem, 3]
        
        # Assemble to global
        integral_flat = integral.reshape(*batch_shape, -1)  # [..., n_elem*3]
        
        # Sparse matrix multiply
        ele2msh = torch.sparse_coo_tensor(
            self.ele2msh_node_indices, 
            self.ele2msh_node_values,
            self.ele2msh_node_shape
        ).to(integral_flat.device)
        
        if len(batch_shape) == 0:
            R = torch.sparse.mm(ele2msh, integral_flat.unsqueeze(-1)).squeeze(-1)
        else:
            # Handle batch dimension
            R = torch.einsum('ij,...j->...i', ele2msh.to_dense(), integral_flat)
        
        R = self.apply_zero_boundary(R)
        return R


# --- FDM Loss (Graph-based gradient regression) ---

class WaveFdmLoss(nn.Module):
    """FDM wave loss using graph Laplacian (simplified, more stable)"""
    
    def __init__(self, mesh, edge_index, dt, c=4.0, dtype=torch.float64):
        super().__init__()
        self.dt = dt
        self.c = c
        self.dtype = dtype
        
        points = mesh.points[:, :2] if mesh.points.shape[1] > 2 else mesh.points
        n_points = len(points)
        points_t = torch.from_numpy(points).type(dtype)
        
        # Move edge_index to CPU for preprocessing
        edge_index_cpu = edge_index.cpu()
        edge_index_no_self, _ = pyg_utils.remove_self_loops(edge_index_cpu)
        row, col = edge_index_no_self
        
        # Build graph Laplacian with cotangent-like weights
        # Weight = 1/distance for each edge
        diffs = points_t[row] - points_t[col]
        dists = torch.norm(diffs, dim=-1)
        weights = 1.0 / (dists + 1e-8)
        
        # Normalize weights per node (without torch_scatter dependency)
        weight_sum = torch.zeros(n_points, dtype=dtype)
        weight_sum.scatter_add_(0, row, weights)
        
        weights_norm = weights / (weight_sum[row] + 1e-8)
        
        # Build sparse Laplacian: L[i,j] = w_ij (for i != j), L[i,i] = -1
        # Laplacian(u)[i] = sum_j w_ij * (u[j] - u[i]) = sum_j w_ij * u[j] - u[i]
        
        # Scale factor for discrete Laplacian
        h = dists.mean()
        scale = 2.0 / (h ** 2)
        
        self.register_buffer('edge_index', edge_index_no_self)
        self.register_buffer('weights', weights_norm * scale)
        self.register_buffer('points', points_t)
        
        # Boundary
        boundary_mask = mesh.point_data.get('boundary_mask', np.zeros(n_points, dtype=bool))
        self.register_buffer('boundary_mask', torch.from_numpy(boundary_mask).bool())
        self.n_points = n_points
    
    def laplacian(self, u):
        """Compute graph Laplacian: L(u)[i] = sum_j w_ij * (u[j] - u[i])"""
        row, col = self.edge_index
        
        # u: [..., n_points]
        u_diff = u[..., col] - u[..., row]  # [..., n_edges]
        
        # Weighted sum
        weighted_diff = u_diff * self.weights  # [..., n_edges]
        
        # Scatter to nodes
        lap = torch.zeros_like(u)
        lap.scatter_add_(-1, row.expand_as(weighted_diff), weighted_diff)
        
        return lap
    
    def compute(self, u_next, u_curr, u_prev):
        """Compute FDM residual
        
        Residual: u_tt - c^2 * laplacian(u) = 0
        """
        u_tt = (u_next - 2 * u_curr + u_prev) / (self.dt ** 2)
        lap_u = self.laplacian(u_curr)
        
        residual = u_tt - self.c**2 * lap_u
        
        # Zero out boundary
        residual = residual.clone()
        residual[..., self.boundary_mask] = 0
        
        return (residual ** 2).mean()


# --- PINN Loss ---

class WavePinnLoss(nn.Module):
    """PINN loss for wave equation (aligned with PINNWaveTrainer)"""
    
    def __init__(self, c=4.0, lambda_pde=1.0, lambda_sb=0.1, lambda_init=0.001):
        super().__init__()
        self.c = c
        self.lambda_pde = lambda_pde
        self.lambda_sb = lambda_sb
        self.lambda_init = lambda_init
    
    def compute_pde_residual(self, model, xyt):
        """Compute PDE residual: u_tt - c^2 * (u_xx + u_yy) = 0"""
        xyt = xyt.requires_grad_(True)
        u = model(xyt)
        
        grad_u = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
        u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1]
        u_tt = torch.autograd.grad(u_t.sum(), xyt, create_graph=True)[0][:, 2]
        
        residual = u_tt - self.c**2 * (u_xx + u_yy)
        return residual
    
    def compute(self, model, inp_int, inp_sb, inp_tb, output_tb):
        """Compute PINN loss
        
        Parameters:
            model: PINN model mapping (x, y, t) -> u
            inp_int: interior points [n_int, 3]
            inp_sb: boundary points [n_sb, 3]
            inp_tb: initial condition points [n_tb, 3]
            output_tb: initial condition values [n_tb, 2] (u0, v0)
        """
        # PDE residual loss
        loss_pde = (self.compute_pde_residual(model, inp_int) ** 2).mean()
        
        # Boundary loss (u = 0 on boundary)
        u_sb = model(inp_sb)
        loss_sb = (u_sb ** 2).mean()
        
        # Initial condition loss
        inp_tb = inp_tb.requires_grad_(True)
        u_tb = model(inp_tb)
        loss_ic_u = ((u_tb.squeeze() - output_tb[:, 0]) ** 2).mean()
        
        # Initial velocity loss
        grad_u_tb = torch.autograd.grad(u_tb.sum(), inp_tb, create_graph=True)[0]
        u_t_tb = grad_u_tb[:, 2]
        loss_ic_v = ((u_t_tb - output_tb[:, 1]) ** 2).mean()
        
        loss_ic = loss_ic_u + loss_ic_v
        
        total_loss = self.lambda_pde * loss_pde + self.lambda_sb * loss_sb + self.lambda_init * loss_ic
        
        return torch.log10(total_loss), loss_pde, loss_sb, loss_ic


# --- PINN Model ---

class PINNModel(nn.Module):
    """PINN model for wave equation"""
    
    def __init__(self, hidden_dim=32, num_layers=4, activation='tanh'):
        super().__init__()
        # Input: (x, y, t), Output: u
        layers = [nn.Linear(3, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.extend([Activation(activation), nn.Linear(hidden_dim, hidden_dim)])
        layers.extend([Activation(activation), nn.Linear(hidden_dim, 1)])
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, xyt):
        return self.net(xyt)


# --- Training ---

def build_graph(mesh, chara_length=0.08):
    """Build graph structure from mesh"""
    cells = mesh.cells_dict['triangle']
    n_points = len(mesh.points)
    
    # Build edges from triangles
    edges = set()
    for tri in cells:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((tri[i], tri[j]))
    
    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    
    # Add self-loops
    edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=n_points)
    
    return edge_index


def train_galerkin(mesh, u_gt, device, output_dir, config):
    """Train with Galerkin loss (aligned with LongTermWaveTrainer)
    
    Key alignment with Graph-Galerkin-Learning:
    - Input: [u_1, v_1, u_2, v_2, ..., u_ws, v_ws] (no c appended)
    - Loss computed over multiple time steps within prediction window
    - Uses discount_factor for temporal weighting
    - float64 precision for loss computation
    """
    print("\n--- Training Galerkin ---")
    
    dtype = torch.float64
    window_size = config['window_size']
    dt = config['dt']
    c = config['c']
    epochs = config['epochs']
    n_hidden = config['n_hidden']
    n_samples = config.get('n_samples', 16)  # Multiple initial conditions
    
    # Build model (input: ws*2, output: ws)
    model = build_wave_model(mesh, window_size=window_size, n_hidden=n_hidden,
                             n_layers=config['n_layers'], num_hops=config['num_hops'],
                             encoder_frequency=config['encoder_frequency'],
                             decoder_frequency=config['decoder_frequency'],
                             activation=config['activation'])
    model = model.to(dtype).to(device)
    
    # Build equation
    equation = WaveEquation(mesh, c=c, dt=dt, dtype=dtype)
    equation = equation.to(device)
    
    # Build graph
    edge_index = build_graph(mesh).to(device)
    
    # Points tensor
    points = torch.from_numpy(mesh.points[:, :2]).type(dtype).to(device)
    boundary_mask = torch.from_numpy(mesh.point_data['boundary_mask']).to(device)
    boundary_value = torch.from_numpy(mesh.point_data['boundary_value']).type(dtype).to(device)
    
    n_points = len(mesh.points)
    
    # Generate multiple training samples with different initial conditions
    print(f"Generating {n_samples} training samples...")
    np.random.seed(config.get('seed', 100))
    K = config['K']
    T_train = config['T']  # Training time (half of total)
    
    all_ndata = []  # [n_samples, n_points, ws*2]
    all_labels = []  # [n_samples, n_steps+1, n_points]
    
    for sample_idx in range(n_samples):
        a = np.random.uniform(-1.0, 1.0, (K, K))
        
        # Generate trajectory
        us = MultiAnalytical.generate_trajectory(mesh.points, a, c=c, T=T_train, dt=dt, r=0.5)
        us[:, mesh.point_data['boundary_mask']] = 0
        
        # Compute velocities
        vs = np.zeros_like(us)
        vs[1:-1] = (us[2:] - us[:-2]) / (2 * dt)
        
        # Initial window: [u_0, v_0, u_1, v_1, ..., u_{ws-1}, v_{ws-1}]
        window_u = us[:window_size]  # [ws, n_points]
        window_v = vs[:window_size]
        uv = np.stack([window_u, window_v], axis=-1)  # [ws, n_points, 2]
        ndata = uv.transpose(1, 0, 2).reshape(n_points, -1)  # [n_points, ws*2]
        
        all_ndata.append(ndata)
        all_labels.append(us[window_size-1:])  # [n_steps+1, n_points]
    
    ndata = torch.from_numpy(np.stack(all_ndata)).type(dtype).to(device)  # [n_samples, n_points, ws*2]
    labels = torch.from_numpy(np.stack(all_labels)).type(dtype).to(device)  # [n_samples, n_steps+1, n_points]
    
    # n_steps = number of window_size blocks to predict (aligned with Graph-Galerkin-Learning)
    # In their code: n_steps = len(ts) // window_size where ts = arange(dt, t+dt, dt)
    n_steps = int(T_train / dt) // window_size  # e.g., 200 // 4 = 50
    
    print(f"Training data: ndata {ndata.shape}, labels {labels.shape}, n_steps={n_steps}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], 
                                                  gamma=config['scheduler_gamma'])
    
    history = {"loss": [], "error": []}
    discount_factor = config.get('discount_factor', 1.0)
    
    for epoch in tqdm(range(epochs), desc="Galerkin"):
        model.train()
        optimizer.zero_grad()
        optimizer.zero_grad()
            
        # Sample batch of initial conditions
        batch_size = min(config['batch_size'], n_samples)
        batch_idx = torch.randperm(n_samples)[:batch_size]
        
        uvs = ndata[batch_idx]  # [batch, n_points, ws*2]
        
        # Multi-step prediction (like multisteps function in Graph-Galerkin-Learning)
        recording = [uvs]
        for step in range(n_steps):
            uv_curr = recording[-1]  # [batch, n_points, ws*2]
            
            # Model predicts acceleration for window_size steps
            accel = model(uv_curr, edge_index)  # [batch, n_points, ws]
            
            # Get last u and v from current window
            u_t = uv_curr[:, :, -2]  # [batch, n_points]
            v_t = uv_curr[:, :, -1]  # [batch, n_points]
            
            # Integrate window_size steps
            dt_vec = torch.arange(1, window_size + 1, device=device, dtype=dtype) * dt
            
            # v_{t+k} = v_t + sum(dt * a_i for i=1..k)
            # cumulative acceleration effect
            v_new = v_t.unsqueeze(-1) + dt_vec * accel  # [batch, n_points, ws]
            
            # u_{t+k} = u_t + sum(dt * (v_i + v_{i+1})/2)
            v_prev = torch.cat([v_t.unsqueeze(-1), v_new[:, :, :-1]], dim=-1)
            v_diff = (v_new + v_prev) / 2 * dt
            v_diff_cum = torch.cumsum(v_diff, dim=-1)
            u_new = u_t.unsqueeze(-1) + v_diff_cum  # [batch, n_points, ws]
            
            # Apply boundary conditions
            u_new[:, boundary_mask, :] = 0
            
            # Build next window: take last values and interleave
            # New UV: [u_new, v_new] interleaved -> [u_1, v_1, ..., u_ws, v_ws]
            uv_next = torch.stack([u_new, v_new], dim=-1)  # [batch, n_points, ws, 2]
            uv_next = uv_next.reshape(batch_size, n_points, -1)  # [batch, n_points, ws*2]
            
            recording.append(uv_next)
        
        # Collect all predicted u values
        # recording: list of [batch, n_points, ws*2], length = n_steps + 1
        all_uvs = torch.cat(recording, dim=-1)  # [batch, n_points, (n_steps+1)*ws*2]
        all_uvs = all_uvs.permute(0, 2, 1)  # [batch, (n_steps+1)*ws*2, n_points]
        
        # Extract u (every other starting from 0)
        us = all_uvs[:, ::2, :]  # [batch, (n_steps+1)*ws, n_points]
        
        # Compute residual for consecutive triplets: u_{k-1}, u_k, u_{k+1}
        # Starting from window_size-1 to avoid using initial values directly
        u_t1 = us[:, window_size-1:-2, :]  # [batch, n_pred_steps, n_points]
        u_t2 = us[:, window_size:-1, :]
        u_t3 = us[:, window_size+1:, :]
        
        n_pred_steps = u_t1.shape[1]
        
        # Compute residual for each triplet
        R_list = []
        for t in range(n_pred_steps):
            R = equation.compute_residual(u_t1[:, t], u_t2[:, t], u_t3[:, t])
            R_list.append((R ** 2).mean(dim=-1))  # [batch]
        
        R_all = torch.stack(R_list, dim=-1)  # [batch, n_pred_steps]
        
        # Apply discount factor
        discount_factors = torch.tensor([discount_factor ** i for i in range(n_pred_steps)], 
                                         device=device, dtype=dtype)
        loss = (R_all * discount_factors).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Compute validation error
        with torch.no_grad():
            # Compare predicted u at step window_size with label
            pred_u_check = us[:, window_size, :]  # [batch, n_points]
            label_check = labels[batch_idx, 1, :]  # [batch, n_points]
            error = (pred_u_check - label_check).pow(2).mean().sqrt().item()
        
        history["loss"].append(loss.item())
        history["error"].append(error)
    
    # Rollout on the original u_gt for comparison
    print("Rolling out predictions...")
    model.eval()
    
    u_gt_t = torch.from_numpy(u_gt).type(dtype).to(device) if isinstance(u_gt, np.ndarray) else u_gt.type(dtype).to(device)
    n_total_steps = u_gt_t.shape[0]
    
    vs_gt = torch.zeros_like(u_gt_t)
    vs_gt[1:-1] = (u_gt_t[2:] - u_gt_t[:-2]) / (2 * dt)
    
    preds = list(u_gt_t[:window_size].cpu().numpy())
    
    with torch.no_grad():
        u_hist = u_gt_t[:window_size].clone()
        v_hist = vs_gt[:window_size].clone()
        
        for t in range(window_size, n_total_steps):
            # Build input: [u_1, v_1, ..., u_ws, v_ws]
            window_u = u_hist[-window_size:]
            window_v = v_hist[-window_size:]
            uv = torch.stack([window_u, window_v], dim=-1)  # [ws, n_points, 2]
            uv = uv.permute(1, 0, 2).reshape(n_points, -1)  # [n_points, ws*2]
            inp = uv.unsqueeze(0)  # [1, n_points, ws*2]
            
            accel = model(inp, edge_index)
            
            # Take first predicted step
            u_t = u_hist[-1]
            v_t = v_hist[-1]
            a_step = accel[0, :, 0]
            v_new = v_t + dt * a_step
            u_new = u_t + dt * (v_t + v_new) / 2
            u_new[boundary_mask] = 0
            
            preds.append(u_new.cpu().numpy())
            u_hist = torch.cat([u_hist[1:], u_new.unsqueeze(0)])
            v_hist = torch.cat([v_hist[1:], v_new.unsqueeze(0)])
    
    preds = np.stack(preds)
    np.savez(os.path.join(output_dir, "result_Galerkin.npz"), preds=preds, 
             loss=history["loss"], error=history["error"])
    
    return history


def train_fdm(mesh, u_gt, device, output_dir, config):
    """Train with FDM loss (aligned with FDMWaveTrainer)"""
    print("\n--- Training FDM ---")
    
    dtype = torch.float64
    window_size = config['window_size']
    dt = config['dt']
    c = config['c']
    epochs = config.get('fdm_epochs', config['epochs'])
    
    # Build model (input: ws*2, output: ws)
    model = build_wave_model(mesh, window_size=window_size, n_hidden=config['n_hidden'],
                             n_layers=config['n_layers'], num_hops=config['num_hops'],
                             encoder_frequency=config['encoder_frequency'],
                             decoder_frequency=config['decoder_frequency'],
                             activation=config['activation'])
    model = model.to(dtype).to(device)
    
    # Build graph and FDM loss
    edge_index = build_graph(mesh).to(device)
    fdm_loss = WaveFdmLoss(mesh, edge_index, dt=dt, c=c, dtype=dtype)
    fdm_loss = fdm_loss.to(device)
    
    # Points and boundary
    n_points = len(mesh.points)
    boundary_mask = torch.from_numpy(mesh.point_data['boundary_mask']).to(device)
    
    # Prepare data
    u_gt_t = torch.from_numpy(u_gt).type(dtype).to(device) if isinstance(u_gt, np.ndarray) else u_gt.type(dtype).to(device)
    n_steps = u_gt_t.shape[0]
    
    vs = torch.zeros_like(u_gt_t)
    vs[1:-1] = (u_gt_t[2:] - u_gt_t[:-2]) / (2 * dt)
    
    n_train = int(n_steps * 0.5)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('fdm_scheduler_step_size', 1000),
                                                  gamma=config['scheduler_gamma'])
    
    history = {"loss": [], "error": []}
    
    for epoch in tqdm(range(epochs), desc="FDM"):
        model.train()
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()
        
        batch_size = config['batch_size']
        valid_start = window_size
        valid_end = n_train - window_size
        if valid_end <= valid_start:
            valid_end = valid_start + 1
        
        idx = torch.randint(valid_start, valid_end, (batch_size,), device=device)
        
        inputs = []
        for i in idx:
            window_u = u_gt_t[i-window_size+1:i+1]
            window_v = vs[i-window_size+1:i+1]
            uv = torch.stack([window_u, window_v], dim=-1)
            uv = uv.permute(1, 0, 2).reshape(n_points, -1)  # [n_points, ws*2]
            inputs.append(uv)
        
        inputs = torch.stack(inputs)  # [batch, n_points, ws*2]
        accel = model(inputs, edge_index)
        
        u_t = inputs[:, :, -2]  # Last u
        v_t = inputs[:, :, -1]  # Last v
        
        pred_u = []
        for step in range(window_size):
            a_step = accel[:, :, step]
            v_new = v_t + dt * a_step
            u_new = u_t + dt * (v_t + v_new) / 2
            u_new[:, boundary_mask] = 0
            pred_u.append(u_new)
            u_t = u_new
            v_t = v_new
        
        pred_u = torch.stack(pred_u, dim=-1)  # [batch, n_points, ws]
        
        # FDM loss on consecutive triplets
        u_prev = pred_u[:, :, -3] if window_size >= 3 else inputs[:, :, -4]
        u_curr = pred_u[:, :, -2] if window_size >= 2 else inputs[:, :, -2]
        u_next = pred_u[:, :, -1]
        
        loss = fdm_loss.compute(u_next, u_curr, u_prev)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            error = (pred_u[:, :, -1] - u_gt_t[idx + 1]).pow(2).mean().sqrt().item()
        
        history["loss"].append(loss.item())
        history["error"].append(error)
        
        # Rollout
    model.eval()
    preds = list(u_gt_t[:window_size].cpu().numpy())
    
    with torch.no_grad():
        u_hist = u_gt_t[:window_size].clone()
        v_hist = vs[:window_size].clone()
        
        for t in range(window_size, n_steps):
            window_u = u_hist[-window_size:]
            window_v = v_hist[-window_size:]
            uv = torch.stack([window_u, window_v], dim=-1)
            uv = uv.permute(1, 0, 2).reshape(n_points, -1)
            inp = uv.unsqueeze(0)  # [1, n_points, ws*2]
            
            accel = model(inp, edge_index)
            
            u_t = u_hist[-1]
            v_t = v_hist[-1]
            a_step = accel[0, :, 0]
            v_new = v_t + dt * a_step
            u_new = u_t + dt * (v_t + v_new) / 2
            u_new[boundary_mask] = 0
            
            preds.append(u_new.cpu().numpy())
            u_hist = torch.cat([u_hist[1:], u_new.unsqueeze(0)])
            v_hist = torch.cat([v_hist[1:], v_new.unsqueeze(0)])
    
    preds = np.stack(preds)
    np.savez(os.path.join(output_dir, "result_FDM.npz"), preds=preds,
             loss=history["loss"], error=history["error"])
    
    return history


def train_data(mesh, u_gt, device, output_dir, config):
    """Train with supervised MSE loss"""
    print("\n--- Training Data ---")
    
    dtype = torch.float64
    window_size = config['window_size']
    dt = config['dt']
    epochs = config.get('data_epochs', config['epochs'] // 4)
    
    model = build_wave_model(mesh, window_size=window_size, n_hidden=config['n_hidden'],
                             n_layers=config['n_layers'], num_hops=config['num_hops'],
                             encoder_frequency=config['encoder_frequency'],
                             decoder_frequency=config['decoder_frequency'],
                             activation=config['activation'])
    model = model.to(dtype).to(device)
    
    edge_index = build_graph(mesh).to(device)
    
    n_points = len(mesh.points)
    boundary_mask = torch.from_numpy(mesh.point_data['boundary_mask']).to(device)
    
    u_gt_t = torch.from_numpy(u_gt).type(dtype).to(device) if isinstance(u_gt, np.ndarray) else u_gt.type(dtype).to(device)
    n_steps = u_gt_t.shape[0]
    
    vs = torch.zeros_like(u_gt_t)
    vs[1:-1] = (u_gt_t[2:] - u_gt_t[:-2]) / (2 * dt)
    
    n_train = int(n_steps * 0.5)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    history = {"loss": [], "error": []}
    
    for epoch in tqdm(range(epochs), desc="Data"):
        model.train()
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()
        
        batch_size = config['batch_size']
        valid_start = window_size
        valid_end = n_train - window_size
        if valid_end <= valid_start:
            valid_end = valid_start + 1
        
        idx = torch.randint(valid_start, valid_end, (batch_size,), device=device)
        
        inputs = []
        targets = []
        for i in idx:
            window_u = u_gt_t[i-window_size+1:i+1]
            window_v = vs[i-window_size+1:i+1]
            uv = torch.stack([window_u, window_v], dim=-1)
            uv = uv.permute(1, 0, 2).reshape(n_points, -1)  # [n_points, ws*2]
            inputs.append(uv)
            
            # Target: next window_size u values
            target = u_gt_t[i+1:i+1+window_size]
            targets.append(target.T)
        
        inputs = torch.stack(inputs)  # [batch, n_points, ws*2]
        targets = torch.stack(targets)  # [batch, n_points, ws]
        
        accel = model(inputs, edge_index)
        
        u_t = inputs[:, :, -2]  # Last u
        v_t = inputs[:, :, -1]  # Last v
        
        pred_u = []
        for step in range(window_size):
            a_step = accel[:, :, step]
            v_new = v_t + dt * a_step
            u_new = u_t + dt * (v_t + v_new) / 2
            u_new[:, boundary_mask] = 0
            pred_u.append(u_new)
            u_t = u_new
            v_t = v_new
        
        pred_u = torch.stack(pred_u, dim=-1)  # [batch, n_points, ws]
        
        loss = ((pred_u - targets) ** 2).mean()
        
        loss.backward()
        optimizer.step()
            
        with torch.no_grad():
            error = (pred_u[:, :, 0] - targets[:, :, 0]).pow(2).mean().sqrt().item()
        
        history["loss"].append(loss.item())
        history["error"].append(error)
    
    # Rollout
    model.eval()
    preds = list(u_gt_t[:window_size].cpu().numpy())
    
    with torch.no_grad():
        u_hist = u_gt_t[:window_size].clone()
        v_hist = vs[:window_size].clone()
        
        for t in range(window_size, n_steps):
            window_u = u_hist[-window_size:]
            window_v = v_hist[-window_size:]
            uv = torch.stack([window_u, window_v], dim=-1)
            uv = uv.permute(1, 0, 2).reshape(n_points, -1)
            inp = uv.unsqueeze(0)  # [1, n_points, ws*2]
            
            accel = model(inp, edge_index)
            
            u_t = u_hist[-1]
            v_t = v_hist[-1]
            a_step = accel[0, :, 0]
            v_new = v_t + dt * a_step
            u_new = u_t + dt * (v_t + v_new) / 2
            u_new[boundary_mask] = 0
            
            preds.append(u_new.cpu().numpy())
            u_hist = torch.cat([u_hist[1:], u_new.unsqueeze(0)])
            v_hist = torch.cat([v_hist[1:], v_new.unsqueeze(0)])
    
    preds = np.stack(preds)
    np.savez(os.path.join(output_dir, "result_Data.npz"), preds=preds,
             loss=history["loss"], error=history["error"])
    
    return history


def train_pinn(mesh, u_gt, device, output_dir, config):
    """Train PINN (aligned with PINNWaveTrainer)"""
    print("\n--- Training PINN ---")
    
    dtype = torch.float32  # PINN uses float32 for stability
    dt = config['dt']
    c = config['c']
    epochs = config.get('pinn_epochs', 30000)
    
    model = PINNModel(hidden_dim=config.get('pinn_hidden', 32), 
                      num_layers=config.get('pinn_layers', 4),
                      activation='tanh')
    model = model.to(device)
    
    pinn_loss = WavePinnLoss(c=c, lambda_pde=config.get('lambda_pde', 1.0),
                              lambda_sb=config.get('lambda_sb', 0.1),
                              lambda_init=config.get('lambda_init', 0.001))
    
    points = torch.from_numpy(mesh.points[:, :2]).type(dtype).to(device)
    boundary_mask = torch.from_numpy(mesh.point_data['boundary_mask'])
    
    u_gt_np = u_gt if isinstance(u_gt, np.ndarray) else u_gt.numpy()
    u_gt_t = torch.from_numpy(u_gt_np).type(dtype).to(device)
    n_steps, n_points = u_gt_t.shape
    
    n_train = int(n_steps * 0.5)
    ts = torch.arange(n_train, dtype=dtype, device=device) * dt
    
    # Interior points
    node_coords = points.unsqueeze(0).expand(n_train, -1, -1)  # [n_steps, n_points, 2]
    time_coords = ts[:, None, None].expand(-1, n_points, 1)
    inp_int = torch.cat([node_coords[:, ~boundary_mask, :], 
                         time_coords[:, ~boundary_mask, :]], dim=-1).reshape(-1, 3)
    
    # Sample interior points
    pde_sample_ratio = config.get('pde_sample_ratio', 0.1)
    n_sample = int(pde_sample_ratio * len(inp_int))
    sample_idx = torch.randperm(len(inp_int))[:n_sample]
    inp_int = inp_int[sample_idx]
    
    # Boundary points
    inp_sb = torch.cat([node_coords[:, boundary_mask, :],
                        time_coords[:, boundary_mask, :]], dim=-1).reshape(-1, 3)
    
    # Sample boundary points
    bd_sample_ratio = config.get('bd_sample_ratio', 0.1)
    n_sample = int(bd_sample_ratio * len(inp_sb))
    sample_idx = torch.randperm(len(inp_sb))[:n_sample]
    inp_sb = inp_sb[sample_idx]
    
    # Initial condition points
    inp_tb = torch.cat([points, torch.zeros(n_points, 1, device=device, dtype=dtype)], dim=-1)
    output_tb = torch.stack([u_gt_t[0], torch.zeros(n_points, device=device, dtype=dtype)], dim=-1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('pinn_lr', 1e-5))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=config.get('pinn_scheduler_step', 1000),
                                                  gamma=config.get('pinn_scheduler_gamma', 0.98))
    
    history = {"loss": [], "error": []}
    
    for epoch in tqdm(range(epochs), desc="PINN"):
        model.train()
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()
        
        loss, loss_pde, loss_sb, loss_ic = pinn_loss.compute(model, inp_int, inp_sb, inp_tb, output_tb)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            check_step = n_train // 2
            t_vec = torch.full((n_points, 1), check_step * dt, device=device, dtype=dtype)
            xyt = torch.cat([points, t_vec], dim=-1)
            u_pred = model(xyt).squeeze()
            error = (u_pred - u_gt_t[check_step]).pow(2).mean().sqrt().item()
        
        history["loss"].append(10 ** loss.item())
        history["error"].append(error)
    
    # Rollout
    model.eval()
    preds = []
    
    with torch.no_grad():
        for t_idx in range(n_steps):
            t_vec = torch.full((n_points, 1), t_idx * dt, device=device, dtype=dtype)
            xyt = torch.cat([points, t_vec], dim=-1)
            u_pred = model(xyt).squeeze()
            preds.append(u_pred.cpu().numpy())
    
    preds = np.stack(preds)
    np.savez(os.path.join(output_dir, "result_PINN.npz"), preds=preds,
             loss=history["loss"], error=history["error"])
    
    return history


def plot_results(output_dir, methods):
    """Plot training curves"""
    print("Plotting results...")
    
    colors = sns.color_palette()
    markers = ["o", "s", "^", "D"]
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, m in enumerate(methods):
        path = os.path.join(output_dir, f"result_{m}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        loss = data["loss"]
        error = data["error"]
        
        epochs = np.arange(len(loss))
        ax[0].plot(epochs, loss, color=colors[i], label=m, 
                   marker=markers[i % len(markers)], markevery=max(1, len(epochs)//10), markersize=8)
        ax[1].plot(epochs, error, color=colors[i], label=m,
                   marker=markers[i % len(markers)], markevery=max(1, len(epochs)//10), markersize=8)
    
    ax[0].set_yscale('log')
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[0].grid(color='gray', linestyle='--', linewidth=0.5)
    
    ax[1].set_yscale('log')
    ax[1].set_title("RMSE Error")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].grid(color='gray', linestyle='--', linewidth=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wave_extrapolation_loss.png"), dpi=150)
    print(f"Saved plot to {os.path.join(output_dir, 'wave_extrapolation_loss.png')}")


def create_animation(output_dir, mesh, u_gt, methods, dt):
    """Create comparison animation"""
    print("Generating Animation...")
    
    preds_dict = {"FEM": u_gt if isinstance(u_gt, np.ndarray) else u_gt.numpy()}
    for m in methods:
        path = os.path.join(output_dir, f"result_{m}.npz")
        if os.path.exists(path):
            preds_dict[m] = np.load(path)["preds"]
    
    n_frame = preds_dict["FEM"].shape[0]
    n_in_frame = int(n_frame * 0.5)
    
    colors = sns.color_palette()
    markers = ["1", "2", "3", "4"]
    zone_colors = sns.color_palette("PRGn", n_colors=2)
    
    # Calculate ARMSE
    armses = {}
    for m in methods:
        if m not in preds_dict:
            continue
        preds = preds_dict[m]
        us_gt = preds_dict["FEM"][:len(preds)]
        mse = np.mean((preds - us_gt)**2, axis=-1)
        armse = np.sqrt(np.cumsum(mse) / np.arange(1, len(mse)+1))
        armses[m] = armse
    
    # Value range
    all_values = np.stack(list(preds_dict.values()))
    umin, umax = all_values.min(), all_values.max()
    
    # Setup figure
    points = mesh.points[:, :2] if mesh.points.shape[1] > 2 else mesh.points
    triangles = Triangulation(points[:, 0], points[:, 1], mesh.cells_dict['triangle'])
    
    fig = plt.figure(figsize=(16, 10))
    n_methods = len(preds_dict)
    gs = gridspec.GridSpec(2, n_methods, height_ratios=[1, 1])
    
    mesh_axes = [fig.add_subplot(gs[0, i]) for i in range(n_methods)]
    armse_ax = fig.add_subplot(gs[1, :])
    
    mesh_imgs = {}
    for i, (method, values) in enumerate(preds_dict.items()):
        ax = mesh_axes[i]
        img = ax.tripcolor(triangles, values[0], cmap='jet', shading='gouraud', vmin=umin, vmax=umax)
        ax.triplot(triangles, color='k', lw=0.5, alpha=0.5)
        ax.set_title(f"{method} Solution")
        ax.set_aspect('equal')
        ax.axis('off')
        
        if i > 0:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.scatter([xlim[1]*0.75], [ylim[1]*0.75], color=colors[i-1], 
                      marker=markers[i-1], s=800, linewidth=8)
        
        mesh_imgs[method] = img
    
    # ARMSE plot
    armse_ax.axvspan(0, n_in_frame * dt, color=zone_colors[0], alpha=0.2)
    armse_ax.axvspan(n_in_frame * dt, n_frame * dt, color=zone_colors[1], alpha=0.2)
    armse_ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
    armse_ax.text(0.25, 0.85, "Interpolation", transform=armse_ax.transAxes, fontsize=16,
                  fontweight='bold', color=zone_colors[0], alpha=0.9, ha='center')
    armse_ax.text(0.75, 0.85, "Extrapolation", transform=armse_ax.transAxes, fontsize=16,
                  fontweight='bold', color=zone_colors[1], alpha=0.9, ha='center')
    
    armse_ax.set_xlabel("Time (s)", fontsize=14)
    armse_ax.set_ylabel("Accumulated RMSE", fontsize=14)
    armse_ax.set_xlim(0, n_frame * dt)
    
    max_armse = max([a.max() for a in armses.values()]) if armses else 1.0
    armse_ax.set_ylim(0, max_armse * 1.1)
    
    armse_dots = {}
    for i, (method, armse) in enumerate(armses.items()):
        time_axis = np.arange(len(armse)) * dt
        armse_ax.plot(time_axis, armse, color=colors[i], lw=2)
        dot = armse_ax.scatter([0], [armse[0]], label=method, c=[colors[i]], 
                               marker=markers[i], linewidth=4, s=400)
        armse_dots[method] = (armse, dot)
    
    armse_ax.legend(fontsize=12, labelspacing=1.5)
    vline = armse_ax.axvline(0, color=zone_colors[0], lw=2, linestyle='--')
    
    fig.tight_layout()
    pbar = tqdm(range(n_frame), desc="Animation")
    
    def update(frame):
        for method, img in mesh_imgs.items():
            values = preds_dict[method]
            img.set_array(values[min(frame, len(values)-1)])
        
        current_time = frame * dt
        vline.set_xdata([current_time])
        if frame > n_in_frame:
            vline.set_color(zone_colors[1])
        else:
            vline.set_color(zone_colors[0])
        
        for method, (armse, dot) in armse_dots.items():
            if frame < len(armse):
                dot.set_offsets(np.array([[current_time, armse[frame]]]))
        
        fig.suptitle(f"Time: {current_time:.4f}s", fontsize=16)
        pbar.update()
    
    anim = FuncAnimation(fig, update, frames=n_frame, interval=100)
    anim.save(os.path.join(output_dir, "wave_extrapolation.mp4"), dpi=150, fps=16)
    pbar.close()
    print(f"Animation saved to {os.path.join(output_dir, 'wave_extrapolation.mp4')}")


def run_experiment():
    """Run the full experiment"""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration (aligned with Graph-Galerkin-Learning config)
    config = {
        'c': 4.0,
        'dt': 0.0005,
        'T': 0.1,
        'K': 6,
        'chara_length': 0.08,
        'window_size': 4,
        'batch_size': 8,
        'n_samples': 16,  # Number of different initial conditions for training
        'n_hidden': 64,
        'n_layers': 3,
        'num_hops': 8,
        'encoder_frequency': 4,
        'decoder_frequency': 4,
        'activation': 'relu',
        'lr': 0.001,
        'epochs': 4000,
        'scheduler_step_size': 500,
        'scheduler_gamma': 0.8,
        'discount_factor': 1.0,
        'seed': 100,
        # FDM specific
        'fdm_epochs': 4000,
        'fdm_scheduler_step_size': 1000,
        # Data specific
        'data_epochs': 1000,
        # PINN specific
        'pinn_epochs': 30000,
        'pinn_lr': 1e-5,
        'pinn_hidden': 32,
        'pinn_layers': 4,
        'pinn_scheduler_step': 1000,
        'pinn_scheduler_gamma': 0.98,
        'lambda_pde': 1.0,
        'lambda_sb': 0.1,
        'lambda_init': 0.001,
        'pde_sample_ratio': 0.1,
        'bd_sample_ratio': 0.1,
    }
    
    # Generate mesh (square domain for sin mode initial condition)
    # Sin modes sin(πix)*sin(πjy) are naturally 0 on square [0,1]x[0,1] boundary
    print("Generating mesh...")
    mesh = GmshGen.gen_tri_square(side=1.0, chara_length=config['chara_length'])
    print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells_dict['triangle'])} elements")
    
    # Generate initial condition
    print("Generating initial condition...")
    np.random.seed(100)
    a = np.random.uniform(-1.0, 1.0, (config['K'], config['K']))
    
    # Initial condition using MultiAnalytical (sine modes)
    u0, v0 = MultiAnalytical.initial_condition(mesh.points, a, r=0.5)
    
    # For analytical solution with sin modes, v0 should be 0
    # (the original code uses v0=ones but that's for general FEM, not analytical)
    v0 = np.zeros_like(v0)
    
    # Apply boundary condition (sin modes are naturally 0 on square boundary)
    boundary_mask = mesh.point_data['boundary_mask']
    boundary_value = mesh.point_data['boundary_value']
    u0[boundary_mask] = boundary_value[boundary_mask]
    
    # Generate ground truth using analytical solution (more stable than FEM)
    # For sin mode initial condition with v0=0, analytical solution is exact
    print("Generating ground truth (Analytical Solution)...")
    T_total = config['T'] * 2  # Double time for extrapolation
    u_gt = MultiAnalytical.generate_trajectory(
        mesh.points, a, c=config['c'], T=T_total, dt=config['dt'], r=0.5
    )
    
    # Enforce boundary condition
    u_gt[:, boundary_mask] = 0
    
    print(f"Ground truth shape: {u_gt.shape}, range: [{u_gt.min():.4f}, {u_gt.max():.4f}]")
    
    # Train methods
    methods = ["Galerkin", "FDM", "Data", "PINN"]
    
    train_galerkin(mesh, u_gt, device, output_dir, config)
    train_fdm(mesh, u_gt, device, output_dir, config)
    train_data(mesh, u_gt, device, output_dir, config)
    train_pinn(mesh, u_gt, device, output_dir, config)
    
    # Plot and animate
    plot_results(output_dir, methods)
    create_animation(output_dir, mesh, u_gt, methods, config['dt'])


if __name__ == "__main__":
    run_experiment()
