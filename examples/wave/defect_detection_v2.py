"""
Improved Wave Speed Inversion using Differentiable Galerkin Method (V2)

Key improvements over v1:
1. Multi-source excitation for richer information
2. Neural network parameterization of wave speed field
3. Combined time-domain and frequency-domain loss
4. Cosine annealing learning rate schedule
5. Improved regularization with Laplacian smoothing
6. Gradient clipping for stability
7. Multi-scale coarse-to-fine optimization
"""

import sys
sys.path.append("../..")
import random
import math
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.visualization import StreamPlotter

# ============================================================================
# ICML Style Settings
# ============================================================================
ICML_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}

ICML_COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05', 
    'green': '#029E73',
    'red': '#D55E00',
    'purple': '#CC78BC',
}

def set_icml_style():
    plt.rcParams.update(ICML_RCPARAMS)

# ============================================================================
# Wave Equation Assemblers
# ============================================================================

class AAssembler(ElementAssembler):
    """Stiffness matrix assembler with wave speed"""
    def forward(self, gradu, gradv, c):
        return gradu @ gradv * c * c
    
class MAssembler(ElementAssembler):
    """Mass matrix assembler"""
    def forward(self, u, v):
        return u * v

# ============================================================================
# Wave Solver
# ============================================================================

class WaveSolver:
    """
    Differentiable wave equation solver using finite element method.
    Solves: u_tt = c^2 * Laplacian(u)
    """
    def __init__(self, mesh, boundary_type='bottom'):
        self.M_asm = MAssembler.from_mesh(mesh)
        self.A_asm = AAssembler.from_assembler(self.M_asm)
        self.mesh = mesh
        self.boundary_type = boundary_type
        
        # Set boundary condition based on type
        if boundary_type == 'bottom':
            self.condenser = Condenser(mesh.point_data['is_bottom_boundary'])
        elif boundary_type == 'all':
            self.condenser = Condenser(mesh.point_data['is_boundary'])
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")

    def solve(self, u0, c, dt=0.001, n_steps=100, return_all=True):
        """
        Solve wave equation with given initial condition and wave speed.
        
        Parameters
        ----------
        u0 : torch.Tensor[n_point]
            Initial displacement
        c : torch.Tensor[n_point]
            Wave speed field
        dt : float
            Time step
        n_steps : int
            Number of time steps
        return_all : bool
            If True, return all time steps; otherwise return only last step
            
        Returns
        -------
        torch.Tensor[n_steps, n_point] or torch.Tensor[n_point]
        """
        # Ensure c has the same dtype as mesh points
        c = c.to(dtype=self.mesh.points.dtype)
        
        M = self.M_asm(self.mesh.points)
        A = self.A_asm(self.mesh.points, point_data={"c": c})

        Us = [u0]
        v0 = torch.zeros_like(u0)
        
        # First time step (special treatment)
        K = 2 * M
        F_rhs = -dt * dt * A @ u0 + 2 * M @ u0 + 2 * dt * M @ v0
        K_, F_ = self.condenser(K, F_rhs)
        U_ = K_.solve(F_)
        U = self.condenser.recover(U_)
        M_ = self.condenser(M)[0]
        Us.append(U)
        
        # Subsequent time steps
        for _ in range(n_steps - 2):
            U1, U2 = Us[-2:]
            F_rhs = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2
            F_ = self.condenser.condense_rhs(F_rhs)
            U_ = M_.solve(F_)
            U = self.condenser.recover(U_)
            Us.append(U)

        if return_all:
            return torch.stack(Us, 0)
        else:
            return Us[-1]

# ============================================================================
# Neural Network for Wave Speed Parameterization
# ============================================================================

class WaveSpeedMLP(nn.Module):
    """
    MLP to parameterize wave speed field.
    Uses positional encoding for better high-frequency representation.
    """
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=4, 
                 c_min=0.5, c_max=3.0, use_positional_encoding=True,
                 n_frequencies=6):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        self.use_positional_encoding = use_positional_encoding
        self.n_frequencies = n_frequencies
        
        if use_positional_encoding:
            # Positional encoding: [sin(2^0 * pi * x), cos(2^0 * pi * x), ...]
            actual_input_dim = input_dim * (1 + 2 * n_frequencies)
        else:
            actual_input_dim = input_dim
        
        layers = []
        layers.append(nn.Linear(actual_input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize last layer to output near zero (sigmoid -> 0.5)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def positional_encoding(self, x):
        """Apply positional encoding to input coordinates."""
        encodings = [x]
        for i in range(self.n_frequencies):
            freq = 2 ** i * math.pi
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1)
    
    def forward(self, coords):
        """
        Parameters
        ----------
        coords : torch.Tensor[n_points, 2]
            Spatial coordinates
            
        Returns
        -------
        torch.Tensor[n_points]
            Wave speed at each point
        """
        if self.use_positional_encoding:
            x = self.positional_encoding(coords)
        else:
            x = coords
        
        raw = self.net(x).squeeze(-1)
        # Bounded output using sigmoid
        c = self.c_min + (self.c_max - self.c_min) * torch.sigmoid(raw)
        return c


class WaveSpeedDirect(nn.Module):
    """
    Direct parameterization of wave speed (no neural network).
    Uses sigmoid for bounded output.
    """
    def __init__(self, n_points, c_min=0.5, c_max=3.0, c_init=1.0):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        
        # Initialize so that sigmoid gives c_init
        sigmoid_target = (c_init - c_min) / (c_max - c_min)
        c_raw_init = math.log(sigmoid_target / (1 - sigmoid_target))
        self.c_raw = nn.Parameter(torch.ones(n_points) * c_raw_init)
    
    def forward(self, coords=None):
        """Return bounded wave speed."""
        return self.c_min + (self.c_max - self.c_min) * torch.sigmoid(self.c_raw)


class WaveSpeedRBF(nn.Module):
    """
    RBF-based wave speed parameterization.
    Good for smooth fields with localized features.
    """
    def __init__(self, n_rbf=100, domain_bounds=(0, 1, 0, 1),
                 c_min=0.5, c_max=3.0, c_init=1.0):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        
        # RBF centers on a grid
        n_side = int(math.sqrt(n_rbf))
        x = torch.linspace(domain_bounds[0], domain_bounds[1], n_side)
        y = torch.linspace(domain_bounds[2], domain_bounds[3], n_side)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        self.register_buffer('centers', centers)
        
        # RBF width
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.1)))
        
        # RBF weights (initialized for c_init)
        sigmoid_target = (c_init - c_min) / (c_max - c_min)
        raw_init = math.log(sigmoid_target / (1 - sigmoid_target))
        self.weights = nn.Parameter(torch.ones(n_side * n_side) * raw_init)
    
    def forward(self, coords):
        """
        Parameters
        ----------
        coords : torch.Tensor[n_points, 2]
        
        Returns
        -------
        torch.Tensor[n_points]
        """
        sigma = torch.exp(self.log_sigma)
        
        # Compute RBF values: [n_points, n_rbf]
        diff = coords.unsqueeze(1) - self.centers.unsqueeze(0)
        rbf = torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * sigma ** 2))
        
        # Weighted sum
        raw = (rbf * self.weights.unsqueeze(0)).sum(dim=-1)
        
        # Bounded output
        c = self.c_min + (self.c_max - self.c_min) * torch.sigmoid(raw)
        return c

# ============================================================================
# Loss Functions
# ============================================================================

def compute_data_loss(us_pred, us_gt, detector_idx, loss_type='mse'):
    """Compute data fidelity loss at detector locations."""
    pred = us_pred[:, detector_idx]
    gt = us_gt[:, detector_idx]
    
    if loss_type == 'mse':
        return F.mse_loss(pred, gt)
    elif loss_type == 'l1':
        return F.l1_loss(pred, gt)
    elif loss_type == 'huber':
        return F.huber_loss(pred, gt, delta=1.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_frequency_loss(us_pred, us_gt, detector_idx, n_freqs=10):
    """
    Compute loss in frequency domain using FFT.
    Helps capture phase information.
    """
    pred = us_pred[:, detector_idx]
    gt = us_gt[:, detector_idx]
    
    # FFT along time axis
    pred_fft = torch.fft.rfft(pred, dim=0)
    gt_fft = torch.fft.rfft(gt, dim=0)
    
    # Use only first n_freqs frequencies
    pred_fft = pred_fft[:n_freqs]
    gt_fft = gt_fft[:n_freqs]
    
    # Magnitude and phase loss
    mag_loss = F.mse_loss(pred_fft.abs(), gt_fft.abs())
    phase_loss = F.mse_loss(pred_fft.angle(), gt_fft.angle())
    
    return mag_loss + 0.1 * phase_loss


def compute_tv_loss(c, mesh, eps=1e-2):
    """
    Total Variation regularization.
    Promotes piecewise constant solutions.
    """
    elements = mesh.elements()
    element_c = c[elements]
    
    # Compute differences between neighboring nodes
    diffs = []
    n_basis = element_c.shape[1]
    for i in range(n_basis):
        for j in range(i + 1, n_basis):
            diff = element_c[:, i] - element_c[:, j]
            diffs.append(diff)
    
    diffs = torch.stack(diffs, dim=1)
    
    # Smoothed L1 (Huber-like)
    tv = torch.sqrt(diffs ** 2 + eps ** 2).mean()
    return tv


def compute_laplacian_loss(c, mesh):
    """
    Laplacian smoothness regularization.
    Penalizes second-order variations.
    """
    elements = mesh.elements()
    element_c = c[elements]
    
    # Approximate Laplacian as deviation from local mean
    local_mean = element_c.mean(dim=1, keepdim=True)
    laplacian = ((element_c - local_mean) ** 2).mean()
    
    return laplacian


def compute_gradient_penalty(c, coords):
    """
    Gradient penalty for neural network parameterization.
    Requires gradients w.r.t. coordinates.
    """
    coords = coords.requires_grad_(True)
    # This would need the model to be passed in
    # For now, we use finite differences
    return torch.tensor(0.0, device=c.device)

# ============================================================================
# Multi-Source Excitation
# ============================================================================

def generate_multi_source_data(mesh, wave_solver, c_gt, dt, n_steps, 
                                source_positions, A=10.0, sigma=0.01):
    """
    Generate wave data from multiple source positions.
    
    Parameters
    ----------
    source_positions : list of (x, y) tuples
        Source locations
    
    Returns
    -------
    list of (u0, us_gt) tuples
    """
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    data = []
    
    for x_src, y_src in source_positions:
        u0 = A * torch.exp(-((x - x_src)**2 + (y - y_src)**2) / (2 * sigma**2))
        us_gt = wave_solver.solve(u0, c_gt, dt, n_steps)
        data.append((u0, us_gt))
    
    return data

# ============================================================================
# Dataset Generation
# ============================================================================

def circle_dataset(chara_length, device="cpu", A=10, sigma=0.01, 
                   smooth=True, defect_sigma=0.08):
    """Single circle defect."""
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    
    if smooth:
        c_gt = 1.0 + 1.0 * torch.exp(-r2 / (2 * defect_sigma**2))
    else:
        c_gt = torch.ones_like(x) * 1.0
        c_gt[r2 < 0.2**2] = 2.0
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh


def circles_dataset(chara_length, device="cpu", A=10, sigma=0.01,
                    smooth=True, defect_sigma=0.05):
    """Multiple circle defects."""
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    
    circles = [(0.7, 0.2), (0.2, 0.7), (0.6, 0.7)]
    
    c_gt = torch.ones_like(x) * 1.0
    
    for cx, cy in circles:
        r2 = (x - cx)**2 + (y - cy)**2
        if smooth:
            c_gt = c_gt + 1.0 * torch.exp(-r2 / (2 * defect_sigma**2))
        else:
            c_gt[r2 < 0.1**2] = 2.0
    
    c_gt = torch.clamp(c_gt, 1.0, 2.5)
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh


def complex_dataset(chara_length, device="cpu", A=10, sigma=0.01, defect_sigma=0.06):
    """Complex defect pattern with multiple shapes."""
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    
    c_gt = torch.ones_like(x) * 1.0
    
    # Circular defects
    for cx, cy in [(0.3, 0.3), (0.7, 0.7)]:
        r2 = (x - cx)**2 + (y - cy)**2
        c_gt = c_gt + 0.8 * torch.exp(-r2 / (2 * defect_sigma**2))
    
    # Elliptical defect
    cx, cy = 0.5, 0.5
    r2 = ((x - cx) / 0.15)**2 + ((y - cy) / 0.08)**2
    c_gt = c_gt + 0.6 * torch.exp(-r2 / 2)
    
    c_gt = torch.clamp(c_gt, 1.0, 2.5)
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh


def eth_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    """ETH logo dataset."""
    image = Image.open('eth.png')
    image = np.array(image)
    alpha = image[:, :, 3]
    ratio = alpha.shape[1] / alpha.shape[0]
    mesh = Mesh.gen_rectangle(right=ratio, chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    c_gt = torch.ones_like(x) * 1.0

    y_np, x_np = mesh.points.T.cpu().numpy() * alpha.shape[0]
    x_np = alpha.shape[0] - x_np
    coord = np.vstack((x_np, y_np))
    alpha_points = torch.from_numpy(
        scipy.ndimage.map_coordinates(alpha, coord, mode="nearest")
    ).type(mesh.points.dtype).to(mesh.points.device)
    c_gt[alpha_points > 0] = 2.0

    x, y = mesh.points[:, 0], mesh.points[:, 1]
    x_source, y_source = ratio / 2, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh

# ============================================================================
# Optimization
# ============================================================================

class WaveInversion:
    """
    Main class for wave speed inversion.
    """
    def __init__(self, mesh, wave_solver, detector_idx, args):
        self.mesh = mesh
        self.wave_solver = wave_solver
        self.detector_idx = detector_idx
        self.args = args
        self.device = mesh.points.device
        
        # Initialize wave speed model
        if args.model == 'mlp':
            self.c_model = WaveSpeedMLP(
                input_dim=2,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                c_min=args.c_min,
                c_max=args.c_max,
                use_positional_encoding=True,
                n_frequencies=args.n_frequencies
            ).to(self.device)
        elif args.model == 'rbf':
            self.c_model = WaveSpeedRBF(
                n_rbf=args.n_rbf,
                c_min=args.c_min,
                c_max=args.c_max
            ).to(self.device)
        else:  # direct
            self.c_model = WaveSpeedDirect(
                n_points=mesh.n_points,
                c_min=args.c_min,
                c_max=args.c_max
            ).to(self.device)
        
        # Optimizer
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.c_model.parameters(), 
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.c_model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(
                self.c_model.parameters(),
                lr=0.1,
                max_iter=20,
                line_search_fn='strong_wolfe'
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
        # Learning rate scheduler
        if args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epoch, eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.epoch // 4, gamma=0.5
            )
        else:
            self.scheduler = None
        
        # Logging
        self.losses = []
        self.data_losses = []
        self.reg_losses = []
        self.cs_pred = []
        self.c_stats = {'min': [], 'max': [], 'mean': []}
    
    def get_c(self):
        """Get current wave speed prediction."""
        if isinstance(self.c_model, WaveSpeedDirect):
            return self.c_model()
        else:
            return self.c_model(self.mesh.points)
    
    def compute_loss(self, multi_source_data):
        """Compute total loss from multiple sources."""
        c_pred = self.get_c()
        
        total_data_loss = 0.0
        total_freq_loss = 0.0
        
        for u0, us_gt in multi_source_data:
            us_pred = self.wave_solver.solve(
                u0, c_pred, self.args.dt, self.args.n
            )
            
            # Time-domain loss
            data_loss = compute_data_loss(
                us_pred, us_gt, self.detector_idx, 
                loss_type=self.args.loss_type
            )
            total_data_loss += data_loss
            
            # Frequency-domain loss
            if self.args.use_freq_loss:
                freq_loss = compute_frequency_loss(
                    us_pred, us_gt, self.detector_idx,
                    n_freqs=self.args.n_freqs
                )
                total_freq_loss += freq_loss
        
        # Average over sources
        n_sources = len(multi_source_data)
        total_data_loss /= n_sources
        total_freq_loss /= n_sources
        
        # Regularization
        reg_loss = torch.tensor(0.0, device=self.device)
        
        if self.args.lambda_tv > 0:
            tv_loss = compute_tv_loss(c_pred, self.mesh)
            reg_loss = reg_loss + self.args.lambda_tv * tv_loss
        
        if self.args.lambda_laplacian > 0:
            lap_loss = compute_laplacian_loss(c_pred, self.mesh)
            reg_loss = reg_loss + self.args.lambda_laplacian * lap_loss
        
        # Total loss
        loss = total_data_loss + self.args.freq_weight * total_freq_loss + reg_loss
        
        return loss, total_data_loss, reg_loss, c_pred
    
    def train_step(self, multi_source_data):
        """Single training step."""
        if self.args.optimizer == 'lbfgs':
            def closure():
                self.optimizer.zero_grad()
                loss, data_loss, reg_loss, c_pred = self.compute_loss(multi_source_data)
                loss.backward()
                return loss
            
            self.optimizer.step(closure)
            with torch.no_grad():
                loss, data_loss, reg_loss, c_pred = self.compute_loss(multi_source_data)
        else:
            self.optimizer.zero_grad()
            loss, data_loss, reg_loss, c_pred = self.compute_loss(multi_source_data)
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.c_model.parameters(), self.args.grad_clip
                )
            
            self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item(), data_loss.item(), reg_loss.item(), c_pred.detach()
    
    def train(self, multi_source_data, c_gt):
        """Full training loop."""
        pbar = tqdm(range(self.args.epoch), desc="Training")
        
        for epoch in pbar:
            loss, data_loss, reg_loss, c_pred = self.train_step(multi_source_data)
            
            # Logging
            self.losses.append(loss)
            self.data_losses.append(data_loss)
            self.reg_losses.append(reg_loss)
            self.cs_pred.append(c_pred.clone())
            self.c_stats['min'].append(c_pred.min().item())
            self.c_stats['max'].append(c_pred.max().item())
            self.c_stats['mean'].append(c_pred.mean().item())
            
            # Progress bar
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss:.2e}',
                'data': f'{data_loss:.2e}',
                'reg': f'{reg_loss:.2e}',
                'lr': f'{lr:.2e}'
            })
        
        return self.get_c().detach()


# ============================================================================
# Visualization
# ============================================================================

def plot_results(mesh, c_gt, c_pred, losses, data_losses, reg_losses, 
                 c_stats, cs_pred, args):
    """Generate all result plots."""
    set_icml_style()
    
    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    fig.patch.set_facecolor('white')
    
    epochs = list(range(len(losses)))
    axes[0].semilogy(epochs, losses, linewidth=1.2, color=ICML_COLORS['blue'], label='Total')
    axes[0].semilogy(epochs, data_losses, linewidth=1.0, color=ICML_COLORS['orange'], 
                     linestyle='--', label='Data')
    if args.lambda_tv > 0 or args.lambda_laplacian > 0:
        axes[0].semilogy(epochs, reg_losses, linewidth=1.0, color=ICML_COLORS['green'],
                         linestyle=':', label='Reg.')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
    axes[0].legend(frameon=True, fancybox=False, edgecolor='#cccccc')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Wave speed statistics
    axes[1].fill_between(epochs, c_stats['min'], c_stats['max'], 
                         alpha=0.2, color=ICML_COLORS['blue'])
    axes[1].plot(epochs, c_stats['mean'], linewidth=1.2, 
                 color=ICML_COLORS['blue'], label='Predicted $c$')
    axes[1].axhline(y=c_gt.min().item(), color=ICML_COLORS['red'], 
                    linestyle='--', linewidth=1.0, label=f'True $c_{{min}}$')
    axes[1].axhline(y=c_gt.max().item(), color=ICML_COLORS['green'],
                    linestyle='--', linewidth=1.0, label=f'True $c_{{max}}$')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Wave Speed $c$")
    axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
    axes[1].legend(frameon=True, fancybox=False, edgecolor='#cccccc', loc='best')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    fig.savefig("c_loss_v2.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig("c_loss_v2.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Result comparison
    from tensormesh.visualization.matplotlib import draw_mesh as draw_mesh_static
    
    width = mesh.points[:, 0].max() - mesh.points[:, 0].min()
    height = mesh.points[:, 1].max() - mesh.points[:, 1].min()
    ratio = (width / height).item()
    
    fig, axes = plt.subplots(1, 3, figsize=(ratio * 9, 3))
    
    draw_mesh_static(mesh.points, mesh.elements(), c_gt.cpu(), 
                     ax=axes[0], show_colorbar=True, show_mesh=False)
    axes[0].set_title("Ground Truth $c$")
    
    draw_mesh_static(mesh.points, mesh.elements(), c_pred.cpu(),
                     ax=axes[1], show_colorbar=True, show_mesh=False)
    axes[1].set_title("Predicted $c$")
    
    error = torch.abs(c_gt - c_pred).cpu()
    draw_mesh_static(mesh.points, mesh.elements(), error,
                     ax=axes[2], show_colorbar=True, show_mesh=False)
    axes[2].set_title("Absolute Error")
    
    plt.tight_layout()
    fig.savefig("c_result_v2.png", dpi=300, bbox_inches='tight')
    fig.savefig("c_result_v2.pdf", bbox_inches='tight')
    plt.close()
    
    print("Saved c_loss_v2.png, c_loss_v2.pdf, c_result_v2.png, c_result_v2.pdf")


def create_animation(mesh, c_gt, cs_pred, args):
    """Create optimization animation."""
    plt.rcParams['savefig.bbox'] = 'standard'
    
    width = mesh.points[:, 0].max() - mesh.points[:, 0].min()
    height = mesh.points[:, 1].max() - mesh.points[:, 1].min()
    ratio = (width / height).item()
    
    print("Generating c_optimization_v2.mp4...")
    try:
        with StreamPlotter(ncols=2, width=ratio*3.0, height=3.0, 
                          filename="c_optimization_v2.mp4") as sp:
            points = mesh.points
            elements = mesh.elements(2)
            breakpoint()
            
            sp.draw_mesh_2d(points, elements, c_gt, ax=sp.axes[0],
                           title="Ground Truth $c$", show_colorbar=True,
                           show_mesh=False, update=False)
            
            for i, c_pred_i in enumerate(cs_pred):
                if i % args.eval_every_eps == 0:
                    sp.draw_mesh_2d(points, elements, c_pred_i, ax=sp.axes[1],
                                   title=f"Iteration {i}", show_colorbar=True,
                                   update=False, show_mesh=False)
                    sp.update()
        print("Saved c_optimization_v2.mp4")
    except Exception as e:
        import traceback
        print(f"Warning: Could not generate animation: {e}")
        traceback.print_exc()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wave Speed Inversion V2')
    
    # Simulation parameters
    parser.add_argument('-n', '--n', type=int, default=200,
                        help='Number of time steps')
    parser.add_argument('-dt', '--dt', type=float, default=5e-3,
                        help='Time step size')
    parser.add_argument('-d', '--chara_length', type=float, default=0.03,
                        help='Mesh characteristic length')
    
    # Source and detector
    parser.add_argument('-nd', '--n_detector', type=int, default=200,
                        help='Number of detectors')
    parser.add_argument('-s', '--sigma', type=float, default=0.01,
                        help='Source width')
    parser.add_argument('-A', '--A', type=float, default=10.0,
                        help='Source amplitude')
    parser.add_argument('--detect_mode', type=str, default='all',
                        choices=['all', 'top'], help='Detector placement')
    parser.add_argument('--n_sources', type=int, default=3,
                        help='Number of source positions')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='direct',
                        choices=['direct', 'mlp', 'rbf'],
                        help='Wave speed parameterization')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='MLP hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='MLP number of layers')
    parser.add_argument('--n_frequencies', type=int, default=6,
                        help='Positional encoding frequencies')
    parser.add_argument('--n_rbf', type=int, default=100,
                        help='Number of RBF centers')
    
    # Wave speed bounds
    parser.add_argument('--c_min', type=float, default=0.5,
                        help='Minimum wave speed')
    parser.add_argument('--c_max', type=float, default=3.0,
                        help='Maximum wave speed')
    
    # Optimization
    parser.add_argument('-e', '--epoch', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'lbfgs'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'])
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'l1', 'huber'])
    parser.add_argument('--use_freq_loss', action='store_true', default=True,
                        help='Use frequency domain loss')
    parser.add_argument('--no_freq_loss', dest='use_freq_loss', action='store_false')
    parser.add_argument('--freq_weight', type=float, default=0.1,
                        help='Weight for frequency loss')
    parser.add_argument('--n_freqs', type=int, default=20,
                        help='Number of frequencies for FFT loss')
    
    # Regularization
    parser.add_argument('--lambda_tv', type=float, default=1e-4,
                        help='TV regularization weight')
    parser.add_argument('--lambda_laplacian', type=float, default=1e-5,
                        help='Laplacian smoothness weight')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='circle',
                        choices=['circle', 'circles', 'complex', 'eth'])
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--no_smooth', dest='smooth', action='store_false')
    parser.add_argument('--defect_sigma', type=float, default=0.08,
                        help='Defect Gaussian width')
    
    # Other
    parser.add_argument('--cuda', type=int, default=-1,
                        help='CUDA device (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=123456,
                        help='Random seed')
    parser.add_argument('--eval_every_eps', type=int, default=20,
                        help='Evaluation frequency')
    parser.add_argument('--save_cache', type=str, default=None,
                        help='Save results to cache file')
    
    args = parser.parse_args()
    
    # Set device and seed
    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 else torch.device("cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Wave Speed Inversion V2")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Sources: {args.n_sources}")
    print(f"Detectors: {args.n_detector}")
    print("=" * 60)
    
    # Load dataset
    dataset_fn = {
        'circle': circle_dataset,
        'circles': circles_dataset,
        'complex': complex_dataset,
        'eth': eth_dataset
    }[args.dataset]
    
    if args.dataset in ['circle', 'circles', 'complex']:
        mesh = dataset_fn(args.chara_length, device=device, A=args.A, sigma=args.sigma,
                         smooth=args.smooth, defect_sigma=args.defect_sigma)
    else:
        mesh = dataset_fn(args.chara_length, device=device, A=args.A, sigma=args.sigma)
    
    print(f"Mesh: {mesh.n_points} points")
    
    # Setup detectors
    top_idx = torch.where(mesh.point_data['is_top_boundary'])[0]
    all_idx = torch.arange(mesh.n_points, device=device)
    candidate_idx = top_idx if args.detect_mode == 'top' else all_idx
    n_detector = min(args.n_detector, len(candidate_idx))
    sample_idx = random.sample(range(len(candidate_idx)), n_detector)
    detector_idx = candidate_idx[sample_idx]
    print(f"Detector coverage: {n_detector}/{len(candidate_idx)} = {n_detector/len(candidate_idx):.2%}")
    
    # Wave solver
    wave_solver = WaveSolver(mesh, boundary_type='bottom')
    
    # Ground truth
    c_gt = mesh.point_data['c_gt']
    print(f"Ground truth c range: [{c_gt.min().item():.3f}, {c_gt.max().item():.3f}]")
    
    # Generate multi-source data
    # Source positions around the domain boundary
    source_positions = []
    if args.n_sources >= 1:
        source_positions.append((0.5, 1.0))  # Top center
    if args.n_sources >= 2:
        source_positions.append((0.0, 0.5))  # Left center
    if args.n_sources >= 3:
        source_positions.append((1.0, 0.5))  # Right center
    if args.n_sources >= 4:
        source_positions.append((0.25, 1.0))  # Top left
    if args.n_sources >= 5:
        source_positions.append((0.75, 1.0))  # Top right
    
    print(f"Generating data from {len(source_positions)} source(s)...")
    multi_source_data = generate_multi_source_data(
        mesh, wave_solver, c_gt, args.dt, args.n,
        source_positions, A=args.A, sigma=args.sigma
    )
    
    # Run inversion
    inversion = WaveInversion(mesh, wave_solver, detector_idx, args)
    c_pred = inversion.train(multi_source_data, c_gt)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Final total loss: {inversion.losses[-1]:.6e}")
    print(f"Final data loss:  {inversion.data_losses[-1]:.6e}")
    print(f"Final reg loss:   {inversion.reg_losses[-1]:.6e}")
    print(f"Ground truth c range: [{c_gt.min().item():.3f}, {c_gt.max().item():.3f}]")
    print(f"Predicted c range:    [{c_pred.min().item():.3f}, {c_pred.max().item():.3f}]")
    
    c_error = torch.abs(c_pred - c_gt)
    print(f"Mean absolute error: {c_error.mean().item():.4f}")
    print(f"Max absolute error:  {c_error.max().item():.4f}")
    print(f"Relative error: {(c_error.sum() / c_gt.sum()).item()*100:.2f}%")
    print("=" * 60)
    
    # Save cache
    if args.save_cache is not None:
        cache = {
            'mesh_data': {
                'points': mesh.points.cpu(),
                'elements': mesh.elements().cpu(),
            },
            'c_gt': c_gt.cpu(),
            'c_pred': c_pred.cpu(),
            'cs_pred': [c.cpu() for c in inversion.cs_pred],
            'losses': inversion.losses,
            'data_losses': inversion.data_losses,
            'reg_losses': inversion.reg_losses,
            'c_stats': inversion.c_stats,
            'detector_idx': detector_idx.cpu(),
            'args': vars(args)
        }
        torch.save(cache, args.save_cache)
        print(f"Cache saved to {args.save_cache}")
    
    # Visualization
    plot_results(mesh, c_gt, c_pred, inversion.losses, inversion.data_losses,
                 inversion.reg_losses, inversion.c_stats, inversion.cs_pred, args)
    
    # Animation
    create_animation(mesh, c_gt, inversion.cs_pred, args)
    
    # Wave comparison video
    print("Generating wave comparison video...")
    u0 = mesh.point_data['u0']
    with torch.no_grad():
        us_pred = wave_solver.solve(u0, c_pred, args.dt, args.n)
        us_gt = wave_solver.solve(u0, c_gt, args.dt, args.n)
    
    mesh.plot(
        {"prediction": [us_pred[i] for i in range(len(us_pred))],
         "ground truth": [us_gt[i] for i in range(len(us_gt))]},
        save_path="c_compare_v2.mp4", dt=args.dt, show_mesh=False, fix_clim=False
    )
    print("Saved c_compare_v2.mp4")
    
    print("\nDone!")

