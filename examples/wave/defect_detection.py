from ast import Not
from math import e
import sys
import wave 
sys.path.append("../..")
import random
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm
from PIL import Image
import argparse
import torch
import numpy as np
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.dataset import WaveMultiFrequency
from tensormesh.visualization import StreamPlotter

# ============================================================================
# ICML Style Settings (for publication-quality figures)
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
    """Apply ICML publication style to matplotlib."""
    plt.rcParams.update(ICML_RCPARAMS)
# ============================================================================



class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv, c):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
                c: torch.Tensor[]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return gradu @ gradv * c * c
    
class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return u * v
    
class Wave:
    def __init__(self, mesh):
        self.M_asm = MAssembler.from_mesh(mesh)
        self.A_asm = AAssembler.from_assembler(self.M_asm)
        self.mesh = mesh
        self.condenser = Condenser(mesh.point_data['is_bottom_boundary'])

    def __call__(self, u0, c, dt=0.001, n=100):
        """
        Parameters
        ----------
            u0: torch.Tensor[n_point]
                initial condition
            c: torch.Tensor[n_point]
                wave speed
            dt: float, default 0.001
                time interval
            n: int,  default 100
                number of time steps
        Returns:
        --------
            Us: torch.Tensor[n, n_point]
        """

        M = self.M_asm(mesh.points)
        A = self.A_asm(mesh.points, point_data={"c":c})

        Us  = [u0] 
        v0 = torch.zeros_like(u0) 
        A = A
        K = 2 * M
        F = -dt * dt * A @ u0 + 2 * M @ u0 + 2 * dt * M @ v0
        K_, F_ = self.condenser(K, F)
        U_     = K_.solve(F_)
        U      = self.condenser.recover(U_)
        M_     = self.condenser(M)[0]
        Us.append(U)
        for _ in range(n-2):
            U1, U2 = Us[-2:]

            F = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2
            
            F_ = self.condenser.condense_rhs(F)

            U_ = M_.solve(F_)

            U  = self.condenser.recover(U_)

            Us.append(U)

        return torch.stack(Us, 0)


def circle_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    # ground truth c
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:,0], mesh.points[:,1]
    c_gt = torch.ones_like(x) * 1.0 
    c_gt[(x-0.5)**2+(y-0.5)**2 < 0.2**2] = 2.0 
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh 

def circles_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    # ground truth c
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:,0], mesh.points[:,1]
    c_gt = torch.ones_like(x) * 1.0 
    c_gt[(x-0.7)**2+(y-0.2)**2 < 0.1**2] = 2.0 
    c_gt[(x-0.2)**2+(y-0.7)**2 < 0.1**2] = 2.0
    c_gt[(x-0.6)**2+(y-0.7)**2 < 0.1**2] = 2.0
    
    x_source, y_source = 0.5, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh 

def eth_dataset(chara_length, device="cpu", A=10, sigma=0.01):
    image = Image.open('eth.png')
    # get the height and width of the image
    image = np.array(image)
    alpha = image[:,:,3]
    ratio = alpha.shape[1] / alpha.shape[0]
    mesh = Mesh.gen_rectangle(right=ratio, chara_length=chara_length, element_type="quad").to(device)
    x, y = mesh.points[:,0], mesh.points[:,1]
    c_gt = torch.ones_like(x) * 1.0 

    y,x = mesh.points.T.cpu().numpy() * alpha.shape[0]
    x    = alpha.shape[0] - x
    coord = np.vstack((x,y))
    alpha_points = torch.from_numpy(scipy.ndimage.map_coordinates(alpha, coord, mode="nearest")).type(mesh.points.dtype).to(mesh.points.device)
    c_gt[alpha_points >0] = 2.0
  

    x, y = mesh.points[:,0], mesh.points[:,1]
    x_source, y_source = ratio/2, 1.0
    u0 = A * torch.exp(-((x - x_source)**2 + (y - y_source)**2) / (2 * sigma**2))
    mesh.register_point_data("c_gt", c_gt)
    mesh.register_point_data("u0", u0)
    return mesh

def compute_tv_loss(c, mesh):
    """
    Total Variation regularization for wave speed field.
    Promotes piecewise constant solutions (sharp boundaries).
    """
    elements = mesh.elements()
    points = mesh.points
    
    # Compute gradients on each element
    element_coords = points[elements]  # [n_element, n_basis, n_dim]
    element_c = c[elements]  # [n_element, n_basis]
    
    # Simple approximation: compute variance within each element
    c_mean = element_c.mean(dim=1, keepdim=True)
    tv = ((element_c - c_mean)**2).sum()
    
    return tv


def compute_smooth_tv_loss(c, mesh, eps=1e-2):
    """
    Smoothed Total Variation using Huber-like approximation.
    Better for gradient-based optimization.
    """
    elements = mesh.elements()
    points = mesh.points
    
    element_c = c[elements]  # [n_element, n_basis]
    
    # Compute differences between neighboring nodes in each element
    diffs = []
    n_basis = element_c.shape[1]
    for i in range(n_basis):
        for j in range(i+1, n_basis):
            diff = element_c[:, i] - element_c[:, j]
            diffs.append(diff)
    
    diffs = torch.stack(diffs, dim=1)  # [n_element, n_pairs]
    
    # Smoothed L1 (Huber-like)
    tv = torch.sqrt(diffs**2 + eps**2).sum()
    
    return tv


def compute_tikhonov_loss(c, c_prior=1.0):
    """
    Tikhonov (L2) regularization - penalizes deviation from prior.
    """
    return ((c - c_prior)**2).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n', type=int, default=200)  # Increased time steps
    parser.add_argument('-dt','--dt', type=float, default=5e-3)  # Smaller dt for stability
    parser.add_argument('-nd','--n_detector', type=int, default=200)  # More detectors
    parser.add_argument('-s','--sigma', type=float, default=0.1)
    parser.add_argument('-A','--A', type=float, default=10.0)
    parser.add_argument('-e','--epoch', type=int, default=1000)  # More epochs
    parser.add_argument('--eval_every_eps', type=int, default=20)
    parser.add_argument("-d","--chara_length", type=float, default=0.03)  # Finer mesh
    parser.add_argument('--lr', type=float, default=0.1)  # Smaller learning rate
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--detect_mode', type=str, default="all", choices=["all", "top"])
    parser.add_argument('--dataset', type=str, default="circles", choices=['circle','circles','eth'])
    parser.add_argument('--optimizer', type=str,default="adam", choices=["adam", "lbfgs"])
    # Regularization parameters
    parser.add_argument('--reg_type', type=str, default="tv", choices=["none", "tv", "tikhonov", "both"])
    parser.add_argument('--lambda_tv', type=float, default=0)  # TV regularization weight
    parser.add_argument('--lambda_tik', type=float, default=0)  # Tikhonov weight
    # Wave speed bounds
    parser.add_argument('--c_min', type=float, default=0.5)
    parser.add_argument('--c_max', type=float, default=1.5)
    
    args = parser.parse_args()
    dt = args.dt
    n  = args.n
    n_detector = args.n_detector
    sigma = args.sigma
    A     = args.A
    epoch = args.epoch
    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 else torch.device("cpu")
    torch.random.manual_seed(123456)
    
    mesh = {
        "circle":circle_dataset,
        "circles":circles_dataset,
        "eth":eth_dataset
    }[args.dataset](args.chara_length, device=device, A=A, sigma=sigma)
    top_idx = torch.where(mesh.point_data['is_top_boundary'])[0]
    all_idx = torch.arange(mesh.n_points, device=device)
    candidiate_idx = top_idx if args.detect_mode == "top" else all_idx
    n_detector = min(n_detector, len(candidiate_idx))
    sample_idx = random.sample(range(len(candidiate_idx)), n_detector)
    print(f"select portion:{n_detector/len(candidiate_idx)}")
    detector_idx = candidiate_idx[sample_idx]
    dbc_constraint = mesh.point_data['is_bottom_boundary']
    
    wave = Wave(mesh)


    c_gt  = mesh.point_data['c_gt']
    u0    = mesh.point_data['u0']
    us_gt = wave(u0, c_gt, dt, n)

    # prediction c - use parametrization for bounded wave speed
    # c = c_min + (c_max - c_min) * sigmoid(c_raw) ensures c is always in [c_min, c_max]
    c_min, c_max = args.c_min, args.c_max
    
    # Initialize c_raw so that sigmoid(c_raw) gives c=1.0 initially
    # sigmoid(x) = 0.5 when x = 0, so (c_init - c_min) / (c_max - c_min) = sigmoid(c_raw_init)
    c_init = 1.0
    sigmoid_target = (c_init - c_min) / (c_max - c_min)
    c_raw_init = torch.log(torch.tensor(sigmoid_target / (1 - sigmoid_target)))  # inverse sigmoid
    c_raw = (torch.ones_like(c_gt) * c_raw_init).requires_grad_(True)
    
    def get_c_from_raw(c_raw):
        """Convert unconstrained c_raw to bounded c using sigmoid"""
        return c_min + (c_max - c_min) * torch.sigmoid(c_raw)
    
    loss_fn = torch.nn.MSELoss()
   
    cs_pred = []
    losses  = []
    data_losses = []
    reg_losses = []
    
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([c_raw], lr=0.1, max_iter=50000, line_search_fn="strong_wolfe", tolerance_change=1e-10)
        pbar = tqdm(total = 50000)

    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam([c_raw], lr=args.lr)
        pbar = tqdm(total = epoch)

    else:
        raise NotImplementedError(f"optimizer {args.optimizer} not implemented")
    
    def closure():
        optimizer.zero_grad()
        
        # Get bounded c from raw parameters
        c_pred = get_c_from_raw(c_raw)
        
        us_pred = wave(u0, c_pred, dt, n)
        
        # Data fidelity loss
        data_loss = loss_fn(us_pred[:, detector_idx], us_gt[:, detector_idx])
        
        # Regularization losses
        reg_loss = torch.tensor(0.0, device=device)
        
        if args.reg_type in ["tv", "both"]:
            tv_loss = compute_smooth_tv_loss(c_pred, mesh)
            reg_loss = reg_loss + args.lambda_tv * tv_loss
            
        if args.reg_type in ["tikhonov", "both"]:
            tik_loss = compute_tikhonov_loss(c_pred, c_prior=1.0)
            reg_loss = reg_loss + args.lambda_tik * tik_loss
        
        # Total loss
        loss = data_loss + reg_loss
        loss.backward()
        
        pbar.set_postfix({
            "loss": loss.item(), 
            "data": data_loss.item(), 
            "reg": reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        })
        cs_pred.append(c_pred.detach().clone())
        losses.append(loss.item())
        data_losses.append(data_loss.item())
        reg_losses.append(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)
        pbar.update(1)
        return loss
  
    if args.optimizer == "lbfgs":
        optimizer.step(closure)
    else:
        for i in range(epoch):
            closure()
            optimizer.step()

    # Get final bounded c_pred
    c_pred = get_c_from_raw(c_raw)
    
    with torch.no_grad():
        us_pred = wave(u0, c_pred, dt, n)

    # Apply ICML style for publication-quality figures
    set_icml_style()
    
    # Plot loss curve with breakdown (ICML column width ~3.25in)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    fig.patch.set_facecolor('white')
    
    # Total loss
    axes[0].plot(losses, linewidth=1.2, color=ICML_COLORS['blue'], label='Total')
    axes[0].plot(data_losses, linewidth=1.0, color=ICML_COLORS['orange'], linestyle='--', label='Data')
    if args.reg_type != "none":
        axes[0].plot(reg_losses, linewidth=1.0, color=ICML_COLORS['green'], linestyle=':', label='Reg.')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
    axes[0].legend(frameon=True, fancybox=False, edgecolor='#cccccc', framealpha=1.0)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # c_pred statistics over epochs
    c_pred_max = [c.max().item() for c in cs_pred]
    c_pred_min = [c.min().item() for c in cs_pred]
    c_pred_mean = [c.mean().item() for c in cs_pred]
    
    axes[1].fill_between(range(len(c_pred_min)), c_pred_min, c_pred_max, alpha=0.2, color=ICML_COLORS['blue'])
    axes[1].plot(c_pred_mean, linewidth=1.2, color=ICML_COLORS['blue'], label='Predicted $c$')
    axes[1].axhline(y=c_gt.min().item(), color=ICML_COLORS['red'], linestyle='--', linewidth=1.0, 
                    label=f'True $c_{{min}}$={c_gt.min().item():.1f}')
    axes[1].axhline(y=c_gt.max().item(), color=ICML_COLORS['green'], linestyle='--', linewidth=1.0, 
                    label=f'True $c_{{max}}$={c_gt.max().item():.1f}')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Wave Speed $c$")
    axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
    axes[1].legend(frameon=True, fancybox=False, edgecolor='#cccccc', framealpha=1.0, loc='best')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.5)
    fig.savefig("c_loss.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig("c_loss.pdf", bbox_inches='tight', facecolor='white')  # Vector format for paper
    plt.close()
    # breakpoint()
    # Print summary statistics
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Final total loss: {losses[-1]:.6e}")
    print(f"Final data loss:  {data_losses[-1]:.6e}")
    print(f"Final reg loss:   {reg_losses[-1]:.6e}")
    print(f"Ground truth c range: [{c_gt.min().item():.3f}, {c_gt.max().item():.3f}]")
    print(f"Predicted c range:    [{c_pred.min().item():.3f}, {c_pred.max().item():.3f}]")
    
    # Compute reconstruction error
    c_error = torch.abs(c_pred - c_gt)
    print(f"Mean absolute error in c: {c_error.mean().item():.4f}")
    print(f"Max absolute error in c:  {c_error.max().item():.4f}")
    print(f"Relative error: {(c_error.sum() / c_gt.sum()).item()*100:.2f}%")
    print("="*60)
    
    mesh.plot({"prediction":[us_pred[i] for i in range(len(us_pred))], "ground truth":[us_gt[i] for i in range(len(us_gt))]},
              save_path="c_compare.mp4", dt=dt, show_mesh=True, fix_clim=False)


    # Plot optimization process (ICML style)
    width = mesh.points[:,0].max() - mesh.points[:,0].min()
    height = mesh.points[:,1].max() - mesh.points[:,1].min()
    ratio = (width / height).item()
    
    # Use viridis colormap for ICML style (perceptually uniform, colorblind-friendly)
    with StreamPlotter(ncols=2, width=ratio*5.0, height=5.0, filename="c_optimization.mp4") as sp:
        # Ground truth plot
        sp.draw_mesh_2d(mesh.points, mesh.elements(2), c_gt, ax=sp.axes[0], title="Ground Truth $c$", 
                     show_colorbar=True, show_mesh=False, update=False, cmap="viridis")
        
        # Optimization progress plot
        for i, c_pred_i in enumerate(cs_pred):
            if i % args.eval_every_eps == 0:
                sp.draw_mesh_2d(mesh.points, mesh.elements(2), c_pred_i, ax=sp.axes[1], 
                            title=f"Iteration {i}", 
                            show_colorbar=True, update=False, show_mesh=False, cmap="viridis")
                
                # Add detector points - subtle academic style
                sp.axes[1].scatter(
                    mesh.points[detector_idx, 0].cpu().numpy(), 
                    mesh.points[detector_idx, 1].cpu().numpy(), 
                    c='white',
                    edgecolors='#333333',
                    s=2,
                    alpha=0.8,
                    linewidths=0.5,
                    zorder=10,
                    marker='o',
                    label="Sensors" if i == 0 else None
                )
                
                if i == 0:
                    sp.axes[1].legend(loc='upper right', fontsize=7, frameon=True, 
                                      fancybox=False, edgecolor='#cccccc', framealpha=0.95)
                
                sp.update()
