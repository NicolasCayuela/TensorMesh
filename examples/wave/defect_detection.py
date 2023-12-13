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
from torch_fem import ElementAssembler, Mesh,Condenser
from torch_fem.dataset import WaveMultiFrequency
from torch_fem.functional import mul, dot
from torch_fem.visualization import StreamPlotter



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
        return dot(gradu, gradv) * c * c
    
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
        return mul(u, v)
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n', type=int, default=100)
    parser.add_argument('-dt','--dt', type=float, default=1e-2)
    parser.add_argument('-nd','--n_detector', type=int, default=100)
    parser.add_argument('-s','--sigma', type=float, default=0.1)
    parser.add_argument('-A','--A', type=float, default=10.0)
    parser.add_argument('-e','--epoch', type=int, default=400)
    parser.add_argument('--eval_every_eps', type=int, default=10)
    parser.add_argument("-d","--chara_length", type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--detect_mode', type=str, default="all", choices=["all", "top"])
    parser.add_argument('--dataset', type=str, default="circles", choices=['circle','circles','eth'])
    parser.add_argument('--optimizer', type=str,default="adam", choices=["adam", "lbfgs"])
    
    args = parser.parse_args()
    dt = args.dt
    n  = args.n
    n_detector = args.n_detector
    sigma = args.sigma
    A     = args.A
    epoch = args.epoch
    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    torch.random.manual_seed(123456)
    
    mesh = {
        "circle":circle_dataset,
        "circles":circles_dataset,
        "eth":eth_dataset
    }[args.dataset](args.chara_length, device=device, A=A, sigma=sigma)
    top_idx = torch.where(mesh.point_data['is_top_boundary'])[0]
    all_idx = torch.arange(mesh.n_point, device=device)
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

    # prediction c
    c_pred = torch.ones_like(c_gt).requires_grad_(True) 
    loss_fn = torch.nn.MSELoss()
   
    cs_pred = []
    losses  = []
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([c_pred], lr=0.1, max_iter=50000, line_search_fn="strong_wolfe", tolerance_change=1e-10)
        pbar = tqdm(total = 50000)

    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam([c_pred], lr=args.lr)
        pbar = tqdm(total = epoch)

    else:
        raise NotImplementedError(f"optimizer {args.optimizer} not implemented")
    
    def closure():
        optimizer.zero_grad()
        us_pred = wave(u0, c_pred, dt, n)
        loss = loss_fn(us_pred[:, detector_idx], us_gt[:, detector_idx])
        loss.backward()
        pbar.set_postfix({"loss":loss.item()})
        cs_pred.append(c_pred.detach().clone())
        losses.append(loss.item())
        pbar.update(1)
        return loss
  
    if args.optimizer == "lbfgs":
        optimizer.step(closure)
    else:
        for i in range(epoch):
            closure()
            optimizer.step()

    with torch.no_grad():
        us_pred = wave(u0, c_pred, dt, n)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    fig.savefig("c_loss.png", dpi=400)
    # breakpoint()
    mesh.plot({"prediction":[us_pred[i] for i in range(len(us_pred))], "ground truth":[us_gt[i] for i in range(len(us_gt))]},
              save_path="c_compare.mp4", dt=dt, show_mesh=True, fix_clim=False)


    width = mesh.points[:,0].max() - mesh.points[:,0].min()
    height = mesh.points[:,1].max() - mesh.points[:,1].min()
    ratio = (width / height).item()
    with StreamPlotter(ncols=2, width=ratio*5,height=5, filename="c_optimization.mp4") as sp:
        sp.draw_mesh(mesh, c_gt, ax=sp.axes[0], title="ground truth", update=False)
        for i, c_pred in enumerate(cs_pred):
            if i % args.eval_every_eps == 0:
                sp.draw_mesh(mesh, c_pred, ax=sp.axes[1], title=f"epoch {i}", update=False, show_mesh=False)
                sp.axes[1].scatter(mesh.points[detector_idx,0], mesh.points[detector_idx,1], c="r", label="detector", s=2)
                if i == 0:
                    sp.axes[1].legend()
                sp.update()
