from importlib.metadata import requires
import sys
from colorama import init 
import warnings
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse 
import scipy.interpolate
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch_fem import ElementAssembler, NodeAssembler, Condenser, Mesh, dot, mul
from torch_fem.dataset import WaveMultiFrequency






def ac_general(args):

    image = args.image 

    image = Image.open(image)
    # get the height and width of the image
    width, height = image.size
    # turn image into binary mask 
    image = image.convert('L')
    image = np.array(image)
    image = np.round(image / 255, 0) * 2 - 1

    dt = args.dt
    
    class KAssembler(ElementAssembler):
        def __post_init__(self):
            epsilon = args.epsilon
            self.dt = args.dt
            self.dcdotdc = 1.0/dt
            self.D  = lambda x: 1.0e0 
            self.dD = lambda x: 0.0e0
            self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
            self.df = lambda x: -epsilon**2*(3*x**2 - 1)
        def forward(self, u, v, gradu, gradv, c, gradc, cold):
        
            return -1.0 * (self.dcdotdc * mul(u, v) +
                    self.dD(c) * u * (gradv  @ gradc) + 
                    self.D(c) * dot(gradu, gradv) -
                    self.df(c) * mul(u, v))
            # return -1.0 * (self.dcdotdc * mul(u, v) +
            #         self.dD(c) * u * dot(gradc, gradv) + 
            #         self.D(c) * dot(gradu, gradv) -
            #         self.df(c) * mul(u, v))
    
    class RAssembler(NodeAssembler):
        def __post_init__(self):
            self.dt = args.dt
            epsilon = args.epsilon
            self.dcdotdc = 1.0/dt
            self.D  = lambda x: 1.0e0 
            self.dD = lambda x: 0.0e0
            self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
            self.df = lambda x: -epsilon**2*(3*x**2 - 1)

        def forward(self, v, gradv, c, gradc, cold):
            cdot = (c - cold) / self.dt
            
            R =  cdot * v + self.D(c) * (gradv @ gradc) - self.f(c) * v
        
            return R


    mesh = Mesh.gen_rectangle(chara_length=args.chara_length,  element_type=args.element_type, right=width/height)
    if args.device.startswith("cuda"):
        mesh.to(args.device)

    y, x = mesh.points.T.cpu().numpy() * height
    x    = width - x
    coord = np.vstack((x,y))
    label = torch.from_numpy(scipy.ndimage.map_coordinates(image, coord, mode="nearest")).type(mesh.points.dtype).to(mesh.points.device)
    label = (label - label.min()) /(label.max() - label.min()) * 2 - 1
    # breakpoint()

    initial = label 
    initial[mesh.boundary_mask] = 0.0 # initial value of the boundary is 0
    initial.requires_grad = True

    K_asm = KAssembler.from_mesh(mesh)
    R_asm = RAssembler.from_mesh(mesh)

    condenser = Condenser(mesh.boundary_mask)
    loss_fn   = nn.MSELoss()

    def evolve(cold, recording=False, progressbar=False, steps=100, max_iter=50):
        if recording:
            cs = [cold.detach()]

        if progressbar:
            pbar = tqdm(total=steps)
    
        for _ in range(steps):
            c = cold
            for n_iter in range(max_iter):  
                point_data = {"c":c,"cold":cold}  
            
                K = K_asm(mesh.points, point_data=point_data)
                R = R_asm(mesh.points, point_data=point_data)
                
                if args.use_dirichlet_boundary:
                    K_, R_ = condenser(K, R)
                    dC_ = K_.solve(R_)
                    dC  = condenser.recover(dC_)
                else:
                    dC = K.solve(R)

                c  = c + dC
                n_iter += 1

                rnorm = torch.linalg.norm(R)
                if rnorm < args.threshold:
                    break
            
            if rnorm > args.threshold:
                warnings.warn(f"not converged after {n_iter} iterations with rnorm {rnorm}")

            if progressbar:
                pbar.update(1)
                pbar.set_postfix({"rnorm":rnorm})

            cold = c 
            if recording:
                cs.append(cold.detach())

        if recording:
            return cs
        else:
            return cold
    
    optimizer = torch.optim.Adam([initial], lr=args.lr)
    # training
    losses = []
    pbar = tqdm(range(args.epochs))
    for _ in pbar:
        optimizer.zero_grad()
        final = evolve(initial, recording=False, progressbar=False, steps=args.steps, max_iter=args.max_iter)
        loss  = loss_fn(final, label)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss":loss.item()})
        losses.append(loss.item())
    # inference
    with torch.no_grad():
        cs = evolve(initial, recording=True, progressbar=True, steps=args.steps, max_iter=args.max_iter)

    mesh.plot(values={
        "cs":cs
    },show_mesh=True, dt=dt, save_path="cs.mp4")   

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(losses)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    fig.savefig("loss.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="eth.png")
    parser.add_argument("--element_type", type=str, default="quad")
    parser.add_argument("--chara_length", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--threshold", type=float,default=1e-5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--use_dirichlet_boundary", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    ac_general(args)

