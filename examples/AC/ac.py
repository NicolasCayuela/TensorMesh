import sys 
sys.path.append("../..")
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch_fem import ElementAssembler, NodeAssembler, Condenser, Mesh, dot, mul
from torch_fem.dataset import WaveMultiFrequency, PoissonMultiFrequency


dt = 1e-6


class KAssembler(ElementAssembler):
    def __post_init__(self):
        epsilon = 220
        self.dt = dt
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
        self.dt = dt
        epsilon = 220
        self.dcdotdc = 1.0/dt
        self.D  = lambda x: 1.0e0 
        self.dD = lambda x: 0.0e0
        self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
        self.df = lambda x: -epsilon**2*(3*x**2 - 1)

    def forward(self, v, gradv, c, gradc, cold):
        cdot = (c - cold) / self.dt
        
        R =  cdot * v + self.D(c) * (gradv @ gradc) - self.f(c) * v
      
        return R
    

if __name__ == '__main__':
    mesh = Mesh.gen_rectangle(chara_length=0.02, element_type="quad")
    # dataset = WaveMultiFrequency(K=24, r=1)
    dataset = PoissonMultiFrequency(K=24, r=1)
    
    cold = dataset.initial_condition(mesh.points)
    # cold = torch.zeros(mesh.points.shape[0]).type(mesh.points.dtype)
    # cold[mesh.boundary_mask] = 0.0

    # image = Image.open("eth.png")
    # width, height = image.size
    # image = image.convert('L')
    # image = np.array(image)
    # image = np.round(image / 255, 0) * 2 - 1
    # y, x = mesh.points.T.cpu().numpy() * height
    # x    = width - x
    # coord = np.vstack((x,y))
    # label = torch.from_numpy(scipy.ndimage.map_coordinates(image, coord, mode="nearest")).type(mesh.points.dtype).to(mesh.points.device)
    # cold  = label

    cs = []
    cs.append(cold)

    K_asm = KAssembler.from_mesh(mesh)
    R_asm = RAssembler.from_mesh(mesh)
    new_boundary_mask = torch.zeros_like(mesh.boundary_mask, dtype=torch.bool)
    #mesh.boundary_mask = new_boundary_mask
    condenser = Condenser(new_boundary_mask)

    max_iter = 50
    steps    = 200
    pbar = tqdm(total=steps)
    for step in range(steps):
        c = cold
        n_iter, converged = 0, False
        while n_iter < max_iter and (not converged):  
            point_data = {"c":c,"cold":cold}  
          
            K = K_asm(mesh.points, point_data=point_data)
            R = R_asm(mesh.points, point_data=point_data)
            
            K_, R_ = condenser(K, R)
            dC_ = K_.solve(R_)
            dC  = condenser.recover(dC_)

            c  = c + dC
            n_iter += 1

            rnorm = torch.linalg.norm(R_)
            if rnorm < 1e-10:
                converged = True

        pbar.update(1)
        pbar.set_postfix({"rnorm":rnorm})

        cold = c 
        cs.append(cold)

    mesh.plot(values={
        "cs":cs
    },show_mesh=True, dt=1e-6, save_path="Allen-Cahn.mp4")      

