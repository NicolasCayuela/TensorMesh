import torch
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
import matplotlib.cm as cm



# Plot Functions

def plot_1d(basis, basis_fns, resolution=100):
    fig, ax = plt.subplots()

    x = torch.linspace(0, 1, resolution)[:, None]

    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

    for i,basis_fn in enumerate(basis_fns):
        y = basis_fn(x)
        basis_x = basis[i] # [n_dim]
        basis_y = basis_fn(basis_x)
        ax.plot(x, y, color=colors[i])
        ax.scatter(basis_x, basis_y, color=colors[i])

    plt.show()

def plot_2d(basis, basis_fns, resolution=100):

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    x, y = torch.meshgrid(torch.linspace(0, 1, resolution), 
                          torch.linspace(0, 1, resolution), indexing='ij') 
    x, y = x.flatten(), y.flatten()
    xy   = torch.stack([x,y], -1)

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple']

    for i,basis_fn in enumerate(basis_fns):
        basis_z = basis_fn(basis[i])
        z = basis_fn(xy) 
        x = x.reshape(resolution, resolution)
        y = y.reshape(resolution, resolution)
        z = z.reshape(resolution, resolution)
        ax.plot_surface(x, y, z, color=colors[i], alpha=0.2)
        ax.scatter(basis[i,0], basis[i,1], basis_z, color=colors[i])
    
    plt.show()

def plot_3d(element, basis, basis_fns, resolution=15):

    n_basis = len(basis_fns)
    
    n_cols  = 4 
    n_rows  = (n_basis + 3 ) // 4

    fig = plt.figure()

    x, y, z = torch.meshgrid(torch.linspace(0, 1, resolution), 
                             torch.linspace(0, 1, resolution), 
                             torch.linspace(0, 1, resolution), indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    xyz     = torch.stack([x,y,z], -1)
  
    cmap    = plt.get_cmap('Spectral_r')
    
    
    for i,basis_fn in enumerate(basis_fns):
        ax  = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        basis_w = basis_fn(basis[i])
        w = basis_fn(xyz) 
        # x = x.reshape(resolution, resolution, resolution)
        # y = y.reshape(resolution, resolution, resolution)
        # z = z.reshape(resolution, resolution, resolution)
        # w = w.reshape(resolution, resolution, resolution)
        norm = mcolors.Normalize(vmin=w.min(), vmax=w.max())
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.scatter(x, y, z, c=scalar_map.to_rgba(w), alpha=0.1)
        ax.scatter(basis[i, 0], basis[i,1], basis[i,2], color=scalar_map.to_rgba(basis_w))
        ax.text(basis[i,0], basis[i,1], basis[i,2], f"{basis_w}")
        # breakpoint()
    
        cbar = fig.colorbar(scalar_map, ax=ax)
        print(f"basis:{basis_fn.basis}")
        edges = element.points[element.edge]
        # breakpoint()
        for edge in edges:
            # breakpoint()
            ax.plot(edge[:, 0].numpy(), edge[:,1].numpy(), edge[:,2].numpy(), color='black')      
        ax.set_title(f'Basis {i+1}')
  
    plt.show()

    