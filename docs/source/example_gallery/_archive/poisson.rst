Poisson Equation
================

Adaptive Mesh Refinement
------------------------

.. code-block:: python 

    import torch 
    import numpy as np
    from tqdm import tqdm
    from torch_fem import LaplaceElementAssembler, Mesh,  Condenser
    from torch_fem.dataset import PoissonMultiFrequency
    from torch_fem.visualization import StreamPlotter
    import matplotlib.pyplot as plt

    if __name__ == "__main__":
        torch.random.manual_seed(123456)
        mesh      = Mesh.gen_rectangle(chara_length=0.1)
        assembler = LaplaceElementAssembler.from_mesh(mesh)
        equation  = PoissonMultiFrequency(K=8)
        condenser = Condenser(mesh.boundary_mask)

        optimizer = torch.optim.Adam(mesh.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        epoch = 100

        f = equation.initial_condition(mesh.points)
        # u = equation.solution(mesh.points)
        loss_fn = torch.nn.MSELoss()

        losses = []

        with StreamPlotter(filename="poisson.mp4") as plotter:
            plotter.draw_mesh(mesh, f)
            pbar = tqdm(total=epoch)
            for i in range(epoch):
                optimizer.zero_grad()
                K = assembler(mesh.points)
                u = K.solve(f)
                loss = loss_fn(K @ u, f)
                # TODO: why retain_graph=True?
                loss.backward(retain_graph=True) 
                optimizer.step()
                scheduler.step()
                plotter.draw_mesh(mesh, f)
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                losses.append(loss.item())

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(losses)), losses, label="loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_yscale("log")
        fig.savefig("loss.png")


.. raw:: html

    <div style="display: flex; justify-content: center; align-items: center;">
    <video width="600" height="600" controls>
      <source src="../_static/poisson_adaptive.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    </div>
    
    
3D Poisson Equation (Cube)
--------------------------

This example solves the Poisson equation :math:`-\Delta u = f` on a unit cube domain :math:`\Omega = [0, 1]^3` with zero Dirichlet boundary conditions.

.. math::

    \begin{aligned}
    -\Delta u &= f \quad \text{in } \Omega \\
    u &= 0 \quad \text{on } \partial \Omega
    \end{aligned}

For the analytic solution :math:`u(x, y, z) = \sin(\pi x) \sin(\pi y) \sin(\pi z)`, the source term is :math:`f = 3\pi^2 u`.

.. image:: ../_static/poisson_3d_half_from_cut.png
   :width: 80%
   :align: center
   :alt: 3D Poisson Solution

.. code-block:: python

    import os
    import sys
    sys.path.append("../..")

    import numpy as np
    import torch

    from tensormesh import Mesh, LaplaceElementAssembler, MassElementAssembler, NodeAssembler, Condenser
    from tensormesh.visualization import setup_headless


    class LoadAssembler(NodeAssembler):
        """Assemble RHS f_i = ∫ f v_i dx."""

        def forward(self, v, f):
            return v * f


    def main():
        torch.manual_seed(0)
        setup_headless()

        out_dir = os.path.dirname(os.path.abspath(__file__))

        # --------------------
        # Mesh (unit cube)
        # --------------------
        # NOTE: Mesh.gen_cube uses gmsh and may generate tetra/hex depending on settings.
        # We keep it as-is but enforce 3D Poisson correctness independent of element type.
        # Smaller => denser mesh (clearer visualization, slower mesh generation/solve)
        chara_length = 0.12
        order = 1
        mesh = Mesh.gen_cube(chara_length=chara_length, order=order).double()

        x = mesh.points  # [n_points, 3]

        # --------------------
        # Analytic solution (zero Dirichlet on boundary)
        # u = sin(pi x) sin(pi y) sin(pi z)
        # -Δu = 3*pi^2 * u
        # --------------------
        pi = torch.pi
        u_exact = torch.sin(pi * x[:, 0]) * torch.sin(pi * x[:, 1]) * torch.sin(pi * x[:, 2])
        f = 3.0 * (pi ** 2) * u_exact

        # Dirichlet boundary: u = 0 on boundary
        # Mesh has boundary_mask already for gmsh meshes
        if hasattr(mesh, "boundary_mask"):
            boundary_mask = mesh.boundary_mask
        else:
            # Fallback: infer boundary nodes from coordinates
            eps = 1e-12
            boundary_mask = (
                (x[:, 0] < eps)
                | (x[:, 0] > 1 - eps)
                | (x[:, 1] < eps)
                | (x[:, 1] > 1 - eps)
                | (x[:, 2] < eps)
                | (x[:, 2] > 1 - eps)
            )

        dirichlet_value = torch.zeros_like(x[:, 0])
        condenser = Condenser(boundary_mask, dirichlet_value)

        # --------------------
        # Assemble system
        # --------------------
        K_asm = LaplaceElementAssembler.from_mesh(mesh)
        K = K_asm(mesh.points)

        F_asm = LoadAssembler.from_mesh(mesh)
        rhs = F_asm(mesh.points, point_data={"f": f})

        Kc, rhsc = condenser(K, rhs)
        uc = Kc.solve(rhsc)
        u = condenser.recover(uc)

        # --------------------
        # Error (mass-weighted L2)
        # --------------------
        e = u - u_exact
        M_asm = MassElementAssembler.from_mesh(mesh)
        M = M_asm(mesh.points)
        l2_err = torch.sqrt((e * (M @ e)).sum())
        l2_ref = torch.sqrt((u_exact * (M @ u_exact)).sum())
        rel_l2 = (l2_err / (l2_ref + 1e-30)).item()

        print(f"[poisson_3d] n_points={mesh.points.shape[0]}  rel_L2={rel_l2:.3e}")

    if __name__ == "__main__":
        main()