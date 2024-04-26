import sys 
sys.path.append("../..")
import torch
import torch_fem as thfem 
from torch_fem.profile import TimeProfiler, CPUProfiler, CUDAProfiler
from torch_fem.dataset import PoissonMultiFrequency

class ThFEM:
    def __init__(self, mesh):
        self.mesh = mesh
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)
        self.condenser = thfem.Condenser(self.mesh.boundary_mask)
    def __call__(self):
        # f     = PoissonMultiFrequency().initial_condition(self.mesh.points
        K     = self.K_asm(self.mesh.points, batch_size=1)
        f     = torch.ones(K.shape[0])
        # f     = self.f_asm(self.mesh.points, batch_size=1)
        K_,f_ = self.condenser(K, f)
        # backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        backend = "torch"
        u_    = K_.solve(f_, backend=backend)
        u     = self.condenser.recover(u_)
        return u 

if __name__ == "__main__":
   
    # with CPUProfiler() as cpu_profiler:
    #     with cpu_profiler.scope("create mesh"):
    #         mesh = thfem.Mesh.gen_rectangle(chara_length=0.002, element_type="tri")
    #     with cpu_profiler.scope("create assembler"):
    #         th_fem = ThFEM(mesh.cuda())
    #     with cpu_profiler.scope("solve"):
    #         th_fem()
    
    # cpu_profiler.plot("cpu_mem.png")
    # print(f"Max CPU memory usage: {cpu_profiler.max()} MB")

    with CUDAProfiler() as cuda_profiler:
        with cuda_profiler.scope("create mesh"):
            # mesh = thfem.Mesh.gen_rectangle(chara_length=0.0012, element_type="tri")
            mesh = thfem.Mesh.gen_cube(chara_length=0.005)
        with cuda_profiler.scope("create assembler"):
            th_fem = ThFEM(mesh.cuda())
        with cuda_profiler.scope("solve"):
            th_fem()
    cuda_profiler.plot("cuda_mem.png")
    print(f"dofs:{mesh.n_points}")
    print(f"Max GPU memory usage: {cuda_profiler.max()} MB")

    with TimeProfiler() as time_profiler:
        with time_profiler.scope("create mesh"):
            # mesh = thfem.Mesh.gen_rectangle(chara_length=0.0012, element_type="tri")
            mesh = thfem.Mesh.gen_cube(chara_length=0.008)
        with time_profiler.scope("create assembler"):
            th_fem = ThFEM(mesh.cuda())
        with time_profiler.scope("solve"):
            th_fem()
    time_profiler.plot("time.png")

    print(f"n_dofs:{(~mesh.boundary_mask).sum()}")
