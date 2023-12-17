import sys 
sys.path.append("../..")
import torch_fem as thfem 
from torch_fem.profile import TimeProfiler, CPUProfiler, CUDAProfiler


class ThFEM:
    def __init__(self, mesh):
        self.mesh = mesh
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)

    def __call__(self):
        K     = self.K_asm(self.mesh.points, batch_size=1)
        f     = self.f_asm(self.mesh.points, batch_size=1)
       
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        u     = K.solve(f, backend=backend)
        return u 

if __name__ == "__main__":
   
    with CPUProfiler() as cpu_profiler:
        with cpu_profiler.scope("create mesh"):
            mesh = thfem.Mesh.gen_rectangle(chara_length=0.004, element_type="tri")
        with cpu_profiler.scope("create assembler"):
            th_fem = ThFEM(mesh)
        with cpu_profiler.scope("solve"):
            th_fem()
    
    cpu_profiler.plot("cpu_mem.png")
    print(f"Max CPU memory usage: {cpu_profiler.max()} MB")

    with CUDAProfiler() as cuda_profiler:
        with cuda_profiler.scope("create mesh"):
            mesh = thfem.Mesh.gen_rectangle(chara_length=0.004, element_type="tri")
        with cuda_profiler.scope("create assembler"):
            th_fem = ThFEM(mesh.cuda())
        with cuda_profiler.scope("solve"):
            th_fem()
    cuda_profiler.plot("cuda_mem.png")
    print(f"Max GPU memory usage: {cuda_profiler.max()} MB")

    with TimeProfiler() as time_profiler:
        with time_profiler.scope("create mesh"):
            mesh = thfem.Mesh.gen_rectangle(chara_length=0.004, element_type="tri")
        with time_profiler.scope("create assembler"):
            th_fem = ThFEM(mesh)
        with time_profiler.scope("solve"):
            th_fem()
    time_profiler.plot("time.png")