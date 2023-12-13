import sys 
import os
import torch
sys.path.append("../..")
import torch_fem as thfem
import skfem
import skfem.helpers
import pandas as pd
import time
import cupy as cp
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import fenics
from memory_profiler import memory_usage


class SkFEM:
    def __init__(self, mesh, element="tri"):
        assert element in ["tri", "tetra"]
        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")

        self.mesh  = mesh
        self.element = element

    def __call__(self):
        element = skfem.ElementTriP1() if self.element == "tri" else skfem.ElementTetP1()
        @skfem.BilinearForm
        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v)) 
        
        @skfem.LinearForm
        def load(v, w):
            return 1.0*v
        basis = skfem.Basis(self.mesh, element)
        K     = skfem.asm(laplace, basis)

        f     = skfem.asm(load, basis)
        u     = skfem.solve(K,f)
        return  u 
    
class ThFEM:
    def __init__(self, mesh, batch_size=None):
        self.mesh = mesh
        self.batch_size = batch_size
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_element_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)

    def __call__(self):
        K     = self.K_asm(self.mesh.points, batch_size=self.batch_size)
        f     = self.f_asm(self.mesh.points, batch_size=self.batch_size)
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        u     = K.solve(f, backend=backend)
        return u 

class feFEM:
    def __init__(self, mesh):
        mesh.save("tmp.xdmf")
        fenics_mesh = fenics.Mesh()
        with fenics.XDMFFile("tmp.xdmf") as infile:
            infile.read(fenics_mesh)

        # Create a FunctionSpace on the fenics_mesh, not the original mesh
        self.fenics_mesh = fenics_mesh

    def __call__(self):
        V = fenics.FunctionSpace(self.fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        # Use fenics, not fem, for the methods inner, grad, and dx
        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        L = fenics.inner(fenics.Constant(1), v) * fenics.dx
        u = fenics.Function(a.arguments()[0].function_space())

        fenics.solve(a == L, u)
        return u

       
class SimpleMemoryHook(cp.cuda.MemoryHook):
    def __init__(self):
        self.alloc_size = 0
        self.free_size = 0
        self.cur_size = 0
        self.peak_size = 0

    def malloc_preprocess(self, device_id, size, mem_size):
        self.alloc_size += size
        self.cur_size += size
        self.peak_size = max(self.peak_size, self.cur_size)

    def free_preprocess(self, device_id, mem_size, mem_ptr, pmem_id):
        self.free_size += mem_size
        self.cur_size -= mem_size

    def print_report(self):
        print(f'Total allocated: {self.alloc_size} bytes')
        print(f'Total freed: {self.free_size} bytes')
        print(f'Current memory usage: {self.alloc_size - self.free_size} bytes')



def plot_comparison(element_type, chara_lengths, n_times, csv_path, ax_time, ax_mem):
    data = {
        "chara_length":[],
        "backend":[],
        "time":[],
        "memory in MB":[]
    }
    pbar = tqdm(total=len(chara_lengths)*n_times*6)

    for chara_length in chara_lengths:
        if element_type == "tri":
            mesh = thfem.Mesh.gen_rectangle(chara_length=chara_length, element_type=element_type)
        elif element_type == "tetra":
            mesh = thfem.Mesh.gen_cube(chara_length=chara_length)
        else:
            raise NotImplementedError(f"element_type={element_type} is not supported")
        th_fem_cpu = ThFEM(mesh.clone())
        th_fem_cpu_1 = ThFEM(mesh.clone(), batch_size=1)
        th_fem_gpu   = ThFEM(mesh.clone().to("cuda:0"))
        th_fem_gpu_1 = ThFEM(mesh.clone().to("cuda:0"), batch_size=1)
        sk_fem       = SkFEM(mesh, element=element_type)
        fe_fem       = feFEM(mesh)
        for _ in range(n_times):
            for name, fem in zip(["torch_fem cpu(None)", 
                                        "torch_fem cpu(1)",
                                        "torch_fem cuda(None)", 
                                        "torch_fem cuda(1)",
                                        "scikit-fem", 
                                        "fenics"], [th_fem_cpu, 
                                                    th_fem_cpu_1,
                                                    th_fem_gpu, 
                                                    th_fem_gpu_1,
                                                    sk_fem, 
                                                    fe_fem]):
                start = time.perf_counter()
                fem()
                end = time.perf_counter()
                if "cuda" in name:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    with SimpleMemoryHook() as hook:
                        fem()
                    peak = hook.alloc_size - hook.free_size
                    torch.cuda.synchronize()
                    peak_mem = torch.cuda.max_memory_allocated() 
                    peak_mem = max(peak, peak_mem)/ 1e6
                else:
                    peak_mem = memory_usage(fem, max_usage=True)
                data["chara_length"].append(chara_length)
                data["backend"].append(name)
                data["time"].append(end-start)
                data["memory in MB"].append(peak_mem)
                pbar.update(1)
                pbar.set_postfix({
                    "chara_length":chara_length,
                    "backend":name,
                    "time":f"{end-start:7.5f}s"
                })
    df = pd.DataFrame(data)
    df.to_csv(csv_path)
    sns.lineplot(x="chara_length", y="time",
             hue="backend",data=df,ax=ax_time)
    sns.lineplot(x="chara_length", y="memory in MB",
                hue="backend",data=df,ax=ax_mem)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")

if __name__ == '__main__':

    fig, ax = plt.subplots(ncols=2, figsize=(12,  6))
    plot_comparison(
        element_type="tri",
        chara_lengths=[0.05, 0.01, 0.007],
        n_times=5,
        csv_path="compare_2d.csv",
        ax_time = ax[0],
        ax_mem   = ax[1]
    )

    fig.savefig("compare_2d.png")

    fig, ax = plt.subplots(ncols=2, figsize=(12,  6))

    plot_comparison(
        element_type="tetra",
        chara_lengths=[0.1,  0.08, 0.06],
        n_times=5,
        csv_path="compare_3d.csv",
        ax_time = ax[0],
        ax_mem  = ax[1]
    )

    fig.savefig("compare_3d.png")

    



    # from torch.profiler import profile, record_function, ProfilerActivity
    # mesh = thfem.Mesh.gen_rectangle(chara_length=0.007, element_type="tri")
    # skfem_ = SkFEM(mesh)
    # start = time.perf_counter()
    # skfem_()
    # end = time.perf_counter()
    # print(f"skfem: {end-start}s")
    # fefem = feFEM(mesh)
    # start = time.perf_counter()
    # fefem()
    # end = time.perf_counter()
    # print(f"fenics: {end-start}s")
    # fem = ThFEM(mesh, batch_size=1)
    # start = time.perf_counter()
    # fem()
    # end = time.perf_counter()
    # print(f"torch_fem: {end-start}s")
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True) as prof:
    #     with record_function("fem"):
    #         fem()
        
    # prof.export_chrome_trace("trace.json")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))