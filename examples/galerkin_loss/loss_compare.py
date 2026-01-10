
from tensormesh import Mesh, NodeAssembler
from tensormesh.dataset import PoissonMultiFrequency
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import List, Tuple
import os
import json
import torch
import numpy as np
import meshio
import time
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# ICML 论文风格设置
plt.rcParams.update({
    # 字体设置
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    
    # 图形设置
    'figure.figsize': (4.5, 3.5),  # ICML 单栏宽度
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    
    # 线条和标记
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    
    # 坐标轴
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    
    # 图例
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    
    # LaTeX
    'text.usetex': False,  # 设为 True 如果有 LaTeX 环境
    'mathtext.fontset': 'stix',
})

def gen_regular_grid(n: int, left=0.0, right=1.0, bottom=0.0, top=1.0) -> Mesh:
    """
    生成 n×n 的规则四边形网格
    
    Parameters
    ----------
    n : int
        每个方向的节点数，总共 n×n 个节点
    left, right, bottom, top : float
        区域边界
        
    Returns
    -------
    Mesh
        规则网格，节点按行优先顺序排列
    """
    # 生成规则网格点
    x = torch.linspace(left, right, n)
    y = torch.linspace(bottom, top, n)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # yy[i,j], xx[i,j] 对应第 i 行第 j 列
    
    # 展平成 (n*n, 2) 的点集，按行优先顺序
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1).numpy()
    
    # 向量化生成四边形单元连接
    # 每个单元由 4 个节点组成: (i,j), (i,j+1), (i+1,j+1), (i+1,j)
    # 使用 meshgrid 生成所有 (i, j) 索引对
    ii, jj = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing='ij')
    ii, jj = ii.flatten(), jj.flatten()
    
    # 计算四个角的节点索引
    n0 = ii * n + jj            # 左下
    n1 = ii * n + (jj + 1)      # 右下
    n2 = (ii + 1) * n + (jj + 1)  # 右上
    n3 = (ii + 1) * n + jj      # 左上
    
    quads = np.stack([n0, n1, n2, n3], axis=1)
    
    # 创建 meshio 网格
    cells = [("quad", quads)]
    meshio_mesh = meshio.Mesh(points, cells)
    
    # 转换为 tensormesh.Mesh
    mesh = Mesh.from_meshio(meshio_mesh)
    
    # 注册边界信息
    pts = mesh.points
    is_left = pts[:, 0] == left
    is_right = pts[:, 0] == right
    is_bottom = pts[:, 1] == bottom
    is_top = pts[:, 1] == top
    is_boundary = is_left | is_right | is_bottom | is_top
    
    mesh.register_point_data("is_boundary", is_boundary)
    mesh.register_point_data("is_left_boundary", is_left)
    mesh.register_point_data("is_right_boundary", is_right)
    mesh.register_point_data("is_bottom_boundary", is_bottom)
    mesh.register_point_data("is_top_boundary", is_top)
    
    return mesh 


@dataclass 
class LiteDatabase:
    path:str = "output/loss_compare.jsonl"
    rows:List["Row"] = field(default_factory=list)
    count:Counter[Tuple[int, str, str]] = field(default_factory=Counter)
    
    def __post_init__(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                for line in f:
                    row = Row(**json.loads(line))
                    self.rows.append(row)
                    self.count[row.key] += 1
                    
    def __len__(self):
        return len(self.rows)

    def add(self, row: "Row"):
        self.rows.append(row)
        self.count[row.key] += 1
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(asdict(row)) + "\n")

    def plot(self, show_title=True):
        df = pd.DataFrame([asdict(row) for row in self.rows])
        
        # ICML 风格配色 (colorblind-friendly)
        palette = {
            'GalerkinLoss': '#0173B2',    # 蓝色
            'FdmLoss': '#DE8F05',          # 橙色
            'PinnLoss': '#029E73',         # 绿色
            'DataDrivenLoss': '#D55E00',   # 红橙色
        }
        
        # 分离 CUDA 和 CPU 数据
        df['device_type'] = df['device'].apply(lambda x: 'cuda' if x.startswith('cuda') else 'cpu')
        df['num_threads'] = df['device'].apply(lambda x: int(x.split(':')[1]) if x.startswith('cpu') else -1)
        
        # 绘制 CUDA 图
        df_cuda = df[df['device_type'] == 'cuda']
        if len(df_cuda) > 0:
            self._plot_cuda(df_cuda, palette, show_title=show_title)
        
        # 绘制 CPU 图
        df_cpu = df[df['device_type'] == 'cpu']
        if len(df_cpu) > 0:
            self._plot_cpu(df_cpu, palette, show_title=show_title)
    
    def _plot_cuda(self, df, palette, show_title=True):
        """绘制 CUDA 设备的图"""
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        
        sns.lineplot(
            x="dof", 
            y="time", 
            hue="loss", 
            data=df,
            palette=palette,
            markers=True,
            style="loss",
            dashes=False,
            errorbar=("ci", 95),
            err_style="bars",
            err_kws={'capsize': 3, 'capthick': 1.2, 'elinewidth': 1.2},
            ax=ax
        )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degrees of Freedom (DOF)')
        ax.set_ylabel('Time (s)')
        if show_title:
            ax.set_title('CUDA Performance')
        
        ax.legend(
            title='Method',
            loc='upper left',
            frameon=True,
            fancybox=False,
            edgecolor='0.8'
        )
        
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        plt.tight_layout()
        base_path = self.path.replace(".jsonl", "")
        plt.savefig(f"{base_path}_cuda.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base_path}_cuda.pdf", bbox_inches='tight')
        plt.close()
    
    def _plot_cpu(self, df, palette, show_title=True):
        """绘制 CPU 设备的图，颜色表示 loss 类型，透明度表示线程数"""
        # 获取所有线程数并排序
        thread_counts = sorted(df['num_threads'].unique())
        loss_types = sorted(df['loss'].unique())
        n_threads = len(thread_counts)
        n_losses = len(loss_types)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 存储 handles 用于自定义图例排列
        # handles_dict[loss][thread_idx] = handle
        handles_dict = {loss: {} for loss in loss_types}
        
        for loss in loss_types:
            color = palette.get(loss, '#333333')
            df_loss = df[df['loss'] == loss]
            
            for i, n_thread in enumerate(thread_counts):
                df_thread = df_loss[df_loss['num_threads'] == n_thread]
                if len(df_thread) == 0:
                    continue
                
                # 按 dof 分组计算均值和标准差
                grouped = df_thread.groupby('dof')['time'].agg(['mean', 'std']).reset_index()
                
                # 透明度从 0.3 到 1.0，线程数越多越不透明
                alpha = 0.3 + 0.7 * (i / max(n_threads - 1, 1))
                
                line, = ax.plot(
                    grouped['dof'], 
                    grouped['mean'], 
                    color=color, 
                    linewidth=2,
                    marker='o',
                    markersize=5,
                    alpha=alpha,
                )
                handles_dict[loss][i] = line
                
                # 绘制误差条
                ax.errorbar(
                    grouped['dof'],
                    grouped['mean'],
                    yerr=grouped['std'],
                    color=color,
                    alpha=alpha * 0.5,
                    fmt='none',
                    capsize=2,
                    capthick=1,
                    elinewidth=1
                )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degrees of Freedom (DOF)')
        ax.set_ylabel('Time (s)')
        if show_title:
            ax.set_title('CPU Performance (darker = more threads)')
        
        # 自定义图例: 每行是一个 loss，每列是一个线程数
        # matplotlib legend ncol 是行优先排列，所以按 loss -> thread 顺序排列
        from matplotlib.lines import Line2D
        handles_ordered = []
        labels_ordered = []
        
        for loss in loss_types:
            for i, n_thread in enumerate(thread_counts):
                if i in handles_dict[loss]:
                    handles_ordered.append(handles_dict[loss][i])
                    # 第一列显示 loss 名，后续列显示线程数
                    if i == 0:
                        labels_ordered.append(f'{loss} ({n_thread}T)')
                    else:
                        labels_ordered.append(f'({n_thread}T)')
                else:
                    handles_ordered.append(Line2D([], [], color='none'))
                    labels_ordered.append('')
        
        ax.legend(
            handles_ordered, labels_ordered,
            loc='upper left',
            frameon=True,
            fancybox=False,
            edgecolor='0.8',
            fontsize=8,
            ncol=n_threads,
            columnspacing=1.0,
            handletextpad=0.3,
        )
        
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        plt.tight_layout()
        base_path = self.path.replace(".jsonl", "")
        plt.savefig(f"{base_path}_cpu.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base_path}_cpu.pdf", bbox_inches='tight')
        plt.close()

    def del_key(self, row: "Row"):
        valid_indices = []
        for _row in self.rows:
            if _row.key!= row.key:
                valid_indices.append(_row)
        self.rows = [x for i, x in enumerate(self.rows) if i not in valid_indices]
        self.count = Counter([x.key for x in self.rows])
        self.save()


@dataclass 
class Row:
    dof:int 
    loss:str 
    device:str

    time:float = -1.0

    def __post_init__(self):
        if isinstance(self.device, torch.device):
            self.device=  str(self.device)
        if self.device.startswith("device"):
            self.device = self.device.split(":")[0]
        

    @property
    def key(self):
        return (self.dof, self.loss, self.device)

    def __hash__(self):
        return hash((self.dof, self.time, self.loss, self.device))

class PoissonGalerkinAssembler(NodeAssembler):
    def forward(self, gradphi, gradv, v, f):
        """
        Galerkin 弱形式残差: ∫(∇φ·∇v - f·v) dx
        
        Parameters
        ----------
        gradphi : [n_dim]
            φ 在单个积分点的梯度
        gradv : [n_basis, n_dim]
            测试函数 v 在单个积分点的梯度
        v : [n_basis]
            测试函数 v 在单个积分点的值
        f : scalar
            源项 f 在单个积分点的值
        """
        # 扩散项: ∇φ·∇v
        diffusion = (gradphi[..., None, :] * gradv).sum(-1)  # [n_basis]
        # 源项: f·v
        source = f[..., None] * v  # [n_basis]
        return diffusion - source



class GalerkinLoss(nn.Module):

    def __init__(self, mesh, dataset, lambda_bd=10.0):
        super().__init__()
        self.mesh = mesh
        self.dataset = dataset
        self.lambda_bd = lambda_bd

        # self.phi = self.dataset.solution(self.mesh.points)
        self.phi = nn.Parameter(torch.randn(self.mesh.n_points)).requires_grad_(True)
        # 注册为 buffer，这样会随模型一起移动到正确的设备
        self.register_buffer("f", self.dataset.source_term(self.mesh.points))
        self.poisson_galerkin_assembler = PoissonGalerkinAssembler.from_mesh(self.mesh)
        
        # 边界掩码
        boundary_mask = self.mesh.point_data["is_boundary"]
        if isinstance(boundary_mask, np.ndarray):
            boundary_mask = torch.from_numpy(boundary_mask)
        self.register_buffer("boundary_mask", boundary_mask.bool())
        self.register_buffer("interior_mask", ~boundary_mask.bool())
        self.register_buffer("loss_mask", (~self.mesh.point_data["is_boundary"]).float())
        
        # 边界值 (Dirichlet BC: u = 0)
        self.register_buffer("boundary_value", torch.zeros(self.mesh.n_points))
        
        # 训练记录
        self.losses = []
        self.pde_losses = []
        self.bd_losses = []


    def prepare(self):
        # 使用 flat_mode() 只启用 broadcast 优化，不使用 torch.compile
        # 这样可以避免 autotune 的开销，同时获得 ~8x 加速
        self.poisson_galerkin_assembler.flat_mode()

    def compute_loss(self):
        residual = self.poisson_galerkin_assembler(
            point_data = {
                "phi": self.phi,
                "f" : self.f
            },
            scalar_data = {
                "a": torch.tensor(1.0, device=self.phi.device)
            }
        )
        # PDE loss: 只在内部节点上计算
        pde_residual = residual * self.loss_mask
        pde_loss = (pde_residual ** 2).sum()
        
        # Boundary loss: Dirichlet BC
        bd_loss = ((self.phi[self.boundary_mask] - self.boundary_value[self.boundary_mask]) ** 2).mean()
        
        # 返回总损失和分项
        total_loss = pde_loss + self.lambda_bd * bd_loss
        return total_loss, pde_loss, bd_loss
    
    def fit_adam(self, epochs=1000, lr=0.001):
        """使用 Adam 优化器训练"""
        optimizer = torch.optim.Adam([self.phi], lr=lr)
        pbar = tqdm(range(epochs), desc="Training[adam]", unit="epoch", colour="green")
        for ep in pbar:
            optimizer.zero_grad()
            loss, pde_loss, bd_loss = self.compute_loss()
            loss.backward()
            optimizer.step()
            
            self.losses.append(loss.item())
            self.pde_losses.append(pde_loss.item())
            self.bd_losses.append(bd_loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6e}", pde=f"{pde_loss.item():.6e}", bd=f"{bd_loss.item():.6e}")
        return self
    
    def fit_lbfgs(self, epochs=100, lr=0.01):
        """使用 L-BFGS 优化器训练"""
        optimizer = torch.optim.LBFGS([self.phi], lr=lr)
        pbar = tqdm(range(epochs), desc="Training[lbfgs]", unit="epoch", colour="green")
        
        captured = {"pde_loss": 0, "bd_loss": 0}
        
        for ep in pbar:
            def closure():
                optimizer.zero_grad()
                loss, pde_loss, bd_loss = self.compute_loss()
                captured["pde_loss"] = pde_loss.item()
                captured["bd_loss"] = bd_loss.item()
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            self.losses.append(loss.item())
            self.pde_losses.append(captured["pde_loss"])
            self.bd_losses.append(captured["bd_loss"])
            pbar.set_postfix(loss=f"{loss.item():.6e}", pde=f"{captured['pde_loss']:.6e}", bd=f"{captured['bd_loss']:.6e}")
        return self
    
    def fit(self, epochs=1000, epochs2=100, optimizer='adam'):
        """训练接口"""
        self.losses = []
        self.pde_losses = []
        self.bd_losses = []
        
        if optimizer.lower() == 'adam':
            self.fit_adam(epochs)
        elif optimizer.lower() == 'lbfgs':
            self.fit_lbfgs(epochs2)
        elif optimizer.lower() == 'combine':
            self.fit_adam(epochs)
            self.fit_lbfgs(epochs2)
        return self

class DataDrivenLoss(GalerkinLoss):
    def compute_loss(self):
        mse = (self.phi + self.f) ** 2 
        mse = mse * self.loss_mask
        return mse.sum()

class FdmLoss(GalerkinLoss):
    """
    FDM Loss: 使用有限差分法计算拉普拉斯算子
    要求网格是 gen_regular_grid 生成的规则网格，节点按行优先顺序排列
    """
    def prepare(self):
        # 网格大小
        self.size = int(round(np.sqrt(self.mesh.n_points)))
        assert self.size * self.size == self.mesh.n_points, \
            f"网格节点数 {self.mesh.n_points} 不是完全平方数"
        
        # 网格间距
        x = self.mesh.points[:, 0]
        self.h = (x.max() - x.min()) / (self.size - 1)
        
    def compute_loss(self):
        # 节点已经按行优先顺序排列，直接 reshape
        phi_grid = self.phi.reshape(self.size, self.size)
        
        # FDM 拉普拉斯算子: Δu ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h^2
        # 使用 padding 处理边界 (Dirichlet BC: u=0)
        phi_pad = torch.nn.functional.pad(phi_grid, (1, 1, 1, 1), mode='constant', value=0)
        laplacian = (
            phi_pad[2:, 1:-1] + phi_pad[:-2, 1:-1] +  # 上下
            phi_pad[1:-1, 2:] + phi_pad[1:-1, :-2] -  # 左右
            4 * phi_grid
        ) / (self.h * self.h)
        
        # 获取 f 的网格形式
        f_grid = self.f.reshape(self.size, self.size)
        
        # PDE 残差: -Δu - f = 0
        residual = -laplacian - f_grid
        
        # 只计算内部点的损失 (排除边界)
        interior_residual = residual[1:-1, 1:-1]

        return interior_residual.pow(2).sum()

class PinnLoss(GalerkinLoss):
    """
    PINN Loss: 使用自动微分计算 PDE 残差
    参考 Graph-Galerkin-Learning/loss_landscape_v2.py 中的实现
    """
    def __init__(self, mesh, dataset, lambda_bd=10.0, hidden_layers=None):
        self.hidden_layers = hidden_layers or [32, 32]  # 默认网络结构
        super().__init__(mesh, dataset, lambda_bd)
    
    def prepare(self):
        # 获取当前设备
        device = self.phi.device
        dtype = self.phi.dtype
        
        # 确保 points 在正确的设备上
        self.register_buffer("points", self.mesh.points.clone().to(device=device, dtype=dtype))
        # 需要 requires_grad，但 buffer 不能 requires_grad，所以在 compute_loss 中处理
        
        # 动态构建网络
        layers = []
        input_dim = 2
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())  # 使用 Tanh 更适合 PDE 求解
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.lin = nn.Sequential(*layers).to(device=device, dtype=dtype)

    def _update_pred(self):
        """更新预测值，确保 points 需要梯度"""
        # 每次调用时确保设备一致
        device = self.phi.device
        if not hasattr(self, '_points_grad') or self._points_grad.device != device:
            self._points_grad = self.points.clone().detach().requires_grad_(True)
        self.pred = self.lin(self._points_grad).squeeze()

    def compute_loss(self):
        # 确保 pred 是最新的
        self._update_pred()
        
        # 使用 grad_outputs 处理向量输出
        ones = torch.ones_like(self.pred)
        
        # 一阶导数: ∂u/∂x, ∂u/∂y
        gradphi = torch.autograd.grad(
            self.pred, self._points_grad, 
            grad_outputs=ones, 
            create_graph=True
        )[0]
        dx = gradphi[:, 0]  # ∂u/∂x
        dy = gradphi[:, 1]  # ∂u/∂y
        
        # 二阶导数: ∂²u/∂x², ∂²u/∂y²
        graddx = torch.autograd.grad(
            dx, self._points_grad, 
            grad_outputs=ones, 
            create_graph=True
        )[0]
        graddy = torch.autograd.grad(
            dy, self._points_grad, 
            grad_outputs=ones, 
            create_graph=True
        )[0]
        ddx = graddx[:, 0]  # ∂²u/∂x²
        ddy = graddy[:, 1]  # ∂²u/∂y²
        
        # 拉普拉斯算子: Δu = ∂²u/∂x² + ∂²u/∂y²
        nablaphi = ddx + ddy 
        
        # PDE 残差: -Δu - f = 0
        residual = -nablaphi - self.f
        
        # 只在内部节点计算 PDE loss
        pde_loss = (residual[self.interior_mask] ** 2).mean()

        # 边界条件损失: u = 0 on boundary
        bd_loss = (self.pred[self.boundary_mask] ** 2).mean()
        
        # 使用 log10 形式的 loss (参考 loss_landscape_v2.py)
        total_loss = torch.log10(self.lambda_bd * bd_loss + pde_loss + 1e-10)
        
        return total_loss, pde_loss, bd_loss
    
    def fit_adam(self, epochs=1000, lr=0.001):
        """使用 Adam 优化器训练"""
        optimizer = torch.optim.Adam(self.lin.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc="PINN[adam]", unit="epoch", colour="blue")
        for ep in pbar:
            optimizer.zero_grad()
            loss, pde_loss, bd_loss = self.compute_loss()
            loss.backward()
            optimizer.step()
            
            self.losses.append(loss.item())
            self.pde_losses.append(pde_loss.item())
            self.bd_losses.append(bd_loss.item())
            pbar.set_postfix(loss=f"{loss.item():.6e}", pde=f"{pde_loss.item():.6e}", bd=f"{bd_loss.item():.6e}")
        return self
    
    def fit_lbfgs(self, epochs=100, lr=0.01):
        """使用 L-BFGS 优化器训练"""
        optimizer = torch.optim.LBFGS(self.lin.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc="PINN[lbfgs]", unit="epoch", colour="blue")
        
        captured = {"pde_loss": 0, "bd_loss": 0}
        
        for ep in pbar:
            def closure():
                optimizer.zero_grad()
                loss, pde_loss, bd_loss = self.compute_loss()
                captured["pde_loss"] = pde_loss.item()
                captured["bd_loss"] = bd_loss.item()
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            self.losses.append(loss.item())
            self.pde_losses.append(captured["pde_loss"])
            self.bd_losses.append(captured["bd_loss"])
            pbar.set_postfix(loss=f"{loss.item():.6e}", pde=f"{captured['pde_loss']:.6e}", bd=f"{captured['bd_loss']:.6e}")
        return self
    
    def fit(self, epochs=1000, epochs2=100, optimizer='adam'):
        """训练接口"""
        self.losses = []
        self.pde_losses = []
        self.bd_losses = []
        
        if optimizer.lower() == 'adam':
            self.fit_adam(epochs)
        elif optimizer.lower() == 'lbfgs':
            self.fit_lbfgs(epochs2)
        elif optimizer.lower() == 'combine':
            self.fit_adam(epochs)
            self.fit_lbfgs(epochs2)
        return self
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1",
                            help="cpu:N or cuda:N")
    parser.add_argument("--times", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--force", action="store_true", default=False,
                            help="force to run the experiment")
    parser.add_argument("--only_plot", action="store_true", default=False,
                            help="only plot the results without running experiments")
    parser.add_argument("--no-title", action="store_true", default=False,
                            help="do not draw titles on plots")
    args = parser.parse_args()

    db = LiteDatabase()

    # 如果只绘图，直接绘制并退出
    if args.only_plot:
        db.plot(show_title=not args.no_title)
        print("Plot saved.")
        exit(0)

    if args.device.startswith("cpu"):
        device = torch.device("cpu")
        num_threads = int(args.device.split(":")[1])
        torch.set_num_threads(num_threads)
    else:
        device = torch.device(args.device)

    torch.manual_seed(0)
    np.random.seed(0)

    sizes = [10, 50, 100, 500, 1000, 5000]
    losses = [FdmLoss, PinnLoss, DataDrivenLoss, GalerkinLoss]


    with tqdm(total=len(sizes) * len(losses) * args.times) as pbar:
        for size in sizes:
            # 使用 meshgrid 生成 size x size 的规则网格
            # 节点按行优先顺序排列，可以直接用于 FDM
            mesh = gen_regular_grid(n=size, left=0.0, right=1.0, bottom=0.0, top=1.0)
            dataset = PoissonMultiFrequency()
            for LossFunc in losses:
                if args.force:
                    db.del_key(Row(dof=mesh.n_points, loss=LossFunc.__name__, device=args.device))
                loss = LossFunc(mesh, dataset)
                loss.prepare()
                loss = loss.to(device)
                for _ in range(args.warmup):
                    loss.compute_loss()
                for _ in range(args.times):
                    row = Row(dof=mesh.n_points, loss=LossFunc.__name__, device=args.device)

                    if db.count[row.key] >= args.times:
                        pbar.update(1)
                        continue

                    torch.cuda.synchronize()
                    start_time = time.time()
                    loss.compute_loss()
                    torch.cuda.synchronize()
                    end_time = time.time()

                    row.time = end_time - start_time

                    db.add(row)
                    pbar.update(1)

    db.plot(show_title=not args.no_title)

     

        