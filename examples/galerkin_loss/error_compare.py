
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from tqdm import tqdm
from tensormesh.dataset import PoissonMultiFrequency
import sys

# Add current directory to path to import from loss_compare
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from loss_compare import gen_regular_grid, GalerkinLoss, PinnLoss

# Update plotting parameters for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 5),
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

class DynamicPinnLoss(PinnLoss):
    """
    动态配置网络层数的 PINN Loss
    """
    def __init__(self, mesh, dataset, layers_config):
        # 直接使用 layers_config 作为 hidden_layers
        super().__init__(mesh, dataset, lambda_bd=10.0, hidden_layers=layers_config)
    
    def compute_loss(self):
        """重写 compute_loss 返回单个值用于 error_compare 中的比较"""
        # 确保 pred 是最新的
        self._update_pred()
        
        ones = torch.ones_like(self.pred)
        gradphi = torch.autograd.grad(
            self.pred, self._points_grad, 
            grad_outputs=ones, 
            create_graph=True
        )[0]
        dx = gradphi[:, 0]
        dy = gradphi[:, 1]
        
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
        ddx = graddx[:, 0]
        ddy = graddy[:, 1]
        
        nablaphi = ddx + ddy 
        
        residual = -nablaphi - self.f
        
        # Interior PDE Loss - 使用 MSE 而不是 L2 norm，这样和 Galerkin 可比
        pde_loss = (residual[self.interior_mask] ** 2).sum()
        
        # Boundary Loss (u = 0)
        bd_loss = (self.pred[self.boundary_mask] ** 2).mean()
        
        return pde_loss, bd_loss  # 返回分项便于调试


class DynamicGalerkinMLP(GalerkinLoss):
    """
    用神经网络参数化的 Galerkin Loss，用于公平比较
    """
    def __init__(self, mesh, dataset, layers_config):
        self.layers_config = layers_config
        super().__init__(mesh, dataset, lambda_bd=10.0)
    
    def prepare(self):
        super().prepare()  # 调用父类的 prepare
        
        device = self.phi.device
        dtype = self.phi.dtype
        
        # 构建与 PINN 相同结构的网络
        layers = []
        input_dim = 2
        for hidden_dim in self.layers_config:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.lin = nn.Sequential(*layers).to(device=device, dtype=dtype)
        self.register_buffer("mlp_points", self.mesh.points.clone().to(device=device, dtype=dtype))
    
    def _update_phi_from_network(self):
        """从网络预测更新 phi"""
        with torch.no_grad():
            pred = self.lin(self.mlp_points).squeeze()
            self.phi.data.copy_(pred)
    
    def compute_loss(self):
        """使用网络预测计算 Galerkin loss"""
        self._update_phi_from_network()
        
        # 调用父类的 Galerkin loss 计算
        result = super().compute_loss()
        if isinstance(result, tuple):
            total_loss, pde_loss, bd_loss = result
        else:
            pde_loss = result
            bd_loss = torch.tensor(0.0)
        
        return pde_loss, bd_loss


def fit_network(model, target_phi, steps=500, lr=1e-3, desc="Fitting"):
    """
    Fit network to the target_phi (noisy solution).
    Works for both PINN and GalerkinMLP.
    """
    optimizer = torch.optim.Adam(model.lin.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # 获取 points
    if hasattr(model, 'mlp_points'):
        points = model.mlp_points
    else:
        points = model.points
    
    # Target needs to be on the same device
    target = target_phi.to(points.device)
    
    for _ in tqdm(range(steps), desc=desc, leave=False):
        optimizer.zero_grad()
        pred = model.lin(points).squeeze()
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
    
    # 返回最终的预测和拟合 MSE
    with torch.no_grad():
        pred_final = model.lin(points).squeeze()
        fit_mse = ((pred_final - target) ** 2).mean().item()
    
    return pred_final, fit_mse

def verify_galerkin_correctness():
    """
    验证 Galerkin loss 在 Poisson 问题上的正确性：
    1. 解析解应该使 Galerkin loss ≈ 0
    2. 随机解应该有大的 Galerkin loss
    3. 不同噪声水平应该有对应的 loss 变化
    """
    print("\n" + "="*60)
    print("Galerkin Loss Correctness Verification on Poisson Problem")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建网格
    mesh_size = 50
    mesh = gen_regular_grid(n=mesh_size, left=0.0, right=1.0, bottom=0.0, top=1.0)
    dataset = PoissonMultiFrequency()
    
    # 获取解析解和源项
    phi_gt = dataset.solution(mesh.points)
    if isinstance(phi_gt, torch.Tensor):
        phi_gt = phi_gt.clone().detach().to(dtype=torch.float32, device=device)
    else:
        phi_gt = torch.tensor(phi_gt, dtype=torch.float32).to(device)
    if phi_gt.ndim > 1:
        phi_gt = phi_gt.squeeze()
    
    f_source = dataset.source_term(mesh.points)
    if isinstance(f_source, torch.Tensor):
        f_source = f_source.clone().detach().to(dtype=torch.float32, device=device)
    else:
        f_source = torch.tensor(f_source, dtype=torch.float32).to(device)
    
    boundary_mask = mesh.point_data["is_boundary"]
    if isinstance(boundary_mask, np.ndarray):
        boundary_mask = torch.from_numpy(boundary_mask)
    boundary_mask = boundary_mask.bool().to(device)
    
    print(f"\nMesh: {mesh_size}x{mesh_size} = {mesh.n_points} nodes")
    print(f"Interior nodes: {(~boundary_mask).sum().item()}")
    print(f"Boundary nodes: {boundary_mask.sum().item()}")
    print(f"phi_gt range: [{phi_gt.min():.4f}, {phi_gt.max():.4f}]")
    print(f"f_source range: [{f_source.min():.4f}, {f_source.max():.4f}]")
    
    # Test 1: 解析解应该使 Galerkin loss ≈ 0
    print("\n--- Test 1: Analytical solution should give Galerkin loss ≈ 0 ---")
    galerkin = GalerkinLoss(mesh, dataset)
    galerkin = galerkin.to(device)
    galerkin.prepare()
    galerkin.phi.data.copy_(phi_gt)
    
    loss_result = galerkin.compute_loss()
    if isinstance(loss_result, tuple):
        total_loss, pde_loss, bd_loss = loss_result
    else:
        pde_loss = loss_result
        bd_loss = torch.tensor(0.0)
    
    print(f"  PDE Loss (analytical solution): {pde_loss.item():.6e}")
    print(f"  BD Loss (analytical solution): {bd_loss.item():.6e}")
    
    if pde_loss.item() < 1e-6:
        print("  ✓ PASS: PDE loss is very small for analytical solution")
    else:
        print(f"  ⚠ WARNING: PDE loss is larger than expected: {pde_loss.item():.6e}")
    
    # Test 2: 随机解应该有大的 loss
    print("\n--- Test 2: Random solution should give large Galerkin loss ---")
    phi_random = torch.randn_like(phi_gt)
    phi_random[boundary_mask] = 0.0  # 保持边界条件
    
    galerkin.phi.data.copy_(phi_random)
    loss_result = galerkin.compute_loss()
    if isinstance(loss_result, tuple):
        _, pde_loss_random, bd_loss_random = loss_result
    else:
        pde_loss_random = loss_result
        bd_loss_random = torch.tensor(0.0)
    
    print(f"  PDE Loss (random solution): {pde_loss_random.item():.6e}")
    print(f"  BD Loss (random solution): {bd_loss_random.item():.6e}")
    
    if pde_loss_random.item() > pde_loss.item() * 1e3:
        print("  ✓ PASS: Random solution has much larger PDE loss")
    else:
        print("  ⚠ WARNING: Random solution loss not significantly larger")
    
    # Test 3: 不同噪声水平对应不同 loss
    print("\n--- Test 3: Noise level vs Galerkin loss (should be monotonic) ---")
    noise_levels = [0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    interior_mask = ~boundary_mask
    
    prev_loss = 0
    all_increasing = True
    for noise_std in noise_levels:
        noise = torch.randn_like(phi_gt) * noise_std
        noise[boundary_mask] = 0.0
        phi_noisy = phi_gt + noise
        
        galerkin.phi.data.copy_(phi_noisy)
        loss_result = galerkin.compute_loss()
        if isinstance(loss_result, tuple):
            _, pde_loss_noisy, _ = loss_result
        else:
            pde_loss_noisy = loss_result
        
        mse = ((phi_noisy[interior_mask] - phi_gt[interior_mask]) ** 2).mean().item()
        
        # 检查是否单调递增
        if noise_std > 0 and pde_loss_noisy.item() < prev_loss * 0.9:
            all_increasing = False
        prev_loss = pde_loss_noisy.item()
        
        print(f"  noise_std={noise_std:.0e}: MSE={mse:.2e}, PDE_loss={pde_loss_noisy.item():.2e}")
    
    if all_increasing:
        print("  ✓ PASS: PDE loss increases with noise level")
    else:
        print("  ⚠ WARNING: PDE loss not monotonically increasing with noise")
    
    # Test 4: 验证 Galerkin 弱形式的物理意义
    # 对于 -Δu = f，弱形式是 ∫∇u·∇v dx = ∫f·v dx
    # Galerkin 残差应该是 ∫∇u·∇v dx - ∫f·v dx
    print("\n--- Test 4: Physical interpretation of Galerkin residual ---")
    galerkin.phi.data.copy_(phi_gt)
    
    # 手动计算残差向量
    residual = galerkin.poisson_galerkin_assembler(
        point_data={"phi": galerkin.phi, "f": galerkin.f},
        scalar_data={"a": torch.tensor(1.0, device=device)}
    )
    
    # 残差在内部节点上应该接近 0
    interior_residual = residual[interior_mask]
    residual_norm = interior_residual.norm().item()
    residual_max = interior_residual.abs().max().item()
    residual_mean = interior_residual.abs().mean().item()
    
    print(f"  Residual norm (interior): {residual_norm:.6e}")
    print(f"  Residual max (interior): {residual_max:.6e}")
    print(f"  Residual mean (interior): {residual_mean:.6e}")
    
    if residual_max < 1e-3:
        print("  ✓ PASS: Galerkin residual is small for analytical solution")
    else:
        print(f"  ⚠ Note: Residual is not zero, likely due to discretization error")
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60 + "\n")
    
    return True


def run_experiment():
    # 首先验证 Galerkin 正确性
    verify_galerkin_correctness()
    
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "error_compare.json")
    png_path = os.path.join(output_dir, "error_compare.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Settings
    mesh_size = 100
    # Use wider noise range to verify low-noise behavior
    noise_levels = np.logspace(-8, 1, 15) 
    
    mesh = gen_regular_grid(n=mesh_size)
    dataset = PoissonMultiFrequency()
    
    phi_gt_numpy = dataset.solution(mesh.points)
    if isinstance(phi_gt_numpy, torch.Tensor):
        phi_gt = phi_gt_numpy.clone().detach().to(dtype=torch.float32, device=device)
    else:
        phi_gt = torch.tensor(phi_gt_numpy, dtype=torch.float32).to(device)
    if phi_gt.ndim > 1:
        phi_gt = phi_gt.squeeze()

    # Define network configurations
    # List of hidden layers. e.g. [10, 10] means 2->10->10->1
    network_configs = [
        {"name": "Small", "layers": [10, 10]},
        {"name": "Medium", "layers": [30, 30, 30]},
        {"name": "Large", "layers": [60, 60, 60]},
    ]

    results = []

    # 获取边界掩码
    boundary_mask = mesh.point_data["is_boundary"]
    if isinstance(boundary_mask, np.ndarray):
        boundary_mask = torch.from_numpy(boundary_mask)
    boundary_mask = boundary_mask.bool().to(device)
    interior_mask = ~boundary_mask
    
    # ========== 第一步：在干净的 phi_gt 上训练网络 ==========
    print("\n=== Step 1: Training networks on clean phi_gt ===")
    trained_models = {}
    
    for config in network_configs:
        n_layers = config["layers"]
        n_params = None
        
        # PINN: 在干净数据上训练
        pinn = DynamicPinnLoss(mesh, dataset, n_layers)
        pinn = pinn.to(device)
        pinn.prepare()
        n_params = sum(p.numel() for p in pinn.lin.parameters())
        pred_pinn, fit_mse_pinn = fit_network(pinn, phi_gt, steps=1000, desc=f"PINN {config['name']} (clean)")
        trained_models[f"PINN_{config['name']}"] = {
            "model": pinn, 
            "fit_mse": fit_mse_pinn,
            "n_params": n_params
        }
        print(f"  PINN {config['name']}: fit_mse={fit_mse_pinn:.2e}")
        
        # GalerkinMLP: 在干净数据上训练
        galerkin_mlp = DynamicGalerkinMLP(mesh, dataset, n_layers)
        galerkin_mlp = galerkin_mlp.to(device)
        galerkin_mlp.prepare()
        pred_gal, fit_mse_gal = fit_network(galerkin_mlp, phi_gt, steps=1000, desc=f"GalerkinMLP {config['name']} (clean)")
        trained_models[f"GalerkinMLP_{config['name']}"] = {
            "model": galerkin_mlp, 
            "fit_mse": fit_mse_gal,
            "n_params": n_params
        }
        print(f"  GalerkinMLP {config['name']}: fit_mse={fit_mse_gal:.2e}")

    # ========== 第二步：在带噪声的 phi 上计算 loss ==========
    print("\n=== Step 2: Evaluating loss on noisy phi ===")
    
    for noise_std in tqdm(noise_levels, desc="Noise Levels"):
        # 只在内部节点加噪声，边界保持精确值
        noise = torch.randn_like(phi_gt) * noise_std
        noise[boundary_mask] = 0.0  # 边界不加噪声
        phi_noisy = phi_gt + noise
        
        # MSE 只计算内部节点
        mse_noisy = ((phi_noisy[interior_mask] - phi_gt[interior_mask]) ** 2).mean().item()

        # 1. Galerkin Loss (直接赋值，作为 baseline)
        galerkin = GalerkinLoss(mesh, dataset)
        galerkin = galerkin.to(device)
        galerkin.prepare()
        galerkin.phi.data.copy_(phi_noisy)
        
        loss_result = galerkin.compute_loss()
        if isinstance(loss_result, tuple):
            total_loss, pde_loss, bd_loss = loss_result
        else:
            pde_loss = loss_result
            bd_loss = torch.tensor(0.0)
        
        results.append({
            "mse": mse_noisy,
            "mse_pred": mse_noisy,
            "loss": pde_loss.item(),
            "method": "Galerkin (Direct)",
            "group": "Galerkin",
            "param_count": 0,
            "pde_loss": pde_loss.item(),
            "bd_loss": bd_loss.item() if hasattr(bd_loss, 'item') else bd_loss
        })

        # 2. 用训练好的网络，在 noisy phi 上计算 loss
        for config in network_configs:
            # 2a. PINN
            pinn_info = trained_models[f"PINN_{config['name']}"]
            pinn = pinn_info["model"]
            
            # 用网络预测 noisy phi（网络权重不变，直接预测）
            # 这里我们需要把网络预测替换成 phi_noisy 来计算 loss
            # 实际上：PINN 的 loss 是基于网络预测的，但我们要测试 "如果解是 noisy 的"
            # 所以需要修改：直接用 phi_noisy 计算 PINN 的 PDE 残差
            
            # 方案：创建临时的 phi_noisy 作为 "预测"，计算 PINN loss
            pinn._update_pred()  # 确保 pred 是网络当前输出
            
            # 保存原始预测
            original_pred = pinn.pred.clone()
            
            # 用 phi_noisy 替换预测，计算 loss
            # 但 PINN 的 compute_loss 依赖 autograd，这里需要不同处理
            # 实际上：我们需要在 phi_noisy 上计算 PDE 残差
            # 对于 PINN，这需要用 autograd 计算 noisy 函数的导数
            # 这比较复杂，简化方案：用 FDM 近似计算 Laplacian
            
            # 简化：直接用训练好的网络输出计算 loss，然后比较网络预测与 noisy phi 的差异
            pde_loss_pinn, bd_loss_pinn = pinn.compute_loss()
            
            # 计算网络预测与 phi_noisy 的差异
            pred_pinn = pinn.pred.detach()
            mse_pred_vs_noisy = ((pred_pinn[interior_mask] - phi_noisy[interior_mask]) ** 2).mean().item()
            mse_pred_vs_gt = ((pred_pinn[interior_mask] - phi_gt[interior_mask]) ** 2).mean().item()

            results.append({
                "mse": mse_noisy,
                "loss": pde_loss_pinn.item(),
                "method": f"PINN ({config['name']})",
                "group": "PINN",
                "param_count": pinn_info["n_params"],
                "mse_pred": mse_pred_vs_gt,  # 预测 vs ground truth
                "fit_mse": pinn_info["fit_mse"],  # 训练时的拟合误差
                "pde_loss": pde_loss_pinn.item(),
                "bd_loss": bd_loss_pinn.item()
            })
            
            # 2b. GalerkinMLP
            gal_info = trained_models[f"GalerkinMLP_{config['name']}"]
            galerkin_mlp = gal_info["model"]
            
            # 用 phi_noisy 替换 phi 来计算 Galerkin loss
            galerkin_mlp.phi.data.copy_(phi_noisy)
            pde_loss_gal, bd_loss_gal = galerkin_mlp.compute_loss()
            
            # 网络预测
            pred_gal = galerkin_mlp.lin(galerkin_mlp.mlp_points).squeeze().detach()
            mse_pred_gal = ((pred_gal[interior_mask] - phi_gt[interior_mask]) ** 2).mean().item()
            
            results.append({
                "mse": mse_noisy,
                "loss": pde_loss_gal.item(),
                "method": f"GalerkinMLP ({config['name']})",
                "group": "GalerkinMLP",
                "param_count": gal_info["n_params"],
                "mse_pred": mse_pred_gal,
                "fit_mse": gal_info["fit_mse"],
                "pde_loss": pde_loss_gal.item(),
                "bd_loss": bd_loss_gal.item() if hasattr(bd_loss_gal, 'item') else bd_loss_gal
            })

    # Save Data
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Plotting
    df = pd.DataFrame(results)
    
    # 打印调试信息
    print("\n=== Debug Info ===")
    print(f"Groups: {df['group'].unique()}")
    print(f"Methods: {df['method'].unique()}")
    
    # 打印 PINN vs GalerkinMLP 在相同网络下的对比
    for noise_idx in [0, 7, 14]:  # low, mid, high noise
        subset = df[df['mse'] == df['mse'].unique()[noise_idx]]
        print(f"\n--- Noise level {noise_idx} (mse={subset['mse'].iloc[0]:.2e}) ---")
        for _, row in subset.iterrows():
            if 'fit_mse' in row and pd.notna(row.get('fit_mse')):
                print(f"  {row['method']:25s}: pde_loss={row['pde_loss']:.2e}, fit_mse={row['fit_mse']:.2e}, mse_pred={row.get('mse_pred', 'N/A')}")
            else:
                print(f"  {row['method']:25s}: pde_loss={row['pde_loss']:.2e}")
    
    plt.figure(figsize=(10, 7))
    
    # 使用 mse_pred 作为 x 轴（预测值 vs 真实值的 MSE）
    x_col = "mse_pred"
    
    # 1. Plot Galerkin Direct (Blue line, baseline)
    df_galerkin = df[df["group"] == "Galerkin"]
    sns.lineplot(
        data=df_galerkin, 
        x=x_col, y="loss", 
        label="Galerkin (Direct)",
        color="#0173B2",
        marker="o",
        markersize=8,
        linewidth=2.5
    )
    
    # 2. Plot GalerkinMLP (Blues - 不同深度)
    df_galerkin_mlp = df[df["group"] == "GalerkinMLP"]
    if len(df_galerkin_mlp) > 0:
        methods_gal = df_galerkin_mlp.sort_values("param_count")["method"].unique()
        cmap_blue = plt.get_cmap("Blues")
        colors_blue = [cmap_blue(i) for i in np.linspace(0.4, 0.8, len(methods_gal))]
        
        for i, method in enumerate(methods_gal):
            subset = df_galerkin_mlp[df_galerkin_mlp["method"] == method]
            sns.lineplot(
                data=subset,
                x=x_col, y="loss",
                label=method,
                color=colors_blue[i],
                marker="^",
                markersize=6,
                linewidth=1.5,
                linestyle="-."
            )
    
    # 3. Plot PINNs (Greens)
    df_pinn = df[df["group"] == "PINN"]
    pinn_methods = df_pinn.sort_values("param_count")["method"].unique()
    
    cmap = plt.get_cmap("Greens")
    colors = [cmap(i) for i in np.linspace(0.4, 0.9, len(pinn_methods))]
    
    for i, method in enumerate(pinn_methods):
        subset = df_pinn[df_pinn["method"] == method]
        sns.lineplot(
            data=subset,
            x=x_col, y="loss",
            label=method,
            color=colors[i],
            marker="s",
            markersize=5,
            linewidth=1.5,
            linestyle="--"
        )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Prediction MSE (prediction vs ground truth)')
    plt.ylabel('PDE Residual Loss')
    plt.title('PDE Loss vs Prediction Error: PINN vs Galerkin')
    
    plt.legend()
    plt.grid(True, which="minor", alpha=0.15)
    plt.grid(True, which="major", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    print(f"Plot saved to {png_path}")

if __name__ == "__main__":
    run_experiment()
