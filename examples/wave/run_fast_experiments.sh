#!/bin/bash
source ~/miniforge3/bin/activate tensormesh-bench
cd /minimax-dialogue/users/walker/projects/TensorMesh/examples/wave

# 使用更粗的网格和更少的时间步来加速

# 实验1: Direct模型 基准
echo "========== Experiment 1: Direct Baseline =========="
python defect_detection_v2.py \
  --dataset circle \
  --model direct \
  --epoch 200 \
  --lr 0.1 \
  --n_sources 3 \
  --n_detector 100 \
  --lambda_tv 5e-4 \
  --lambda_laplacian 0 \
  --use_freq_loss \
  --freq_weight 0.1 \
  --chara_length 0.05 \
  --n 100 \
  --dt 0.008 \
  --save_cache exp1_direct.pt \
  --cuda 0

# 实验2: Direct + 更高学习率 + 更多检测器
echo "========== Experiment 2: Direct + High LR =========="
python defect_detection_v2.py \
  --dataset circle \
  --model direct \
  --epoch 200 \
  --lr 0.15 \
  --n_sources 5 \
  --n_detector 150 \
  --lambda_tv 1e-3 \
  --lambda_laplacian 0 \
  --use_freq_loss \
  --freq_weight 0.2 \
  --chara_length 0.05 \
  --n 100 \
  --dt 0.008 \
  --save_cache exp2_highLR.pt \
  --cuda 0

# 实验3: Direct + 强TV正则
echo "========== Experiment 3: Direct + Strong TV =========="
python defect_detection_v2.py \
  --dataset circle \
  --model direct \
  --epoch 200 \
  --lr 0.12 \
  --n_sources 3 \
  --n_detector 100 \
  --lambda_tv 2e-3 \
  --lambda_laplacian 1e-4 \
  --use_freq_loss \
  --freq_weight 0.15 \
  --chara_length 0.05 \
  --n 100 \
  --dt 0.008 \
  --save_cache exp3_strongTV.pt \
  --cuda 0

# 实验4: RBF模型
echo "========== Experiment 4: RBF Model =========="
python defect_detection_v2.py \
  --dataset circle \
  --model rbf \
  --n_rbf 100 \
  --epoch 300 \
  --lr 0.02 \
  --n_sources 3 \
  --n_detector 100 \
  --lambda_tv 1e-4 \
  --lambda_laplacian 0 \
  --use_freq_loss \
  --freq_weight 0.1 \
  --chara_length 0.05 \
  --n 100 \
  --dt 0.008 \
  --save_cache exp4_rbf.pt \
  --cuda 0

# 实验5: MLP模型 (较小网络)
echo "========== Experiment 5: MLP Model =========="
python defect_detection_v2.py \
  --dataset circle \
  --model mlp \
  --hidden_dim 64 \
  --n_layers 4 \
  --epoch 300 \
  --lr 0.005 \
  --n_sources 3 \
  --n_detector 100 \
  --lambda_tv 0 \
  --lambda_laplacian 5e-5 \
  --use_freq_loss \
  --freq_weight 0.1 \
  --chara_length 0.05 \
  --n 100 \
  --dt 0.008 \
  --save_cache exp5_mlp.pt \
  --cuda 0

echo "All experiments completed!"
echo ""
echo "=== 结果汇总 ==="
for f in exp*.pt; do
  echo "--- $f ---"
  python -c "import torch; d=torch.load('$f'); print(f'Final loss: {d[\"losses\"][-1]:.6e}'); print(f'Mean error: {(d[\"c_pred\"]-d[\"c_gt\"]).abs().mean():.4f}')"
done

