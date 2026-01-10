#!/bin/bash
source ~/miniforge3/bin/activate tensormesh-bench
cd /minimax-dialogue/users/walker/projects/TensorMesh/examples/wave

# 实验1: Direct模型 + 高学习率
echo "========== Experiment 1: Direct + High LR =========="
python defect_detection_v2.py \
  --dataset circle \
  --model direct \
  --epoch 500 \
  --lr 0.1 \
  --n_sources 3 \
  --n_detector 200 \
  --lambda_tv 1e-3 \
  --lambda_laplacian 0 \
  --use_freq_loss \
  --freq_weight 0.2 \
  --chara_length 0.03 \
  --n 200 \
  --save_cache exp1_direct_highLR.pt \
  --cuda 0

# 实验2: Direct模型 + 更多源
echo "========== Experiment 2: Direct + 5 Sources =========="
python defect_detection_v2.py \
  --dataset circle \
  --model direct \
  --epoch 500 \
  --lr 0.05 \
  --n_sources 5 \
  --n_detector 300 \
  --lambda_tv 5e-4 \
  --lambda_laplacian 1e-5 \
  --use_freq_loss \
  --freq_weight 0.1 \
  --chara_length 0.03 \
  --n 200 \
  --save_cache exp2_direct_5src.pt \
  --cuda 0

# 实验3: MLP模型
echo "========== Experiment 3: MLP Model =========="
python defect_detection_v2.py \
  --dataset circle \
  --model mlp \
  --hidden_dim 128 \
  --n_layers 5 \
  --epoch 800 \
  --lr 0.001 \
  --n_sources 3 \
  --n_detector 200 \
  --lambda_tv 0 \
  --lambda_laplacian 1e-4 \
  --use_freq_loss \
  --freq_weight 0.1 \
  --chara_length 0.03 \
  --n 200 \
  --save_cache exp3_mlp.pt \
  --cuda 0

# 实验4: RBF模型
echo "========== Experiment 4: RBF Model =========="
python defect_detection_v2.py \
  --dataset circle \
  --model rbf \
  --n_rbf 225 \
  --epoch 600 \
  --lr 0.01 \
  --n_sources 3 \
  --n_detector 200 \
  --lambda_tv 1e-4 \
  --lambda_laplacian 0 \
  --use_freq_loss \
  --freq_weight 0.15 \
  --chara_length 0.03 \
  --n 200 \
  --save_cache exp4_rbf.pt \
  --cuda 0

# 实验5: Direct + AdamW + 强正则
echo "========== Experiment 5: Direct + AdamW + Strong Reg =========="
python defect_detection_v2.py \
  --dataset circle \
  --model direct \
  --optimizer adamw \
  --epoch 600 \
  --lr 0.08 \
  --weight_decay 1e-4 \
  --n_sources 3 \
  --n_detector 200 \
  --lambda_tv 2e-3 \
  --lambda_laplacian 1e-4 \
  --use_freq_loss \
  --freq_weight 0.2 \
  --chara_length 0.03 \
  --n 200 \
  --save_cache exp5_adamw_strongreg.pt \
  --cuda 0

echo "All experiments completed!"
