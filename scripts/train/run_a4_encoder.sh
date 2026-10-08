#!/usr/bin/env bash
# A4 编码器消融: 4 数据集 × 3 encoder (BWGNN/GCN/GAT), 下游流程完全一致
# 用法: nohup bash scripts/run_a4_encoder.sh > outputs/a4_encoder_all.log 2>&1 &
set -e
cd /media/lixin/新加卷/数据集/test/StructCP
source <CONDA_PREFIX>/bin/activate CBP
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"

for DS in Amazon YelpChi Elliptic TFinance; do
  echo "========== [A4] encoder ablation $DS =========="
  python -u scripts/run_graph_subspace_ad_ablation.py \
    --ablation encoder --dataset $DS --device cpu \
    --alpha_base 0.6 --out outputs/graph_subspace_ad 2>&1 | tee -a outputs/a4_encoder_${DS}.log
  echo "---------- $DS done ----------"
done
echo "ALL A4 ENCODER ABLATION DONE"
