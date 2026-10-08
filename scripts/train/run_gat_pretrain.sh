#!/usr/bin/env bash
# A4 消融: GAT 三个大图预训练 (CPU, 8GB RAM 约束; GPU 8GB 跑不下 7.7M+ 边 GAT)
# 用法: nohup bash scripts/run_gat_pretrain.sh > outputs/gat_pretrain_all.log 2>&1 &
set -e
cd /media/lixin/新加卷/数据集/test/StructCP
source <CONDA_PREFIX>/bin/activate CBP
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"

for DS in YelpChi TFinance Amazon; do
  echo "========== GAT pretrain $DS (CPU) =========="
  python -u scripts/pretrain_bwgnn.py \
    --dataset $DS --encoder gat --hidden 128 --epochs 100 \
    --device cpu --resume 2>&1 | tee -a outputs/gat_pretrain_${DS}.log
  echo "---------- $DS done ----------"
done
echo "ALL GAT PRETRAIN DONE"
