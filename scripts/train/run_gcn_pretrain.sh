#!/usr/bin/env bash
# A4 消融: GCN 四个数据集预训练 (当前稀疏 matmul 版, 结构对齐 gcn_encoder.py)
# 旧 checkpoint 是 GCNConv 版, 结构不匹配, 需重训。
# 用法: nohup bash scripts/run_gcn_pretrain.sh > outputs/gcn_pretrain_all.log 2>&1 &
set -e
cd /media/lixin/新加卷/数据集/test/StructCP
source <CONDA_PREFIX>/bin/activate CBP
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"

for DS in Amazon YelpChi Elliptic TFinance; do
  echo "========== GCN pretrain $DS (CPU, sparse) =========="
  python -u scripts/pretrain_bwgnn.py \
    --dataset $DS --encoder gcn --hidden 128 --epochs 100 \
    --device cpu --resume 2>&1 | tee -a outputs/gcn_pretrain_${DS}.log
  echo "---------- $DS done ----------"
done
echo "ALL GCN PRETRAIN DONE"
