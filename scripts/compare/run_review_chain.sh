#!/usr/bin/env bash
# 评审回应实验链：等待校准基线结束 → 流式/归纳 → 骨干无关性 → bootstrap CI
set -e
cd /media/lixin/新加卷/数据集/test/StructCP
source <CONDA_PREFIX>/bin/activate CBP
export PYTHONPATH="$PWD"

echo "[chain] waiting for calibration_baselines to finish..."
while pgrep -f run_calibration_baselines.py > /dev/null; do sleep 20; done
echo "[chain] calibration done, running streaming eval..."
python scripts/compare/run_structcp_streaming.py --seeds 42,123,456 --device cuda 2>&1 | tail -20

echo "[chain] running backbone sensitivity..."
python scripts/ablation/run_backbone_sensitivity.py \
  --datasets Amazon,YelpChi,Elliptic,TFinance \
  --backbones BWGNN,GCN,GAT --seeds 42,123,456,789,1011 --device cuda 2>&1 | tail -30

echo "[chain] computing bootstrap CI..."
python scripts/analysis/compute_bootstrap_ci.py 2>&1 | tail -30
echo "[chain] ALL DONE"
