#!/bin/bash
set -e
source /home/lixin/miniconda3/bin/activate CBP
export PYTHONPATH=/media/lixin/新加卷/数据集/test/StructCP
cd /media/lixin/新加卷/数据集/test/StructCP
python scripts/run_all_baselines.py \
    --methods UNPrompt \
    --datasets main,tune \
    --device cpu \
    --out outputs/compare_all