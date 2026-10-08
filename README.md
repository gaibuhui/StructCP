# StructCP: Structure-Aware Empirical Calibration for Graph Anomaly Detection under Normality Shift

Code for the journal submission *"StructCP: Structure-Aware Empirical Calibration for Graph Anomaly Detection under Normality Shift"*.

## Overview

Graph anomaly detection (GAD) systems face *normality shift*: legitimate normal classes unseen at calibration appear at test time, breaking the exchangeability assumption of split conformal prediction and causing uncontrolled false positive rates (FPR). StructCP is a **drop-in, optimization-free recalibration wrapper** for any frozen GAD detector that:

- reuses **one k-NN graph** for both test-time feature propagation and personalized-PageRank (PPR) structure weighting;
- recalibrates the frozen detector's scores with a **weighted quantile over test-time pseudo-normal samples**;
- adds a **dual threshold with an abstain option** to bound FPR and FNR jointly on weak backbones.

StructCP holds FPR <= alpha = 0.05 on all four benchmarks (Amazon, YelpChi, Elliptic, T-Finance) at the 5-seed mean, is test-time label-free, and requires no retraining of the backbone.

## Repository layout

```
adapters/    Core recalibration pipeline (hypergraph TTA, conformal recalibration, pipeline, registry)
models/      Backbone models (BWGNN, GAD-NR, encoders, graph-subspace AD)
utils/       Shared utilities (paths, seeds, config, checkpoint, results, metrics)
data/        GAD data loading and normality-shift construction
configs/     Default configuration
scripts/     Experiments and analyses (train / evaluate / compare / ablation / analysis / tables / figures / data_prep)
tests/       Unit tests
```

## Requirements

- Python 3.x, PyTorch (tested with torch 2.0+), DGL / PyTorch Geometric
- See `requirements.txt`

## Quick start

```bash
# 1. Pre-train a frozen backbone (optional; pretrained checkpoints may be used)
python scripts/train/pretrain_bwgnn.py --dataset amazon

# 2. Run StructCP recalibration on a frozen detector
python scripts/evaluate/run_structcp.py --dataset amazon --mode hard

# 3. Reproduce main-table numbers across seeds
python scripts/evaluate/run_structcp_seeds.py --dataset all
```

## Data

The benchmark datasets are public:

- **Amazon, YelpChi** (CARE-GNN / BWGNN format)
- **Elliptic, T-Finance** (TFS format)
- **TUNE** graph benchmarks (photo, computer, weibo, tolokers, questions, reddit)

## Citation

```
(to be added)
```

## License

(to be added)
