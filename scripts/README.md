# scripts/ 目录说明

2026-08-23 重构后按功能分层。全部脚本内置**深度无关的项目根 bootstrap**，
可在任意子目录下直接运行，无需 `export PYTHONPATH`。

| 子目录 | 用途 | 典型脚本 |
|---|---|---|
| `train/` | 骨干预训练（冻结） | `pretrain_bwgnn.py`, `pretrain_gadnr.py`, `run_*_pretrain.sh` |
| `evaluate/` | 主推理与多种子 | `run_structcp.py`, `run_structcp_seeds.py`, `run_graph_subspace_ad.py`, `run_tune_baseline.py` |
| `compare/` | 基线对比（20+ 方法） | `run_compare_baselines*.py`, `run_all_baselines.py`, `run_gadnr_official.py` |
| `ablation/` | 消融与鲁棒性 | `run_ablation*.py`, `run_*_ablation.py`, `sweep_*.py`, `p5_*.py`, `p9_*.py` |
| `analysis/` | 聚合 / 计算 / 诊断 | `aggregate_*.py`, `collect_*.py`, `compute_*.py`, `diagnose_*.py` |
| `tables/` | 论文表格 | `gen_*.py`, `make_*.py` |
| `figures/` | 论文图 | `plot_*.py`, `p9_plot_fpr_alpha.py` |
| `data_prep/` | 数据转换 | `convert_tfs.py`, `build_reddit_mat.py`, `build_taggad_mats.py` |
| `orchestrate/` | 编排器 | `run_experiments.py`, `collect_manifest.py` |
| `archive/` | 一次性/下载脚本（不再维护） | `gadbench_*.sh`, `_dedup_notes.py` |

完整脚本索引见 `docs/SCRIPT_INDEX.md`；工作流见 `docs/RUNBOOK.md`。
