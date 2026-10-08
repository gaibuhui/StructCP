"""GADT3 多 seed 稳健性验证 (仅 Amazon, P1 任务)。

目的: 审稿人看到 GADT3 在 Amazon 的 TPR=0.198 是单 seed=42 的结果, 可能怀疑
"初始化运气差 / 抽中坏 seed"。本脚本串行跑多个 seed, 验证 TPR 崩塌是
系统性的 (近零 TPR 跨 seed 稳定出现), 而非随机坏运气, 从而坐实论文
"GADT3 低 FPR 是 score collapse 的退化伪影" 论点。

复用 run_compare_baselines 的统一口径: split_conformal (alpha=0.05) 定阈,
GADT3 独立 GraphSAGE+MLP 检测器 + 同配性自监督 TTA。

运行 (前台, 实时 flush):
  cd /media/lixin/新加卷/数据集/test/StructCP && \
  source <CONDA_PREFIX>/bin/activate CBP && \
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \
  python scripts/run_gadt3_seeds.py --dataset Amazon --seeds 42,123,456
"""
from __future__ import annotations

# --- StructCP 项目根定位（深度无关）：任意脚本深度下均可定位根目录 ---
import os as _os, sys as _sys
def _structcp_root():
    _p = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(10):
        if _os.path.isfile(_os.path.join(_p, 'configs', 'default.yaml')):
            return _p
        _p = _os.path.dirname(_p)
    return _p
if _structcp_root() not in _sys.path:
    _sys.path.insert(0, _structcp_root())


import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, _structcp_root())

from adapters.conformal import evaluate_coverage, split_conformal_threshold  # noqa: E402
# 2026-09-24: baselines.tune 模块在发布版中不存在，且此处并不实际使用该符号，
# 注释掉以避免 ModuleNotFoundError 阻断 GADT3 self-run。
# from baselines.tune import TuneAligner  # noqa: E402
from data.loader import load_gad  # noqa: E402

_BASELINES = os.path.join(_structcp_root(), "baselines")
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)
from gadt3_pyg.adapter import GADT3Adapter  # noqa: E402

ROOT = _structcp_root()

METRICS = ["AUROC", "AUPRC", "FPR", "TPR", "F1", "FPR_novel"]


def gadt3_metrics_for_seed(dataset, relation, seed, alpha, k, device="cpu"):
    print(f"\n{'#'*80}\n# GADT3 seed={seed}  [{dataset}/{relation}] dev={device}\n{'#'*80}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    print("[data]", data, flush=True)

    gadt3 = GADT3Adapter(ndim_s=data.num_features, device=device)
    t0 = time.time()
    gadt3.fit_source(data)
    gadt3.fit_target(data)
    t_gadt3 = time.time() - t0
    s_gadt3 = gadt3.score(data)

    cal, te = data.cal_mask, data.test_mask
    y_te = data.y[te].cpu().numpy()
    y = data.y
    nov = data.novel_mask & te

    s_dev = s_gadt3.device  # 分数可能在 cuda/cpu，掩码对齐到分数设备
    cal_norm_full = (cal & (y == 0)).to(s_dev)
    thr = split_conformal_threshold(s_gadt3[cal_norm_full].cpu(), alpha)
    from sklearn.metrics import average_precision_score, roc_auc_score
    from adapters.conformal import evaluate_coverage as _ec
    r = {"AUROC": float(roc_auc_score(y_te, s_gadt3[te.to(s_dev)].cpu().numpy())),
         "AUPRC": float(average_precision_score(y_te, s_gadt3[te.to(s_dev)].cpu().numpy())),
         "infer_time_s": float(t_gadt3)}
    r.update(_ec(s_gadt3[te.to(s_dev)], data.y[te].to(s_dev), thr))
    if int(nov.sum()) > 0:
        r["FPR_novel"] = float((s_gadt3[nov.to(s_dev)] > thr).float().mean())
    r["trainable_params"] = int(sum(p.numel() for p in gadt3.model.parameters()))
    print(f"[seed={seed}] AUROC={r['AUROC']:.4f} FPR={r['FPR']:.4f} "
          f"TPR={r['TPR']:.4f} F1={r['F1']:.4f} FPR_nov={r.get('FPR_novel', float('nan')):.4f}",
          flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon",
                    choices=["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--seeds", default="42,123,456", help="逗号分隔 seed 列表")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cuda", help="cuda/cpu (GPU 8GB 可能 OOM 时回退 cpu)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    relation = args.relation if args.dataset != "Elliptic" else "homo"

    all_r = {}
    for seed in seeds:
        all_r[seed] = gadt3_metrics_for_seed(
            args.dataset, relation, seed, args.alpha, args.k, device=args.device)

    # ---- 汇总均值 / 标准差 ----
    agg = {"seeds": seeds, "n": len(seeds), "alpha": args.alpha}
    per_seed = {m: [] for m in METRICS}
    for seed in seeds:
        for m in METRICS:
            if m in all_r[seed]:
                per_seed[m].append(all_r[seed][m])
    for m in METRICS:
        vals = per_seed[m]
        if vals:
            arr = np.array(vals, dtype=float)
            agg[f"{m}_mean"] = arr.mean()
            agg[f"{m}_std"] = arr.std(ddof=0)
            agg[f"{m}_min"] = float(arr.min())
            agg[f"{m}_max"] = float(arr.max())

    print(f"\n{'='*80}\nGADT3 multi-seed summary  {args.dataset}/{relation}  "
          f"seeds={seeds}  alpha={args.alpha}\n{'='*80}", flush=True)
    for m in METRICS:
        if m in agg:
            print(f"  {m:<10} mean={agg[m+'_mean']:.4f}  std={agg[m+'_std']:.4f}  "
                  f"min={agg[m+'_min']:.4f}  max={agg[m+'_max']:.4f}", flush=True)
    print("=" * 80, flush=True)

    out = os.path.join(ROOT, "outputs", f"gadt3_seeds_{args.dataset}_{relation}.json")
    json.dump({"agg": agg, "per_seed": all_r}, open(out, "w"), indent=2)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
