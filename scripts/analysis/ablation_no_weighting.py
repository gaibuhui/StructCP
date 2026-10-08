# -*- coding: utf-8 -*-
"""
C1 消融: 分离"加权分位"与"伪正常扩增"的独立贡献。

对照:
  * Full        = contrastive 一致性权重 + 伪正常扩增 (主表口径, 已有)
  * w/o-weight  = uniform 权重 (1.0) + 伪正常扩增  ← 本脚本
  * w/o-pseudo  = contrastive 权重, 无扩增 (已有消融数据见 README 6.3.1, 口径为 adaptive)

本脚本只跑 w/o-weight 一行 (4 主数据集 × 5 seeds, fixed alpha=0.05),
与 outputs/structcp_contrastive_seeds_fixedA_all.json 的 Full 行配对成表。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/ablation_no_weighting.py --datasets Amazon,YelpChi,Elliptic,TFinance \
      --seeds 42,123,456,789,1011 --out ablation_no_weighting.json
"""
import argparse, io, json, os, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp, DEFAULT_DATASET_CFG  # noqa: E402

METRICS = ["AUROC", "AUPRC", "FPR", "TPR", "F1", "FPR_novel_normality"]


def aggregate(per_seed, metrics):
    agg = {}
    for m in metrics:
        vals = [per_seed[s]["StructCP"][m] for s in per_seed if m in per_seed[s]["StructCP"]]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"] = float(np.std(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456,789,1011")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="ablation_no_weighting.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    summary = {}
    for ds in datasets:
        if ds not in DEFAULT_DATASET_CFG:
            print(f"[skip] unknown dataset {ds}")
            continue
        cfg = dict(DEFAULT_DATASET_CFG[ds])
        per_seed = {}
        for seed in seeds:
            out = evaluate_structcp(
                ds, seed=seed, alpha=args.alpha, device=args.device, root=_ROOT,
                adaptive_alpha=False,  # 与主表口径一致: fixed alpha=0.05
                weight_mode="uniform",  # C1: 去结构加权, 保留伪正常扩增
            )
            r = out["results"]["StructCP"]
            per_seed[seed] = {"StructCP": r}
            print(f"  [{ds} seed={seed}] w/o-weight FPR={r['FPR']:.4f} TPR={r['TPR']:.4f} "
                  f"F1={r['F1']:.4f} n_pseudo={r.get('n_pseudo_normal')}", flush=True)
        summary[ds] = {
            "config": cfg, "alpha": args.alpha, "seeds": seeds,
            "method": "StructCP-uniform (w/o structural weighting, pseudo kept)",
            "agg": aggregate(per_seed, METRICS), "per_seed": per_seed,
        }
        a = summary[ds]["agg"]
        print(f"{ds}: w/o-weight FPR={a['FPR_mean']:.4f}±{a['FPR_std']:.4f}  "
              f"TPR={a['TPR_mean']:.4f}±{a['TPR_std']:.4f}", flush=True)

    out_path = os.path.join(_ROOT, "outputs", args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
