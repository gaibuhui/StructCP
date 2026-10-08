"""StructCP-PPR 多 seed 稳健性验证（Elliptic/TFinance 社区型为主）。

在 `adapters.pipeline.evaluate_structcp` 上开启 weight_mode="ppr" 跨多个 seed
运行，聚合 mean±std，验证 PPR 伪正常门槛对 TPR 修复的稳定性（非单 seed 偶然）。

运行（前台, 实时 flush）:
  source <CONDA_PREFIX>/bin/activate CBP
  python scripts/evaluate/run_structcp_ppr_seeds.py --datasets Elliptic,TFinance --seeds 42,123,456,789,1011
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys

import numpy as np

# 项目根 bootstrap（深度无关）——必须在任何项目内 import 之前执行
_ROOT = None
_p = os.path.dirname(os.path.abspath(__file__))
for _ in range(10):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)
if _ROOT and _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp, DEFAULT_DATASET_CFG, METHOD_NAMES  # noqa: E402
from utils.results import save_json  # noqa: E402

METRICS = ["AUROC", "AUPRC", "FPR", "TPR", "F1", "FPR_novel_normality"]


def run_one(dataset, seed, cfg, alpha, ppr_quantile, device="cpu"):
    """单 seed 运行（内部走共享管线 evaluate_structcp, weight_mode="ppr"）。"""
    out = evaluate_structcp(
        dataset, seed=seed, alpha=alpha, device=device, root=_ROOT,
        relation=cfg["relation"], mode=cfg["mode"], score_quantile=cfg["score_quantile"],
        tau=cfg["tau"], k=cfg["k"], layers=cfg["layers"], alpha_res=cfg["alpha_res"],
        adaptive_alpha=False, weight_mode="ppr", ppr_quantile=ppr_quantile,
    )
    r = out["results"]["StructCP-PPR"]
    print(f"  StructCP-PPR FPR={r['FPR']:.4f} TPR={r['TPR']:.4f} "
          f"(n_pseudo={r.get('n_pseudo_normal')}, tau_ppr={r.get('tau_ppr')})",
          flush=True)
    # weight_mode="ppr" 时主方法名为 "StructCP-PPR"，复制到 "StructCP" 键
    # 以兼容 METHOD_NAMES 聚合（下游按 Frozen/HG-TTA/StructCP 取数）。
    out["results"]["StructCP"] = r
    return out["results"]


def aggregate(per_seed, methods, metrics):
    agg = {}
    for m in methods:
        agg[m] = {}
        for met in metrics:
            vals = [per_seed[s][m][met] for s in per_seed
                    if met in per_seed[s][m]]
            if vals:
                arr = np.array(vals, dtype=float)
                agg[m][f"{met}_mean"] = float(arr.mean())
                agg[m][f"{met}_std"] = float(arr.std(ddof=0))
                agg[m][f"{met}_min"] = float(arr.min())
                agg[m][f"{met}_max"] = float(arr.max())
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456,789,1011")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ppr_quantile", type=float, default=0.4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    methods = list(METHOD_NAMES)

    summary = {}
    for ds in datasets:
        if ds not in DEFAULT_DATASET_CFG:
            print(f"[skip] unknown dataset {ds}")
            continue
        cfg = DEFAULT_DATASET_CFG[ds]
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = run_one(ds, seed, cfg, args.alpha, args.ppr_quantile,
                                     device=args.device)

        agg = aggregate(per_seed, methods, METRICS)

        summary[ds] = {"config": cfg, "alpha": args.alpha, "ppr_quantile": args.ppr_quantile,
                       "seeds": seeds, "agg": agg, "per_seed": per_seed}

        print(f"\n{'='*90}\n{ds}  multi-seed summary (StructCP-PPR)  "
              f"seeds={seeds}  alpha={args.alpha}  ppr_q={args.ppr_quantile}\n{'='*90}",
              flush=True)
        hdr = f"{'Method':<10}" + "".join(f"{m:>14}" for m in METRICS)
        print(hdr, flush=True)
        for m in methods:
            line = f"{m:<10}"
            for met in METRICS:
                key = f"{met}_mean"
                if key in agg[m]:
                    line += f"{agg[m][key]:>7.4f}±{agg[m][met+'_std']:<6.4f}"
                else:
                    line += f"{'':>14}"
            print(line, flush=True)
        print("=" * 90, flush=True)

    out = os.path.join(_ROOT, "outputs", "structcp_ppr_seeds_all.json")
    save_json(summary, out)
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
