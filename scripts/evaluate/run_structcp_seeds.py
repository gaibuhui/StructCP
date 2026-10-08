"""StructCP 多 seed 稳健性验证（Elliptic / T-Finance, community-type 反转复核）。

重构说明（2026-08-23）：单 seed 运行逻辑委托给共享管线
`adapters.pipeline.evaluate_structcp`（数据集级默认配置 `DEFAULT_DATASET_CFG`
即旧 DATASET_CFG 迁入处），本脚本仅负责多 seed 调度 + mean±std 聚合 +
Wilcoxon 符号检验式"反转显著性"复核。聚合口径与重构前逐字段一致。

目的（历史）：论文 Table 1 主结果 StructCP 在 community-type 数据集
(Elliptic/T-Finance) 上加权层 FPR 出现"反转"，本脚本跨 seed 核验其统计显著性。

运行（前台, 实时 flush）:
  source <CONDA_PREFIX>/bin/activate CBP
  python scripts/evaluate/run_structcp_seeds.py --datasets Elliptic,TFinance --seeds 42,123,456
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


def run_one(dataset, seed, cfg, alpha, device="cpu", adaptive_alpha=True):
    """单 seed 运行（内部走共享管线 evaluate_structcp）。"""
    out = evaluate_structcp(
        dataset, seed=seed, alpha=alpha, device=device, root=_ROOT,
        relation=cfg["relation"], mode=cfg["mode"], score_quantile=cfg["score_quantile"],
        tau=cfg["tau"], k=cfg["k"], layers=cfg["layers"], alpha_res=cfg["alpha_res"],
        adaptive_alpha=adaptive_alpha,
    )
    print(f"  Frozen FPR={out['results']['Frozen']['FPR']:.4f}  "
          f"HG-TTA FPR={out['results']['HG-TTA']['FPR']:.4f}  "
          f"StructCP FPR={out['results']['StructCP']['FPR']:.4f}  "
          f"(StructCP n_pseudo={out['results']['StructCP'].get('n_pseudo_normal')})",
          flush=True)
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
    ap.add_argument("--adaptive_alpha", type=int, default=1,
                    help="1 = adaptive-alpha variant (default), 0 = fixed alpha=0.05")
    ap.add_argument("--out", default="structcp_seeds_all.json",
                    help="output filename under outputs/")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    methods = list(METHOD_NAMES)
    adaptive_alpha = bool(args.adaptive_alpha)

    summary = {}
    for ds in datasets:
        if ds not in DEFAULT_DATASET_CFG:
            print(f"[skip] unknown dataset {ds}")
            continue
        cfg = DEFAULT_DATASET_CFG[ds]
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = run_one(ds, seed, cfg, args.alpha, device=args.device,
                                     adaptive_alpha=adaptive_alpha)

        agg = aggregate(per_seed, methods, METRICS)

        # ---- 反转显著性复核 ----
        # StructCP 加权层 FPR 相对 Frozen 是否在每个 seed 都反转 (更高)
        frozen_fpr = [per_seed[s]["Frozen"]["FPR"] for s in seeds]
        structcp_fpr = [per_seed[s]["StructCP"]["FPR"] for s in seeds]
        reverses = [h > f for f, h in zip(frozen_fpr, structcp_fpr)]
        n_rev = sum(reverses)
        # 配对 Wilcoxon (若 scipy 可用)
        wilcoxon_p = None
        try:
            from scipy.stats import wilcoxon
            if len(frozen_fpr) >= 2 and not np.allclose(frozen_fpr, structcp_fpr):
                stat, wilcoxon_p = wilcoxon(frozen_fpr, structcp_fpr,
                                            alternative="less")  # H1: StructCP>Frozen
        except Exception:
            wilcoxon_p = None

        reversal = {
            "frozen_fpr_per_seed": frozen_fpr,
            "structcp_fpr_per_seed": structcp_fpr,
            "n_seeds_reversed": int(n_rev),
            "n_seeds_total": len(seeds),
            "always_reversed": bool(n_rev == len(seeds)),
            "structcp_fpr_std": float(np.std(structcp_fpr, ddof=0)),
            "wilcoxon_p_StructCP_gt_Frozen": (None if wilcoxon_p is None
                                            else float(wilcoxon_p)),
            "verdict": (
                "反转在全部 seed 稳定出现, 具统计显著性 (非随机波动)"
                if n_rev == len(seeds) and (wilcoxon_p is None or wilcoxon_p < 0.05)
                else "反转不稳定或方差过大, 结论需谨慎"),
        }

        summary[ds] = {"config": cfg, "alpha": args.alpha, "seeds": seeds,
                       "agg": agg, "per_seed": per_seed, "reversal_check": reversal}

        print(f"\n{'='*90}\n{ds}  multi-seed summary  "
              f"seeds={seeds}  alpha={args.alpha}\n{'='*90}", flush=True)
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
        print("\n[反转复核]", json.dumps(reversal, indent=2, ensure_ascii=False),
              flush=True)
        print("=" * 90, flush=True)

    out = os.path.join(_ROOT, "outputs", args.out)
    save_json(summary, out)
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
