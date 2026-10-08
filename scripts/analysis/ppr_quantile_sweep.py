# -*- coding: utf-8 -*-
"""
A2 修复实验: PPR 伪正常准入门槛 ppr_quantile 的 FPR-TPR 杠杆扫描。
目标: 在保持 FPR<=α=0.05 的前提下, 用 PPR 的 FPR 余量换取 TPR。

机制: ppr_quantile ↓ (门槛更松) → 更多伪正常进池 → 池右移小 → FPR↑, TPR↑。
     ppr_quantile ↑ (门槛更严) → 池更保守 → FPR↓, TPR↓。
当前默认 ppr_quantile=0.4 在 Amazon/YelpChi 上 FPR≈0.017, 有 0.033 余量。

用法 (fov):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/ppr_quantile_sweep.py --datasets Amazon,YelpChi --seeds 42,123,456
"""
import argparse, io, json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp  # noqa: E402

QS = [0.2, 0.3, 0.4, 0.5, 0.6]
ALPHA = 0.05


def agg(per_seed, metric):
    vals = [s.get(metric) for s in per_seed if s.get(metric) is not None]
    vals = [v for v in vals if v == v]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456,789,1011")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {}
    for ds in datasets:
        print(f"\n{'='*90}\n{ds}\n{'='*90}", flush=True)
        results[ds] = {}
        for q in QS:
            per_seed = []
            for seed in seeds:
                out = evaluate_structcp(
                    ds, seed=seed, alpha=ALPHA, device=args.device, root=_ROOT,
                    adaptive_alpha=False, weight_mode="ppr",
                    dual=True, dual_beta=0.05, ppr_quantile=q,
                )
                r = out["results"].get("StructCP-PPR-Dual",
                     out["results"].get("StructCP-Dual",
                     out["results"].get("StructCP", {})))
                per_seed.append(dict(r))
            fpr_m, fpr_s = agg(per_seed, "FPR")
            tpr_m, tpr_s = agg(per_seed, "TPR")
            ab_m, ab_s = agg(per_seed, "abstain_rate")
            flag = "≤α" if fpr_m is not None and fpr_m <= ALPHA else ">α"
            results[ds][q] = {"FPR": [fpr_m, fpr_s], "TPR": [tpr_m, tpr_s],
                              "abstain": [ab_m, ab_s]}
            print(f"  ppr_q={q}: FPR={fpr_m:.4f}±{fpr_s:.4f}  TPR={tpr_m:.4f}±{tpr_s:.4f}  "
                  f"abstain={ab_m:.4f}  ({flag})", flush=True)

    out = os.path.join(_ROOT, "outputs", "ppr_quantile_sweep.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"QS": QS, "alpha": ALPHA, "results": results}, f, indent=1)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
