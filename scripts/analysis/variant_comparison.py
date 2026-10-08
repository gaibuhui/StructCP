# -*- coding: utf-8 -*-
"""
变体组合对比实验: 找 FPR-TPR 联合最优的 StructCP 变体作为论文主体。

变体:
  A. contrastive   (现状主体, contrastive 权重 + 伪正常扩增)
  B. ppr           (PPR 结构权重 + 伪正常扩增, 社区型 TPR 恢复)
  C. contrastive+Dual (双阈值: FPR≤α 且 FNR≤β, abstain)
  D. ppr+Dual      (PPR 权重 + 双阈值)

协议: fixed alpha=0.05 (与主表一致, 非 adaptive), 5 seeds, GPU。
输出: outputs/variant_comparison.json + 控制台对比表。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/variant_comparison.py --datasets Amazon,YelpChi,Elliptic,TFinance \
      --seeds 42,123,456,789,1011
"""
import argparse, io, json, os, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp, DEFAULT_DATASET_CFG  # noqa: E402

VARIANT_KEYS = {
    "contrastive":      dict(weight_mode="contrastive", dual=False),
    "ppr":              dict(weight_mode="ppr", dual=False),
    "contrastive_dual": dict(weight_mode="contrastive", dual=True),
    "ppr_dual":         dict(weight_mode="ppr", dual=True),
}
METHOD_KEYS = {"contrastive": "StructCP", "ppr": "StructCP-PPR",
               "contrastive_dual": "StructCP-Dual", "ppr_dual": "StructCP-PPR-Dual"}


def agg_metric(per_seed, variant, metric):
    vals = [per_seed[s][variant].get(metric)
            for s in per_seed if variant in per_seed[s]
            and per_seed[s][variant].get(metric) is not None]
    vals = [v for v in vals if v == v]  # 去 NaN
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456,789,1011")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="variant_comparison.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    summary = {}
    for ds in datasets:
        print(f"\n{'='*100}\n{ds}\n{'='*100}", flush=True)
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = {}
            for vname, vcfg in VARIANT_KEYS.items():
                out = evaluate_structcp(
                    ds, seed=seed, alpha=args.alpha, device=args.device, root=_ROOT,
                    adaptive_alpha=False,           # fixed alpha, 与主表一致
                    weight_mode=vcfg["weight_mode"],
                    dual=vcfg["dual"], dual_beta=args.beta,
                )
                r = out["results"]
                # 定位变体结果: ppr 模式方法名是 StructCP-PPR; dual 时是 *-Dual
                if vname == "ppr":
                    key = "StructCP-PPR" if "StructCP-PPR" in r else "StructCP"
                elif vname == "contrastive_dual":
                    key = "StructCP-Dual" if "StructCP-Dual" in r else "StructCP"
                elif vname == "ppr_dual":
                    key = ("StructCP-PPR-Dual" if "StructCP-PPR-Dual" in r
                           else "StructCP-Dual" if "StructCP-Dual" in r else "StructCP")
                else:
                    key = "StructCP"
                res = dict(r.get(key, {}))
                per_seed[seed][vname] = res
                print(f"  [seed={seed} {vname:<16}] "
                      f"FPR={res.get('FPR', float('nan')):.4f} "
                      f"TPR={res.get('TPR', float('nan')):.4f} "
                      f"F1={res.get('F1', float('nan')):.4f} "
                      f"abstain={res.get('abstain_rate', 0.0):.3f}", flush=True)
        summary[ds] = {"seeds": seeds, "per_seed": per_seed,
                       "agg": {v: {"FPR": agg_metric(per_seed, v, "FPR"),
                                   "TPR": agg_metric(per_seed, v, "TPR"),
                                   "F1": agg_metric(per_seed, v, "F1"),
                                   "abstain": agg_metric(per_seed, v, "abstain_rate")}
                               for v in VARIANT_KEYS}}

    # 控制台对比表
    print(f"\n\n{'='*110}\n变体对比 (fixed alpha={args.alpha}, mean±std over {len(seeds)} seeds)\n{'='*110}")
    hdr = f"{'Dataset':<10}" + "".join(f"{v:>30}" for v in VARIANT_KEYS)
    print(hdr)
    for ds in datasets:
        row = f"{ds:<10}"
        for v in VARIANT_KEYS:
            a = summary[ds]["agg"][v]
            fpr = a["FPR"]; tpr = a["TPR"]
            if fpr[0] is None:
                row += f"{'n/a':>30}"
            else:
                flag = "≤α" if fpr[0] <= args.alpha else ">α"
                row += f"  {fpr[0]:.3f}/{tpr[0]:.3f}({flag})"
        print(row)
    print("\n注: 格式为 FPR/TPR(≤α/超过α)。")

    out_path = os.path.join(_ROOT, "outputs", args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
