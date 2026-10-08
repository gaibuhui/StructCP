# -*- coding: utf-8 -*-
"""
PPR+Dual 完整 α 敏感性扫描（补全 tab:app_alpha_sweep 为新主体口径）。
α ∈ {0.01, 0.02, 0.05, 0.06, 0.075, 0.10, 0.15, 0.20}, 4 数据集 × 5 seeds。
输出: outputs/ppr_alpha_sweep.json
"""
import argparse, io, json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp  # noqa: E402

ALPHAS = [0.01, 0.02, 0.05, 0.06, 0.075, 0.10, 0.15, 0.20]


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
        print(f"\n=== {ds} ===", flush=True)
        results[ds] = {}
        for a in ALPHAS:
            fps, tps = [], []
            for seed in seeds:
                out = evaluate_structcp(
                    ds, seed=seed, alpha=a, device=args.device, root=_ROOT,
                    adaptive_alpha=False, weight_mode="ppr",
                    dual=True, dual_beta=0.05,
                )
                r = out["results"].get("StructCP-PPR-Dual",
                     out["results"].get("StructCP-Dual",
                     out["results"].get("StructCP", {})))
                fps.append(r.get("FPR", 0.0)); tps.append(r.get("TPR", 0.0))
            results[ds][a] = {"FPR": [float(np.mean(fps)), float(np.std(fps))],
                              "TPR": [float(np.mean(tps)), float(np.std(tps))]}
            print(f"  α={a:.3f}: FPR={np.mean(fps):.4f}±{np.std(fps):.4f} "
                  f"TPR={np.mean(tps):.4f}±{np.std(tps):.4f}", flush=True)

    out = os.path.join(_ROOT, "outputs", "ppr_alpha_sweep.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"ALPHAS": ALPHAS, "results": results}, f, indent=1)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
