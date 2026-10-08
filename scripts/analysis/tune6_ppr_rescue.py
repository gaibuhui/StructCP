# -*- coding: utf-8 -*-
"""
TUNE6 PPR 试救实验: 对轻度超限的 computer/weibo/reddit 用 PPR 权重重跑,
检验能否把 FPR 拉回 ≤α (结构性弱 shift 假设的挑战)。

协议: 与 TUNE6 审计一致 (fixed alpha=0.05, 3 seeds, BWGNN)。
输出: outputs/tune6_ppr_rescue.json + 对比表。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/tune6_ppr_rescue.py --datasets computer,weibo,reddit
"""
import argparse, io, json, os, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp, DEFAULT_DATASET_CFG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="computer,weibo,reddit")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="tune6_ppr_rescue.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # TUNE6 审计基线 (contrastive) FPR —— 对照
    base = {}
    try:
        d = json.load(io.open(os.path.join(_ROOT, "outputs",
                                           "structcp_tune6_fixedA_seeds3.json"),
                              encoding="utf-8"))
        for ds in datasets:
            if ds in d and "agg" in d[ds]:
                base[ds] = {"FPR_contrastive": d[ds]["agg"]["StructCP"]["FPR_mean"],
                            "FPR_frozen": d[ds]["agg"]["Frozen"]["FPR_mean"]}
    except Exception as e:
        print("WARN 读取基线失败:", e)

    summary = {}
    for ds in datasets:
        if ds not in DEFAULT_DATASET_CFG:
            print(f"[skip] {ds} 未注册")
            continue
        print(f"\n===== {ds} (PPR 权重) =====", flush=True)
        per_seed = {}
        for seed in seeds:
            try:
                out = evaluate_structcp(
                    ds, seed=seed, alpha=args.alpha, device=args.device, root=_ROOT,
                    adaptive_alpha=False,
                    weight_mode="ppr",
                )
                r = out["results"]
                key = "StructCP-PPR" if "StructCP-PPR" in r else "StructCP"
                per_seed[seed] = dict(r.get(key, {}))
                res = per_seed[seed]
                print(f"  [seed={seed}] FPR={res.get('FPR', float('nan')):.4f} "
                      f"TPR={res.get('TPR', float('nan')):.4f} "
                      f"F1={res.get('F1', float('nan')):.4f}", flush=True)
            except Exception as e:
                print(f"  [seed={seed}] ERR {type(e).__name__}: {e}", flush=True)
                per_seed[seed] = {}
        # 聚合
        agg = {}
        for m in ["FPR", "TPR", "F1"]:
            vals = [per_seed[s][m] for s in per_seed if m in per_seed[s]
                    and per_seed[s][m] == per_seed[s][m]]
            agg[f"{m}_mean"] = float(np.mean(vals)) if vals else None
            agg[f"{m}_std"] = float(np.std(vals)) if len(vals) > 1 else None
        summary[ds] = {"seeds": seeds, "agg": agg, "per_seed": per_seed,
                       "baseline": base.get(ds, {})}
        flag = "≤α ✓" if (agg["FPR_mean"] is not None and agg["FPR_mean"] <= args.alpha) else ">α ✗"
        print(f"  => PPR FPR={agg['FPR_mean']:.4f} {flag}  "
              f"(contrastive 基线: {base.get(ds, {}).get('FPR_contrastive', 'n/a')})", flush=True)

    out_path = os.path.join(_ROOT, "outputs", args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
