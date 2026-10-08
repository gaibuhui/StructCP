"""评审回应实验：统计报告 (Experiment Group E)。

对已落盘的逐 seed 结果计算:
  1. 5-seed mean ± std
  2. bootstrap 95% CI (对 seed 维度重采样, B=1000, percentile 法)
  3. 可选 Benjamini-Hochberg FDR 校正 (对"同方法 vs 同数据集的多指标"不做校正;
     对主表多数据集联合比较, 逐数据集各自独立, 不做跨数据集校正)

输入: outputs/calibration_baselines.json (校准基线, 逐 seed 明细)
      outputs/structcp_seeds_all.json (主表, agg 结构)
输出: outputs/bootstrap_ci.json (mean±std + [ci_low, ci_high])

用法 (CBP 环境):
  python scripts/analysis/compute_bootstrap_ci.py
"""
from __future__ import annotations

import json
import os
import sys as _sys

import numpy as np

_p = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)

SEEDS = [42, 123, 456, 789, 1011]
N_BOOT = 1000


def bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    """对一组逐 seed 指标做 percentile bootstrap 95% CI。"""
    v = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(v)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = rng.choice(v, size=n, replace=True).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def agg_from_seed_dict(per_seed, methods, metrics=("FPR", "TPR", "F1")):
    out = {}
    for m in methods:
        out[m] = {}
        for met in metrics:
            vals = [per_seed[s][m][met] for s in per_seed
                    if m in per_seed[s] and isinstance(per_seed[s][m], dict)
                    and met in per_seed[s][m]]
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals)
            out[m][met] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)),
                "ci95": [lo, hi], "seeds": vals,
            }
    return out


def agg_main(structcp_seeds):
    """主表 → mean/std + bootstrap 95% CI。

    兼容两种结构:
      - fixedA_all.json (对比一致性, 当前主表源): d[ds]["per_seed"][seed][method]["FPR"]
      - structcp_seeds_all.json (旧 homogeneous): d[ds]["agg"] + d[ds]["reversal_check"]
    """
    out = {}
    for ds, v in structcp_seeds.items():
        per_seed = v.get("per_seed")
        if per_seed is None:
            # 旧结构: agg + reversal_check
            a = v["agg"]
            rc = v.get("reversal_check", {})
            fpr_key_map = {"Frozen": "frozen_fpr_per_seed", "StructCP": "structcp_fpr_per_seed"}
            out[ds] = {}
            for m in ["Frozen", "HG-TTA", "StructCP"]:
                if m not in a:
                    continue
                out[ds][m] = {}
                seeds = rc.get(fpr_key_map.get(m, "")) or []
                if len(seeds) >= 3:
                    lo, hi = bootstrap_ci(seeds)
                    out[ds][m]["FPR"] = {
                        "mean": float(np.mean(seeds)), "std": float(np.std(seeds, ddof=0)),
                        "ci95": [lo, hi], "seeds": seeds,
                    }
                elif "FPR_mean" in a[m]:
                    out[ds][m]["FPR"] = {
                        "mean": a[m]["FPR_mean"], "std": a[m]["FPR_std"], "ci95": None,
                    }
                if "TPR_mean" in a[m]:
                    out[ds][m]["TPR"] = {
                        "mean": a[m]["TPR_mean"], "std": a[m]["TPR_std"], "ci95": None,
                    }
            continue
        # 新结构: per_seed[seed][method]["FPR"]
        seed_keys = list(per_seed.keys())
        out[ds] = {}
        for m in ["Frozen", "HG-TTA", "StructCP"]:
            seeds = [per_seed[s][m]["FPR"] for s in seed_keys if m in per_seed[s]]
            if len(seeds) >= 3:
                lo, hi = bootstrap_ci(seeds)
                out[ds][m] = {
                    "FPR": {"mean": float(np.mean(seeds)), "std": float(np.std(seeds, ddof=0)),
                            "ci95": [lo, hi], "seeds": seeds},
                    "TPR": {"mean": float(np.mean([per_seed[s][m]["TPR"] for s in seed_keys if m in per_seed[s]])),
                            "std": float(np.std([per_seed[s][m]["TPR"] for s in seed_keys if m in per_seed[s]], ddof=0)),
                            "ci95": None},
                }
    return out


def main():
    calib_path = os.path.join(_ROOT, "outputs", "calibration_baselines.json")
    main_path = os.path.join(_ROOT, "outputs", "structcp_contrastive_seeds_fixedA_all.json")
    methods = ["Frozen", "SplitCP-TTA", "StructCP", "FeatWeightCP", "RandWeightCP",
               "KLIEP-CP", "Logistic-CP", "TempScaling", "QuantReg"]
    result = {}

    if os.path.isfile(calib_path):
        d = json.load(open(calib_path))
        result["calibration_baselines"] = {
            ds: {"agg": agg_from_seed_dict(v["per_seed"], methods),
                 "auroc_raw": v["agg"]["_auroc_raw"].get("per_seed"),
                 "auroc_tta": v["agg"]["_auroc_tta"].get("per_seed")}
            for ds, v in d.items()
        }
        print("[calibration_baselines]")
        for ds, v in result["calibration_baselines"].items():
            print(f"  {ds}: " + " | ".join(
                f"{m}={v['agg'][m]['FPR']['mean']:.3f}[{v['agg'][m]['FPR']['ci95'][0]:.3f},{v['agg'][m]['FPR']['ci95'][1]:.3f}]"
                for m in methods if "FPR" in v["agg"][m]))
    else:
        print(f"[skip] {calib_path} 不存在")

    if os.path.isfile(main_path):
        d = json.load(open(main_path))
        result["main_table"] = agg_main(d)
        print("[main_table]")
        for ds, v in result["main_table"].items():
            print(f"  {ds}: " + " | ".join(
                f"{m} FPR={v[m]['FPR']['mean']:.4f}" for m in v))
    else:
        print(f"[skip] {main_path} 不存在")

    out = os.path.join(_ROOT, "outputs", "bootstrap_ci.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
