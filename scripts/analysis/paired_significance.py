# -*- coding: utf-8 -*-
"""
⑤ 配对显著性检验 (回应审稿人 Minor 6).

配对 1: StructCP (ppr_dual, 主配置) vs Frozen —— per-seed 配对 t 检验, 4 数据集 × 5 seeds。
配对 2: StructCP vs GADT3 self-run —— 3 seeds 配对 (GADT3 自跑仅 3 seeds)。
同时报告 worst-case seed。

数据源:
  - StructCP per-seed: outputs/variant_comparison.json (ppr_dual)
  - Frozen per-seed:   outputs/structcp_contrastive_seeds_fixedA_all.json
  - GADT3 per-seed:    outputs/gadt3_seeds_{ds}_homo.json

用法: python scripts/analysis/paired_significance.py
"""
import json, math, os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
SEEDS = [42, 123, 456, 789, 1011]
ALPHA = 0.05


def paired_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2 or d.std() == 0:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    # 双尾 p 值 (t 分布近似, df=n-1)
    df = n - 1
    p = 2.0 * _t_sf(abs(t), df)
    return float(t), float(p)


def _t_sf(x, df):
    """Student t 生存函数 (数值近似, 用不完全 beta; 这里用简单 Simpson 积分足够)。"""
    if x <= 0:
        return 0.5
    # p = P(T > x) = 0.5 * I_{df/(df+x^2)}(df/2, 1/2)
    from math import lgamma, exp
    a = df / 2.0
    b = 0.5
    z = df / (df + x * x)
    # 正则不完全 beta 用 scipy 不存在时退化为正态近似
    try:
        from scipy import stats as sps
        return float(sps.t.sf(x, df))
    except Exception:
        return float(0.5 * math.erfc(x / math.sqrt(2)))  # 正态近似


def main():
    vc = json.load(open(os.path.join(_ROOT, "outputs", "variant_comparison.json")))
    fz = json.load(open(os.path.join(_ROOT, "outputs", "structcp_contrastive_seeds_fixedA_all.json")))

    print("=== 配对 1: StructCP(ppr_dual) vs Frozen, 5 seeds ===")
    out1 = {}
    for ds in DATASETS:
        scp = [vc[ds]["per_seed"][str(sd)]["ppr_dual"]["FPR"] for sd in SEEDS]
        frozen = [fz[ds]["per_seed"][str(sd)]["Frozen"]["FPR"] for sd in SEEDS]
        t, p = paired_t(scp, frozen)
        out1[ds] = {"StructCP_fpr": scp, "Frozen_fpr": frozen,
                    "StructCP_mean": float(np.mean(scp)), "Frozen_mean": float(np.mean(frozen)),
                    "t": t, "p": p, "sig_0.05": bool(p < 0.05),
                    "worst_seed_scp": float(max(scp))}
        print(f"  {ds:<9} StructCP={np.mean(scp):.4f}±{np.std(scp):.3f}  "
              f"Frozen={np.mean(frozen):.4f}±{np.std(frozen):.3f}  "
              f"t={t:+.2f} p={p:.4f} {'SIG' if p < 0.05 else 'n.s.'}  worstSCP={max(scp):.4f}")

    print("\n=== 配对 2: StructCP(ppr_dual) vs GADT3 self-run, 3 seeds ===")
    out2 = {}
    for ds in DATASETS:
        g = json.load(open(os.path.join(_ROOT, "outputs", f"gadt3_seeds_{ds}_homo.json")))
        gseeds = [int(sd) for sd in g["per_seed"].keys()]
        scp = [vc[ds]["per_seed"][str(sd)]["ppr_dual"]["FPR"] for sd in gseeds]
        gad = [g["per_seed"][str(sd)]["FPR"] for sd in gseeds]
        t, p = paired_t(scp, gad)
        out2[ds] = {"StructCP_fpr": scp, "GADT3_fpr": gad, "t": t, "p": p}
        print(f"  {ds:<9} StructCP={np.mean(scp):.4f}  GADT3={np.mean(gad):.4f}  "
              f"t={t:+.2f} p={p:.4f} {'SIG' if p < 0.05 else 'n.s.'}")

    json.dump({"alpha": ALPHA, "paired1_vs_Frozen": out1, "paired2_vs_GADT3": out2},
              open(os.path.join(_ROOT, "outputs", "paired_significance.json"), "w"),
              indent=1)
    print("\n[saved] outputs/paired_significance.json")


if __name__ == "__main__":
    main()
