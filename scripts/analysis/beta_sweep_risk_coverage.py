# -*- coding: utf-8 -*-
"""
③ 2B: beta 扫描 → risk-coverage 曲线 (回应审稿人 Major 6 / 问题 7).

对 4 数据集 × 5 seeds (主配置 PPR+dual): 每 seed 主干只算一次
(structcp_threshold 拿 lambda_anomaly + 校准异常分数), 然后对
beta ∈ {0.05,0.1,0.2,0.5,1.0} 计算 lambda_normal = 异常分数 beta 分位,
evaluate_prediction_set 得 (FPR, TPR, FNR, abstain)。
coverage = 1 - abstain (被分类的节点比例)。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/beta_sweep_risk_coverage.py
"""
import io, json, os, sys, time
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import (KNNHypergraph, hypergraph_propagation,  # noqa: E402
                                     hyperedge_consistency)
from adapters.conformal import structcp_threshold, evaluate_prediction_set  # noqa: E402
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHA = 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75
BETAS = [0.05, 0.1, 0.2, 0.5, 1.0]
SEEDS = [42, 123, 456, 789, 1011]


def run_seed(ds, seed, ppr_times):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    data = load_gad(ds, relation="homo", seed=seed).to(dev)
    model, _ = load_frozen_detector(ds, "homo", root=_ROOT, device="cpu")
    hg = KNNHypergraph(data.x, k=K).to(dev)

    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    from adapters.pipeline import get_scores
    s_sp = get_scores(model, x_enh, data.edge_index)

    cal_bool = data.cal_mask.bool()
    test_bool = data.test_mask.bool()
    cal_normal = cal_bool & (data.y == 0)
    c_sp = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)

    t0 = time.perf_counter()
    c_ppr = ppr_structure_score(
        x_enh, data.edge_index, cal_normal,
        pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
    )
    ppr_times[ds][seed] = time.perf_counter() - t0  # ⑤ PPR 计时

    test_ppr_t = c_ppr[test_bool].float()
    y_cal = data.y[cal_bool]
    thr3, info = structcp_threshold(
        s_sp[cal_bool], y_cal, c_sp[cal_bool],
        s_sp[test_bool], c_sp[test_bool],
        alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
        use_calib_stat=True, weight_mode="struct",
        test_ppr=test_ppr_t, ppr_quantile=0.4,
    )

    # 校准异常分数 (用于 lambda_normal 的 beta 分位) —— 与 structcp_dual_threshold 一致
    anom = y_cal == 1
    s_anom = s_sp[cal_bool][anom].flatten()
    n_anom = int(s_anom.numel())

    s_t = s_sp[test_bool]
    y_t = data.y[test_bool]
    out = {}
    for beta in BETAS:
        if n_anom == 0:
            lb = torch.full_like(thr3, -float("inf"))
        else:
            k = min(n_anom, max(1, int(np.ceil(beta * (n_anom + 1)))))
            lb = torch.sort(s_anom).values[k - 1].to(thr3)
        ps = evaluate_prediction_set(s_t, y_t, lb, thr3)
        out[str(beta)] = {
            "FPR": ps["FPR"], "TPR": ps["TPR"], "FNR": ps["FNR"],
            "abstain": ps["abstain_rate"],
            "coverage": 1.0 - ps["abstain_rate"],
            "TPR_rejected": ps["TPR_rejected"], "FPR_rejected": ps["FPR_rejected"],
        }
    return out


def main():
    results = {}
    ppr_times = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        results[ds] = {str(b): [] for b in BETAS}
        for sd in SEEDS:
            try:
                r = run_seed(ds, sd, ppr_times)
                for b in BETAS:
                    results[ds][str(b)].append(r[str(b)])
            except Exception as e:
                print(f"{ds} seed{sd} ERROR: {e}")
        print(f"{ds}: beta sweep done ({SEEDS} seeds)")

    agg = {}
    for ds in DATASETS:
        agg[ds] = {}
        for b in BETAS:
            rows = results[ds][str(b)]
            agg[ds][str(b)] = {m: {"mean": float(np.mean([r[m] for r in rows])),
                                   "std": float(np.std([r[m] for r in rows]))}
                               for m in ("FPR", "TPR", "FNR", "abstain", "coverage",
                                         "TPR_rejected", "FPR_rejected")}
    agg["ppr_time_s"] = {ds: {str(sd): float(t) for sd, t in ppr_times[ds].items()}
                         for ds in DATASETS}

    out = os.path.join(_ROOT, "outputs", "beta_sweep_risk_coverage.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps({"alpha": ALPHA, "betas": BETAS, "seeds": SEEDS,
                    "per_seed": results, "agg": agg}, indent=1, ensure_ascii=False))

    print("\n=== YelpChi risk-coverage (5-seed mean) ===")
    for b in BETAS:
        r = agg["YelpChi"][str(b)]
        print(f"  beta={b:<5} FPR={r['FPR']['mean']:.4f} TPR={r['TPR']['mean']:.4f} "
              f"FNR={r['FNR']['mean']:.4f} abstain={r['abstain']['mean']:.4f} "
              f"coverage={r['coverage']['mean']:.4f}")
    print("\n=== 4 数据集 beta=0.05 vs beta=1.0 (5-seed mean) ===")
    for ds in DATASETS:
        r05, r10 = agg[ds]["0.05"], agg[ds]["1.0"]
        print(f"  {ds:<9} beta=.05: FPR={r05['FPR']['mean']:.4f} TPR={r05['TPR']['mean']:.4f} "
              f"abst={r05['abstain']['mean']:.4f} | beta=1: FPR={r10['FPR']['mean']:.4f} "
              f"TPR={r10['TPR']['mean']:.4f} abst={r10['abstain']['mean']:.4f}")
    print("\n=== PPR 求解时间 (s) ===")
    for ds in DATASETS:
        ts = list(ppr_times[ds].values())
        print(f"  {ds:<9} mean={np.mean(ts):.2f}s  per-seed={[f'{t:.2f}' for t in ts]}")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
