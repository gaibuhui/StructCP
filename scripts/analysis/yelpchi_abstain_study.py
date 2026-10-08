# -*- coding: utf-8 -*-
"""
2A: YelpChi "强制分类 vs abstain" 对比研究。

目标: 把 abstain 从"掩盖失败"翻转为"严格优于强制分类"的实证。
输出 per-seed + 5-seed agg:
  dual   : FPR / TPR / FNR / abstain_rate / FPR_rejected / TPR_rejected (PPR+dual, 主表口径)
  force  : FPR / TPR / F1 —— 只用 lambda_anomaly 的强制单阈值 (无 abstain)
  oracle : TPR@FPR=0.05 (诊断上界, 用测试正常 95 分位, 泄漏仅供对比)
  band   : abstain 带异常密度 / 全局异常密度 / 富集比 (abstain 带是否富含高风险节点)

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/yelpchi_abstain_study.py
"""
import io, json, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import (KNNHypergraph, hypergraph_propagation,  # noqa: E402
                                     hyperedge_consistency)
from adapters.conformal import (structcp_threshold, structcp_dual_threshold,  # noqa: E402
                                evaluate_prediction_set)
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DS = "YelpChi"
ALPHA, BETA = 0.05, 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75
SEEDS = [42, 123, 456, 789, 1011]


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    data = load_gad(DS, relation="homo", seed=seed).to(dev)
    model, _ = load_frozen_detector(DS, "homo", root=_ROOT, device="cpu")
    hg = KNNHypergraph(data.x, k=K).to(dev)

    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    from adapters.pipeline import get_scores
    s_sp = get_scores(model, x_enh, data.edge_index)

    cal_bool = data.cal_mask.bool()
    test_bool = data.test_mask.bool()
    cal_normal = cal_bool & (data.y == 0)
    c_sp = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)
    c_ppr = ppr_structure_score(
        x_enh, data.edge_index, cal_normal,
        pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
    )
    test_ppr_t = c_ppr[test_bool].float()

    y_cal = data.y[cal_bool]
    thr3, _ = structcp_threshold(
        s_sp[cal_bool], y_cal, c_sp[cal_bool],
        s_sp[test_bool], c_sp[test_bool],
        alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
        use_calib_stat=True, weight_mode="struct",
        test_ppr=test_ppr_t, ppr_quantile=0.4,
    )
    lb, la, dinfo = structcp_dual_threshold(
        s_sp[cal_bool], y_cal, c_sp[cal_bool],
        s_sp[test_bool], c_sp[test_bool],
        alpha=ALPHA, beta=BETA, lambda_anomaly=thr3,
        tau_quantile=TAU, mode="hard", score_quantile=SQ,
        use_calib_stat=True, test_ppr=test_ppr_t, ppr_quantile=0.4,
        weight_mode="struct",
    )

    s_t = s_sp[test_bool]
    y_t = data.y[test_bool]
    ps = evaluate_prediction_set(s_t, y_t, lb, la)

    # ---- 强制单阈值 (无 abstain): pred_anomaly iff s >= la ----
    force_pred = (s_t >= la)
    n0 = int((y_t == 0).sum()); n1 = int((y_t == 1).sum())
    fpr_f = float((force_pred & (y_t == 0)).float().sum()) / max(n0, 1)
    tpr_f = float((force_pred & (y_t == 1)).float().sum()) / max(n1, 1)
    prec = float((force_pred & (y_t == 1)).float().sum()) / max(int(force_pred.sum()), 1)
    f1_f = 2 * prec * tpr_f / max(prec + tpr_f, 1e-12)

    # ---- Oracle 强制上界 (诊断, 泄漏): 测试正常 95 分位 ----
    s_tn = s_t[y_t == 0]
    t_or = torch.quantile(s_tn, 1 - ALPHA)
    tpr_or = float((s_t[y_t == 1] >= t_or).float().mean())

    # ---- abstain 带构成 (2D 辅助): 带内异常密度 vs 全局 ----
    pred = torch.where((s_t >= la) & (s_t > lb), 1,
             torch.where((s_t <= lb) & (s_t < la), 0, 2))
    band = pred == 2
    nband = int(band.sum())
    if nband > 0:
        band_anom = float(((band & (y_t == 1)).float().sum()) / nband)
        band_norm = float(((band & (y_t == 0)).float().sum()) / nband)
    else:
        band_anom = band_norm = float("nan")
    global_anom = float((y_t == 1).float().mean())
    enrichment = band_anom / max(global_anom, 1e-12)

    return {
        "seed": seed,
        "dual": {k: ps[k] for k in ("FPR", "TPR", "FNR", "abstain_rate",
                                    "FPR_rejected", "TPR_rejected")},
        "force": {"FPR": fpr_f, "TPR": tpr_f, "F1": f1_f,
                  "FPR_le_alpha": bool(fpr_f <= ALPHA + 1e-9)},
        "oracle_force": {"TPR_at_FPR_alpha": tpr_or},
        "band": {"abstain_rate": ps["abstain_rate"], "band_anomaly_density": band_anom,
                 "band_normal_density": band_norm, "global_anomaly_density": global_anom,
                 "enrichment": enrichment,
                 "lambda_anomaly": float(la), "lambda_normal": float(lb)},
    }


def mean_std(vals):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)])
    return (float(a.mean()), float(a.std())) if len(a) else (None, None)


def main():
    per_seed = []
    for sd in SEEDS:
        try:
            r = run_seed(sd)
            per_seed.append(r)
            d = r["dual"]; f = r["force"]; b = r["band"]
            print(f"seed{sd:<5} dual FPR={d['FPR']:.4f} TPR={d['TPR']:.4f} "
                  f"abstain={d['abstain_rate']:.4f} rejTPR={d['TPR_rejected']:.4f} | "
                  f"force FPR={f['FPR']:.4f} TPR={f['TPR']:.4f} | "
                  f"oracle TPR={r['oracle_force']['TPR_at_FPR_alpha']:.4f} | "
                  f"band enrich={b['enrichment']:.2f}")
        except Exception as e:
            print(f"seed {sd} ERROR: {e}")
            per_seed.append({"seed": sd, "error": str(e)})

    agg = {}
    for key in ("dual", "force", "oracle_force", "band"):
        agg[key] = {}
    for r in per_seed:
        if "error" in r:
            continue
        for grp, fields in [("dual", ("FPR", "TPR", "FNR", "abstain_rate",
                                      "FPR_rejected", "TPR_rejected")),
                            ("force", ("FPR", "TPR", "F1")),
                            ("oracle_force", ("TPR_at_FPR_alpha",)),
                            ("band", ("abstain_rate", "band_anomaly_density",
                                      "global_anomaly_density", "enrichment"))]:
            for f in fields:
                m, s = mean_std([r[grp][f] for r in per_seed if f in r[grp]])
                agg[grp][f] = {"mean": m, "std": s} if m is not None else None
    agg["n_seeds"] = sum(1 for r in per_seed if "error" not in r)

    out = os.path.join(_ROOT, "outputs", "yelpchi_abstain_study.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps({"dataset": DS, "alpha": ALPHA, "beta": BETA,
                    "per_seed": per_seed, "agg": agg}, indent=1, ensure_ascii=False))
    print("\n=== AGG (5-seed) ===")
    print(json.dumps(agg, indent=1))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
