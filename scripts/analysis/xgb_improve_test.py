# -*- coding: utf-8 -*-
"""
XGBGraph 骨干 FPR 改进空间验证: XGB 分数 × {contrastive/PPR} × {单阈值/Dual}。
判断 novel-normal 失守能否通过权重/双阈值救回。

用法 (fov):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/xgb_improve_test.py --datasets Amazon,YelpChi --seeds 42
"""
import argparse, io, json, os, sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.loader import load_gad  # noqa: E402
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation, contrastive_consistency  # noqa: E402
from adapters.conformal import (structcp_threshold, structcp_dual_threshold,
                                split_conformal_threshold, evaluate_coverage,
                                evaluate_prediction_set)  # noqa: E402

K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75
ALPHA, BETA = 0.05, 0.05


def eval_pred(y_test, s_test, thr):
    ev = evaluate_coverage(s_test, y_test, thr)
    return {"FPR": ev["FPR"], "TPR": ev["TPR"], "F1": ev["F1"]}


def run_one(dataset, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    data = load_gad(dataset, relation="homo", seed=seed).to(dev)
    npz_path = os.path.join(_ROOT, "checkpoint", "xgbgraph", f"{dataset}_seed{seed}.npz")
    if not os.path.isfile(npz_path):
        return {"error": f"missing {npz_path}"}
    s_xgb = torch.from_numpy(np.load(npz_path)["scores"]).float().to(dev)

    hg = KNNHypergraph(data.x, k=K).to(dev)
    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    cal_b = data.cal_mask.bool(); test_b = data.test_mask.bool()
    cal_normal = cal_b & (data.y == 0)
    c_sp = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_normal)

    y_test = data.y[test_b].cpu().numpy()
    out = {}

    # Frozen
    thr_f = split_conformal_threshold(s_xgb[cal_b][data.y[cal_b] == 0], ALPHA)
    out["Frozen"] = eval_pred(data.y[test_b], s_xgb[test_b], thr_f)
    out["Frozen"]["abstain"] = 0.0

    # 4 变体
    for vname, wmode, dual in [("contrastive", "contrastive", False),
                               ("ppr", "ppr", False),
                               ("contrastive+dual", "contrastive", True),
                               ("ppr+dual", "ppr", True)]:
        try:
            if not dual:
                thr, info = structcp_threshold(
                    s_xgb[cal_b], data.y[cal_b], c_sp[cal_b],
                    s_xgb[test_b], c_sp[test_b],
                    alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
                    use_calib_stat=True, weight_mode=wmode,
                )
                ev = eval_pred(data.y[test_b], s_xgb[test_b], thr)
                ev["abstain"] = 0.0
            else:
                ln, la, info = structcp_dual_threshold(
                    s_xgb[cal_b], data.y[cal_b], c_sp[cal_b],
                    s_xgb[test_b], c_sp[test_b],
                    alpha=ALPHA, beta=BETA, tau_quantile=TAU, mode="hard",
                    score_quantile=SQ, use_calib_stat=True, weight_mode=wmode,
                )
                ev = evaluate_prediction_set(s_xgb[test_b], data.y[test_b], la, ln)
            out[vname] = {"FPR": float(ev["FPR"]), "TPR": float(ev["TPR"]),
                          "F1": float(ev.get("F1", float("nan"))),
                          "abstain": float(ev.get("abstain_rate", ev.get("abstain", 0.0)))}
        except Exception as e:
            out[vname] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--out", default="xgb_improve_test.json")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {}
    for ds in datasets:
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = run_one(ds, seed)
            r = per_seed[seed]
            if "error" not in r:
                print(f"[{ds} s={seed}] " + " | ".join(
                    f"{k}: FPR={r[k]['FPR']:.3f} TPR={r[k]['TPR']:.3f} abs={r[k].get('abstain', 0.0):.2f}"
                    for k in ["Frozen", "contrastive", "ppr", "contrastive+dual", "ppr+dual"] if "error" not in r[k]),
                    flush=True)
        # 聚合
        agg = {}
        for v in ["Frozen", "contrastive", "ppr", "contrastive+dual", "ppr+dual"]:
            agg[v] = {}
            for met in ["FPR", "TPR", "F1", "abstain"]:
                vals = [per_seed[s][v][met] for s in per_seed
                        if v in per_seed[s] and "error" not in per_seed[s][v] and met in per_seed[s][v]]
                vals = [x for x in vals if x == x]
                agg[v][f"{met}_mean"] = float(np.mean(vals)) if vals else None
        results[ds] = {"seeds": seeds, "agg": agg, "per_seed": per_seed}
        print(f"== {ds} ==")
        for v in ["Frozen", "contrastive", "ppr", "contrastive+dual", "ppr+dual"]:
            a = agg[v]
            if a["FPR_mean"] is not None:
                flag = "≤α" if a["FPR_mean"] <= ALPHA else ">α"
                print(f"  {v:<16} FPR={a['FPR_mean']:.3f} TPR={a['TPR_mean']:.3f} "
                      f"abstain={a['abstain_mean']:.3f}  ({flag})", flush=True)

    out_path = os.path.join(_ROOT, "outputs", args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
