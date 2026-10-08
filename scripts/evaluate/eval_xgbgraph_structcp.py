# -*- coding: utf-8 -*-
"""
用 XGBGraph 骨干分数评估 StructCP (Frozen split-CP vs StructCP 加权分位)。

XGBGraph 是分数级骨干: 只需节点级异常分数即可接入 StructCP wrapper。
本脚本直接读取预计算的 XGBGraph 分数 (checkpoint/xgbgraph/*.npz),
复用 pipeline 的超图传播/对比一致性/加权分位组件, 不依赖 BWGNN 推理。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/evaluate/eval_xgbgraph_structcp.py --datasets Amazon,YelpChi --seeds 42
"""
import argparse, io, json, os, sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.loader import load_gad  # noqa: E402
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation, contrastive_consistency  # noqa: E402
from adapters.conformal import structcp_threshold, split_conformal_threshold, evaluate_coverage  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance",
            "photo", "computer", "weibo", "tolokers", "questions", "reddit"]
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75


def run_one(dataset, seed, alpha=0.05):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")

    data = load_gad(dataset, relation="homo", seed=seed).to(dev)
    # XGBGraph 分数 (预计算, 全图)
    npz_path = os.path.join(_ROOT, "checkpoint", "xgbgraph", f"{dataset}_seed{seed}.npz")
    if not os.path.isfile(npz_path):
        return {"error": f"missing {npz_path}"}
    npz = np.load(npz_path)
    s_xgb = torch.from_numpy(npz["scores"]).float().to(dev)

    # 超图传播特征 (骨干无关) -> 对比一致性
    hg = KNNHypergraph(data.x, k=K).to(dev)
    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    cal_bool = data.cal_mask.bool()
    test_bool = data.test_mask.bool()
    cal_normal = cal_bool & (data.y == 0)
    c_sp = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_normal)

    y_test = data.y[test_bool].cpu().numpy()

    # ---- Frozen: split conformal (校准正常分数) ----
    thr_f = split_conformal_threshold(s_xgb[cal_bool][data.y[cal_bool] == 0], alpha)
    ev_f = evaluate_coverage(s_xgb[test_bool], data.y[test_bool], thr_f)
    auroc_f = float(roc_auc_score(y_test, s_xgb[test_bool].cpu().numpy()))

    # ---- StructCP: 加权分位 + 伪正常扩增 ----
    thr_s, info = structcp_threshold(
        s_xgb[cal_bool], data.y[cal_bool], c_sp[cal_bool],
        s_xgb[test_bool], c_sp[test_bool],
        alpha=alpha, tau_quantile=TAU, mode="hard", score_quantile=SQ,
        use_calib_stat=True, weight_mode="contrastive",
    )
    ev_s = evaluate_coverage(s_xgb[test_bool], data.y[test_bool], thr_s)
    auroc_s = float(roc_auc_score(y_test, s_xgb[test_bool].cpu().numpy()))

    return {
        "Frozen": {"AUROC": auroc_f, "FPR": ev_f["FPR"], "TPR": ev_f["TPR"], "F1": ev_f["F1"]},
        "StructCP": {"AUROC": auroc_s, "FPR": ev_s["FPR"], "TPR": ev_s["TPR"], "F1": ev_s["F1"],
                     "n_pseudo": info.get("n_pseudo_normal")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="xgbgraph_structcp_eval.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {}
    for ds in datasets:
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = run_one(ds, seed, args.alpha)
            r = per_seed[seed]
            if "error" not in r:
                print(f"[{ds} seed={seed}] XGBGraph Frozen FPR={r['Frozen']['FPR']:.4f} TPR={r['Frozen']['TPR']:.4f} "
                      f"| StructCP FPR={r['StructCP']['FPR']:.4f} TPR={r['StructCP']['TPR']:.4f}", flush=True)
        # 聚合
        agg = {}
        for m in ["Frozen", "StructCP"]:
            agg[m] = {}
            for met in ["FPR", "TPR", "F1", "AUROC"]:
                vals = [per_seed[s][m][met] for s in per_seed if m in per_seed[s] and met in per_seed[s][m]]
                vals = [v for v in vals if v == v]
                agg[m][f"{met}_mean"] = float(np.mean(vals)) if vals else None
                agg[m][f"{met}_std"] = float(np.std(vals)) if len(vals) > 1 else None
        results[ds] = {"seeds": seeds, "agg": agg, "per_seed": per_seed}
        a = agg
        print(f"{ds}: XGBGraph-Frozen FPR={a['Frozen']['FPR_mean']:.4f} TPR={a['Frozen']['TPR_mean']:.4f} "
              f"| XGBGraph-StructCP FPR={a['StructCP']['FPR_mean']:.4f} TPR={a['StructCP']['TPR_mean']:.4f}",
              flush=True)

    out_path = os.path.join(_ROOT, "outputs", args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
