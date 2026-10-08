"""消融: 一致性特征来源 (raw vs propagated) 对 FPR 控制的影响。

动机 (回应审稿硬伤 #4):
  paper_iclr.tex Eq.(4) 在 RAW 特征上算超边一致性 c_i, 而检测器分数 s_i 来自
  propagated 特征 X^(L)。正文直觉是 "propagated features blur local discontinuity",
  但此前缺对照证据。本脚本在 StructCP 完整流程下, 仅切换一致性特征来源, 固定其余
  所有超参, 验证 raw 特征一致性是否确实带来更紧的 FPR 控制。

变量:
  raw        : consistency_features = data.x            (默认, 论文采用)
  propagated : consistency_features = x_enh = X^(L)     (消融对照)

其余与 scripts/run_structcp.py 的 StructCP(full) 分支完全一致。
"""
from __future__ import annotations

# --- StructCP 项目根定位（深度无关）：任意脚本深度下均可定位根目录 ---
import os as _os, sys as _sys
def _structcp_root():
    _p = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(10):
        if _os.path.isfile(_os.path.join(_p, 'configs', 'default.yaml')):
            return _p
        _p = _os.path.dirname(_p)
    return _p
if _structcp_root() not in _sys.path:
    _sys.path.insert(0, _structcp_root())


import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from adapters.conformal import (  # noqa: E402
    evaluate_coverage,
    structcp_threshold,
    split_conformal_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph,
    hyperedge_consistency,
    hypergraph_propagation,
)
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402

ROOT = _structcp_root()


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_once(args, consistency_src):
    relation = args.relation if args.dataset not in ("Elliptic", "TFinance", "TSocial") else "homo"
    subsample = args.subsample
    knn_backbone = args.knn_backbone
    if args.dataset == "TSocial" and subsample is None:
        subsample = 100000
        knn_backbone = 10
    data = load_gad(args.dataset, relation=relation, seed=args.seed,
                    subsample=subsample, knn_backbone=knn_backbone)
    data = data.to(args.device)

    ckpt = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{args.dataset}_{relation}.pth"),
                      map_location=args.device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal_m, test_m = data.cal_mask, data.test_mask
    y_test = data.y[test_m].cpu().numpy()
    y_cal = data.y[cal_m]

    t0 = time.time()
    hg = KNNHypergraph(data.x, k=args.k).to(args.device)
    x_enh = hypergraph_propagation(data.x, hg, args.layers, args.alpha_res)
    s_tta = get_scores(model, x_enh, data.edge_index)
    t_hg = time.time() - t0

    # 唯一变量: 一致性特征来源
    cons_feat = data.x if consistency_src == "raw" else x_enh
    cons = hyperedge_consistency(s_tta, hg, features=x_enh,
                                 consistency_features=cons_feat)

    thr, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=args.alpha, tau_quantile=args.tau, mode=args.mode,
        score_quantile=args.score_quantile,
    )
    cov = evaluate_coverage(s_tta[test_m], data.y[test_m], thr)
    res = {
        "AUROC": float(roc_auc_score(y_test, s_tta[test_m].cpu().numpy())),
        "AUPRC": float(average_precision_score(y_test, s_tta[test_m].cpu().numpy())),
        "threshold": float(thr),
        "infer_time_s": t_hg,
        **cov,
        **info,
    }
    # novel normality 误报率
    nov = data.novel_mask & test_m
    if int(nov.sum()) > 0:
        res["FPR_novel_normality"] = float((s_tta[nov] > thr).float().mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon",
                    choices=["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--mode", default="hard", choices=["hard", "soft"])
    ap.add_argument("--score_quantile", type=float, default=None)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--knn_backbone", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = {"config": vars(args), "results": {}}
    for src in ["raw", "propagated"]:
        print(f"\n[consistency_ablation] dataset={args.dataset} src={src}")
        out["results"][src] = run_once(args, src)

    # 打印对照
    print(f"\n{'='*70}")
    print(f"Consistency-feature ablation on {args.dataset} (alpha={args.alpha})")
    print(f"{'src':<12}{'FPR':>9}{'TPR':>9}{'F1':>9}{'FPR_novel':>12}")
    print("-" * 70)
    for src in ["raw", "propagated"]:
        r = out["results"][src]
        print(f"{src:<12}{r['FPR']:>9.4f}{r['TPR']:>9.4f}{r['F1']:>9.4f}"
              f"{r.get('FPR_novel_normality', float('nan')):>12.4f}")
    print("=" * 70)

    opath = os.path.join(ROOT, "outputs", f"consistency_ablation_{args.dataset}_{args.relation}.json")
    os.makedirs(os.path.dirname(opath), exist_ok=True)
    with open(opath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {opath}")


if __name__ == "__main__":
    main()
