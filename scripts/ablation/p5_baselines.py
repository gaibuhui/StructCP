"""P5-b: 对照基线 — Temp Scaling & Distribution Alignment (feature/score-level 校准)。

论点 (论文核心):
  正常性偏移下, 校准集根本不含新正常类别。任何"仅缩放分数/对齐分布"
  的免训练校准都无法凭空造出校准集里缺失的高分正常样本, 因此标准
  split conformal 的 FPR 仍会超标 (可交换性未被恢复)。StructCP 的加权
  CP + 伪正常扩增才是真正恢复可交换性的机制。

基线:
  (A) Temp Scaling       : 在校准集上搜最优温度 T 缩放 logits, 标准 split CP
  (B) Distribution Align : 将测试期特征均值/方差对齐到校准集 (feature DA), 标准 split CP
  (对照) StructCP          : 同上 run_structcp 的加权 CP 结果 (本脚本内重算以同条件对比)

输出:
  outputs/p5_baselines_{dataset}.json
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

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = _structcp_root()
sys.path.insert(0, ROOT)

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


@torch.no_grad()
def get_logits(model, x, edge_index):
    return model(x, edge_index)


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


@torch.no_grad()
def best_temp_scaling(logits_cal, y_cal, grid=(0.2, 0.5, 1.0, 2.0, 5.0, 10.0)):
    """在**校准正常集**上选使 split-CP FPR 最接近 α 的温度 (无法恢复, 仅演示)。"""
    normal = y_cal == 0
    best, best_err = 1.0, 1e9
    for T in grid:
        s = F.softmax(logits_cal / T, dim=1)[:, 1]
        thr = split_conformal_threshold(s[normal], alpha=0.05)
        fpr = evaluate_coverage(s, y_cal, thr)["FPR"]
        err = abs(fpr - 0.05)
        if err < best_err:
            best, best_err = T, err
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon",
                    choices=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    relation = args.relation if args.dataset not in ("Elliptic", "TFinance") else "homo"
    device = torch.device(args.device)
    data = load_gad(args.dataset, relation=relation, seed=args.seed).to(device)
    ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{args.dataset}_{relation}.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal, y_test = data.y[cal_m], data.y[test_m].cpu().numpy()
    normal = y_cal == 0

    # 原始分数 (Frozen split CP)
    s_raw = get_scores(model, data.x, data.edge_index)
    thr0 = split_conformal_threshold(s_raw[cal_m][normal], args.alpha)
    frozen = {"AUROC": roc_auc_score(y_test, s_raw[test_m].cpu().numpy()),
              "AUPRC": average_precision_score(y_test, s_raw[test_m].cpu().numpy()),
              **evaluate_coverage(s_raw[test_m], data.y[test_m], thr0)}

    # ---- (A) Temp Scaling ----
    logits = get_logits(model, data.x, data.edge_index)
    T = best_temp_scaling(logits[cal_m], y_cal)
    s_ts = F.softmax(logits / T, dim=1)[:, 1]
    thr_ts = split_conformal_threshold(s_ts[cal_m][normal], args.alpha)
    temp_scaling = {"AUROC": roc_auc_score(y_test, s_ts[test_m].cpu().numpy()),
                    "AUPRC": average_precision_score(y_test, s_ts[test_m].cpu().numpy()),
                    **evaluate_coverage(s_ts[test_m], data.y[test_m], thr_ts),
                    "temperature": float(T)}

    # ---- (B) Distribution Alignment (feature-level) ----
    mu_cal = data.x[cal_m][normal].mean(0, keepdim=True)
    std_cal = data.x[cal_m][normal].std(0, keepdim=True).clamp(min=1e-6)
    mu_test = data.x[test_m].mean(0, keepdim=True)
    std_test = data.x[test_m].std(0, keepdim=True).clamp(min=1e-6)
    x_da = (data.x - mu_test) / std_test * std_cal + mu_cal  # 测试特征对齐校准分布
    s_da = get_scores(model, x_da, data.edge_index)
    thr_da = split_conformal_threshold(s_da[cal_m][normal], args.alpha)
    dist_align = {"AUROC": roc_auc_score(y_test, s_da[test_m].cpu().numpy()),
                  "AUPRC": average_precision_score(y_test, s_da[test_m].cpu().numpy()),
                  **evaluate_coverage(s_da[test_m], data.y[test_m], thr_da)}

    # ---- (C) StructCP (加权 CP, 同条件) ----
    hg = KNNHypergraph(data.x, k=args.k).to(device)
    x_enh = hypergraph_propagation(data.x, hg, args.layers, 0.5)
    s_tta = get_scores(model, x_enh, data.edge_index)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh)
    thr_h, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=args.alpha, tau_quantile=0.5, mode="hard",
        score_quantile=0.75 if args.dataset in ("Elliptic", "TFinance") else None,
    )
    structcp = {"AUROC": roc_auc_score(y_test, s_tta[test_m].cpu().numpy()),
              "AUPRC": average_precision_score(y_test, s_tta[test_m].cpu().numpy()),
              **evaluate_coverage(s_tta[test_m], data.y[test_m], thr_h),
              **info}

    results = {"Frozen": frozen, "TempScaling": temp_scaling,
               "DistAlign": dist_align, "StructCP": structcp}

    print(f"\n{'='*90}\nP5 baselines on {args.dataset} (α={args.alpha})")
    hdr = f"{'Method':<12}{'AUROC':>9}{'AUPRC':>9}{'FPR':>9}{'TPR':>9}{'F1':>9}"
    print(hdr); print("-" * 90)
    for n, r in results.items():
        print(f"{n:<12}{r['AUROC']:>9.4f}{r['AUPRC']:>9.4f}"
              f"{r['FPR']:>9.4f}{r['TPR']:>9.4f}{r['F1']:>9.4f}")
    print("=" * 90)

    out = os.path.join(ROOT, "outputs", f"p5_baselines_{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
