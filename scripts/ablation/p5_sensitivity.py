"""P5-a: StructCP 超参数敏感性扫描 (η = alpha_res, γ = tau_quantile)。

目的:
  证明 StructCP 对两个核心超参数 (η 传播残差系数, γ 伪正常一致性分位数)
  在合理区间内 FPR 控制稳定、TPR 不崩, 即方法对超参选择不敏感, 鲁棒。

复用 run_structcp.py 的核心推理流程, 仅外层增加网格扫描 + 落盘 JSON + 出热力图。

输出:
  outputs/p5_sensitivity_{dataset}.json        # 每个 (η,γ) 的 FPR/TPR/F1
  outputs/submission/figures/fig_sensitivity_{dataset}.png   # 热力图 (FPR | TPR)
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
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys  # noqa: E402

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

# 扫描网格 (论文用默认值 η=0.5, γ=0.5 已验证)
ETA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
GAMMA_GRID = [0.3, 0.5, 0.7, 0.9]


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_one(dataset, relation, eta, gamma, alpha, k, layers, seed, device):
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal = data.y[cal_m]

    # Frozen 参考阈值 (split CP)
    s_raw = get_scores(model, data.x, data.edge_index)
    thr0 = split_conformal_threshold(s_raw[cal_m][y_cal == 0], alpha)

    # 超图 TTA (特征增强)
    hg = KNNHypergraph(data.x, k=k).to(device)
    x_enh = hypergraph_propagation(data.x, hg, layers, eta)
    s_tta = get_scores(model, x_enh, data.edge_index)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh)

    thr, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=alpha, tau_quantile=gamma, mode="hard",
        score_quantile=0.75 if dataset in ("Elliptic", "TFinance") else None,
    )
    cov = evaluate_coverage(s_tta[test_m], data.y[test_m], thr)
    return {
        "eta": eta, "gamma": gamma,
        "FPR": cov["FPR"], "TPR": cov["TPR"], "F1": cov["F1"],
        "threshold": float(thr),
        "FPR_frozen": evaluate_coverage(s_raw[test_m], data.y[test_m], thr0)["FPR"],
    }


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
    ap.add_argument("--eta_grid", type=float, nargs="*", default=ETA_GRID)
    ap.add_argument("--gamma_grid", type=float, nargs="*", default=GAMMA_GRID)
    args = ap.parse_args()

    relation = args.relation if args.dataset not in ("Elliptic", "TFinance") else "homo"
    device = torch.device(args.device)

    grid = []
    t0 = time.time()
    for eta in args.eta_grid:
        for gamma in args.gamma_grid:
            r = run_one(args.dataset, relation, eta, gamma,
                        args.alpha, args.k, args.layers, args.seed, device)
            grid.append(r)
            print(f"[scan] η={eta:.1f} γ={gamma:.1f} | "
                  f"FPR={r['FPR']:.4f} TPR={r['TPR']:.4f} F1={r['F1']:.4f}")
    print(f"[done] {args.dataset} scanned {len(grid)} configs in {time.time()-t0:.1f}s")

    out = os.path.join(ROOT, "outputs", f"p5_sensitivity_{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "grid": grid}, f, indent=2)
    print(f"[saved] {out}")

    # ---------------- 出图 (热力图) ----------------
    etas = args.eta_grid
    gammas = args.gamma_grid
    fpr_mat = np.zeros((len(etas), len(gammas)))
    tpr_mat = np.zeros((len(etas), len(gammas)))
    for r in grid:
        i = etas.index(r["eta"])
        j = gammas.index(r["gamma"])
        fpr_mat[i, j] = r["FPR"]
        tpr_mat[i, j] = r["TPR"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mat, title, cmap in [
        (axes[0], fpr_mat, f"FPR (target α={args.alpha})", "YlOrRd"),
        (axes[1], tpr_mat, "TPR", "YlGnBu"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(gammas)))
        ax.set_xticklabels(gammas)
        ax.set_yticks(range(len(etas)))
        ax.set_yticklabels(etas)
        ax.set_xlabel("γ = tau_quantile")
        ax.set_ylabel("η = alpha_res")
        ax.set_title(title)
        for i in range(len(etas)):
            for j in range(len(gammas)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"StructCP hyperparameter sensitivity — {args.dataset}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_dir = os.path.join(ROOT, "outputs", "submission", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"fig_sensitivity_{args.dataset}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[fig] {fig_path}")


if __name__ == "__main__":
    main()
