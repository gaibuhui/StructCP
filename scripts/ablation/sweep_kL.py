"""超参数 k (k-NN 大小) / L (传播层数) 敏感性扫描.

目的: 系统评估 StructCP 在 4 个基准数据集上, k 与 L 变化对
目标 α=0.05 下 FPR / TPR 的影响, 以回答"k=10, L=2 仅在 Amazon 上
sweep 后固定"的审稿质疑. 各数据集独立扫描, 结果落盘 + 绘制热图.

复用 run_structcp.py 的核心推理流程 (Frozen / HG-TTA / StructCP),
但只跑 StructCP 主配置 (hard + score_quantile=0.75), 网格扫描 (k, L).

运行:
  cd StructCP && source activate CBP && export PYTHONPATH=StructCP && \
  python scripts/sweep_kL.py
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
K_GRID = [5, 10, 15, 20]
L_GRID = [1, 2, 3]


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_one(dataset, relation, k, layers, alpha, tau, score_quantile,
            device, seed=42):
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    cal_m, test_m = data.cal_mask, data.test_mask
    y_test = data.y[test_m].cpu().numpy()
    y_cal = data.y[cal_m]

    ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    s_raw = get_scores(model, data.x, data.edge_index)
    # Frozen baseline (固定 alpha 标准 split conformal)
    thr_f = split_conformal_threshold(s_raw[cal_m][y_cal == 0], alpha)
    frozen = evaluate_coverage(s_raw[test_m], data.y[test_m], thr_f)

    # TTA 增强分数
    hg = KNNHypergraph(data.x, k=k, metric="cosine", approx=False).to(device)
    x_enh = hypergraph_propagation(data.x, hg, layers, 0.5)
    s_tta = get_scores(model, x_enh, data.edge_index)

    cons = hyperedge_consistency(s_tta, hg, features=x_enh)
    thr_s, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=alpha, tau_quantile=tau, mode="hard",
        score_quantile=score_quantile,
    )
    structcp = evaluate_coverage(s_tta[test_m], data.y[test_m], thr_s)
    structcp.update({f"w_{kk}": v for kk, v in info.items()})
    return {"frozen": frozen, "structcp": structcp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--score_quantile", type=float, default=0.75)
    ap.add_argument("--k_grid", nargs="+", type=int, default=K_GRID)
    ap.add_argument("--l_grid", nargs="+", type=int, default=L_GRID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    out = {"config": vars(args), "grids": {"k": args.k_grid, "L": args.l_grid},
           "results": {}}

    for dataset in args.datasets:
        relation = "homo" if dataset in ("Elliptic", "TFinance") else "homo"
        print(f"\n########## dataset={dataset} ##########")
        ds_res = {}
        for k in args.k_grid:
            for L in args.l_grid:
                t0 = time.time()
                r = run_one(dataset, relation, k, L, args.alpha,
                            args.tau, args.score_quantile, device, args.seed)
                dt = time.time() - t0
                ds_res[f"k{k}_L{L}"] = {
                    "frozen_FPR": r["frozen"]["FPR"],
                    "frozen_TPR": r["frozen"]["TPR"],
                    "structcp_FPR": r["structcp"]["FPR"],
                    "structcp_TPR": r["structcp"]["TPR"],
                    "structcp_ESS": r["structcp"].get("w_ess"),
                    "elapsed_s": dt,
                }
                print(f"  k={k:2d} L={L} | FPR={r['structcp']['FPR']:.4f} "
                      f"TPR={r['structcp']['TPR']:.4f} "
                      f"(Frozen FPR={r['frozen']['FPR']:.4f})  {dt:.1f}s")
        out["results"][dataset] = ds_res
        # 每数据集即时落盘, 支持断点续跑
        os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
        with open(os.path.join(ROOT, "outputs", "kl_sensitivity.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"[saved] outputs/kl_sensitivity.json ({dataset} done)")

    # ---- 绘制热图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric in ["structcp_FPR", "structcp_TPR"]:
            fig, axes = plt.subplots(1, len(args.datasets), figsize=(4 * len(args.datasets), 3.2))
            if len(args.datasets) == 1:
                axes = [axes]
            for ax, dataset in zip(axes, args.datasets):
                mat = np.zeros((len(args.l_grid), len(args.k_grid)))
                for i, L in enumerate(args.l_grid):
                    for j, k in enumerate(args.k_grid):
                        mat[i, j] = out["results"][dataset][f"k{k}_L{L}"][metric]
                im = ax.imshow(mat, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(args.k_grid)))
                ax.set_xticklabels(args.k_grid)
                ax.set_yticks(range(len(args.l_grid)))
                ax.set_yticklabels(args.l_grid)
                ax.set_xlabel("k")
                ax.set_ylabel("L (layers)")
                ax.set_title(f"{dataset}\n{metric}")
                for i in range(len(args.l_grid)):
                    for j in range(len(args.k_grid)):
                        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                                color="white", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            png = os.path.join(ROOT, "outputs", f"kl_sensitivity_{metric}.png")
            fig.savefig(png, dpi=130, bbox_inches="tight")
            print(f"[saved] {png}")
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    print("\n[done] outputs/kl_sensitivity.json")


if __name__ == "__main__":
    main()
