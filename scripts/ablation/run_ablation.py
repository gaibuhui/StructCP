"""消融与敏感性实验。

1) alpha 敏感性: 检验保形保证 FPR <= alpha 是否在不同目标水平下成立
2) 组件消融: 超图传播 / 一致性加权 / 伪正常扩增 各自的贡献
3) k 与传播层数的敏感性
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

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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


def load_all(dataset, relation="homo", device="cuda", seed=42):
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    ck = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth"),
                    map_location=device)
    a = ck["args"]
    model = BWGNN(ck["in_channels"], a["hidden"], 2, d=a["order"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.freeze()
    return data, model


@torch.no_grad()
def scores_of(model, x, ei):
    return F.softmax(model(x, ei), dim=1)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon", choices=["Amazon", "YelpChi", "Elliptic"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--mode", default="hard", choices=["hard", "soft"],
                    help="StructCP 伪正常处理模式")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    relation = args.relation if args.dataset != "Elliptic" else "homo"
    data, model = load_all(args.dataset, relation, device, args.seed)
    cal, te = data.cal_mask, data.test_mask
    y_cal, y_te = data.y[cal], data.y[te]
    nov = data.novel_mask & te

    out = {"dataset": args.dataset}

    # ============ 1) alpha 敏感性 ============
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta = scores_of(model, x_enh, data.edge_index)
    s_raw = scores_of(model, data.x, data.edge_index)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh)

    rows = []
    for alpha in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
        t_base = split_conformal_threshold(s_raw[cal][y_cal == 0], alpha)
        r_base = evaluate_coverage(s_raw[te], y_te, t_base)
        t_hy, _ = structcp_threshold(s_tta[cal], y_cal, cons[cal],
                                   s_tta[te], cons[te], alpha=alpha, mode=args.mode)
        r_hy = evaluate_coverage(s_tta[te], y_te, t_hy)
        rows.append({
            "alpha": alpha,
            "Frozen_FPR": r_base["FPR"], "Frozen_TPR": r_base["TPR"], "Frozen_F1": r_base["F1"],
            "StructCP_FPR": r_hy["FPR"], "StructCP_TPR": r_hy["TPR"], "StructCP_F1": r_hy["F1"],
            "Frozen_valid": r_base["FPR"] <= alpha, "StructCP_valid": r_hy["FPR"] <= alpha,
        })
    out["alpha_sensitivity"] = rows

    print(f"\n=== [{args.dataset}] alpha 敏感性 (FPR <= alpha 即保证成立) ===")
    print(f"{'alpha':>7}{'Frozen FPR':>12}{'ok':>5}{'StructCP FPR':>12}{'ok':>5}"
          f"{'Frozen F1':>11}{'StructCP F1':>11}")
    for r in rows:
        print(f"{r['alpha']:>7.2f}{r['Frozen_FPR']:>12.4f}"
              f"{'Y' if r['Frozen_valid'] else 'N':>5}"
              f"{r['StructCP_FPR']:>12.4f}{'Y' if r['StructCP_valid'] else 'N':>5}"
              f"{r['Frozen_F1']:>11.4f}{r['StructCP_F1']:>11.4f}")

    # ============ 2) 组件消融 (alpha=0.05) ============
    alpha = 0.05
    ab = {}
    # (a) 全部关闭 = Frozen + split conformal
    t = split_conformal_threshold(s_raw[cal][y_cal == 0], alpha)
    ab["base (no HG, no CP-weight)"] = evaluate_coverage(s_raw[te], y_te, t)
    ab["base (no HG, no CP-weight)"]["FPR_novel"] = float((s_raw[nov] > t).float().mean())

    # (b) 仅超图传播
    t = split_conformal_threshold(s_tta[cal][y_cal == 0], alpha)
    ab["+ HG propagation"] = evaluate_coverage(s_tta[te], y_te, t)
    ab["+ HG propagation"]["FPR_novel"] = float((s_tta[nov] > t).float().mean())

    # (c) 超图 + 伪正常扩增(等权, 无一致性加权)
    t, _ = structcp_threshold(s_tta[cal], y_cal, torch.ones_like(cons[cal]),
                            s_tta[te], cons[te], alpha=alpha)
    ab["+ pseudo-normal (uniform w)"] = evaluate_coverage(s_tta[te], y_te, t)
    ab["+ pseudo-normal (uniform w)"]["FPR_novel"] = float((s_tta[nov] > t).float().mean())

    # (d) 完整 StructCP
    t, info = structcp_threshold(s_tta[cal], y_cal, cons[cal], s_tta[te], cons[te], alpha=alpha)
    ab["StructCP (full)"] = evaluate_coverage(s_tta[te], y_te, t)
    ab["StructCP (full)"]["FPR_novel"] = float((s_tta[nov] > t).float().mean())
    out["ablation"] = ab
    out["pseudo_info"] = info

    print(f"\n=== [{args.dataset}] 组件消融 (alpha=0.05) ===")
    print(f"{'variant':<32}{'FPR':>9}{'TPR':>9}{'F1':>9}{'FPR_novel':>12}")
    for k, v in ab.items():
        print(f"{k:<32}{v['FPR']:>9.4f}{v['TPR']:>9.4f}{v['F1']:>9.4f}{v['FPR_novel']:>12.4f}")

    # ============ 3) k / layers 敏感性 ============
    sens = []
    for k in [5, 10, 20]:
        h = KNNHypergraph(data.x, k=k).to(device)
        for L in [1, 2, 3]:
            xe = hypergraph_propagation(data.x, h, L, 0.5)
            s = scores_of(model, xe, data.edge_index)
            c = hyperedge_consistency(s, h, features=xe)
            t, _ = structcp_threshold(s[cal], y_cal, c[cal], s[te], c[te], alpha=0.05)
            r = evaluate_coverage(s[te], y_te, t)
            sens.append({"k": k, "layers": L, "AUROC": roc_auc_score(y_te.cpu(), s[te].cpu()),
                         "FPR": r["FPR"], "TPR": r["TPR"], "F1": r["F1"]})
    out["knn_sensitivity"] = sens
    print(f"\n=== [{args.dataset}] k / layers 敏感性 (alpha=0.05) ===")
    print(f"{'k':>4}{'L':>4}{'AUROC':>9}{'FPR':>9}{'TPR':>9}{'F1':>9}")
    for r in sens:
        print(f"{r['k']:>4}{r['layers']:>4}{r['AUROC']:>9.4f}{r['FPR']:>9.4f}"
              f"{r['TPR']:>9.4f}{r['F1']:>9.4f}")

    p = os.path.join(ROOT, "outputs", f"ablation_{args.dataset}_{args.relation}_{args.mode}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
