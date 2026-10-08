"""任务9(d): YelpChi + 更强冻结骨干 GAD-NR 的 StructCP TPR 恢复实验。

目的: 验证论文核心 claim —— YelpChi 主表的近零 TPR 是 *弱骨干 (BWGNN) 天花板*,
而非 StructCP 共形层失效。固定 StructCP 的 TTA + 加权共形管线不变, 仅把冻结
打分器从 BWGNN 换成 GAD-NR (WSDM-2024 邻域重建骨干, 无监督预训练后冻结)。

复用 adapters.conformal / adapters.hypergraph_tta 的整套加权共形逻辑, 与
run_structcp.py 完全一致, 仅替换 get_scores 为 GAD-NR 的 anomaly_score
(重建损失, 越高越异常, 语义与 BWGNN 类-1 概率一致)。

输出: outputs/p9_yelpchi_gadnr.json
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
from models.gadnr import GADNR  # noqa: E402

ROOT = _structcp_root()


@torch.no_grad()
def get_scores_gadnr(model, x, edge_index):
    """GAD-NR 异常分数 = 邻域重建损失 (越高越异常)。"""
    return model.anomaly_score(x, edge_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YelpChi")
    ap.add_argument("--relation", default="homo")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--mode", default="hard", choices=["hard", "soft"])
    ap.add_argument("--score_quantile", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data = load_gad(args.dataset, relation=args.relation, seed=args.seed).to(device)

    ckpt = torch.load(os.path.join(ROOT, "checkpoint", f"gadnr_{args.dataset}_homo.pth"),
                      map_location=device)
    model = GADNR(ckpt["in_channels"], ckpt["args"]["hidden"],
                  lambda_n=ckpt["args"]["lambda_n"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    print(f"[model] loaded GAD-NR (gadnr_{args.dataset}_homo.pth), trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    cal_m, test_m = data.cal_mask, data.test_mask
    y_test = data.y[test_m].cpu().numpy()
    y_cal = data.y[cal_m]

    results = {}

    # 1. Frozen (GAD-NR 原始分数 + 标准 split conformal)
    t0 = time.time()
    s_raw = get_scores_gadnr(model, data.x, data.edge_index)
    t_frozen = time.time() - t0
    thr = split_conformal_threshold(s_raw[cal_m][y_cal == 0], args.alpha)
    results["Frozen-GADNR"] = {
        "AUROC": roc_auc_score(y_test, s_raw[test_m].cpu().numpy()),
        "AUPRC": average_precision_score(y_test, s_raw[test_m].cpu().numpy()),
        "threshold": float(thr), "infer_time_s": t_frozen,
        **evaluate_coverage(s_raw[test_m], data.y[test_m], thr),
    }

    # 2. HG-TTA (k-NN 邻域传播增强特征 + 标准 split conformal)
    t0 = time.time()
    hg = KNNHypergraph(data.x, k=args.k).to(device)
    t_build = time.time() - t0
    t0 = time.time()
    x_enh = hypergraph_propagation(data.x, hg, args.layers, args.alpha_res)
    s_tta = get_scores_gadnr(model, x_enh, data.edge_index)
    t_tta = time.time() - t0

    thr2 = split_conformal_threshold(s_tta[cal_m][y_cal == 0], args.alpha)
    results["HG-TTA-GADNR"] = {
        "AUROC": roc_auc_score(y_test, s_tta[test_m].cpu().numpy()),
        "AUPRC": average_precision_score(y_test, s_tta[test_m].cpu().numpy()),
        "threshold": float(thr2), "infer_time_s": t_tta + t_build,
        **evaluate_coverage(s_tta[test_m], data.y[test_m], thr2),
    }

    # 3. StructCP (full) = HG-TTA + 邻域一致性加权保形
    cons = hyperedge_consistency(s_tta, hg, features=x_enh)
    thr3, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=args.alpha, tau_quantile=args.tau, mode=args.mode,
        score_quantile=args.score_quantile,
    )
    results["StructCP-GADNR"] = {
        "AUROC": roc_auc_score(y_test, s_tta[test_m].cpu().numpy()),
        "AUPRC": average_precision_score(y_test, s_tta[test_m].cpu().numpy()),
        "threshold": float(thr3), "infer_time_s": t_tta + t_build,
        **evaluate_coverage(s_tta[test_m], data.y[test_m], thr3), **info,
    }

    # novel normality 误报率
    nov = data.novel_mask & test_m
    if int(nov.sum()) > 0:
        for name, s, t in [("Frozen-GADNR", s_raw, thr),
                           ("HG-TTA-GADNR", s_tta, thr2),
                           ("StructCP-GADNR", s_tta, thr3)]:
            results[name]["FPR_novel_normality"] = float((s[nov] > t).float().mean())

    print(f"\n{'='*100}\nYelpChi + GAD-NR backbone (StructCP pipeline)  alpha={args.alpha}\n{'='*100}")
    hdr = f"{'Method':<16}{'AUROC':>9}{'AUPRC':>9}{'FPR':>9}{'TPR':>9}{'F1':>9}{'FPR_novel':>12}{'time(s)':>10}"
    print(hdr); print("-" * 100)
    for name in ["Frozen-GADNR", "HG-TTA-GADNR", "StructCP-GADNR"]:
        r = results[name]
        print(f"{name:<16}{r['AUROC']:>9.4f}{r['AUPRC']:>9.4f}{r['FPR']:>9.4f}"
              f"{r['TPR']:>9.4f}{r['F1']:>9.4f}"
              f"{r.get('FPR_novel_normality', float('nan')):>12.4f}{r['infer_time_s']:>10.3f}")
    print("=" * 100)

    out = os.path.join(ROOT, "outputs", "p9_yelpchi_gadnr.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
