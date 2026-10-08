"""任务 B: CUR-GAD 等价代理基线间接对标实验.

CUR-GAD 无开源代码; 其与 CRC-SGAD (Bai et al., 2025, 同作者团队) 共享核心
机制 "Dual-Threshold Conformal Risk Control": 不确定性代理校准 + i.i.d. 风险
控制器. 我们复刻该逻辑 (adapters/conformal.curgad_proxy_threshold) 作为公平
代理基线, 在四大数据集上跑 3-seed 对比:

    (1) StructCP       : 超图结构一致性权重 + 伪正常扩增 (测试期对齐 normality shift)
    (2) CUR-GAD-proxy: 不确定性代理(特征距离) + i.i.d. 双阈值风险控制器
                       (不做结构对齐, 不扩增, 严格 i.i.d. 假设)

产出 3-seed 均值 FPR/TPR/F1, 证明 StructCP 在 normality shift 场景显著优于
"纯 i.i.d. 不确定性校准" 思路, 消除评审 "对比仅文字空谈" 质疑.

断点续跑: key = {ds}_{seed}_{method}.
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
    curgad_proxy_threshold,
    graphlcp_proxy_threshold,
    evaluate_coverage,
    structcp_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph,
    hyperedge_consistency,
    hypergraph_propagation,
)
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
# 统一新种子集 (与 run_community_mitigation / run_structcp_seeds 一致)
SEEDS = [42, 123, 456, 789, 1011]
ALPHA = 0.05
K = 10
LAYERS = 2
ALPHA_RES = 0.5
TAU = 0.5
SCORE_QUANTILE = 0.75
METHODS = ("StructCP", "CUR-GAD-proxy", "GRAPHLCP-proxy")


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_once(dataset, seed, method, device):
    relation = "homo"
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    ckpt = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth"),
                      map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal_m, test_m = data.cal_mask, data.test_mask
    y_test = data.y[test_m].cpu().numpy()
    y_cal = data.y[cal_m]

    s_raw = get_scores(model, data.x, data.edge_index)
    hg = KNNHypergraph(data.x, k=K).to(device)
    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    s_tta = get_scores(model, x_enh, data.edge_index)

    if method == "StructCP":
        cons = hyperedge_consistency(s_tta, hg, features=x_enh)
        thr, info = structcp_threshold(
            s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
            alpha=ALPHA, tau_quantile=TAU, mode="hard",
            score_quantile=SCORE_QUANTILE if dataset in ("Elliptic", "TFinance") else None,
        )
    else:  # CUR-GAD proxy
        thr_per_node, info = curgad_proxy_threshold(
            s_tta[cal_m], y_cal, data.x[cal_m], s_tta[test_m], data.x[test_m],
            alpha=ALPHA,
        )
        pred = (s_tta[test_m] > thr_per_node).long()
        cov = evaluate_coverage(s_tta[test_m], data.y[test_m], thr_per_node)
        cov["AUROC"] = roc_auc_score(y_test, s_tta[test_m].cpu().numpy())
        cov["AUPRC"] = average_precision_score(y_test, s_tta[test_m].cpu().numpy())
        cov["info"] = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
        return cov

    cov = evaluate_coverage(s_tta[test_m], data.y[test_m], thr)
    cov["AUROC"] = roc_auc_score(y_test, s_tta[test_m].cpu().numpy())
    cov["AUPRC"] = average_precision_score(y_test, s_tta[test_m].cpu().numpy())
    return cov


@torch.no_grad()
def run_once_graphlcp(dataset, seed, device):
    """GRAPHLCP 合理变体: 图局部性平滑 + 标准 split-conformal (保持可交换性)."""
    relation = "homo"
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    ckpt = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth"),
                      map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal_m, test_m = data.cal_mask, data.test_mask
    y_test = data.y[test_m].cpu().numpy()
    y_cal = data.y[cal_m]

    s_raw = get_scores(model, data.x, data.edge_index)
    hg = KNNHypergraph(data.x, k=K).to(device)
    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    s_tta = get_scores(model, x_enh, data.edge_index)

    pred, info = graphlcp_proxy_threshold(
        s_tta[cal_m], y_cal, data.x[cal_m], s_tta[test_m], data.x[test_m],
        alpha=ALPHA, k=K,
    )
    cov = evaluate_coverage(s_tta[test_m], data.y[test_m], pred)
    cov["AUROC"] = roc_auc_score(y_test, s_tta[test_m].cpu().numpy())
    cov["AUPRC"] = average_precision_score(y_test, s_tta[test_m].cpu().numpy())
    cov["info"] = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_path = os.path.join(ROOT, "outputs", "proxy_baselines.json")
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}

    dsets = [args.dataset] if args.dataset else DATASETS
    for ds in dsets:
        for seed in SEEDS:
            for method in METHODS:
                key = f"{ds}_{seed}_{method}"
                if key in cache:
                    print(f"[skip] {key}")
                    continue
                t0 = time.time()
                try:
                    if method == "GRAPHLCP-proxy":
                        r = run_once_graphlcp(ds, seed, device)
                    else:
                        r = run_once(ds, seed, method, device)
                except Exception as e:
                    print(f"[err] {key}: {e}")
                    continue
                r["_time_s"] = round(time.time() - t0, 2)
                cache[key] = r
                json.dump(cache, open(out_path, "w"), indent=2)
                print(f"[ok] {key} FPR={r['FPR']:.4f} TPR={r['TPR']:.4f} ({r['_time_s']}s)")

    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
