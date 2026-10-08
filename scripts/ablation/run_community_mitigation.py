"""任务 A: 社区型异常 TPR 衰减缓解实验.

验证结构感知联合权重衰减策略 (high-pass beta x deg_gate) + 自适应 alpha
能否在维持 FPR <= alpha 的前提下, 显著回升社区型异常 (Elliptic/T-Finance)
的 TPR, 缩小方法适用边界缺陷.

扫描网格:
    beta      ∈ [0, 0.2, 0.4, 0.6, 0.8]
    deg_gate  ∈ [False, True]
    alpha_mode∈ [fixed(0.05), adaptive]
    对比基线  : StructCP-hard (原 run_structcp 默认, score_quantile=0.75)

每个 (dataset, seed, beta, gate, alpha_mode) 跑完整 StructCP 流程, 记录
FPR / TPR / F1 / AUROC / AUPRC / alpha_eff(adaptive) / ESS.

断点续跑: key = {ds}_{seed}_{beta}_{gate}_{mode}, 已存在则跳过.
3-seed 聚合输出到 outputs/community_mitigation.json.
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
    adaptive_structcp_threshold,
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
SEEDS = [42, 123, 456, 789, 1011]  # 5 标准种子 (扩展自 {42,123,456}; 原 [0,1,2] 已弃用)
BETAS = [0.0, 0.2, 0.4, 0.6, 0.8]
GATES = [False, True]
ALPHA = 0.05
K = 10
LAYERS = 2
ALPHA_RES = 0.5
TAU = 0.5
SCORE_QUANTILE = 0.75  # Elliptic/T-Finance 修复过度校正


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_one(dataset, seed, beta, gate, adaptive, device):
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
    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES,
                                   beta_highpass=beta, deg_gate=gate)
    s_tta = get_scores(model, x_enh, data.edge_index)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh)

    if adaptive:
        thr, info = adaptive_structcp_threshold(
            s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
            alpha=ALPHA, alpha_max=0.10,
            tau_quantile=TAU, score_quantile=SCORE_QUANTILE,
            weight_mode="contrastive",
        )
    else:
        thr, info = structcp_threshold(
            s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
            alpha=ALPHA, tau_quantile=TAU, mode="hard",
            score_quantile=SCORE_QUANTILE, weight_mode="contrastive",
        )

    cov = evaluate_coverage(s_tta[test_m], data.y[test_m], thr)
    cov["AUROC"] = roc_auc_score(y_test, s_tta[test_m].cpu().numpy())
    cov["AUPRC"] = average_precision_score(y_test, s_tta[test_m].cpu().numpy())
    cov["alpha_eff"] = info.get("alpha_eff", ALPHA)
    cov["ess"] = info.get("ess", float("nan"))
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_path = os.path.join(ROOT, "outputs", "community_mitigation.json")
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}

    dsets = [args.dataset] if args.dataset else DATASETS
    for ds in dsets:
        for seed in SEEDS:
            for beta in BETAS:
                for gate in GATES:
                    for adaptive in (False, True):
                        mode = "adaptive" if adaptive else "fixed"
                        key = f"{ds}_{seed}_{beta}_{gate}_{mode}"
                        if key in cache:
                            print(f"[skip] {key}")
                            continue
                        t0 = time.time()
                        try:
                            r = run_one(ds, seed, beta, gate, adaptive, device)
                        except Exception as e:
                            print(f"[err] {key}: {e}")
                            continue
                        r["_time_s"] = round(time.time() - t0, 2)
                        cache[key] = r
                        json.dump(cache, open(out_path, "w"), indent=2)
                        print(f"[ok] {key} FPR={r['FPR']:.4f} TPR={r['TPR']:.4f} "
                              f"a_eff={r['alpha_eff']:.3f} ({r['_time_s']}s)")

    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
