"""Appendix 验证: 特征传播如何"锐化"一致性统计量 (解释 TTA 与 StructCP 的互补性)。

动机 (review-driven):
    审稿指出本论文指出 "TTA 不恢复交换性", 但实验显示 TUNE + StructCP 效果最好,
    二者确有互补性却缺清晰数学解释 —— 为什么特征对齐后的邻域一致性权重更有效?

论点:
    特征传播 (Eq.3 超图拉普拉斯低通) 是特征流形上的低通滤波: 同簇相连节点特征被拉近、
    inter-class 边界保留 (同配/簇结构下边主要连同类)。于是 (a) 邻域特征离散度 shrinks,
    使 c_feat = exp(-cosdist(x_i, centroid_Ni)) 更锐利; (b) 检测器分数 s_i 是 x_enh 的函数,
    故 s_i 也被对齐锐化, c_score = exp(-|s_i - med(s_Ni)|) 在新正常 vs 真异常间的对比度放大。
    关键: 传播无需"恢复交换性"(覆盖性质), 只需 *锐化权重信号* (加权性质) —— 这正是
    TTA(改善权重质量) 与 StructCP(覆盖重标定) 互补的根因。

本脚本量化该锐化 (隔离特征一致性通道, 分数来源固定为 tta):
    - raw  : 一致性特征 = raw X
    - prop : 一致性特征 = propagated X^{(L)}
    比较 (a) 邻域特征离散度 (cosine) 在正常/异常组的收缩 (锐化直接证据),
          (b) 一致性权重 c 对异常的分离 AUROC (越高=权重越能区分异常/正常),
          (c) 完整 StructCP 的 FPR/TPR (prop 应≈raw, 与 tab:consistency_feat 一致, 不损覆盖)。

用法 (CBP 环境):
    cd /media/lixin/新加卷/数据集/test/StructCP
    source <CONDA_PREFIX>/bin/activate CBP
    export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
    python scripts/run_ablation_sharpen.py --datasets Amazon,YelpChi,Elliptic,TFinance
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, _structcp_root())

from adapters.conformal import evaluate_coverage, structcp_threshold  # noqa: E402
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


def consistency_separability(cons, y):
    """异常分离 AUROC: c 高 = 正常, 故 rank 异常为低 c -> roc_auc(y_anom, -c)."""
    y = y.cpu().numpy()
    c = -cons.cpu().numpy()
    if c[y == 1].std() < 1e-12 or c[y == 0].std() < 1e-12:
        return float("nan")
    return float(roc_auc_score(y, c))


def neighborhood_feature_dispersion(features, hg, y_full, test_mask):
    """测试节点到其超边邻域质心的 cosine 离散度均值 (正常组/异常组)。"""
    f = features.detach().cpu().numpy()
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    idx = hg.knn_idx.cpu().numpy()                       # (N, k+1)
    centroid = f[idx].mean(axis=1)
    centroid = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8)
    dist = 1.0 - (f * centroid).sum(axis=1)              # cosine 距离
    test_idx = np.where(test_mask.cpu().numpy())[0]
    d_test = dist[test_idx]
    y_test = y_full.detach().cpu().numpy()[test_idx]
    return float(d_test[y_test == 1].mean()), float(d_test[y_test == 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--relation", default="homo")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--score_quantile", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rows = []
    for ds in datasets:
        t0 = time.time()
        relation = args.relation if ds not in ("Elliptic", "TFinance") else "homo"
        data = load_gad(ds, relation=relation, seed=args.seed).to(args.device)
        ckpt = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{ds}_{relation}.pth"),
                          map_location=args.device)
        ca = ckpt["args"]
        model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.freeze()

        hg = KNNHypergraph(data.x, k=args.k).to(args.device)
        x_raw = data.x
        x_enh = hypergraph_propagation(x_raw, hg, args.layers, args.alpha_res)
        s_tta = get_scores(model, x_enh, data.edge_index)   # 分数来源固定 (TTA)

        # 隔离特征一致性通道: raw vs propagated 一致性特征
        c_raw = hyperedge_consistency(s_tta, hg, features=x_enh, consistency_features=x_raw)
        c_prop = hyperedge_consistency(s_tta, hg, features=x_enh, consistency_features=x_enh)
        y_test = data.y[data.test_mask].cpu()
        sep_raw = consistency_separability(c_raw[data.test_mask], y_test)
        sep_prop = consistency_separability(c_prop[data.test_mask], y_test)

        # 邻域特征离散度收缩 (锐化的直接证据)
        disp_anom_raw, disp_norm_raw = neighborhood_feature_dispersion(x_raw, hg, data.y, data.test_mask)
        disp_anom_prop, disp_norm_prop = neighborhood_feature_dispersion(x_enh, hg, data.y, data.test_mask)

        # 完整 StructCP FPR/TPR: raw 一致性特征 (论文默认) vs propagated
        cal_m, test_m = data.cal_mask, data.test_mask
        thr_raw, _ = structcp_threshold(s_tta[cal_m], data.y[cal_m], c_raw[cal_m],
                                        s_tta[test_m], c_raw[test_m], alpha=args.alpha,
                                        tau_quantile=args.tau, mode="hard",
                                        score_quantile=args.score_quantile)
        thr_prop, _ = structcp_threshold(s_tta[cal_m], data.y[cal_m], c_prop[cal_m],
                                         s_tta[test_m], c_prop[test_m], alpha=args.alpha,
                                         tau_quantile=args.tau, mode="hard",
                                         score_quantile=args.score_quantile)
        cov_raw = evaluate_coverage(s_tta[test_m], data.y[test_m], thr_raw)
        cov_prop = evaluate_coverage(s_tta[test_m], data.y[test_m], thr_prop)

        row = {
            "dataset": ds,
            "consistency_AUROC": {"raw_feat": round(sep_raw, 4), "prop_feat": round(sep_prop, 4),
                                  "delta": round(sep_prop - sep_raw, 4)},
            "neighborhood_disp": {
                "raw_anom": round(disp_anom_raw, 4), "raw_norm": round(disp_norm_raw, 4),
                "prop_anom": round(disp_anom_prop, 4), "prop_norm": round(disp_norm_prop, 4),
                "shrink_norm": round(disp_norm_raw - disp_norm_prop, 4),
            },
            "structcp_FPR": {"raw_feat": round(cov_raw["FPR"], 4), "prop_feat": round(cov_prop["FPR"], 4)},
            "structcp_TPR": {"raw_feat": round(cov_raw["TPR"], 4), "prop_feat": round(cov_prop["TPR"], 4)},
            "time_s": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(f"[sharpen:{ds}] cons_AUROC raw={sep_raw:.4f} prop={sep_prop:.4f} (Δ={sep_prop - sep_raw:+.4f}) "
              f"| disp_norm {disp_norm_raw:.4f}->{disp_norm_prop:.4f} (shrink {disp_norm_raw - disp_norm_prop:+.4f}) "
              f"| FPR raw={cov_raw['FPR']:.4f} prop={cov_prop['FPR']:.4f} "
              f"| TPR raw={cov_raw['TPR']:.4f} prop={cov_prop['TPR']:.4f}")

    out = os.path.join(ROOT, "outputs", "ablation_sharpen.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "per_dataset": rows}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
