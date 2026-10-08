"""扫描 score_quantile 对 StructCP 的影响, 验证联合筛选在社群型异常上的修复作用。

对比纯一致性 (None) vs 分数-一致性联合 (0.5/0.6/0.7/0.75/0.8/0.9) 在
Amazon / Elliptic / T-Finance 上的 FPR / TPR / F1。
"""

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
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, _structcp_root())
from data.loader import load_gad
from models.bwgnn import BWGNN
from adapters.hypergraph_tta import (
    KNNHypergraph, hypergraph_propagation, hyperedge_consistency,
)
from adapters.conformal import structcp_threshold, split_conformal_threshold, evaluate_coverage


def run(dataset, sq, seed=42, device="cuda"):
    relation = "homo"
    subsample = 100000 if dataset == "TSocial" else None
    knn = 10 if dataset == "TSocial" else None
    data = load_gad(dataset, relation=relation, seed=seed,
                    subsample=subsample, knn_backbone=knn).to(device)
    ckpt = torch.load(os.path.join(_structcp_root(),
                                   "checkpoint", f"bwgnn_{dataset}_{relation}.pth"),
                      map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    with torch.no_grad():
        s = F.softmax(model(x_enh, data.edge_index), dim=1)[:, 1]
        cons = hyperedge_consistency(s, hg, features=x_enh)
    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal = data.y[cal_m]
    y_test = data.y[test_m].cpu().numpy()
    thr, info = structcp_threshold(s[cal_m], y_cal, cons[cal_m], s[test_m], cons[test_m],
                                alpha=0.05, tau_quantile=0.5, mode="hard",
                                score_quantile=sq)
    cov = evaluate_coverage(s[test_m], data.y[test_m], thr)
    return cov, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,Elliptic,TFinance")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    qs = [None, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    for ds in args.datasets.split(","):
        print(f"\n===== {ds} =====")
        print(f"{'sq':>6}{'n_pseudo':>10}{'FPR':>9}{'TPR':>9}{'F1':>9}")
        for sq in qs:
            cov, info = run(ds, sq, device=args.device)
            print(f"{str(sq):>6}{info['n_pseudo_normal']:>10}"
                  f"{cov['FPR']:>9.4f}{cov['TPR']:>9.4f}{cov['F1']:>9.4f}")


if __name__ == "__main__":
    main()
