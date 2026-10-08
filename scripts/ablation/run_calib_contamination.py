"""校准集污染鲁棒性实验 (质疑 #1)。

动机: StructCP 假设校准集 CC 仅含源域正常 (Sec.3.1)。实际部署中 CC 正常/异常混合,
若 CC 混入少量异常且被误当正常, 阈值会被污染, FPR 控制可能失效。本脚本压力测试:
  - 注入 p∈{1%,5%,10%} 校准异常 (骨干漏分的"新异异常", 分数取校准正常分布低位)
  - 对比 Frozen (split-conformal, 完全信任 cal_y) / StructCP (一致性重加权) /
    StructCP-robust (校准集统计量 + 一致性自清洗地板) 的 FPR / TPR 退化。
结论预期: Frozen 的 FPR 控制显著退化, StructCP 因一致性重加权天然鲁棒,
robust_calibration 通过一致性地板剔除进一步收紧。
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
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.nn.functional import softmax

from adapters.conformal import (
    structcp_threshold,
    split_conformal_threshold,
)
from adapters.hypergraph_tta import (
    KNNHypergraph,
    hyperedge_consistency,
)
from adapters.pipeline import get_scores
from data.loader import load_gad
from models.bwgnn import BWGNN

MAIN_DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
CONTAM = [0.0, 0.01, 0.05, 0.10]


def load_and_score(ds, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_gad(ds, relation="homo", seed=seed).to(device)
    ckpt_path = os.path.join(_structcp_root(), "checkpoint", f"bwgnn_{ds}_homo.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    cal_m, test_m = data.cal_mask, data.test_mask
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta = get_scores(model, x_enh, data.edge_index).to(device)
    # 一致性尺度项仅从校准集 C 估计 (质疑#2 去泄漏)
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh, cal_mask=cal_bool)
    return data, s_tta, cons, cal_m, test_m


def inject_contamination(s_cal, y_cal, p, rng):
    """把 p 比例的校准节点设为'被误当正常的异常': 其分数取校准正常分布低位 (骨干漏分的新异异常)。"""
    s = s_cal.clone()
    y = y_cal.clone()
    n = s.numel()
    k = int(round(p * n))
    if k <= 0:
        return s, y
    normal_idx = torch.where(y == 0)[0]
    if normal_idx.numel() < k:
        k = normal_idx.numel()
    pick = normal_idx[rng.choice(normal_idx.numel(), size=k, replace=False)]
    norm_scores = s[y == 0]
    low_vals = torch.quantile(norm_scores, torch.linspace(0.0, 0.15, max(k, 1)))
    s[pick] = low_vals[:k].to(s.dtype)
    return s, y  # 标签仍记为正常 (部署者不知其为异常)


def fpr_tpr(s_test, y_test, thr):
    y = y_test.cpu().numpy()
    pred = (s_test.cpu().numpy() > float(thr)).astype(int)
    fpr = pred[y == 0].mean() if (y == 0).any() else float("nan")
    tpr = pred[y == 1].mean() if (y == 1).any() else float("nan")
    return float(fpr), float(tpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    out: Dict[str, Dict] = {}
    for ds in MAIN_DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        data, s_tta, cons, cal_m, test_m = load_and_score(ds, device, args.seed)
        y_cal = data.y[cal_m].to(device)
        y_test = data.y[test_m].to(device)
        s_test, cons_cal, cons_test = s_tta[test_m], cons[cal_m], cons[test_m]
        rec = {}
        for p in CONTAM:
            s_cal_inj, y_cal_inj = inject_contamination(s_tta[cal_m].to(device), y_cal, p, rng)
            # Frozen: 完全信任 cal_y
            thr_f = split_conformal_threshold(s_cal_inj[y_cal_inj == 0], args.alpha)
            fpr_f, tpr_f = fpr_tpr(s_test, y_test, thr_f)
            # StructCP 标准 (一致性重加权 + 校准集统计量)
            thr_s, _ = structcp_threshold(s_cal_inj, y_cal_inj, cons_cal, s_test, cons_test,
                                          alpha=args.alpha, tau_quantile=0.5,
                                          score_quantile=0.75, use_calib_stat=True,
                                          robust_calibration=False)
            fpr_s, tpr_s = fpr_tpr(s_test, y_test, thr_s)
            # StructCP-robust (一致性自清洗地板)
            thr_r, _ = structcp_threshold(s_cal_inj, y_cal_inj, cons_cal, s_test, cons_test,
                                          alpha=args.alpha, tau_quantile=0.5,
                                          score_quantile=0.75, use_calib_stat=True,
                                          robust_calibration=True)
            fpr_r, tpr_r = fpr_tpr(s_test, y_test, thr_r)
            rec[f"p{p:.2f}"] = {
                "frozen_fpr": fpr_f, "frozen_tpr": tpr_f,
                "struct_fpr": fpr_s, "struct_tpr": tpr_s,
                "robust_fpr": fpr_r, "robust_tpr": tpr_r,
            }
            print(f"  p={p:.2f}  Frozen FPR={fpr_f:.3f}/TPR={tpr_f:.3f}  "
                  f"StructCP FPR={fpr_s:.3f}/TPR={tpr_s:.3f}  "
                  f"Robust FPR={fpr_r:.3f}/TPR={tpr_r:.3f}", flush=True)
        out[ds] = rec
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/calib_contamination.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] -> outputs/calib_contamination.json", flush=True)


if __name__ == "__main__":
    main()
