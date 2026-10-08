"""质疑#3 实验: YelpChi 上换不同骨干 (BWGNN/GCN/GAT) 验证 StructCP 的 TPR 崩溃是否为骨干上限。

审稿人质疑: "既然 BWGNN 在 YelpChi AUROC 仅 0.66, 为何不选更强骨干 (如 GAT AUROC 0.705)?
一个在所有正样本上几乎全错的方法, 仅因 FPR 低就声称成功没有意义。"

本脚本用 GADBench 表 1 (tab:auc) 里已有真实 checkpoint 的三种骨干在 YelpChi 上跑
Frozen 与 StructCP, 证明:
  - 即便换到 AUROC 最高的 GAT (0.705), novel-normality 协议下 StructCP 的 TPR 依然
    极低 (因为 0.705 仍接近随机, novel-normal 与 anomaly 分数模式高度重叠);
  - 低 TPR 是骨干判别上限 + 严格 FPR 约束的统计决策困境, 不是 StructCP 权重层缺陷。
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
import time

import numpy as np
import torch
import torch.nn.functional as F

from adapters.registry import _make_bwgnn, _make_gcn, _make_gat
from adapters.conformal import (
    evaluate_coverage,
    split_conformal_threshold,
    structcp_threshold,
)
from adapters.hypergraph_tta import KNNHypergraph, hyperedge_consistency, hypergraph_propagation
from data.loader import load_gad
from sklearn.metrics import roc_auc_score  # 原 `import run_structcp as R` 的 roc_auc_score

MAIN_BACKBONES = ["BWGNN", "GCN", "GAT"]


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_backbone(backbone, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_gad("YelpChi", relation="homo", seed=seed).to(device)
    factory = {"BWGNN": _make_bwgnn, "GCN": _make_gcn, "GAT": _make_gat}[backbone]
    adapter = factory(device=str(device))
    adapter.fit(data)
    model = adapter.model.to(device)
    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal = data.y[cal_m]
    y_test = data.y[test_m].cpu().numpy()

    # Frozen
    s_raw = get_scores(model, data.x, data.edge_index)
    thr_f = split_conformal_threshold(s_raw[cal_m][y_cal == 0], 0.05)
    ev_f = evaluate_coverage(s_raw[test_m], data.y[test_m], thr_f)
    auroc_f = roc_auc_score(y_test, s_raw[test_m].cpu().numpy())

    # StructCP (去泄漏: 一致性尺度项仅从校准集估计)
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta = get_scores(model, x_enh, data.edge_index)
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh, cal_mask=cal_bool)
    thr_s, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=0.05, tau_quantile=0.5, score_quantile=0.75,
        use_calib_stat=True, robust_calibration=False,
    )
    ev_s = evaluate_coverage(s_tta[test_m], data.y[test_m], thr_s)
    auroc_s = roc_auc_score(y_test, s_tta[test_m].cpu().numpy())
    return {
        "backbone": backbone, "seed": seed,
        "AUROC": float(auroc_s),
        "Frozen_FPR": float(ev_f["FPR"]), "Frozen_TPR": float(ev_f["TPR"]), "Frozen_F1": float(ev_f["F1"]),
        "StructCP_FPR": float(ev_s["FPR"]), "StructCP_TPR": float(ev_s["TPR"]), "StructCP_F1": float(ev_s["F1"]),
        "ess": float(info.get("ESS", float("nan"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,456", help="逗号分隔")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]

    all_rows = []
    for backbone in MAIN_BACKBONES:
        for seed in seeds:
            print(f"=== {backbone} seed={seed} ===", flush=True)
            all_rows.append(run_backbone(backbone, device, seed))

    # 聚合 mean±std
    agg = {}
    for backbone in MAIN_BACKBONES:
        rows = [r for r in all_rows if r["backbone"] == backbone]
        def m(k):
            v = [r[k] for r in rows if r[k] == r[k]]
            return [float(np.mean(v)), float(np.std(v))]
        agg[backbone] = {k: m(k) for k in ["AUROC", "Frozen_FPR", "Frozen_TPR", "Frozen_F1",
                                           "StructCP_FPR", "StructCP_TPR", "StructCP_F1"]}

    out = {"config": {"dataset": "YelpChi", "alpha": 0.05, "seeds": seeds},
           "per_seed": all_rows, "agg": agg}
    os.makedirs(os.path.join(_structcp_root(), "outputs"), exist_ok=True)
    path = os.path.join(_structcp_root(), "outputs", "backbone_yelpchi.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {path}")
    print(f"{'backbone':8s} {'AUROC':8s} {'Frz FPR':9s} {'Frz TPR':9s} {'SCP FPR':9s} {'SCP TPR':9s} {'SCP F1':9s}")
    for bb in MAIN_BACKBONES:
        a = agg[bb]
        print(f"{bb:8s} {a['AUROC'][0]:.3f}      {a['Frozen_FPR'][0]:.3f}     {a['Frozen_TPR'][0]:.3f}      "
              f"{a['StructCP_FPR'][0]:.3f}     {a['StructCP_TPR'][0]:.3f}     {a['StructCP_F1'][0]:.3f}")


if __name__ == "__main__":
    main()
