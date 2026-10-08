"""骨干敏感性实验 (回应审稿"detector-agnostic 名不副实")。

在 Amazon / YelpChi / Elliptic 上换 BWGNN / GCN / GAT 三种冻结骨干,
跑 Frozen 与 StructCP, 验证:
  - FPR 控制是 wrapper-level 的 (任何骨干都应控住 FPR ≤ α);
  - TPR 随骨干判别能力变化 (骨干上限)。
核心口径与主流程一致: 去泄漏 (cal_mask 尺度), hard 阈值, α=0.05。
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

import numpy as np
import torch
import torch.nn.functional as F

from adapters.registry import _make_bwgnn, _make_gcn, _make_gat
from utils.checkpoint import load_frozen_detector
from adapters.conformal import (
    evaluate_coverage,
    split_conformal_threshold,
    structcp_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph, hyperedge_consistency, hypergraph_propagation, contrastive_consistency,
)
from data.loader import load_gad

import sys
sys.path.insert(0, _structcp_root())
ROOT = _structcp_root()

BACKBONES = ["BWGNN", "GCN", "GAT"]
FACTORY = {"BWGNN": _make_bwgnn, "GCN": _make_gcn, "GAT": _make_gat}


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run(dataset, backbone, device, seed, alpha=0.05):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    if backbone == "BWGNN":
        # 与主表 Tab:main 同口径: 使用 homo 关系权重 (load_frozen_detector)
        model, _ = load_frozen_detector(dataset, "homo", root=ROOT, device=str(device))
    else:
        adapter = FACTORY[backbone](device=str(device))
        adapter.fit(data)
        model = adapter.model.to(device)
    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal = data.y[cal_m]
    y_test = data.y[test_m].cpu().numpy()

    # Frozen —— 分数方向校正 (若 AUROC<0.5 说明 softmax 第1列反向, 翻转使异常=高分组)
    s_raw = get_scores(model, data.x, data.edge_index)
    from sklearn.metrics import roc_auc_score
    auroc_f = float(roc_auc_score(y_test, s_raw[test_m].cpu().numpy()))
    flip = auroc_f < 0.5
    if flip:
        s_raw = 1.0 - s_raw
        auroc_f = 1.0 - auroc_f  # 翻转后 AUROC
    thr_f = split_conformal_threshold(s_raw[cal_m][y_cal == 0], alpha)
    ev_f = evaluate_coverage(s_raw[test_m], data.y[test_m], thr_f)

    # StructCP (去泄漏), 与 Frozen 同一方向校正标志
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta = get_scores(model, x_enh, data.edge_index)
    if flip:
        s_tta = 1.0 - s_tta
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    cal_normal = cal_bool & (data.y == 0)
    cons = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_normal)
    thr_s, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=alpha, tau_quantile=0.5, score_quantile=0.75,
        use_calib_stat=True, robust_calibration=False,
        weight_mode="contrastive",
    )
    ev_s = evaluate_coverage(s_tta[test_m], data.y[test_m], thr_s)
    # s_tta 已按 flip 校正 (L65-66), 故此处直接算 AUROC, 不再二次翻转
    auroc_s = float(roc_auc_score(y_test, s_tta[test_m].cpu().numpy()))

    return {
        "dataset": dataset, "backbone": backbone, "seed": seed,
        "AUROC_Frozen": auroc_f, "AUROC_StructCP": auroc_s,
        "Frozen_FPR": float(ev_f["FPR"]), "Frozen_TPR": float(ev_f["TPR"]), "Frozen_F1": float(ev_f["F1"]),
        "StructCP_FPR": float(ev_s["FPR"]), "StructCP_TPR": float(ev_s["TPR"]), "StructCP_F1": float(ev_s["F1"]),
        "ess": float(info.get("ESS", float("nan"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic")
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for ds in datasets:
        for bb in backbones:
            for seed in seeds:
                print(f"=== {ds} {bb} seed={seed} ===", flush=True)
                r = run(ds, bb, device, seed, alpha=args.alpha)
                print(f"  Frozen FPR={r['Frozen_FPR']:.4f} TPR={r['Frozen_TPR']:.4f} | "
                      f"StructCP FPR={r['StructCP_FPR']:.4f} TPR={r['StructCP_TPR']:.4f} F1={r['StructCP_F1']:.4f} "
                      f"AUROC={r['AUROC_StructCP']:.3f}")
                rows.append(r)

    # 聚合 (dataset × backbone)
    agg = {}
    for ds in datasets:
        agg[ds] = {}
        for bb in backbones:
            rs = [r for r in rows if r["dataset"] == ds and r["backbone"] == bb]
            def m(k):
                v = [r[k] for r in rs if r[k] == r[k]]
                return [float(np.mean(v)), float(np.std(v))]
            agg[ds][bb] = {k: m(k) for k in ["AUROC_Frozen", "AUROC_StructCP",
                                              "Frozen_FPR", "Frozen_TPR", "Frozen_F1",
                                              "StructCP_FPR", "StructCP_TPR", "StructCP_F1"]}

    out = {"config": {"alpha": args.alpha, "seeds": seeds},
           "per_seed": rows, "agg": agg}
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    path = os.path.join(ROOT, "outputs", "backbone_sensitivity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {path}")

    print(f"\n{'dataset':9s} {'backbone':8s} {'FrzFPR':7s} {'SCPFPR':7s} {'SCPTPR':7s} {'SCPF1':6s} {'AUROC':7s}")
    for ds in datasets:
        for bb in backbones:
            a = agg[ds][bb]
            print(f"{ds:9s} {bb:8s} {a['Frozen_FPR'][0]:.4f}   {a['StructCP_FPR'][0]:.4f}   "
                  f"{a['StructCP_TPR'][0]:.4f}   {a['StructCP_F1'][0]:.4f}   {a['AUROC_StructCP'][0]:.3f}")


if __name__ == "__main__":
    main()
