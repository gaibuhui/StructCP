"""A+B 超图能力提升消融: 高通传播 (beta) x 超边度门控 (deg_gate)。

验证目标: 在 StructCP 框架下引入无参数高通/度门控后, 社区型异常
(Elliptic/T-Finance) 的 TPR 是否回升, 同时 FPR 是否仍 <= alpha (0.05)。

每个 (dataset, seed, beta, deg_gate) 组合跑完整 StructCP 流程, 记录:
  FPR / TPR / F1 / FPR_novel_normality / AUROC / AUPRC

断点续跑: 输出 JSON 已存在则跳过 (key = dataset_seed_beta_gate)。
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
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
SEEDS = [42, 123, 456, 789, 1011]
BETAS = [0.0, 0.2, 0.4]
GATES = [False, True]
ALPHA = 0.05
K = 10
LAYERS = 2
ALPHA_RES = 0.5
TAU = 0.5
SCORE_QUANTILE = 0.75  # Elliptic/T-Finance 修复过度校正 (run_structcp 默认)


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def run_one(dataset, seed, beta, gate, device):
    relation = "homo"
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
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

    # HG-TTA (标准 split conformal)
    thr2 = split_conformal_threshold(s_tta[cal_m][y_cal == 0], ALPHA)
    hgtta = evaluate_coverage(s_tta[test_m], data.y[test_m], thr2)

    # StructCP (full)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh)
    thr3, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=ALPHA, tau_quantile=TAU, mode="hard",
        score_quantile=SCORE_QUANTILE,
    )
    structcp = evaluate_coverage(s_tta[test_m], data.y[test_m], thr3)

    nov = data.novel_mask & test_m
    if int(nov.sum()) > 0:
        structcp["FPR_novel_normality"] = float((s_tta[nov] > thr3).float().mean())
        hgtta["FPR_novel_normality"] = float((s_tta[nov] > thr2).float().mean())

    return {
        "HG-TTA": hgtta,
        "StructCP": structcp,
        "AUROC": roc_auc_score(y_test, s_tta[test_m].cpu().numpy()),
        "AUPRC": average_precision_score(y_test, s_tta[test_m].cpu().numpy()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="限定单数据集 (默认全部)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_path = os.path.join(ROOT, "outputs", "highpass_ablation.json")
    if os.path.exists(out_path):
        cache = json.load(open(out_path))
    else:
        cache = {}

    dsets = [args.dataset] if args.dataset else DATASETS
    for ds in dsets:
        for seed in SEEDS:
            for beta in BETAS:
                for gate in GATES:
                    key = f"{ds}_{seed}_{beta}_{gate}"
                    if key in cache:
                        print(f"[skip] {key}")
                        continue
                    t0 = time.time()
                    try:
                        r = run_one(ds, seed, beta, gate, device)
                    except Exception as e:
                        print(f"[err] {key}: {e}")
                        continue
                    r["_time_s"] = round(time.time() - t0, 2)
                    cache[key] = r
                    json.dump(cache, open(out_path, "w"), indent=2)
                    print(f"[ok] {key}  StructCP FPR={r['StructCP']['FPR']:.4f} "
                          f"TPR={r['StructCP']['TPR']:.4f} "
                          f"NovelFPR={r['StructCP'].get('FPR_novel_normality', float('nan')):.4f} "
                          f"({r['_time_s']}s)")

    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
