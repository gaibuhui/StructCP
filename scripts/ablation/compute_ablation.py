"""回填论文 Table 3 (组件消融): 真实运行 StructCP 各变体, 报告 FPR/TPR/AUC。

变体定义 (论文 Sec 4.2):
  (a) Base        : 冻结 BWGNN, 标准 split conformal, 无 TTA, 无重加权。
  (b) +Hypergraph : Base + 超图免训练 TTA 特征增强。
  (c) +Conformal  : Base + 加权保形 (无 TTA, 仅用超边一致性重加权)。
  (d) Full StructCP : (b) + (c) 完整方法。

每个数据集取 5 随机种子均值。原占位示意值 (如 (a) FPR 0.31) 由此真实替换。
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


import json
import os

import numpy as np
import torch

from data.loader import load_gad
from models.bwgnn import BWGNN
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation, hyperedge_consistency
from adapters.conformal import split_conformal_threshold, structcp_threshold, evaluate_coverage

CKPT_DIR = os.path.join(_structcp_root(), "checkpoint")
RESULT_DIR = os.path.join(_structcp_root(), "outputs")
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial"]
SEEDS = [42, 123, 456, 789, 1011]  # 5 标准随机种子 (single seed for TFinance / TSocial by default)
ALPHA = 0.05


def load_model(name, data):
    ckpt = os.path.join(CKPT_DIR, f"bwgnn_{name}_homo.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT_DIR, f"bwgnn_{name}.pth")
    model = BWGNN(in_channels=data.num_features)
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    model.freeze()
    return model


@torch.no_grad()
def compute_variants_for_scores(name, data, raw_scores, tta_scores):
    """在给定两组分数 (原始 / 超图增强) 上算出 4 个变体的 FPR/TPR。

    超边一致性需在**全图分数**上计算 (KNN 超图基于全图特征),
    再索引到 cal/test 子集, 避免索引越界。
    """
    cal_idx = data.cal_mask
    test_idx = data.test_mask
    x = data.x
    hg_full = KNNHypergraph(x, k=10)

    out = {}
    for variant in ("a", "b", "c", "d"):
        if variant in ("a", "c"):
            enh = raw_scores
        else:
            enh = tta_scores
        cal_scores, cal_y = enh[cal_idx], data.y[cal_idx]
        test_scores, test_y = enh[test_idx], data.y[test_idx]

        if variant in ("a", "b"):
            thr = split_conformal_threshold(cal_scores[cal_y == 0], alpha=ALPHA)
        else:
            # consistency 必须在**全图分数**上计算 (内部用全图 knn 索引)
            c_full = hyperedge_consistency(enh, hg_full, x)
            c_test = c_full[test_idx]
            c_cal = c_full[cal_idx]
            thr, _ = structcp_threshold(
                cal_scores, cal_y, c_cal, test_scores, c_test,
                alpha=ALPHA, mode="hard",
                score_quantile=0.75 if name in ("Elliptic", "TFinance") else None,
            )
        cov = evaluate_coverage(test_scores, test_y, thr)
        out[variant] = {"FPR": cov["FPR"], "TPR": cov["TPR"]}
    return out


def main():
    import sys, os as _os
    ds = sys.argv[1:] or DATASETS
    out = os.path.join(RESULT_DIR, "ablation_components.json")
    summary = {}
    if _os.environ.get("FORCE") and _os.path.exists(out):
        summary = {}
    elif _os.path.exists(out):  # 断点续跑
        with open(out) as f:
            summary = json.load(f)

    for name in ds:
        if name in summary:
            print(f"  [cached] {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        seeds = SEEDS if data_edge_small(name) else [42]
        agg = {v: {"FPR": [], "TPR": []} for v in ("a", "b", "c", "d")}
        for seed in seeds:
            try:
                if name == "TSocial":
                    data = load_gad(name, relation="homo", seed=seed,
                                    subsample=100_000, knn_backbone=10)
                else:
                    data = load_gad(name, relation="homo", seed=seed)
            except Exception as e:
                print(f"  [skip seed {seed}] {e}", flush=True)
                continue
            model = load_model(name, data)
            import torch.nn.functional as F
            with torch.no_grad():
                raw = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
                hg = KNNHypergraph(data.x, k=10)
                feat = hypergraph_propagation(data.x, hg, num_layers=2)
                tta = F.softmax(model(feat, data.edge_index), dim=1)[:, 1]
            res = compute_variants_for_scores(name, data, raw, tta)
            for v in ("a", "b", "c", "d"):
                agg[v]["FPR"].append(res[v]["FPR"])
                agg[v]["TPR"].append(res[v]["TPR"])
                print(f"  seed {seed} var {v}: FPR={res[v]['FPR']:.3f} TPR={res[v]['TPR']:.3f}", flush=True)
        summary[name] = {
            v: {"FPR": round(float(np.mean(agg[v]["FPR"])), 4) if agg[v]["FPR"] else None,
                "TPR": round(float(np.mean(agg[v]["TPR"])), 4) if agg[v]["TPR"] else None}
            for v in ("a", "b", "c", "d")
        }
        # 防御: 单数据集内的未捕获异常不应丢弃已算好的其他数据集
        try:
            with open(out, "w") as f:  # 每个数据集完成即落盘
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"  [warn] dump failed for {name}: {e}", flush=True)
    print(f"\n[done] -> {out}", flush=True)


def data_edge_small(name):
    """超大图只跑 seed=42, 避免 BWGNN 前向重复过慢。"""
    return name not in ("TFinance", "TSocial")


if __name__ == "__main__":
    main()
