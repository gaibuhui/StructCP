"""回填论文 Appendix: 大 α 失效的 ESS (有效样本量) 解释。

核心论点: 加权保形在大 α 区间失效 (α≥0.10 时 FPR 超限),
因为目标分位数水平 (1-α) 落在校准分布尾部, 此时一致性权重分布
越不均匀 -> 有效样本量 ESS = (Σw_i)² / Σw_i² 越小 -> 分位数估计越不稳定。

本脚本扫描 α ∈ {0.01, 0.02, 0.05, 0.10, 0.15, 0.20}, 对每个数据集报告:
  - w_ess  : 最终用于分位数估计的 (真实正常+伪正常) 权重有效样本量
  - fpr     : StructCP 在该 α 下的测试 FPR (理想应 ≤ α)
  - n_pseudo: 注入的伪正常样本数

结果落盘 ess_scan.json (每数据集完成即落盘 + 断点续跑)。
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
import torch.nn.functional as F

from data.loader import load_gad
from models.bwgnn import BWGNN
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation
from adapters.conformal import structcp_threshold

CKPT_DIR = os.path.join(_structcp_root(), "checkpoint")
RESULT_DIR = os.path.join(_structcp_root(), "outputs")
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial"]
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
SEEDS = [42, 123, 456, 789, 1011]  # 5-seed sweep for std reporting


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
def scan(name, data, seed):
    model = load_model(name, data)
    x, ei = data.x, data.edge_index
    with torch.no_grad():
        def _scores(t):
            return F.softmax(model(t, ei), dim=1)[:, 1]
        raw = _scores(x)
        hg = KNNHypergraph(x, k=10)
        feat = hypergraph_propagation(x, hg, num_layers=2, alpha_res=0.5)
        s_tta = _scores(feat)  # softmax 概率口径, 与 run_structcp_seeds.py 统一
    enh = s_tta
    cal_idx, test_idx = data.cal_mask, data.test_mask
    from adapters.hypergraph_tta import hyperedge_consistency
    c_full = KNNHypergraph(x, k=10)
    c_all = hyperedge_consistency(s_tta, c_full, x)  # 一致性基于标量分数
    c_cal, c_test = c_all[cal_idx], c_all[test_idx]
    cal_scores, cal_y = enh[cal_idx], data.y[cal_idx]
    test_scores, test_y = enh[test_idx], data.y[test_idx]

    rows = {}
    for alpha in ALPHAS:
        thr, info = structcp_threshold(
            cal_scores, cal_y, c_cal, test_scores, c_test,
            alpha=alpha, mode="hard",
            score_quantile=0.75 if name in ("Elliptic", "TFinance") else None,
        )
        pred = (test_scores > thr).long()
        fp = int(((pred == 1) & (test_y == 0)).sum())
        tn = int(((pred == 0) & (test_y == 0)).sum())
        fpr = fp / max(fp + tn, 1)
        rows[f"a{alpha:g}"] = {
            "ess": round(info["w_ess"], 2),
            "fpr": round(fpr, 4),
            "n_pseudo": info["n_pseudo_normal"],
            "w_min": round(info["w_min"], 4),
            "w_max": round(info["w_max"], 4),
        }
    return rows


def _aggregate(per_seed_rows):
    """per_seed_rows: list of dict keyed by 'a{alpha}' with fpr/ess.
    Returns aggregated dict: each alpha key -> {fpr_mean,fpr_std,ess_mean,ess_std,seed42}."""
    out = {}
    for a in [f"a{al:g}" for al in ALPHAS]:
        fprs = [r[a]["fpr"] for r in per_seed_rows]
        esss = [r[a]["ess"] for r in per_seed_rows]
        fm = float(np.mean(fprs)); fs = float(np.std(fprs))
        em = float(np.mean(esss)); es = float(np.std(esss))
        out[a] = {
            "fpr_mean": round(fm, 4),
            "fpr_std": round(fs, 4),
            "ess_mean": round(em, 1),
            "ess_std": round(es, 1),
            # seed-42 representative value (kept for table parity with 3-seed mean)
            "seed42": per_seed_rows[0][a]["fpr"] if 42 in SEEDS else round(fm, 4),
            "n_pseudo": per_seed_rows[0][a]["n_pseudo"],
        }
    return out


def main():
    import sys
    ds = sys.argv[1:] or DATASETS
    out = os.path.join(RESULT_DIR, "ess_scan.json")
    # ESS 扫描本身廉价(仅分位数估计, 无重训练), 每次运行全量重算并覆盖。
    # 现输出 3-seed 聚合 (SEEDS=[42,0,1]); 每个 alpha 含 fpr_mean/fpr_std/ess_mean/ess_std。
    summary = {}

    for name in ds:
        print(f"\n=== {name} (3-seed) ===", flush=True)
        per_seed = []
        for seed in SEEDS:
            try:
                if name == "TSocial":
                    data = load_gad(name, relation="homo", seed=seed,
                                    subsample=100_000, knn_backbone=10)
                else:
                    data = load_gad(name, relation="homo", seed=seed)
                r = scan(name, data, seed)
            except Exception as e:
                print(f"  [skip] {name} seed={seed}: {e}", flush=True)
                continue
            per_seed.append(r)
            print(f"  seed={seed}: " + ", ".join(
                f"{a}:FPR={v['fpr']:.3f}" for a, v in r.items()), flush=True)
        if not per_seed:
            continue
        agg = _aggregate(per_seed)
        summary[name] = agg
        # 同时保留单 seed=42 兼容结构 (仅 fpr/ess 旧字段)
        summary[name]["_seed42_compat"] = {
            a: {"ess": per_seed[0][a]["ess"], "fpr": per_seed[0][a]["fpr"],
                "n_pseudo": per_seed[0][a]["n_pseudo"]}
            for a in [f"a{al:g}" for al in ALPHAS]
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
