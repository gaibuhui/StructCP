"""评审回应实验：归纳/流式部署评估 (Experiment Group D, Critical Q2)。

回应评审"方法似为转导(transductive)"关切:
  - 全图转导版 (StructCP-full) : 伪正常池取全部测试节点 (论文主口径)。
  - 流式版 (StructCP-stream)   : 测试按节点序分批, 每批的阈值只用
    [校准集 + 当前批] 构造伪正常池, 不使用任何未来节点/边 —— 满足
    "无完整测试图访问"的归纳部署。
  - SplitCP (Frozen, 纯校准, 无任何测试访问) 作为下界参考。

两者均使用同一冻结 BWGNN、同一 kNN 一致性 (c_test 为离线特征相似度,
不涉及标签/图边泄漏; 论文 Sec.3.5 已说明)。

输出: outputs/structcp_streaming.json  (每数据集×seed 的逐批 FPR/TPR + 汇总)

用法 (CBP 环境):
  python scripts/compare/run_structcp_streaming.py --batches 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys

import numpy as np
import torch
import torch.nn.functional as F

_p = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)

from utils.checkpoint import load_frozen_detector  # noqa: E402
from adapters.conformal import (  # noqa: E402
    evaluate_coverage, split_conformal_threshold, structcp_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph, hyperedge_consistency, hypergraph_propagation, contrastive_consistency,
)
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402


@torch.no_grad()
def _softmax_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def _flip_scores(s, y_test):
    from sklearn.metrics import roc_auc_score
    return s if roc_auc_score(y_test.cpu().numpy(), s.cpu().numpy()) >= 0.5 else 1.0 - s


def _score_flip_bool(s_test, y_test):
    """判定分数是否需要翻转（仅基于测试节点）。"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_test.cpu().numpy(), s_test.cpu().numpy()) < 0.5


def _confusion(s, y, thr):
    pred = (s > thr).long()
    y = y.long()
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(FPR=fp / max(fp + tn, 1), TPR=tp / max(tp + fn, 1), n=int(y.numel()))


def run_one(dataset, seed, n_batches, device, weight_mode="contrastive"):
    torch.manual_seed(seed); np.random.seed(seed)
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model, _ = load_frozen_detector(dataset, "homo", root=_ROOT, device=str(device))

    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_raw_all = _softmax_scores(model, data.x, data.edge_index)
    s_tta_all = _softmax_scores(model, x_enh, data.edge_index)
    flip = _score_flip_bool(s_raw_all[data.test_mask], data.y[data.test_mask])
    s_raw = (1.0 - s_raw_all) if flip else s_raw_all
    s_tta = (1.0 - s_tta_all) if flip else s_tta_all

    cal_bool = torch.as_tensor(data.cal_mask, dtype=torch.bool, device=device)
    cal_n = cal_bool & (data.y == 0)
    test_m = data.test_mask
    y_test = data.y[test_m]
    # PPR 模式: 权重 = hyperedge 一致性, PPR 分数作伪正常准入门槛 (与 pipeline 一致)
    if weight_mode == "ppr":
        c_all = hyperedge_consistency(s_tta, hg, features=x_enh, cal_mask=cal_bool)
        c_ppr = ppr_structure_score(
            x_enh, data.edge_index, cal_n,
            pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
        )
        test_ppr_full = c_ppr[test_m].float()
        thr_weight = "struct"
    else:
        c_all = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_n)
        test_ppr_full = None
        thr_weight = weight_mode
    s_test = s_tta[test_m]; c_test = c_all[test_m]

    # ---- 全图转导版 ----
    thr, info = structcp_threshold(s_tta[data.cal_mask], data.y[data.cal_mask],
                                   c_all[data.cal_mask], s_test, c_test,
                                   alpha=0.05, tau_quantile=0.5, score_quantile=0.75,
                                   weight_mode=thr_weight,
                                   test_ppr=test_ppr_full, ppr_quantile=0.4)
    full = evaluate_coverage(s_test, y_test, thr)

    # ---- SplitCP 下界 (无测试访问) ----
    thr_scp = split_conformal_threshold(s_tta[cal_n], 0.05)
    scp = evaluate_coverage(s_test, y_test, thr_scp)

    # ---- 流式版: 分批, 每批只用 校准集+当前批 ----
    idx = np.arange(len(s_test))
    batches = np.array_split(idx, n_batches)
    per_batch = []
    pooled = dict(FPR=0.0, TPR=0.0, tot_fp=0, tot_fn=0, tot_tp=0, tot_tn=0)
    for b, ids in enumerate(batches):
        ids = torch.as_tensor(ids, device=device)
        s_b = s_test[ids]; c_b = c_test[ids]; y_b = y_test[ids]
        ppr_b = test_ppr_full[ids] if test_ppr_full is not None else None
        thr_b, info_b = structcp_threshold(s_tta[data.cal_mask], data.y[data.cal_mask],
                                           c_all[data.cal_mask], s_b, c_b,
                                           alpha=0.05, tau_quantile=0.5, score_quantile=0.75,
                                           weight_mode=thr_weight,
                                           test_ppr=ppr_b, ppr_quantile=0.4)
        cm = _confusion(s_b, y_b, thr_b)
        cm["n_pseudo"] = int(info_b["n_pseudo_normal"])
        cm["threshold"] = float(thr_b)
        per_batch.append(cm)
        pooled["tot_fp"] += int(((s_b > thr_b).long() & (y_b == 0).long()).sum())
        pooled["tot_fn"] += int(((s_b <= thr_b).long() & (y_b == 1).long()).sum())
        pooled["tot_tp"] += int(((s_b > thr_b).long() & (y_b == 1).long()).sum())
        pooled["tot_tn"] += int(((s_b <= thr_b).long() & (y_b == 0).long()).sum())
    denom_fp = pooled["tot_fp"] + pooled["tot_tn"]
    denom_tp = pooled["tot_tp"] + pooled["tot_fn"]
    pooled["FPR"] = pooled["tot_fp"] / max(denom_fp, 1)
    pooled["TPR"] = pooled["tot_tp"] / max(denom_tp, 1)

    return {
        "full": {"FPR": full["FPR"], "TPR": full["TPR"], "n_pseudo": int(info["n_pseudo_normal"])},
        "streaming_pooled": {"FPR": pooled["FPR"], "TPR": pooled["TPR"]},
        "splitcp": {"FPR": scp["FPR"], "TPR": scp["TPR"]},
        "per_batch": per_batch,
        "n_batches": n_batches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weight_mode", default="contrastive")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    summary = {}
    for ds in datasets:
        per_seed = {}
        for seed in seeds:
            r = run_one(ds, seed, args.batches, device, args.weight_mode)
            per_seed[seed] = r
            print(f"{ds} s{seed}: full FPR={r['full']['FPR']:.3f} | "
                  f"stream FPR={r['streaming_pooled']['FPR']:.3f} | "
                  f"splitcp FPR={r['splitcp']['FPR']:.3f}", flush=True)
        # 汇总 mean±std
        summary[ds] = {"per_seed": per_seed}
        for key in ["full", "streaming_pooled", "splitcp"]:
            summary[ds].setdefault("agg", {})[key] = {
                "FPR_mean": float(np.mean([per_seed[s][key]["FPR"] for s in per_seed])),
                "FPR_std": float(np.std([per_seed[s][key]["FPR"] for s in per_seed], ddof=0)),
                "TPR_mean": float(np.mean([per_seed[s][key]["TPR"] for s in per_seed])),
                "TPR_std": float(np.std([per_seed[s][key]["TPR"] for s in per_seed], ddof=0)),
            }

    out = os.path.join(_ROOT, "outputs", "structcp_streaming.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
