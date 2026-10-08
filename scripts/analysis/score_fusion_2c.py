# -*- coding: utf-8 -*-
"""
2C: 分数融合实验 —— s_fused = s_BWGNN + lambda * (1 - c_struct)。

目标: 用结构一致性信号 (c_cons 超边一致性 / c_ppr PPR 可达性) 增强弱骨干图
(YelpChi) 的判别力, 尝试突破 BWGNN 分数下的 Oracle TPR 天花板 (0.238)。

对每个数据集 (seed 42): 计算 s_sp/c_cons/c_ppr 一次, 然后对每个 lambda:
  - fused 分数
  - AUROC (手写 rank AUC)
  - Oracle TPR@FPR=0.05 (泄漏诊断, 用测试正常 95 分位)
  - StructCP 单阈值 FPR/TPR (structcp_threshold, 分数换成 fused; c_cons/PPR 不变)

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/score_fusion_2c.py
"""
import io, json, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import (KNNHypergraph, hypergraph_propagation,  # noqa: E402
                                     hyperedge_consistency)
from adapters.conformal import structcp_threshold  # noqa: E402
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHA = 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def rank_auc(scores, labels):
    """手写 rank AUC (无 sklearn 依赖)。scores 高 = 异常。"""
    s = scores.numpy() if torch.is_tensor(scores) else np.asarray(scores)
    y = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    n1 = int(y.sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    # 用 sort 算 Mann-Whitney
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理平局: 平均 rank
    s_sorted = s[order]
    # 简化: 用 scipy 不存在时的近似(带平局修正略过, 分数连续近似无平局)
    r1 = ranks[y == 1].sum()
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def run(ds, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    data = load_gad(ds, relation="homo", seed=seed).to(dev)
    model, _ = load_frozen_detector(ds, "homo", root=_ROOT, device="cpu")
    hg = KNNHypergraph(data.x, k=K).to(dev)

    x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
    from adapters.pipeline import get_scores
    s_sp = get_scores(model, x_enh, data.edge_index)   # 异常分数, 高=异常

    cal_bool = data.cal_mask.bool()
    test_bool = data.test_mask.bool()
    cal_normal = cal_bool & (data.y == 0)
    c_cons = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)
    c_ppr = ppr_structure_score(
        x_enh, data.edge_index, cal_normal,
        pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
    ).float()

    y_cal = data.y[cal_bool]
    y_tf = data.y[test_bool]
    s_cal0 = s_sp[cal_bool]; s_test0 = s_sp[test_bool]
    c_cal0 = c_cons[cal_bool]; c_test0 = c_cons[test_bool]
    ppr_cal = c_ppr[cal_bool]; ppr_test = c_ppr[test_bool]

    # 结构信号: 1 - c (低 c = 异常)
    g_cal = (1.0 - c_cal0)
    g_test = (1.0 - c_test0)
    g_ppr_cal = (1.0 - ppr_cal); g_ppr_test = (1.0 - ppr_test)

    rows = []
    for lam in LAMBDAS:
        # 候选融合: 只用 c_cons
        s_cal = s_cal0 + lam * g_cal
        s_test = s_test0 + lam * g_test
        auc = rank_auc(s_test, y_tf)
        # Oracle (泄漏诊断)
        t_or = torch.quantile(s_test[y_tf == 0], 1 - ALPHA)
        orac_tpr = float((s_test[y_tf == 1] > t_or).float().mean())
        # StructCP 单阈值 (分数换成 fused, 一致性权重/PPR gate 不变)
        try:
            thr, info = structcp_threshold(
                s_cal, y_cal, c_cal0, s_test, c_test0,
                alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
                use_calib_stat=True, weight_mode="struct",
                test_ppr=ppr_test, ppr_quantile=0.4,
            )
            fpr = float((s_test[y_tf == 0] > thr).float().mean())
            tpr = float((s_test[y_tf == 1] > thr).float().mean())
        except Exception as e:
            fpr = tpr = float("nan"); print(f"  {ds} lam={lam} structcp ERR {e}")
        rows.append({"lambda": lam, "AUROC": auc, "Oracle_TPR": orac_tpr,
                     "FPR": fpr, "TPR": tpr, "FPR_le_alpha": bool(fpr <= ALPHA + 1e-9)})
        print(f"  {ds:<9} lam={lam:<4} AUROC={auc:.4f} OracleTPR={orac_tpr:.4f} "
              f"FPR={fpr:.4f} TPR={tpr:.4f}")
    return rows


def main():
    results = {}
    for ds in DATASETS:
        print(f"=== {ds} (seed 42) ===")
        try:
            results[ds] = run(ds)
        except Exception as e:
            print(f"{ds} ERROR: {e}")
            results[ds] = {"error": str(e)}
    out = os.path.join(_ROOT, "outputs", "score_fusion_2c.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps({"alpha": ALPHA, "lambdas": LAMBDAS, "datasets": results},
                   indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
