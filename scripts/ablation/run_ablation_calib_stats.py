"""Appendix 消融: 测试批次统计量依赖代价 (Eq.4/5 的 std(c_T)/med(c_T)).

动机 (review-driven, P1):
    论文 L203 已诚实讨论 Eq.4/5 依赖测试批次统计量 (std(c_T), med(c_T), d̄_s, d̄_cos),
    并给部署准则 |T|>=5000。但审稿人可能追问: 既然不主张分布无关保证, 该统计依赖的
    "理论正当性" 是什么? 本脚本给出一个可防御的实证回答 —— 量化把 Eq.4/5 的归一化参考
    从测试批次统计量 (std(c_T), med(c_T)) 替换为纯校准集统计量 (std(c_C), med(c_C)) 的
    FPR/TPR 代价。

变体:
    - test-stat  (默认, use_calib_stat=False): 归一化参考取测试批次一致性分布。
    - calib-stat (use_calib_stat=True)        : 归一化参考取纯校准集正常一致性分布。

若两者 FPR/TPR 差别不大 -> 论证 "测试批次统计量依赖的实际代价可忽略" (纯工程/部署权衡);
若差别大 -> 作为未来工作 (附附录定量证据)。

用法 (CBP 环境):
    cd /media/lixin/新加卷/数据集/test/StructCP
    source <CONDA_PREFIX>/bin/activate CBP
    export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
    python scripts/run_ablation_calib_stats.py --datasets Amazon,YelpChi,Elliptic,TFinance
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
)
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


def run_variant(args, data, s_tta, cons, y_test, use_calib_stat: bool):
    """在给定 TTA 分数/一致性上跑一种统计量变体, 返回 coverage dict。"""
    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal = data.y[cal_m]
    thr, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=args.alpha, tau_quantile=args.tau, mode="hard",
        score_quantile=args.score_quantile,
        use_calib_stat=use_calib_stat,
    )
    cov = evaluate_coverage(s_tta[test_m], data.y[test_m], thr)
    cov.update(info)
    return thr, cov


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

    out_rows = []
    per_dataset = {}
    for ds in datasets:
        t_all = time.time()
        relation = args.relation if ds not in ("Elliptic", "TFinance") else "homo"
        data = load_gad(ds, relation=relation, seed=args.seed).to(args.device)
        ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{ds}_{relation}.pth")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        ca = ckpt["args"]
        model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.freeze()

        x_enh = hypergraph_propagation(data.x, KNNHypergraph(data.x, k=args.k).to(args.device),
                                       args.layers, args.alpha_res)
        s_tta = get_scores(model, x_enh, data.edge_index)
        cons = hyperedge_consistency(s_tta, KNNHypergraph(data.x, k=args.k).to(args.device),
                                     features=x_enh)
        y_test = data.y[data.test_mask].cpu().numpy()
        auroc = roc_auc_score(y_test, s_tta[data.test_mask].cpu().numpy())
        auprc = average_precision_score(y_test, s_tta[data.test_mask].cpu().numpy())

        thr_t, cov_t = run_variant(args, data, s_tta, cons, y_test, use_calib_stat=False)
        thr_c, cov_c = run_variant(args, data, s_tta, cons, y_test, use_calib_stat=True)

        dFPR = cov_c["FPR"] - cov_t["FPR"]
        dTPR = cov_c["TPR"] - cov_t["TPR"]
        row = {
            "dataset": ds, "AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
            "test_stat": {"FPR": round(cov_t["FPR"], 4), "TPR": round(cov_t["TPR"], 4),
                          "threshold": round(float(thr_t), 4),
                          "ESS": round(float(cov_t.get("ESS", float("nan"))), 1)},
            "calib_stat": {"FPR": round(cov_c["FPR"], 4), "TPR": round(cov_c["TPR"], 4),
                           "threshold": round(float(thr_c), 4),
                           "ESS": round(float(cov_c.get("ESS", float("nan"))), 1)},
            "delta_calib_minus_test": {"dFPR": round(dFPR, 4), "dTPR": round(dTPR, 4)},
            "time_s": round(time.time() - t_all, 2),
        }
        out_rows.append(row)
        per_dataset[ds] = row
        print(f"[ablation:{ds}] |T|={int(data.test_mask.sum())}  AUROC={auroc:.4f}")
        print(f"   test-stat  : FPR={cov_t['FPR']:.4f} TPR={cov_t['TPR']:.4f} "
              f"thr={float(thr_t):.4f} ESS={cov_t.get('ESS', float('nan')):.1f}")
        print(f"   calib-stat : FPR={cov_c['FPR']:.4f} TPR={cov_c['TPR']:.4f} "
              f"thr={float(thr_c):.4f} ESS={cov_c.get('ESS', float('nan')):.1f}")
        print(f"   delta (calib - test): FPR {dFPR:+.4f}  TPR {dTPR:+.4f}")

    # 汇总: 是否可忽略
    max_abs_dFPR = max(abs(r["delta_calib_minus_test"]["dFPR"]) for r in out_rows)
    max_abs_dTPR = max(abs(r["delta_calib_minus_test"]["dTPR"]) for r in out_rows)
    summary = {
        "max_abs_dFPR": round(max_abs_dFPR, 4),
        "max_abs_dTPR": round(max_abs_dTPR, 4),
        "verdict": "negligible" if max_abs_dFPR < 0.01 and max_abs_dTPR < 0.02
        else "non-negligible-future-work",
    }
    print(f"\n[summary] max|dFPR|={max_abs_dFPR:.4f}  max|dTPR|={max_abs_dTPR:.4f}  "
          f"-> {summary['verdict']}")

    out = os.path.join(ROOT, "outputs", "ablation_calib_stats.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "per_dataset": per_dataset}, f, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
