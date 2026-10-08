"""图权重 vs 非图权重 消融 (评审报告必做项 #2/#4): 证明超图一致性权重的不可替代性。

对照设置 (统一口径: 同一冻结 BWGNN + 同一超图伪正常扩增逻辑, 仅权重来源不同):
  - StructCP        : 权重 = 超边一致性 (图结构, adapters.conformal.structcp_threshold)
  - Generic-Feat  : 权重 = 节点特征密度比 (_density_ratio, 等价 nonconform 协变量偏移加权)
  - Random        : 权重 = 随机 (阴性对照, 验证"任何加权都行"的零假设)

若 Generic-Feat 的 FPR 无法稳定压到 α 以下, 而 StructCP 可以, 则证明
"图结构提供的高阶一致性信息"是通用加权无法替代的 —— 直接回应审稿人
"StructCP 只是 nonconform 的特例"的质疑。

运行: cd /media/lixin/新加卷/数据集/test/StructCP && \\
      source <CONDA_PREFIX>/bin/activate CBP && \\
      export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
      python scripts/run_weight_ablation.py --dataset Amazon
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

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from adapters.conformal import (  # noqa: E402
    evaluate_coverage,
    generic_weighted_threshold,
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
def scores_of(model, x, ei):
    return F.softmax(model(x, ei), dim=1)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon",
                    choices=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    relation = args.relation if args.dataset != "Elliptic" else "homo"

    data = load_gad(args.dataset, relation=relation, seed=args.seed).to(device)
    ckpt = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{args.dataset}_{relation}.pth"),
                      map_location=device)
    ca = ckpt["args"]
    in_channels = ckpt.get("in_channels", data.num_features)
    model = BWGNN(in_channels, ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal, te = data.cal_mask, data.test_mask
    y_te = data.y[te].cpu().numpy()
    y_cal = data.y[cal]
    nov = data.novel_mask & te

    # 共享超图 + 分数
    hg = KNNHypergraph(data.x, k=args.k).to(device)
    x_hg = hypergraph_propagation(data.x, hg, args.layers, args.alpha_res)
    s_hg = scores_of(model, x_hg, data.edge_index)
    cons = hyperedge_consistency(s_hg, hg, features=x_hg)

    results = {"config": vars(args)}

    # ---------- StructCP (图权重) ----------
    thr_h, info_h = structcp_threshold(
        s_hg[cal], y_cal, cons[cal], s_hg[te], cons[te],
        alpha=args.alpha, tau_quantile=args.tau)
    r_h = evaluate_coverage(s_hg[te], torch.as_tensor(y_te).to(s_hg.device), thr_h)
    r_h["AUROC"] = float(roc_auc_score(y_te, s_hg[te].cpu().numpy()))
    r_h["AUPRC"] = float(average_precision_score(y_te, s_hg[te].cpu().numpy()))
    r_h.update(info_h)
    if int(nov.sum()) > 0:
        r_h["FPR_novel"] = float((s_hg[nov] > thr_h).float().mean())
    results["StructCP(graph-weight)"] = r_h

    # ---------- Generic Feature-weight (非图, 等价 nonconform) ----------
    # 审稿问题12修复: 为公平对比, GenericFeat 也使用与 StructCP 完全相同的
    # hypergraph-propagated 特征 x_hg, 使两者分数与特征信息量一致,
    # 纯比较"图结构一致性权重" vs "特征密度比权重" 两种加权机制本身。
    x_all = x_hg
    thr_g, info_g = generic_weighted_threshold(
        s_hg[cal], y_cal, x_all[cal], s_hg[te], x_all[te],
        alpha=args.alpha, tau_quantile=args.tau,
        weight_source="feature")
    r_g = evaluate_coverage(s_hg[te], torch.as_tensor(y_te).to(s_hg.device), thr_g)
    r_g["AUROC"] = r_h["AUROC"]  # 同分数, 仅阈值不同
    r_g["AUPRC"] = r_h["AUPRC"]
    r_g.update(info_g)
    if int(nov.sum()) > 0:
        r_g["FPR_novel"] = float((s_hg[nov] > thr_g).float().mean())
    results["GenericFeat(weight)"] = r_g

    # ---------- Random weight (阴性对照) ----------
    thr_r, info_r = generic_weighted_threshold(
        s_hg[cal], y_cal, x_all[cal], s_hg[te], x_all[te],
        alpha=args.alpha, tau_quantile=args.tau,
        weight_source="random")
    r_r = evaluate_coverage(s_hg[te], torch.as_tensor(y_te).to(s_hg.device), thr_r)
    r_r["AUROC"] = r_h["AUROC"]
    r_r["AUPRC"] = r_h["AUPRC"]
    r_r.update(info_r)
    if int(nov.sum()) > 0:
        r_r["FPR_novel"] = float((s_hg[nov] > thr_r).float().mean())
    results["Random(control)"] = r_r

    # ---------- 汇总 ----------
    names = ["StructCP(graph-weight)", "GenericFeat(weight)", "Random(control)"]
    print(f"\n{'='*108}\n权重消融 {args.dataset} ({relation})  [alpha={args.alpha}]\n{'='*108}")
    hdr = f"{'Method':<22}{'FPR':>9}{'TPR':>9}{'F1':>9}{'FPR_nov':>10}{'ESS':>9}{'w_min':>9}{'w_max':>9}"
    print(hdr)
    print("-" * 108)
    for name in names:
        r = results[name]
        print(f"{name:<22}{r['FPR']:>9.4f}{r['TPR']:>9.4f}{r['F1']:>9.4f}"
              f"{r.get('FPR_novel', float('nan')):>10.4f}{r.get('w_ess', float('nan')):>9.2f}"
              f"{r.get('w_min', float('nan')):>9.3f}{r.get('w_max', float('nan')):>9.3f}")
    print("=" * 108)
    print(f"[判定] StructCP FPR={r_h['FPR']:.4f} "
          f"vs GenericFeat FPR={r_g['FPR']:.4f} "
          f"(α={args.alpha}): "
          f"{'图权重显著更优' if r_h['FPR'] <= args.alpha < r_g['FPR'] else '需人工核对'}")

    out = os.path.join(ROOT, "outputs", f"weight_ablation_{args.dataset}_{relation}.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
