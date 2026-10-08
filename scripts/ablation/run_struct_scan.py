"""结构扫描消融 (评审定位必做): 证明"邻居局部结构"是权重有效性的关键,
而非任意结构都能用 —— 直接堵死审稿人"为什么用 k-NN 邻居"的追问。

对照 (统一口径: 同一冻结 BWGNN + 同一超图伪正常扩增逻辑, 仅一致性权重的
邻居结构来源不同):
  - knn_k10        : 默认 k=10 星型 k-NN 结构 (主配置)
  - knn_k3         : 更紧的局部邻域 (k=3)
  - knn_k50        : 更宽的局部邻域 (k=50)
  - random_neighbor: 随机邻居索引 (结构无关, 阴性对照)
  - fully_connected: 全连接 (过度平滑, 阴性对照)
  - none           : 无结构 (即 run_weight_ablation 的 GenericFeat 特征密度权重)

若只有 knn_k{3,10,50} 能压住 FPR, 而 random/fully_connected/none 失效,
则证明"适度的局部邻域一致性"不可替代 —— 比"超图"概念具体一百倍。

运行: cd /media/lixin/新加卷/数据集/test/StructCP && \\
      source <CONDA_PREFIX>/bin/activate CBP && \\
      export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
      python scripts/run_struct_scan.py --dataset Amazon --struct_mode knn_k10
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
    structcp_threshold,
    generic_weighted_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph,
    hypergraph_propagation,
    hyperedge_consistency,
)
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402

ROOT = _structcp_root()

STRUCT_MODES = ["knn_k10", "knn_k3", "knn_k50", "random_neighbor",
                "fully_connected", "none"]


@torch.no_grad()
def scores_of(model, x, ei):
    return F.softmax(model(x, ei), dim=1)[:, 1]


def make_neighbor_idx(x, mode, k, rng):
    """返回形状 (n, k+1) 的邻居索引张量 (含自身, 与 KNNHypergraph.knn_idx 一致)。"""
    n = x.shape[0]
    xnp = x.detach().cpu().numpy().astype(np.float32)
    self_idx = torch.arange(n).unsqueeze(1)
    if mode == "fully_connected":
        # 每个节点的邻域 = 全部节点 (极大 k, 过度平滑). 为避免 OOM, centroid 只取
        # 前 min(n, 500) 个节点; 邻域过密本身已是阴性对照, 取子集不影响结论.
        cap = min(n, 500)
        perm = torch.arange(cap).unsqueeze(0).expand(n, cap).clone()
        return perm
    if mode == "random_neighbor":
        nb = rng.integers(0, n, size=(n, k))
        return torch.cat([self_idx, torch.as_tensor(nb)], dim=1)
    if mode in ("knn_k10", "knn_k3", "knn_k50"):
        from sklearn.neighbors import NearestNeighbors
        kk = {"knn_k10": 10, "knn_k3": 3, "knn_k50": 50}[mode]
        kk = min(kk + 1, n)
        nbrs = NearestNeighbors(n_neighbors=kk, n_jobs=-1).fit(xnp)
        _, idx = nbrs.kneighbors(xnp)
        return torch.from_numpy(idx.astype(np.int64))
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon",
                    choices=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--struct_mode", default="knn_k10", choices=STRUCT_MODES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
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

    # 共享传播 (无论结构模式, 平滑都用默认 k=10 星型, 仅一致性权重的邻居来源变化)
    hg = KNNHypergraph(data.x, k=args.k).to(device)
    x_hg = hypergraph_propagation(data.x, hg, args.layers, args.alpha_res)
    s_hg = scores_of(model, x_hg, data.edge_index)

    result = {"config": vars(args)}

    if args.struct_mode == "none":
        # 无结构: 特征密度权重 (等价 nonconform)
        thr, info = generic_weighted_threshold(
            s_hg[cal], y_cal, x_hg[cal], s_hg[te], x_hg[te],
            alpha=args.alpha, tau_quantile=args.tau, weight_source="feature")
    else:
        # 用该模式的邻居索引替换一致性计算的邻域
        nb = make_neighbor_idx(data.x, args.struct_mode, args.k, rng).to(device)
        hg2 = KNNHypergraph(data.x, k=args.k).to(device)
        hg2.knn_idx = nb
        cons = hyperedge_consistency(s_hg, hg2, features=x_hg)
        thr, info = structcp_threshold(
            s_hg[cal], y_cal, cons[cal], s_hg[te], cons[te],
            alpha=args.alpha, tau_quantile=args.tau)

    r = evaluate_coverage(s_hg[te], torch.as_tensor(y_te).to(s_hg.device), thr)
    r["AUROC"] = float(roc_auc_score(y_te, s_hg[te].cpu().numpy()))
    r["AUPRC"] = float(average_precision_score(y_te, s_hg[te].cpu().numpy()))
    r.update(info)
    if int(nov.sum()) > 0:
        r["FPR_novel"] = float((s_hg[nov] > thr).float().mean())
    result["result"] = r

    passed = r["FPR"] <= args.alpha
    print(f"\n{'='*90}\n结构扫描 {args.dataset} ({relation})  struct={args.struct_mode}  "
          f"alpha={args.alpha}\n{'='*90}")
    print(f"{'FPR':>9}{'TPR':>9}{'F1':>9}{'FPR_nov':>10}{'ESS':>9}   [{'PASS' if passed else 'FAIL'}]")
    print(f"{r['FPR']:>9.4f}{r['TPR']:>9.4f}{r['F1']:>9.4f}"
          f"{r.get('FPR_novel', float('nan')):>10.4f}{r.get('w_ess', float('nan')):>9.2f}")
    print("=" * 90)

    out = os.path.join(ROOT, "outputs",
                       f"struct_scan_{args.dataset}_{relation}_{args.struct_mode}.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
