"""异常归因 case study (论文潜力 3: 可解释性)。

对测试集中得分最高的 Top-K 异常节点, 分析其残差在主成分轴的投影贡献,
定位"脱离正态子空间的方向", 输出可解释归因 (哪个主成分偏离最大)。
同时对比训练正常节点的投影分布基线, 给出偏离度 z-score。

用法 (CBP 环境):
  cd /media/lixin/新加卷/数据集/test/StructCP
  source <CONDA_PREFIX>/bin/activate CBP
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
  python scripts/run_attribution.py --device cuda --dataset Amazon
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

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from models.graph_subspace_ad.loader import stratified_subset  # noqa: E402
from models.graph_subspace_ad.pipeline import GraphSubspaceAD  # noqa: E402

ROOT = _structcp_root()
HIDDEN = 128


def load_bwgnn(dataset, hidden, device):
    ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h{hidden}.pth")
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model = BWGNN(sd.get("in_channels") or 100, hidden, 2, d=sd.get("args", {}).get("order", 2))
    model.load_state_dict(sd["state_dict"])
    return model.freeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default="outputs/graph_subspace_ad")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    data = load_gad(args.dataset, relation="homo", seed=args.seed).to(device)
    sub = stratified_subset(data, n_train_norm=3000, n_test=args.n_test,
                            seed=args.seed, degree_quantile=0.98).to(device)
    model = load_bwgnn(args.dataset, args.hidden, device)

    g = GraphSubspaceAD(
        model, device=device, batch_size=512, var_threshold=0.92,
        tta_enabled=False, alpha=1.0, n_components_max=args.hidden, auto_alpha=True,
    )
    parts = g.fit_predict(sub.x, sub.edge_index, sub.train_mask, sub.test_mask,
                          return_parts=True)
    score = parts["score"]
    y = sub.y[sub.test_mask].cpu().numpy().astype(np.int64)
    test_idx = parts["index"]

    # 训练正常节点的投影分布基线
    Z = model.embed(sub.x, sub.edge_index).cpu().numpy().astype(np.float32)
    train_idx = np.where(sub.train_mask.cpu().numpy().astype(bool))[0]
    C_train = g.subspace.component_contributions(Z[train_idx])      # (N_train, K)
    mu = C_train.mean(0)
    sigma = C_train.std(0) + 1e-8

    C_test = g.subspace.component_contributions(Z[test_idx])        # (N_test, K)
    # 按融合分排序 Top-K
    order = np.argsort(-score)[:args.topk]
    cases = []
    for rank, i in enumerate(order, 1):
        true_lbl = int(y[i])
        z = (C_test[i] - mu) / sigma                                  # 偏离度
        top_comp = int(np.argmax(np.abs(z)))                          # 最偏离主成分
        cases.append({
            "rank": rank,
            "node_id": int(test_idx[i]),
            "true_label": true_lbl,
            "score": float(score[i]),
            "top_deviating_component": top_comp,
            "deviation_z": float(z[top_comp]),
            "residual": float(parts["residual"][i]),
            "topo": float(parts["topo"][i]),
        })

    out = {
        "dataset": args.dataset,
        "K": int(parts["k"]),
        "cum_var": float(parts["cum_var"]),
        "n_train_norm": int(len(train_idx)),
        "topk_cases": cases,
    }
    out_path = os.path.join(args.out, f"attribution_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[attribution] {args.dataset}: K={out['K']} cum_var={out['cum_var']:.3f}")
    for c in cases:
        print(f"  #{c['rank']:2d} node={c['node_id']:6d} label={c['true_label']} "
              f"score={c['score']:.3f} -> 主成分#{c['top_deviating_component']} "
              f"偏离 z={c['deviation_z']:+.2f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
