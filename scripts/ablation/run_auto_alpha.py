"""自适应 alpha 验证脚本 (论文鲁棒性核心证据)。

对比:
  (A) 固定 alpha=0.6  (之前主表基准, 但 Amazon/Elliptic 会塌陷)
  (B) 自适应 alpha     (仅用训练正常节点, 无标签)
在真实 4 数据集 + 真实 BWGNN 预训练权重上跑, 证明自适应 alpha 消除手调塌陷。

用法 (CBP 环境):
  cd /media/lixin/新加卷/数据集/test/StructCP
  source <CONDA_PREFIX>/bin/activate CBP
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
  python scripts/run_auto_alpha.py --device cuda
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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from models.graph_subspace_ad.loader import stratified_subset  # noqa: E402
from models.graph_subspace_ad.pipeline import GraphSubspaceAD  # noqa: E402

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
HIDDEN = 128


def load_bwgnn(dataset, hidden, device):
    ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h{hidden}.pth")
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model = BWGNN(sd.get("in_channels") or 100, hidden, 2, d=sd.get("args", {}).get("order", 2))
    model.load_state_dict(sd["state_dict"])
    return model.freeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--out", default="outputs/graph_subspace_ad")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for ds in DATASETS:
        print(f"\n===== {ds} =====", flush=True)
        data = load_gad(ds, relation="homo", seed=args.seed).to(device)
        sub = stratified_subset(data, n_train_norm=3000, n_test=args.n_test,
                                seed=args.seed, degree_quantile=0.98).to(device)
        model = load_bwgnn(ds, args.hidden, device)

        row = {}
        for tag, auto in [("fixed_a0.6", False), ("auto_alpha", True)]:
            g = GraphSubspaceAD(
                model, device=device, batch_size=512,
                var_threshold=0.92, tta_enabled=False,
                alpha=0.6 if not auto else 1.0,   # auto 模式下 alpha 由内部重算
                n_components_max=args.hidden, auto_alpha=auto,
            )
            score = g.fit_predict(sub.x, sub.edge_index, sub.train_mask, sub.test_mask)
            y = sub.y[sub.test_mask].cpu().numpy().astype(np.int64)
            auc = float(roc_auc_score(y, score))
            aps = float(average_precision_score(y, score))
            row[tag] = {"auc": auc, "ap": aps, "alpha": float(g.alpha)}
            print(f"  [{tag}] alpha={g.alpha:.4f} AUC={auc:.4f} AP={aps:.4f}", flush=True)
        results[ds] = row

    out_path = os.path.join(args.out, "auto_alpha_compare.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
