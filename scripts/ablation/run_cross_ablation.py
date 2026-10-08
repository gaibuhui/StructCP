"""跨域泛化消融 (方案 A6): 源域 -> 目标域 全矩阵 (TTA 开启, alpha=1.0)。

每个源数据集用其正常节点构建子空间, 在每个目标数据集测试集打分。
结果落盘 outputs/graph_subspace_ad/cross_matrix.json。
这是论文核心卖点: 跨域下 TTA 对齐能否救回纯 PCA 的分布漂移。
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
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from models.graph_subspace_ad.loader import stratified_subset  # noqa: E402
from models.graph_subspace_ad.subspace import AdaptiveSubspacePCA  # noqa: E402
from models.graph_subspace_ad.tta import lightweight_tta_align  # noqa: E402

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


def load_bwgnn(dataset, hidden, device):
    ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h128.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_homo.pth")
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m = BWGNN(sd.get("in_channels") or 100, hidden, 2, d=2).to(device)
    m.load_state_dict(sd["state_dict"])
    m.freeze()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--out", default="outputs/graph_subspace_ad")
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # 预加载所有数据集子集与嵌入 (同种子保证可比)
    cache = {}
    for ds in DATASETS:
        data = load_gad(ds, relation="homo", seed=args.seed).to(device)
        sub = stratified_subset(data, n_test=args.n_test, seed=args.seed).to(device)
        model = load_bwgnn(ds, args.hidden, device)
        with torch.no_grad():
            Z = model.embed(sub.x, sub.edge_index).cpu().numpy().astype(np.float32)
        y = sub.y[sub.test_mask].cpu().numpy().astype(np.int64)
        Zt = Z[sub.test_mask.cpu().numpy().astype(bool)]
        Zn = Z[sub.train_mask.cpu().numpy().astype(bool)]
        cache[ds] = {"Z": Z, "Zn": Zn, "Zt": Zt, "y": y}

    results = []
    for src in DATASETS:
        for tgt in DATASETS:
            if src == tgt:
                continue
            Zn = cache[src]["Zn"]
            Zt = cache[tgt]["Zt"]
            y = cache[tgt]["y"]
            ss = AdaptiveSubspacePCA(var_threshold=0.92, n_components_max=args.hidden).fit(Zn)
            # 不开 TTA 基线
            r0 = ss.reconstruct_residual(Zt)
            auc0 = roc_auc_score(y, r0)
            # 开 TTA
            Zta = lightweight_tta_align(Zt, Zn, iters=20)
            r1 = ss.reconstruct_residual(Zta)
            auc1 = roc_auc_score(y, r1)
            results.append({"source": src, "target": tgt,
                            "auc_no_tta": float(auc0), "auc_tta": float(auc1)})
            print(f"[cross] {src}->{tgt}: noTTA={auc0:.4f} TTA={auc1:.4f}")

    out_path = os.path.join(args.out, "cross_matrix.json")
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
