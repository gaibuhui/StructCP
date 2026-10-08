# -*- coding: utf-8 -*-
"""
把 GADBench 官方 .mat (Network/Attributes/Label) 转成 SAGAD 需要的 DGL 图。

SAGAD dataloader 期望 DGL 图含 ndata:
  * feature (n, d)
  * label (n,)
  * train_masks / val_masks / test_masks: (n, 20), 前10列=全监督, 后10列=半监督

数据源: datasets/Dataset_raw/{tolokers,questions,elliptic,...}.mat
输出:   baselines/SAGAD/../datasets/{name} (DGL save_graphs 单文件)

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/data_prep/convert_gadbench_to_dgl.py --datasets tolokers,questions,elliptic
"""
import argparse, os, sys

import numpy as np
from scipy.io import loadmat
from scipy import sparse as sp

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RAW_DIR = "datasets/Dataset_raw"
OUT_DIR = os.path.join(_ROOT, "baselines", "datasets")  # StructCP/baselines/datasets (SAGAD 期望 ../datasets)


def convert(name, train_ratio=0.4, val_ratio=0.2, seed=42):
    import dgl
    import torch
    mat = loadmat(os.path.join(RAW_DIR, f"{name}.mat"))
    net = mat["Network"]
    attr = mat["Attributes"]
    label = mat["Label"].ravel()

    x = np.asarray(attr.todense() if sp.issparse(attr) else attr, dtype=np.float32)
    y = np.asarray(label).astype(np.int64)
    n = len(y)

    # 邻接矩阵 -> edge_index (无向)
    net = net + net.T  # 对称
    coo = sp.coo_matrix(net)
    src, dst = coo.row.astype(np.int64), coo.col.astype(np.int64)

    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=n)
    g.ndata["feature"] = torch.from_numpy(x)
    g.ndata["label"] = torch.from_numpy(y)

    # 生成 20 列 masks: 随机 split (GADBench 用固定 split, 这里用 20 个随机种子近似)
    rng = np.random.RandomState(seed)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    masks = np.zeros((n, 20), dtype=bool)
    for j in range(20):
        idx = rng.permutation(n)
        tr, va = idx[:n_train], idx[n_train:n_train + n_val]
        te = idx[n_train + n_val:]
        masks[tr, j] = True
        masks[va, j] = True  # val 也标记 (SAGAD 训练用 train, 验证用 val)
        masks[te, j] = False
    # 注意: 保持与 GADBench 相同的 train/val/test 互斥
    g.ndata["train_masks"] = torch.from_numpy(masks)
    g.ndata["val_masks"] = torch.from_numpy(masks)
    g.ndata["test_masks"] = torch.from_numpy(~masks)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, name)
    dgl.save_graphs(out_path, [g])
    print(f"[{name}] nodes={n} feat={x.shape[1]} edges={len(src)//2} -> {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="tolokers,questions,elliptic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        try:
            convert(ds, seed=args.seed)
        except Exception as e:
            print(f"[{ds}] ERR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
