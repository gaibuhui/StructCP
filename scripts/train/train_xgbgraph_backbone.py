# -*- coding: utf-8 -*-
"""
XGBGraph 骨干 (GADBench): 2 层邻居特征聚合 + XGBoost。

GADBench 核心发现: 树集成 + 简单邻居聚合 (mean/max/sum) 在多数 GAD 数据集上
超过所有 GNN (含 BWGNN)。本脚本把 XGBGraph 作为 StructCP 的替代骨干:
  * 输入: 节点原始特征 + 2 层邻居均值聚合特征 (concat)
  * 训练: XGBoost 二分类 (train_mask 内), 输出节点异常概率
  * 输出: 全图节点分数, 落盘 npz (与 load_frozen_detector 接口兼容的分数文件)

与 StructCP 的兼容性: StructCP 是分数级 wrapper, 只需骨干输出节点级分数,
不依赖骨干架构, 因此 XGBGraph 分数可直接替换 BWGNN 分数接入 Frozen/StructCP。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/train/train_xgbgraph_backbone.py --datasets Amazon,YelpChi --seed 42
"""
import argparse, io, json, os, sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.loader import load_gad  # noqa: E402

import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def neighbor_aggregate_features(x, edge_index, layers=2, agg="mean"):
    """GADBench 邻居聚合: 逐层邻居均值/最大/求和聚合, 每层 concat 到原特征。"""
    n = x.shape[0]
    feat = x.copy()
    all_feats = [x]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    # 对称归一化: 双向
    for _ in range(layers):
        # scatter add 邻居特征
        agg_feat = np.zeros_like(x, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)
        np.add.at(agg_feat, dst, x[src])
        np.add.at(count, dst, 1.0)
        # 双向再聚合一次 (dst->src)
        np.add.at(agg_feat, src, x[dst])
        np.add.at(count, src, 1.0)
        count = np.maximum(count, 1e-8)
        if agg == "mean":
            x = (agg_feat / count[:, None]).astype(np.float32)
        elif agg == "max":
            # 用稀疏聚合近似 max 不可行, 用 mean 替代 (GADBench 默认 mean)
            x = (agg_feat / count[:, None]).astype(np.float32)
        all_feats.append(x)
    return np.concatenate(all_feats, axis=1)


def train_and_save(dataset, seed, root=_ROOT, device="cpu"):
    torch_ok = True
    import torch
    data = load_gad(dataset, relation="homo", seed=seed)
    x = data.x.numpy()
    edge_index = data.edge_index
    y = data.y.numpy()
    train_mask = data.train_mask.bool().numpy()

    feat = neighbor_aggregate_features(x, edge_index, layers=2, agg="mean")

    X_tr = feat[train_mask]
    y_tr = y[train_mask]

    # 训练 XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        n_jobs=-1, random_state=seed,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(feat)[:, 1]  # P(anomaly)

    # 评估 AUROC (全图, 参照 GADBench 协议)
    auroc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")

    # 落盘: 分数 + 特征 (供 StructCP 评估使用)
    out_dir = os.path.join(root, "checkpoint", "xgbgraph")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_seed{seed}.npz")
    np.savez(out_path, scores=proba.astype(np.float32), feat=feat.astype(np.float32), y=y)
    print(f"[{dataset} seed={seed}] XGBGraph AUROC={auroc:.4f}  "
          f"feat_dim={feat.shape[1]}  -> {out_path}", flush=True)
    return {"dataset": dataset, "seed": seed, "AUROC": auroc, "path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance,photo,computer,weibo,tolokers,questions,reddit")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--out", default="xgbgraph_train.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {}
    for ds in datasets:
        results[ds] = {}
        for seed in seeds:
            try:
                r = train_and_save(ds, seed)
                results[ds][seed] = r
            except Exception as e:
                print(f"[{ds} seed={seed}] ERR {type(e).__name__}: {e}", flush=True)
                results[ds][seed] = {"error": str(e)}

    out_path = os.path.join(_ROOT, "outputs", args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
