"""TUNE 基线适配 (稳妥路线 A): 用统一子集 mask 覆盖 TUNE 内部划分后跑实验。

TUNE 官方 DGL 图节点顺序已验证与 StructCP load_gad 完全一致
(Amazon/YelpChi/TFinance 节点数/特征维/标签顺序一致), 故导出 mask 的
原始全量图节点索引可直接复用, 无需映射。

做法: 不改动 TUNE 源码, 本脚本 import 其 ModelWarper/Aligner, 复制官方
run.ipynb CELL 2 训练循环, 仅把 train_mask/test_mask 来源换成统一 mask。

Elliptic 官方 TUNE 无 DGL 图 / 无 models_pkl, 跳过 (compare.json 中缺项如实标注)。

运行 (CPU, 8GB 显存约束 R7):
  cd /media/lixin/新加卷/数据集/test/StructCP
  source <CONDA_PREFIX>/bin/activate CBP
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
  python scripts/run_tune_masked.py --dataset Amazon
  python scripts/run_tune_masked.py --dataset YelpChi
  python scripts/run_tune_masked.py --dataset TFinance
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
import torch.nn as nn
from dgl.data.utils import load_graphs
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- 注入 TUNE 路径, import 其模块 ----
TUNE_ROOT = os.path.join(_structcp_root(),
                         "baselines", "TUNE_official")
for p in (TUNE_ROOT, os.path.join(TUNE_ROOT, "models_GADBench")):
    if p not in sys.path:
        sys.path.insert(0, p)
from pretrained_models import ModelWarper  # noqa: E402
from aligner import Aligner                # noqa: E402

# TUNE 命名映射
TUNE_NAME = {"Amazon": "amazon", "YelpChi": "yelp", "TFinance": "tfinance"}
MASK_DIR = os.path.join(_structcp_root(),
                        "outputs", "baseline_subset")
FILL_DIR = os.path.join(MASK_DIR, "fill")
os.makedirs(FILL_DIR, exist_ok=True)

ADAPTOR_DIM = {
    "BWGNN": {"amazon": 96, "yelp": 96, "tfinance": 96},
    "GCN":   {"amazon": 32, "yelp": 32, "tfinance": 32},
    "GHRN":  {"amazon": 96, "yelp": 96, "tfinance": 96},
}


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_distribution(x):
    x = torch.sigmoid(x)
    return x / x.sum(dim=-1, keepdim=True)


def kld_loss(p, q):
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=-1).mean()


def k_percent_smallest_indices(tensor, k, mask=None):
    flat = tensor.flatten()
    if mask is not None:
        fm = mask.flatten()
        mt = flat[fm]
        kc = max(1, int(mt.numel() * k / 100))
        _, mi = torch.topk(mt, kc, largest=False)
        oi = fm.nonzero(as_tuple=False).squeeze(1)[mi]
    else:
        kc = max(1, int(flat.numel() * k / 100))
        _, oi = torch.topk(flat, kc, largest=False)
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[oi] = True
    return out


def eval_AUCPRC(labels, probs):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    return {
        "AUROC": float(roc_auc_score(labels, probs)),
        "AUPRC": float(average_precision_score(labels, probs)),
    }


def run_one(ours_name, model="BWGNN", device="cpu", trials=5, epochs=200,
            threshold_confident=0.7, lr_aligner=1e-3, lr_estimator=1e-3):
    tune_name = TUNE_NAME[ours_name]
    # 1) 加载 TUNE DGL 图
    graph = load_graphs(os.path.join(TUNE_ROOT, "datasets", tune_name))[0][0].to(device)
    feature = graph.ndata["feature"].to(torch.float32)
    labels = graph.ndata["label"].long()

    # 2) 注入统一 mask (原始全量图节点索引)
    mask_path = os.path.join(MASK_DIR, f"{ours_name}_node_masks.json")
    with open(mask_path) as f:
        mask = json.load(f)
    train_idx = np.array(mask["train"], dtype=np.int64)
    test_idx = np.array(mask["test"], dtype=np.int64)
    # val 在 TUNE 仅用于早停阈值无关, 这里用 train 充当 val
    n = feature.shape[0]
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    print(f"[{ours_name}] 注入统一 mask: train={train_mask.sum().item()} "
          f"test={test_mask.sum().item()} (异常{int(mask['test_anomaly'])}/"
          f"正常{int(mask['test_normal'])})")

    adaptor_dim = ADAPTOR_DIM[model][tune_name]
    all_auc, all_auprc, all_auc_gnn, all_auc_mlp = [], [], [], []
    for trial in range(trials):
        set_random_seed(3600 + trial)
        model_warper = ModelWarper(model, tune_name, folder=os.path.join(TUNE_ROOT, "models_pkl"), device=device)
        aligner = Aligner(feature.shape[1], [64, 64, 1]).to(device)
        adaptor = torch.nn.Linear(adaptor_dim, adaptor_dim).to(device)
        opt_align = torch.optim.Adam(aligner.parameters(), lr=lr_aligner)
        opt_est = torch.optim.Adam(adaptor.parameters(), lr=lr_estimator)

        best_auc, best_auc_gnn, best_auc_mlp, best_auprc = -1, -1, -1, -1
        for epoch in range(epochs):
            aligner.train(); adaptor.train(); model_warper.train()

            # --- estimator 分支: 独立前向 + backward ---
            opt_est.zero_grad()
            X_aligned = aligner.forward(graph, feature)
            pred_gnn, h_gnn, pred_dual, h_dual = model_warper.forward(X_aligned, graph, None)
            conf_dual = k_percent_smallest_indices(pred_dual[:, 1], threshold_confident)
            conf_normal = k_percent_smallest_indices(pred_gnn[:, 1], threshold_confident, conf_dual)
            if conf_normal.sum() > 0:
                loss_nei = kld_loss(adaptor(h_dual[conf_normal]), h_gnn[conf_normal])
                loss_nei.backward()
                opt_est.step()

            # --- aligner 分支: 重新前向 + backward (避免复用计算图) ---
            opt_align.zero_grad()
            X_aligned2 = aligner.forward(graph, feature)
            pred_gnn2, h_gnn2, pred_dual2, h_dual2 = model_warper.forward(X_aligned2, graph, None)
            h_gnn_est = adaptor(h_dual2).detach()
            loss_align = kld_loss(h_gnn2, h_gnn_est) * lr_aligner / lr_estimator
            loss_align.backward()
            opt_align.step()

            probs_gnn = pred_gnn.softmax(1)[:, 1]
            probs_mlp = model_warper.model.mlp.forward(adaptor(h_dual), False).softmax(1)[:, 1]
            probs = (probs_mlp + probs_gnn) / 2

            ev_gnn = eval_AUCPRC(labels[test_mask], probs_gnn[test_mask].detach())
            ev_mlp = eval_AUCPRC(labels[test_mask], probs_mlp[test_mask].detach())
            ev_mean = eval_AUCPRC(labels[test_mask], probs[test_mask].detach())
            if ev_mean["AUROC"] > best_auc:
                best_auc, best_auc_gnn, best_auc_mlp = ev_mean["AUROC"], ev_gnn["AUROC"], ev_mlp["AUROC"]
                best_auprc = ev_mean["AUPRC"]

        print(f"  [trial {trial}] AUROC={best_auc:.4f} AUPRC={best_auprc:.4f} "
              f"(gnn={best_auc_gnn:.4f}, mlp={best_auc_mlp:.4f})")
        all_auc.append(best_auc); all_auprc.append(best_auprc)
        all_auc_gnn.append(best_auc_gnn); all_auc_mlp.append(best_auc_mlp)

    res = {
        "dataset": ours_name, "model": model, "trials": trials, "epochs": epochs,
        "AUROC": float(np.mean(all_auc)), "AUROC_std": float(np.std(all_auc)),
        "AUPRC": float(np.mean(all_auprc)), "AUPRC_std": float(np.std(all_auprc)),
        "AUROC_gnn": float(np.mean(all_auc_gnn)), "AUROC_mlp": float(np.mean(all_auc_mlp)),
    }
    # 回填到 fill 目录 (run_compare_baselines.py 读取)
    out_path = os.path.join(FILL_DIR, f"{ours_name}__TUNE.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[TUNE] {ours_name}/{model}: AUROC={res['AUROC']:.4f}±{res['AUROC_std']:.3f} "
          f"AUPRC={res['AUPRC']:.4f}  -> {out_path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(TUNE_NAME.keys()))
    ap.add_argument("--model", default="BWGNN", choices=["BWGNN", "GCN", "GHRN"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    run_one(args.dataset, model=args.model, device=args.device,
            trials=args.trials, epochs=args.epochs)


if __name__ == "__main__":
    main()
