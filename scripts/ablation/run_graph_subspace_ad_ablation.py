"""Graph SubspaceAD 消融实验脚本 (审稿人最看重, 论文第五章)。

覆盖:
  A1 方差保留阈值 {0.8,0.85,0.9,0.92,0.95}
  A2 TTA 有无对比 (--no_tta)
  A3 融合权重 alpha {0.2,0.4,0.6,0.8,1.0}
  A4 不同 GNN 编码器 (GCN/GAT/BWGNN) -> 需对应 checkpoint, 默认 BWGNN
  A5 少量正常样本鲁棒性 (n_train_norm 500/1000/3000)
  A6 跨域泛化 (源域->多目标域) -> 循环 datasets 列表

结果统一落盘 outputs/graph_subspace_ad/ablation_<type>.json
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
from models.gcn_encoder import GCNEncoder  # noqa: E402
from models.gat_encoder import GATEncoder  # noqa: E402
from models.graph_subspace_ad.loader import stratified_subset  # noqa: E402
from models.graph_subspace_ad.pipeline import GraphSubspaceAD  # noqa: E402

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
VAR_GRID = [0.8, 0.85, 0.9, 0.92, 0.95]
ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
NORM_GRID = [500, 1000, 3000]
ENCODERS = {"BWGNN": ("bwgnn", BWGNN), "GCN": ("gcn", GCNEncoder), "GAT": ("gat", GATEncoder)}


def load_encoder(encoder, dataset, relation, hidden, device):
    """加载指定 encoder 的预训练权重 (A4 消融用)。

    checkpoint 命名约定 (见 scripts/pretrain_bwgnn.py):
      {enc}_{dataset}_h{hidden}.pth  e.g. gcn_Amazon_h128.pth / gat_Elliptic_h128.pth
    """
    enc_tag, EncClass = ENCODERS[encoder]
    ckpt = os.path.join(ROOT, "checkpoint", f"{enc_tag}_{dataset}_h{hidden}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"[A4] missing {ckpt}; 先跑: python scripts/pretrain_bwgnn.py "
            f"--dataset {dataset} --encoder {enc_tag} --hidden {hidden}")
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    in_ch = sd.get("in_channels") or 100
    # 构造与预训练一致的模型: BWGNN 用 d=order; GCN/GAT 用 d=order, 多头参数固定
    if encoder == "BWGNN":
        model = EncClass(in_ch, hidden, 2, d=sd.get("args", {}).get("order", 2))
    elif encoder == "GCN":
        model = EncClass(in_ch, hidden, 2, d=sd.get("args", {}).get("order", 2))
    else:  # GAT
        model = EncClass(in_ch, hidden, 2, d=sd.get("args", {}).get("order", 2),
                          heads=8, att_out=16)
    model.load_state_dict(sd["state_dict"])
    return model.freeze()


def _run(dataset, hidden, var_threshold, tta_enabled, alpha, n_train_norm, n_test, seed, device, degree_quantile=0.98, cached=None, alpha_base=1.0, encoder="BWGNN"):
    relation = "homo"
    use_alpha = alpha if alpha is not None else alpha_base
    # 直接复用主推理脚本 GraphSubspaceAD pipeline (单次 embed, 下游一致),
    # 仅 encoder 不同, 保证 A4 消融的 AUC 与主表完全可比。
    from models.graph_subspace_ad.pipeline import GraphSubspaceAD
    if cached is not None:
        sub, _, _ = cached
        model = load_encoder(encoder, dataset, relation, hidden, device)
    else:
        data = load_gad(dataset, relation=relation, seed=seed).to(device)
        sub = stratified_subset(data, n_train_norm=n_train_norm, n_test=n_test,
                                 seed=seed, degree_quantile=degree_quantile).to(device)
        model = load_encoder(encoder, dataset, relation, hidden, device)
    g = GraphSubspaceAD(
        model, device=device, batch_size=512,
        var_threshold=var_threshold, tta_iters=20,
        tta_enabled=tta_enabled, alpha=use_alpha,
        n_components_max=hidden,
    )
    parts = g.fit_predict(sub.x, sub.edge_index, sub.train_mask, sub.test_mask, return_parts=True)
    score = parts["score"]
    y_test = sub.y[sub.test_mask].cpu().numpy().astype(np.int64)
    return {
        "auc": float(roc_auc_score(y_test, score)),
        "ap": float(average_precision_score(y_test, score)),
        "K": int(parts["k"]), "cum_var": float(parts["cum_var"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, choices=["var", "tta", "alpha", "encoder", "norm", "cross"])
    ap.add_argument("--dataset", default="Amazon")
    ap.add_argument("--source", default="Amazon")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--degree_quantile", type=float, default=0.98)
    ap.add_argument("--alpha_base", type=float, default=0.6,
                    help="非 alpha 消融使用的基准融合系数 (默认 0.6=与主表一致, 残差+拓扑)")
    ap.add_argument("--out", default="outputs/graph_subspace_ad")
    args = ap.parse_args()
    device = torch.device(args.device)

    os.makedirs(args.out, exist_ok=True)
    results = []

    # 同一数据集按 encoder 分别 embed (不同编码器的 Z 不同, 必须各自缓存复用)
    relation = "homo"
    _data = load_gad(args.dataset, relation=relation, seed=args.seed).to(device)
    _sub = stratified_subset(_data, n_train_norm=3000, n_test=args.n_test,
                             seed=args.seed, degree_quantile=args.degree_quantile).to(device)
    _y = _sub.y[_sub.test_mask].cpu().numpy().astype(np.int64)
    cached_by_enc = {}
    enc_list = ["BWGNN"] if args.ablation != "encoder" else list(ENCODERS.keys())
    for enc in enc_list:
        print(f"[prepare] embed {args.dataset} with {enc} ...", flush=True)
        _model = load_encoder(enc, args.dataset, relation, args.hidden, device)
        with torch.no_grad():
            _Z = _model.embed(_sub.x, _sub.edge_index).cpu().numpy().astype(np.float32)
        cached_by_enc[enc] = (_sub, _Z, _y)
        print(f"[prepare] {enc} done. N={_Z.shape[0]}", flush=True)

    if args.ablation == "var":
        for v in VAR_GRID:
            print(f"[var] v={v} ...", flush=True)
            r = _run(args.dataset, args.hidden, v, True, None, 3000, args.n_test, args.seed, device, args.degree_quantile, cached=cached_by_enc["BWGNN"], alpha_base=args.alpha_base)
            r.update({"var_threshold": v}); results.append(r)
            print(f"      -> AUC={r['auc']:.4f}", flush=True)
    elif args.ablation == "tta":
        for en in [True, False]:
            print(f"[tta] tta={en} ...", flush=True)
            r = _run(args.dataset, args.hidden, 0.92, en, None, 3000, args.n_test, args.seed, device, args.degree_quantile, cached=cached_by_enc["BWGNN"], alpha_base=args.alpha_base)
            r.update({"tta_enabled": en}); results.append(r)
            print(f"      -> AUC={r['auc']:.4f}", flush=True)
    elif args.ablation == "alpha":
        for a in ALPHA_GRID:
            print(f"[alpha] a={a} ...", flush=True)
            r = _run(args.dataset, args.hidden, 0.92, False, a, 3000, args.n_test, args.seed, device, args.degree_quantile, cached=cached_by_enc["BWGNN"])
            r.update({"alpha": a}); results.append(r)
            print(f"      -> AUC={r['auc']:.4f}", flush=True)
    elif args.ablation == "norm":
        # norm 消融需不同 n_train_norm 子集 -> 不能复用 cached, 逐档重采样
        for n in NORM_GRID:
            print(f"[norm] n_train={n} ...", flush=True)
            r = _run(args.dataset, args.hidden, 0.92, True, None, n, args.n_test, args.seed, device, args.degree_quantile, alpha_base=args.alpha_base)
            r.update({"n_train_norm": n}); results.append(r)
            print(f"      -> AUC={r['auc']:.4f}", flush=True)
    elif args.ablation == "cross":
        for tgt in DATASETS:
            if tgt == args.source:
                continue
            print(f"[cross] src={args.source} tgt={tgt} ...", flush=True)
            r = _run(tgt, args.hidden, 0.92, True, None, 3000, args.n_test, args.seed, device, args.degree_quantile, alpha_base=args.alpha_base)
            r.update({"target": tgt}); results.append(r)
            print(f"      -> AUC={r['auc']:.4f}", flush=True)
    elif args.ablation == "encoder":
        for enc in ENCODERS.keys():
            print(f"[encoder] enc={enc} ...", flush=True)
            r = _run(args.dataset, args.hidden, 0.92, True, None, 3000, args.n_test, args.seed, device, args.degree_quantile, cached=cached_by_enc[enc], alpha_base=args.alpha_base, encoder=enc)
            r.update({"encoder": enc}); results.append(r)
            print(f"      -> AUC={r['auc']:.4f}", flush=True)

    out_path = os.path.join(args.out, f"ablation_{args.ablation}_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump({"ablation": args.ablation, "dataset": args.dataset,
                   "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
