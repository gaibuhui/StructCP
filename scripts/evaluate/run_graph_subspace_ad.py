"""Graph SubspaceAD 主推理脚本 (统一子集 / 统一特征 / 统一评测)。

用法:
  cd /media/lixin/新加卷/数据集/test/StructCP
  source <CONDA_PREFIX>/bin/activate CBP
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
  python scripts/run_graph_subspace_ad.py --dataset Amazon --hidden 128

固定超参 (论文写死, 可命令行覆盖做消融):
  --var_threshold 0.92   PCA 累计方差阈值
  --tta_iters 20         TTA 迭代步数
  --alpha 0.6            特征残差/拓扑融合系数
  --batch_size 512       冻结 BWGNN 前向批次
  --n_train_norm 3000    源域正常训练节点数
  --n_test 10000         测试集节点数
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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from models.graph_subspace_ad.loader import stratified_subset  # noqa: E402
from models.graph_subspace_ad.pipeline import GraphSubspaceAD  # noqa: E402

ROOT = _structcp_root()


def load_bwgnn(dataset, relation, hidden, device):
    """加载官方预训练 BWGNN (无训练特征提取); 优先 h128 命名, 回退 _homo。

    缺失则提示用 pretrain_bwgnn 生成。
    """
    ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h128.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"missing {ckpt}; 先用: python scripts/pretrain_bwgnn.py "
            f"--dataset {dataset} --relation {relation} --hidden {hidden}"
        )
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    in_ch = sd.get("in_channels") or 100
    model = BWGNN(in_ch, hidden, 2, d=2)
    model.load_state_dict(sd["state_dict"])
    model.freeze()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon")
    ap.add_argument("--source", default=None,
                    help="跨域源域数据集; 默认 None=同源 (用 --dataset 自身正常节点)")
    ap.add_argument("--relation", default="homo")
    ap.add_argument("--hidden", type=int, default=128, help="BWGNN 嵌入维度 d (论文固定 128)")
    ap.add_argument("--var_threshold", type=float, default=0.92)
    ap.add_argument("--tta_iters", type=int, default=20)
    ap.add_argument("--tta_enabled", action="store_true", default=False)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--auto_alpha", action="store_true", default=False,
                    help="自适应 alpha: 仅用训练正常节点 (无标签) 定权, 避免手调塌陷")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--n_train_norm", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--max_degree", type=int, default=None,
                    help="hub 固定阈值; 默认 None -> 按 98 分位动态裁剪")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/graph_subspace_ad")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    source = args.source or args.dataset
    print(f"[1/5] load target {args.dataset} (source={source}) ...")
    relation = args.relation if args.dataset not in ("Elliptic", "TFinance", "TSocial") else "homo"
    data = load_gad(args.dataset, relation=relation, seed=args.seed).to(device)
    print("       full:", data)

    print(f"[2/5] stratified subset (norm={args.n_train_norm}, test={args.n_test}, maxdeg={args.max_degree})")
    sub = stratified_subset(
        data, n_train_norm=args.n_train_norm, n_test=args.n_test,
        max_degree=args.max_degree, seed=args.seed,
    ).to(device)
    print("       subset:", sub)

    print(f"[3/5] load frozen BWGNN (hidden={args.hidden}) for {args.dataset}")
    model = load_bwgnn(args.dataset, relation, args.hidden, device)

    # 跨域: 用源域正常节点构建子空间 (源域需独立加载与特征提取)
    if args.source and args.source != args.dataset:
        srel = "homo"
        sdata = load_gad(args.source, relation=srel, seed=args.seed).to(device)
        ssub = stratified_subset(sdata, n_train_norm=args.n_train_norm,
                                 n_test=args.n_test, max_degree=args.max_degree,
                                 seed=args.seed).to(device)
        smodel = load_bwgnn(args.source, srel, args.hidden, device)
        with torch.no_grad():
            Zsrc = smodel.embed(ssub.x, ssub.edge_index).cpu().numpy().astype(np.float32)
        Z_norm = Zsrc[ssub.train_mask.cpu().numpy().astype(bool)]
        # 跨域默认开启 TTA (分布漂移对齐, 论文 Step4 核心)
        tta = True if args.tta_enabled is False else args.tta_enabled
        print(f"       cross-domain: subspace from {args.source} (n_norm={len(Z_norm)})")
    else:
        Z_norm = None
        tta = args.tta_enabled

    print("[4/5] Graph SubspaceAD pipeline (training-free) ...")
    t0 = time.time()
    g = GraphSubspaceAD(
        model, device=args.device, batch_size=args.batch_size,
        var_threshold=args.var_threshold, tta_iters=args.tta_iters,
        tta_enabled=tta, alpha=args.alpha,
        n_components_max=args.hidden, auto_alpha=args.auto_alpha,
    )
    parts = g.fit_predict(
        sub.x, sub.edge_index, sub.train_mask, sub.test_mask, return_parts=True
    )
    # 跨域: 用源域 Z_norm 覆盖子空间
    if Z_norm is not None:
        from models.graph_subspace_ad.subspace import AdaptiveSubspacePCA
        from models.graph_subspace_ad.tta import lightweight_tta_align
        ss = AdaptiveSubspacePCA(var_threshold=args.var_threshold,
                                 n_components_max=args.hidden).fit(Z_norm)
        Z = g._extract_embeddings(sub.x, sub.edge_index)
        Zt = Z[sub.test_mask.cpu().numpy().astype(bool)]
        if tta:
            Zt = lightweight_tta_align(Zt, Z_norm, iters=args.tta_iters)
        r = ss.reconstruct_residual(Zt)
        from sklearn.preprocessing import MinMaxScaler
        r = MinMaxScaler().fit_transform(r.reshape(-1, 1)).ravel()
        s_topo = g.subspace  # placeholder, 跨域不融合 topo 以公平
        st = parts["topo"]
        s_final = args.alpha * r + (1 - args.alpha) * st
        parts["score"] = MinMaxScaler().fit_transform(s_final.reshape(-1, 1)).ravel()
        parts["k"] = ss.n_components_
        parts["cum_var"] = ss.cum_explained_variance
    dur = time.time() - t0
    score = parts["score"]
    y_test = sub.y[sub.test_mask].cpu().numpy().astype(np.int64)

    auc = roc_auc_score(y_test, score)
    aps = average_precision_score(y_test, score)
    print(f"[5/5] done {dur:.1f}s | AUC={auc:.4f} AP={aps:.4f} | "
          f"K={parts['k']} cumVar={parts['cum_var']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    rec = {
        "method": "GraphSubspaceAD", "dataset": args.dataset, "source": source,
        "var_threshold": args.var_threshold, "tta_iters": args.tta_iters,
        "tta_enabled": bool(tta), "alpha": args.alpha,
        "hidden": args.hidden, "n_train_norm": args.n_train_norm,
        "n_test": args.n_test, "K": int(parts["k"]),
        "cum_var": float(parts["cum_var"]), "auc": float(auc), "ap": float(aps),
        "time_s": dur, "seed": args.seed,
    }
    out_path = os.path.join(args.out, f"{args.dataset}_h{args.hidden}_a{args.alpha}_v{args.var_threshold}.json")
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"      saved -> {out_path}")


if __name__ == "__main__":
    main()
