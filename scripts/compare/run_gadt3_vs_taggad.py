"""在 TA-GGAD 通用 GAD 基准 (cs/weibo/citeseer/cora/ACM/BlogCatalog) 上,
对两个独立 TTA 检测器做公平对比: GADT3 vs TA-GGAD(ARC)。

设定: single-dataset few-shot (与 GADT3 / TA-GGAD 原论文节点级 few-shot 一致)
  * 每个数据集随机采样 shot 个正常 + shot 个异常节点作为 support(train_mask)
  * 全图节点作为 test, 在 (cal|test) 无标签图上做测试时自适应(TTA)
  * 异常分数在全图算 AUROC / AUPRC

与 StructCP 的 run_compare_baselines.py 区别:
  * 那套复用 BWGNN 权重 + novel-normality 偏移, 仅支持 Amazon/YelpChi 等 5 个
  * 本脚本面向 TA-GGAD 通用 GAD 基准 (无 BWGNN 权重、无 novel 偏移),
    直接喂 PyG Data 给 GADT3Adapter / ARCAdapter, 复用已 pyg 改写的两个检测器

依赖: conda 环境 CBP (torch 2.3.1 + torch_geometric 2.8.0)
运行: cd /media/lixin/新加卷/数据集/test/StructCP && \
      source <CONDA_PREFIX>/bin/activate CBP && \
      export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \
      python scripts/run_gadt3_vs_taggad.py --datasets cs weibo citeseer cora ACM BlogCatalog
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
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

# baselines 包根 (gadt3_pyg / ta_ggad_pyg 在此)
_ROOT = _structcp_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASE = os.path.join(_ROOT, "baselines")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from gadt3_pyg.adapter import GADT3Adapter  # noqa: E402
from ta_ggad_pyg.adapter import ARCAdapter   # noqa: E402

MAT_DIR = os.path.join(_BASE, "ta_ggad_src", "dataset")
DEVICE = "cpu"  # 8GB 显存约束(R7): GADT3/ARC 邻居级计算在 CPU 稳定


def load_mat_data(name: str) -> Data:
    """从 TA-GGAD .mat 构造 PyG Data (Network / Attributes / Label)。"""
    m = loadmat(os.path.join(MAT_DIR, f"{name}.mat"))
    net = m["Network"].tocoo()
    row = torch.from_numpy(net.row).long()
    col = torch.from_numpy(net.col).long()
    edge_index = torch.stack([row, col], dim=0)
    attrs = m["Attributes"].tocoo()
    # 转 dense 特征
    x = torch.zeros(attrs.shape[0], attrs.shape[1], dtype=torch.float32)
    x[attrs.row, attrs.col] = torch.from_numpy(np.asarray(attrs.data, dtype=np.float32))
    lab = m["Label"]
    if lab.shape[0] == 1:
        lab = lab.T
    y = torch.from_numpy(np.asarray(lab.flatten(), dtype=np.int64)).long()
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_features = x.shape[1]
    return data


def fewshot_split(data: Data, shot: int, seed: int) -> Data:
    """随机采样 shot 个正常 + shot 个异常作为 support(train_mask)。"""
    rng = np.random.default_rng(seed)
    y = data.y.numpy()
    n_idx = np.where(y == 0)[0]
    a_idx = np.where(y == 1)[0]
    n_sel = rng.choice(n_idx, size=min(shot, len(n_idx)), replace=False)
    a_sel = rng.choice(a_idx, size=min(shot, len(a_idx)), replace=False)
    support = np.union1d(n_sel, a_sel)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(support)] = True
    test_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    cal_mask = test_mask.clone()
    data.train_mask = train_mask
    data.cal_mask = cal_mask
    data.test_mask = test_mask
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cs", "weibo", "citeseer", "cora", "ACM", "BlogCatalog"])
    ap.add_argument("--shot", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--skip_gadt3", action="store_true")
    ap.add_argument("--skip_arc", action="store_true")
    args = ap.parse_args()

    results = {}
    for name in args.datasets:
        print(f"\n{'='*100}\n数据集 {name}\n{'='*100}")
        mat = load_mat_data(name)
        gadt3_auc, gadt3_ap = [], []
        arc_auc, arc_ap = [], []
        for seed in args.seeds:
            data = fewshot_split(mat.clone(), args.shot, seed)
            n = data.num_nodes
            y_te = data.y.numpy()

            # ---------- GADT3 ----------
            if not args.skip_gadt3:
                t0 = time.time()
                gadt3 = GADT3Adapter(ndim_s=data.num_features, device=DEVICE)
                gadt3.fit_source(data)
                gadt3.fit_target(data)
                s_g = gadt3.score(data).numpy()
                gadt3_auc.append(roc_auc_score(y_te, s_g))
                gadt3_ap.append(average_precision_score(y_te, s_g))
                print(f"  [seed {seed}] GADT3  AUROC={gadt3_auc[-1]:.4f} "
                      f"AUPRC={gadt3_ap[-1]:.4f}  ({time.time()-t0:.1f}s)", flush=True)

            # ---------- TA-GGAD / ARC ----------
            if not args.skip_arc:
                t0 = time.time()
                arc = ARCAdapter(dim=data.num_features, device=DEVICE)
                arc.fit(data)
                arc.fit_target(data)
                s_a = arc.score(data).numpy()
                arc_auc.append(roc_auc_score(y_te, s_a))
                arc_ap.append(average_precision_score(y_te, s_a))
                print(f"  [seed {seed}] TA-GGAD AUROC={arc_auc[-1]:.4f} "
                      f"AUPRC={arc_ap[-1]:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        rec = {}
        if not args.skip_gadt3:
            rec["GADT3"] = {"AUROC": float(np.mean(gadt3_auc)), "AUROC_std": float(np.std(gadt3_auc)),
                            "AUPRC": float(np.mean(gadt3_ap)), "AUPRC_std": float(np.std(gadt3_ap))}
        if not args.skip_arc:
            rec["TA-GGAD"] = {"AUROC": float(np.mean(arc_auc)), "AUROC_std": float(np.std(arc_auc)),
                              "AUPRC": float(np.mean(arc_ap)), "AUPRC_std": float(np.std(arc_ap))}
        results[name] = rec

    # ---------- 汇总表 ----------
    print(f"\n{'='*100}\n对比汇总 (shot={args.shot}, seeds={args.seeds})\n{'='*100}")
    methods = [m for m in ["GADT3", "TA-GGAD"] if not (args.skip_gadt3 and m == "GADT3")
               and not (args.skip_arc and m == "TA-GGAD")]
    hdr = f"{'Dataset':<12}" + "".join(f"{m+' AUROC':>16}{m+' AUPRC':>16}" for m in methods)
    print(hdr)
    print("-" * 100)
    for name in args.datasets:
        line = f"{name:<12}"
        for m in methods:
            r = results[name][m]
            line += f"{r['AUROC']:.4f}±{r['AUROC_std']:.3f}{r['AUPRC']:.4f}±{r['AUPRC_std']:.3f}"
        print(line)
    print("=" * 100)

    out = os.path.join(_ROOT, "outputs", "gadt3_vs_taggad_benchmark.json")
    json.dump({"config": vars(args), "results": results}, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
