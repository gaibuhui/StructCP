"""回填论文 Sec 4.3 (交换性失效诊断): 用 KS 检验量化校准/测试分布偏移。

诊断量 (论文定义):
  Δ_exch = | mean(S_cal_normal) - mean(S_test) | / std(S_cal_normal)
      —— 标准 split conformal 的"可交换性间隙"。越大说明正常性偏移越严重。
  KS p 值: 对校准正常分数 vs 测试分数做两样本 Kolmogorov-Smirnov 检验。
      —— p<0.05 即拒绝"同分布"假设, 即交换性被破坏。

比较 (a) 标准 conformal 直接用原始分数, (b) StructCP 用超图 TTA 增强后的分数
重算同样的诊断量, 展示超图传播缩小了分布间隙、恢复了近似可交换性。

原占位示意值 (Amazon Δ=0.287/p=0.92 等) 由此真实替换。
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


import json
import os

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from data.loader import load_gad
from models.bwgnn import BWGNN
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation

CKPT_DIR = os.path.join(_structcp_root(), "checkpoint")
RESULT_DIR = os.path.join(_structcp_root(), "outputs")
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial"]
SEED = 42


def load_model(name, data):
    ckpt = os.path.join(CKPT_DIR, f"bwgnn_{name}_homo.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT_DIR, f"bwgnn_{name}.pth")
    model = BWGNN(in_channels=data.num_features)
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    model.freeze()
    return model


@torch.no_grad()
def diagnostics(name, data):
    model = load_model(name, data)
    x, ei = data.x, data.edge_index

    # (a) 原始分数
    s_raw = model.anomaly_score(x, ei).numpy()

    # (b) 超图 TTA 增强分数
    hg = KNNHypergraph(x, k=10)
    feat = hypergraph_propagation(x, hg, num_layers=2)
    s_tta = model.anomaly_score(feat, ei).numpy()

    cal_normal = s_raw[data.cal_mask.numpy() & (data.y.numpy() == 0)]
    test_all = s_raw[data.test_mask.numpy()]

    def gap(s_cal, s_te):
        mu, sd = s_cal.mean(), s_cal.std() + 1e-12
        delta = abs(mu - s_te.mean()) / sd
        ks = ks_2samp(s_cal, s_te)
        wd = wasserstein_distance(s_cal, s_te)
        mmd = _mmd_rbf(s_cal, s_te)
        return float(delta), float(ks.pvalue), float(wd), float(mmd)

    def _mmd_rbf(a, b, gamma=None):
        a = np.asarray(a, dtype=float).ravel()[:, None]   # (n,1)
        b = np.asarray(b, dtype=float).ravel()[None, :]   # (1,m)
        if gamma is None:
            med = np.median(np.abs(a.ravel()[:, None] - b.ravel()[None, :]).flatten()) + 1e-12
            gamma = 1.0 / (2 * med ** 2)
        aa = np.exp(-gamma * ((a - a.T) ** 2).sum(-1)).mean()
        bb = np.exp(-gamma * ((b - b.T) ** 2).sum(-1)).mean()
        ab = np.exp(-gamma * ((a - b) ** 2).sum(-1)).mean()
        return float(aa + bb - 2 * ab)

    d_raw = gap(cal_normal, test_all)
    d_tta = gap(s_tta[data.cal_mask.numpy() & (data.y.numpy() == 0)], s_tta[data.test_mask.numpy()])
    # QQ 图: 校准正常分数 vs 测试分数 (raw 与 structcp 两版) -> 落盘 supplementary
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = os.path.join(RESULT_DIR, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for tag, s_cal, s_te in [("raw", cal_normal, test_all),
                                 ("structcp", s_tta[data.cal_mask.numpy() & (data.y.numpy() == 0)],
                                  s_tta[data.test_mask.numpy()])]:
            plt.figure(figsize=(3.2, 3.2))
            # 经验分位数 QQ: 排序后两两对齐
            a = np.sort(s_cal); b = np.sort(s_te)
            n = min(len(a), len(b))
            plt.scatter(a[:n], b[:n], s=3, alpha=0.4)
            lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
            plt.plot([lo, hi], [lo, hi], "r--", lw=1)
            plt.xlabel("calibration normal scores"); plt.ylabel("test scores")
            plt.title(f"{name} ({tag}) QQ")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"qq_{name}_{tag}.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"  [warn] QQ plot failed for {name}: {e}", flush=True)
    return {
        "raw": {"delta_exch": round(d_raw[0], 4), "ks_p": round(d_raw[1], 4),
                "wasserstein": round(d_raw[2], 4), "mmd": round(d_raw[3], 5)},
        "structcp": {"delta_exch": round(d_tta[0], 4), "ks_p": round(d_tta[1], 4),
                   "wasserstein": round(d_tta[2], 4), "mmd": round(d_tta[3], 5)},
    }


def main():
    import sys, os as _os
    ds = sys.argv[1:] or DATASETS
    out = os.path.join(RESULT_DIR, "exchangeability_diagnosis.json")
    rows = {}
    if _os.environ.get("FORCE") and _os.path.exists(out):
        # 权重路径或度量已变更, 强制全量重算 (代码内清空, 不依赖外部删除)
        rows = {}
    elif _os.path.exists(out):  # 断点续跑
        with open(out) as f:
            rows = json.load(f)

    print(f"{'Dataset':<10} {'Δraw':>7} {'p_raw':>7} {'W_raw':>7} {'MMD_raw':>8} {'Δhyp':>7} {'p_hyp':>7} {'W_hyp':>7} {'MMD_hyp':>8}", flush=True)
    for name in ds:
        if name in rows and isinstance(rows[name], dict) and "mmd" in rows[name].get("raw", {}):
            print(f"  [cached] {name}", flush=True)
            continue
        try:
            if name == "TSocial":
                data = load_gad(name, relation="homo", seed=SEED,
                                subsample=100_000, knn_backbone=10)
            else:
                data = load_gad(name, relation="homo", seed=SEED)
            d = diagnostics(name, data)
        except Exception as e:
            print(f"  [skip] {name}: {e}", flush=True)
            continue
        rows[name] = d
        print(f"{name:<10} {d['raw']['delta_exch']:>7.3f} {d['raw']['ks_p']:>7.3f} {d['raw']['wasserstein']:>7.3f} {d['raw']['mmd']:>8.4f} "
              f"{d['structcp']['delta_exch']:>7.3f} {d['structcp']['ks_p']:>7.3f} {d['structcp']['wasserstein']:>7.3f} {d['structcp']['mmd']:>8.4f}", flush=True)
        with open(out, "w") as f:  # 每个数据集完成即落盘
            json.dump(rows, f, indent=2)
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
