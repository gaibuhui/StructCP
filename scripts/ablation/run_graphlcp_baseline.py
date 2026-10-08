"""复现 GRaPHLCP (Baghershahi et al., 2026) 的"拓扑加权保形"用于 novel-normality 对比。

GRaPHLCP 核心 (arXiv:2605.08074):
  - PPR 核 π = βs(I-(1-β)M)^{-1}, M=D^{-1}A, β=restart 概率, 幂迭代 K 次近似
  - 拓扑权重: w_i ∝ d_i^{-1}[π_anchor]_i  (结构接近 anchor 的校准点权重大)
  - 加权分位保形阈值: q̂ = Quantile(1-α)(Σ w_i δ_{s(z_i)} + w_{n+1}δ_{∞})

协议适配: novel-normality 检测需要全局阈值 (非逐点预测集), 故把 GRaPHLCP 的
"PPR 拓扑权重"作为校准点权重, 接入 StructCP 的全局加权保形框架 (pseudo-normal
扩增可开关), 从而在 FPR/TPR 上公平对照 "拓扑权重" vs "一致性权重"。
验证审稿论点: PPR 权重仅依赖图拓扑, 无法捕捉特征空间 novelty, 在 normal shift
下 FPR 应失控 (YelpChi 预计 >α)。
"""

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
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, _structcp_root())
ROOT = _structcp_root()

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from adapters.conformal import weighted_quantile, evaluate_coverage, structcp_threshold  # noqa: E402
from adapters.hypergraph_tta import KNNHypergraph, hyperedge_consistency, hypergraph_propagation  # noqa: E402

MAIN_DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


@torch.no_grad()
def get_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def load_model(dataset, device):
    ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h128.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_homo.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    return model


def ppr_kernel(edge_index, n, seed_vec, beta=0.15, K=30):
    """幂迭代近似 PPR: π ← (1-β) M π + β s。返回 (n,) 向量。

    M = D^{-1} A (行随机过渡矩阵)。
    """
    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    deg = np.bincount(row, minlength=n).astype(np.float64)
    deg = np.where(deg == 0, 1.0, deg)
    # M 行随机: (M π)[i] = Σ_j (A_ij / deg_i) π_j = Σ_{j: (i,j)∈E} π_j / deg_i
    pi = seed_vec.astype(np.float64).copy()
    s = seed_vec.astype(np.float64)
    for _ in range(K):
        nxt = np.zeros(n, dtype=np.float64)
        np.add.at(nxt, row, pi[col] / deg[row])
        pi = (1 - beta) * nxt + beta * s
    return pi


def run_dataset(dataset, device, seed=42, alpha=0.05, use_pseudo=True, beta=0.15, K=30):
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model = load_model(dataset, device)
    cal_m, test_m = data.cal_mask.cpu().numpy(), data.test_mask.cpu().numpy()
    y = data.y.cpu().numpy()
    y_test = y[test_m]
    n = data.x.shape[0]
    edge_index = data.edge_index.cpu()

    # 1) Frozen scores (GRaPHLCP 用 frozen 编码器, 无 TTA)
    s_raw = get_scores(model, data.x, data.edge_index).cpu().numpy()

    # 2) PPR 核: seed = 测试节点均匀分布 → 全局 PPR, 校准点按 PPR 拓扑接近度加权
    seed_vec = np.zeros(n)
    seed_vec[test_m] = 1.0 / max(test_m.sum(), 1)
    t0 = time.time()
    ppr = ppr_kernel(edge_index, n, seed_vec, beta=beta, K=K)
    t_ppr = time.time() - t0

    # 校准点 PPR 权重 (拓扑接近测试集), 度归一化 (GRaPHLCP Eq.4: d_i^{-1}[π]_i)
    deg = np.bincount(edge_index[0].cpu().numpy(), minlength=n).astype(np.float64)
    deg = np.where(deg == 0, 1.0, deg)
    cal_ppr = ppr[cal_m]
    cal_deg = deg[cal_m]

    # 3) 加权保形阈值: 用 PPR 权重重加权校准正常分数
    s_cal = s_raw[cal_m]
    # 仅用校准正常节点 (与 Frozen 保形同口径), 权重 = PPR 拓扑权重 (归一化)
    # GRaPHLCP 用拓扑权重 w_i ∝ d_i^{-1}[π_anchor]_i
    w = (cal_ppr / cal_deg)
    w = w / (w.sum() + 1e-12)
    # 加到测试点占位权重: 加权分位 level = (1-α)(n_cal+1)/n_cal (Barber)
    n_cal = w.size
    level = min(1.0, (1 - alpha) * (n_cal + 1) / n_cal)
    q = np.concatenate([s_cal, [np.inf]])
    wq = np.concatenate([w, np.array([w.sum() * 0 + 1.0 / n_cal])])  # 测试点权重占位
    thr_ppr = float(weighted_quantile(torch.tensor(q), torch.tensor(wq), level).item())

    # 4) pseudo-normal 扩增 (对照 StructCP 一致: 高一致性近正常测试点进校准池)
    if use_pseudo:
        hg = KNNHypergraph(data.x, k=10).to(device)
        x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
        cons = hyperedge_consistency(
            torch.tensor(s_raw).to(device), hg, features=x_enh,
            cal_mask=torch.as_tensor(cal_m, dtype=torch.bool, device=device)).cpu().numpy()
        tau = np.quantile(cons[cal_m & (y == 0)], 0.5)
        gamma = 4.0 * (1 - 0.75)
        s_upper = s_cal.max() + gamma * max(s_raw.std(), 1e-6)
        pseudo = (cons[test_m] >= tau) & (s_raw[test_m] <= s_upper)
        n_pseudo = int(pseudo.sum())
        if n_pseudo > 0:
            # 把伪正常并入校准, 权重 = PPR 拓扑权重 (折减 0.5 表示标签不确定, 与 StructCP 一致)
            s_cal2 = np.concatenate([s_cal, s_raw[test_m][pseudo]])
            w2 = np.concatenate([w, ppr[test_m][pseudo] / deg[test_m][pseudo] * 0.5])
            w2 = w2 / (w2.sum() + 1e-12)
            n2 = w2.size
            level2 = min(1.0, (1 - alpha) * (n2 + 1) / n2)
            q2 = np.concatenate([s_cal2, [np.inf]])
            wq2 = np.concatenate([w2, np.array([1.0 / n2])])
            thr_ppr = float(weighted_quantile(torch.tensor(q2), torch.tensor(wq2), level2).item())
    else:
        n_pseudo = 0

    ev = evaluate_coverage(torch.tensor(s_raw[test_m]), torch.tensor(y_test), torch.tensor(thr_ppr))
    return {
        "dataset": dataset, "seed": seed, "alpha": alpha, "use_pseudo": use_pseudo,
        "beta": beta, "K": K, "n_pseudo": n_pseudo,
        "FPR": float(ev["FPR"]), "TPR": float(ev["TPR"]), "F1": float(ev["F1"]),
        "threshold": thr_ppr, "ppr_time_s": t_ppr,
        "ppr_mean_cal": float(cal_ppr.mean()), "ppr_spread": float(cal_ppr.std() / max(cal_ppr.mean(), 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(MAIN_DATASETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-pseudo", action="store_true")
    ap.add_argument("--beta", type=float, default=0.15)
    args = ap.parse_args()
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    use_pseudo = not args.no_pseudo

    rows = []
    for ds in datasets:
        for seed in seeds:
            print(f"=== {ds} seed={seed} (pseudo={use_pseudo}) ===", flush=True)
            r = run_dataset(ds, device, seed=seed, alpha=args.alpha, use_pseudo=use_pseudo, beta=args.beta)
            print(f"  FPR={r['FPR']:.4f}  TPR={r['TPR']:.4f}  F1={r['F1']:.4f}  "
                  f"thr={r['threshold']:.4f}  n_pseudo={r['n_pseudo']}")
            rows.append(r)

    # 聚合
    agg = {}
    for ds in datasets:
        rs = [r for r in rows if r["dataset"] == ds]
        def m(k):
            v = [r[k] for r in rs if r[k] == r[k]]
            return [float(np.mean(v)), float(np.std(v))]
        agg[ds] = {k: m(k) for k in ["FPR", "TPR", "F1"]}

    out = {"config": {"alpha": args.alpha, "use_pseudo": use_pseudo, "beta": args.beta, "seeds": seeds},
           "per_seed": rows, "agg": agg}
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    tag = "pseudo" if use_pseudo else "nopseudo"
    path = os.path.join(ROOT, "outputs", f"graphlcp_baseline_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {path}")
    print(f"\n{'dataset':9s} {'FPR':10s} {'TPR':10s} {'F1':10s}")
    for ds in datasets:
        a = agg[ds]
        print(f"{ds:9s} {a['FPR'][0]:.4f}±{a['FPR'][1]:.4f}  {a['TPR'][0]:.4f}±{a['TPR'][1]:.4f}  "
              f"{a['F1'][0]:.4f}±{a['F1'][1]:.4f}")


if __name__ == "__main__":
    main()
