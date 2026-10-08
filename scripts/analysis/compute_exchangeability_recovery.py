"""显式测量"可交换性恢复程度" (回应创新性建议四: 从 FPR 控制升级为可解释机制验证)。

核心问题: 加权共形恢复可交换性了吗? 直接测 W1:
  - W1(C_normal, T_all): 校准正常 vs 测试全量 (未校正, 基准失配)
  - W1(A_normal, T_all): 一致性加权后的校准池 vs 测试全量 (加权校正)
  - W1(pseudo_aug, T_all): 伪正常扩增后 vs 测试全量

若加权/扩增确实恢复可交换性, 应有:
  W1(加权后) < W1(未加权), 即 exchangeability_gap 下降。

口径: TTA 分数, seed 42, 一致性为去泄漏 (cal_mask 尺度)。
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

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

import sys
sys.path.insert(0, _structcp_root())
ROOT = _structcp_root()

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph, hyperedge_consistency, hypergraph_propagation,
)

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


def weighted_w1(s_cal, w_cal, s_test):
    """加权 W1: 把校准分布按其权重离散化后 vs 测试分布。用 2-Wasserstein on order stats 近似。
    实现: 用累积权重重采样校准分数到与测试等量, 再算 W1。
    """
    if w_cal is None:
        return float(wasserstein_distance(s_cal, s_test))
    n = len(s_test)
    o = np.argsort(s_cal)
    sc, wc = s_cal[o], w_cal[o]
    cw = np.cumsum(wc)
    cw = cw / cw[-1]
    # 逆 CDF 重采样到 n 个均匀分位
    u = (np.arange(n) + 0.5) / n
    res = np.interp(u, cw, sc)
    return float(wasserstein_distance(res, s_test))


def run(dataset, device, seed=42, alpha=0.05, tau_quantile=0.5, score_quantile=0.75):
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model = load_model(dataset, device)
    cal_m = data.cal_mask.cpu().numpy().astype(bool)
    test_m = data.test_mask.cpu().numpy().astype(bool)
    y = data.y.cpu().numpy()
    cal_normal = cal_m & (y == 0)

    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta_t = get_scores(model, x_enh, data.edge_index)
    s_tta = s_tta_t.cpu().numpy()
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    cons = hyperedge_consistency(s_tta_t, hg, features=x_enh, cal_mask=cal_bool).cpu().numpy()

    s_C = s_tta[cal_normal]
    c_C = cons[cal_normal]
    s_T = s_tta[test_m]

    # 1) 未加权: W1(C_normal, T_all)
    w1_raw = float(wasserstein_distance(s_C, s_T))

    # 2) 一致性加权 (去泄漏, 校准集统计量): 复刻 structcp_threshold 的权重
    ref = np.median(c_C); std = max(np.std(c_C), 1e-8)
    w = np.exp((c_C - ref) / std)
    w = np.clip(w, 0.05, 20.0)
    w = w / w.mean()
    w1_cons = weighted_w1(s_C, w, s_T)

    # 3) 伪正常扩增: 高一致性近正常测试节点并入校准, 权重折减 0.5
    tau = np.quantile(c_C, tau_quantile)
    gamma = 4.0 * (1 - score_quantile)
    s_upper = s_C.max() + gamma * max(s_tta.std(), 1e-6)
    pseudo = (cons[test_m] >= tau) & (s_tta[test_m] <= s_upper)
    s_pseudo = s_tta[test_m][pseudo]
    c_pseudo = cons[test_m][pseudo]
    cap = int(len(s_C) * 1.0)
    if len(s_pseudo) > cap:
        keep = np.argsort(c_pseudo)[::-1][:cap]
        s_pseudo, c_pseudo = s_pseudo[keep], c_pseudo[keep]
    s_aug = np.concatenate([s_C, s_pseudo])
    w_aug = np.concatenate([w, np.exp((c_pseudo - ref) / std) * 0.5])
    w_aug = w_aug / w_aug.mean()
    w1_aug = weighted_w1(s_aug, w_aug, s_T)

    # 4) 理想上限: W1(测试正常, 测试全量) —— 若校准=测试正常则不可再降
    s_Tnorm = s_tta[test_m & (y == 0)]
    w1_floor = float(wasserstein_distance(s_Tnorm, s_T)) if len(s_Tnorm) else float("nan")

    return {
        "dataset": dataset, "seed": seed,
        "n_cal_normal": int(cal_normal.sum()), "n_pseudo": int(pseudo.sum()),
        "W1_raw_C_T": w1_raw,
        "W1_consweighted_C_T": w1_cons,
        "W1_pseudoaug_C_T": w1_aug,
        "W1_floor_Tnorm_T": w1_floor,
        "gap_reduction_cons": float(1 - w1_cons / max(w1_raw, 1e-12)),
        "gap_reduction_aug": float(1 - w1_aug / max(w1_raw, 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(MAIN_DATASETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rows = {}
    for ds in datasets:
        print(f"=== {ds} ===", flush=True)
        r = run(ds, device, seed=args.seed)
        print(f"  W1 raw={r['W1_raw_C_T']:.4f} -> cons={r['W1_consweighted_C_T']:.4f} "
              f"(Δ={r['gap_reduction_cons']*100:.1f}%) -> aug={r['W1_pseudoaug_C_T']:.4f} "
              f"(Δ={r['gap_reduction_aug']*100:.1f}%)  floor={r['W1_floor_Tnorm_T']:.4f}")
        rows[ds] = r

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out = os.path.join(ROOT, "outputs", "exchangeability_recovery.json")
    with open(out, "w") as f:
        json.dump({"config": {"seed": args.seed}, "datasets": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
