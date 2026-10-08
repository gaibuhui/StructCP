"""计算 Assumption H 的 W1 诊断表 (回应批评1/5: 把"未证明"转化为"已验证")。

Assumption H (论文 Claim 1): 高一致性集合 H_τ 与测试正常分布 T_normal 在
1-Wasserstein 距离上相差至多 ε。这里直接量化 ε:
  - W1(H_τ, T_normal): 高一致性校准正常节点的分数分布 vs 测试新正常节点
  - W1(C_normal, T_normal): 全部校准正常 vs 测试新正常 (基线/对照, 应更大)
  - 若 W1(H_τ, T_normal) << W1(C_normal, T_normal), 则一致性加权确实把校准分布
    拉向测试正常分布, Assumption H 在数据上成立。

口径:
  - 高一致性: 校准正常节点中 consistency > 中位数 (τ=median, 与主流程一致)
  - T_normal = test_mask & y==0 (新正常, 偏移协议)
  - 使用 StructCP 的 (去泄漏) 一致性 (cal_mask 尺度) 与 TTA 增强分数
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


def run(dataset, device, seed=42):
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model = load_model(dataset, device)
    cal_m = data.cal_mask.cpu().numpy().astype(bool)
    test_m = data.test_mask.cpu().numpy().astype(bool)
    y = data.y.cpu().numpy()
    cal_normal = cal_m & (y == 0)
    test_normal = test_m & (y == 0)  # novel-normal
    test_anom = test_m & (y == 1)

    # StructCP 一致性 (去泄漏: cal_mask 尺度) 与 TTA 分数
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta_t = get_scores(model, x_enh, data.edge_index)
    s_tta = s_tta_t.cpu().numpy()
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    cons = hyperedge_consistency(s_tta_t, hg, features=x_enh, cal_mask=cal_bool).cpu().numpy()

    # 高一致性 = 校准正常中 consistency > 其中位数
    c_cal_normal = cons[cal_normal]
    tau = np.median(c_cal_normal)
    H_tau = cal_normal & (cons >= tau)
    
    # 特征密度权重: 用特征的L2范数作为权重, 取top 50%的节点
    feat_norm = np.linalg.norm(data.x.cpu().numpy(), ord=2, axis=1)
    feat_norm_cal_normal = feat_norm[cal_normal]
    tau_feat = np.median(feat_norm_cal_normal)
    C_feat = cal_normal & (feat_norm >= tau_feat)

    s_H = s_tta[H_tau]
    s_C = s_tta[cal_normal]
    s_C_feat = s_tta[C_feat]
    s_T = s_tta[test_normal]  # 测试新正常
    s_A = s_tta[test_anom]

    w1_H_T = float(wasserstein_distance(s_H, s_T))
    w1_C_T = float(wasserstein_distance(s_C, s_T))
    w1_C_feat_T = float(wasserstein_distance(s_C_feat, s_T))
    # 高一致性 vs 全测试正常: 若一致性有效, 应远小于全校准 vs 全测试
    s_T_all = s_tta[test_m]
    w1_H_Tall = float(wasserstein_distance(s_H, s_T_all))
    w1_C_Tall = float(wasserstein_distance(s_C, s_T_all))
    # 也对照: 高一致性 vs 异常 (若一致性有判别力, 应明显)
    w1_H_A = float(wasserstein_distance(s_H, s_A)) if len(s_A) else float("nan")

    return {
        "dataset": dataset, "seed": seed,
        "n_H_tau": int(H_tau.sum()), "n_T_normal": int(test_normal.sum()),
        "W1(H_tau, T_normal)": w1_H_T,
        "W1(C_normal, T_normal)": w1_C_T,
        "W1(C_feat, T_normal)": w1_C_feat_T,
        "W1(H_tau, T_all)": w1_H_Tall,
        "W1(C_normal, T_all)": w1_C_Tall,
        "W1(H_tau, A_test)": w1_H_A,
        "ratio_H_over_C": float(w1_H_T / max(w1_C_T, 1e-12)),
        "ratio_H_over_C_feat": float(w1_H_T / max(w1_C_feat_T, 1e-12)),
        "assumptionH_holds": bool(w1_H_T < w1_C_T),
        "assumptionH_holds_feat": bool(w1_H_T < w1_C_feat_T),
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
        print(f"  W1(H_τ,T_norm)={r['W1(H_tau, T_normal)']:.4f}  W1(C_norm,T_norm)={r['W1(C_normal, T_normal)']:.4f}  "
              f"ratio={r['ratio_H_over_C']:.2f}x  H_holds={r['assumptionH_holds']}")
        print(f"  W1(H_τ,T_norm)={r['W1(H_tau, T_normal)']:.4f}  W1(C_feat_norm,T_norm)={r['W1(C_feat, T_normal)']:.4f}  "
              f"ratio_feat={r['ratio_H_over_C_feat']:.2f}x  H_holds_feat={r['assumptionH_holds_feat']}")
        print(f"  W1(H_τ,T_all)={r['W1(H_tau, T_all)']:.4f}  W1(C_norm,T_all)={r['W1(C_normal, T_all)']:.4f}  "
              f"W1(H_τ,A_test)={r['W1(H_tau, A_test)']:.4f}")
        rows[ds] = r

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out = os.path.join(ROOT, "outputs", "assumptionH_diagnosis.json")
    with open(out, "w") as f:
        json.dump({"config": {"seed": args.seed}, "datasets": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
