"""评审回应实验：统一校准基线对比（Experiment Group A + E）。

在同一冻结 BWGNN 骨干、同一 kNN 图、同一测试分数上对比校准方法:
  - Frozen (SplitCP raw)      : 原始分数 + 标准 split conformal (评审 A1)
  - SplitCP-TTA               : TTA 分数 + 标准 split conformal (评审 A1, 隔离 TTA 效应)
  - StructCP (ours, adaptive) : 图一致性权重 + 伪正常扩增 (与主表 Tab:main 同口径)
  - FeatWeightCP              : 特征密度比权重 + 伪正常扩增 (评审 A2, generic weighted CP)
  - RandWeightCP              : 随机权重 + 伪正常扩增 (评审 A2, 阴性对照)
  - KLIEP-CP                  : KLIEP 密度比权重 + 伪正常扩增 (评审 A3)
  - Logistic-CP               : Logistic 密度比权重 + 伪正常扩增 (评审 A3)
  - TempScaling               : 温度缩放 (评审 A4, 单调变换 → FPR 不变的阴性对照)
  - QuantReg                  : 分位数回归逐节点自适应阈值 (评审 A5)

统计口径: 每数据集 5 seed, 输出 mean±std + 逐 seed 明细 (供 bootstrap CI 复用)。
运行 (CBP 环境):
  cd /media/lixin/新加卷/数据集/test/StructCP && source <CONDA_PREFIX>/bin/activate CBP \
    && export PYTHONPATH="$PWD" && python scripts/compare/run_calibration_baselines.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys as _sys

import numpy as np
import torch
import torch.nn.functional as F

# --- 项目根定位 ---
_ROOT = None
_p = os.path.dirname(os.path.abspath(__file__))
for _ in range(10):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)
if _ROOT and _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from utils.checkpoint import load_frozen_detector  # noqa: E402
from adapters.conformal import (  # noqa: E402
    evaluate_coverage,
    split_conformal_threshold,
    generic_weighted_threshold,
    structcp_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph, hyperedge_consistency, hypergraph_propagation, contrastive_consistency,
)
from data.loader import load_gad  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

METHODS = ["Frozen", "SplitCP-TTA", "StructCP", "FeatWeightCP", "RandWeightCP",
           "KLIEP-CP", "Logistic-CP", "TempScaling", "QuantReg"]


@torch.no_grad()
def _softmax_scores(model, x, edge_index):
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


def _flip(scores, y_test):
    """若 AUROC<0.5 则翻转分数使异常=高分组; 返回 (校正分数, 翻转标志)。"""
    auroc = float(roc_auc_score(y_test, scores.cpu().numpy()))
    flip = auroc < 0.5
    return (1.0 - scores, True) if flip else (scores, False)


# ---------------------------------------------------------------------------
# 密度比权重实现 (评审 A3)
# ---------------------------------------------------------------------------
def _pca(x: torch.Tensor, n_comp: int = 16):
    """标准化 + PCA 降维 (保持数值稳定)。返回 (降维张量, 拟合器信息)。
    用全量 x 拟合, 仅作为固定特征变换 (无标签信息)。"""
    x = x.double()
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    u, s, v = torch.linalg.svd(xc, full_matrices=False)
    n = min(n_comp, s.numel())
    return (u[:, :n] * s[:n].unsqueeze(0)).float(), (mu.float(), v[:, :n].float())


def _pca_transform(x: torch.Tensor, fitted):
    mu, v = fitted
    return ((x.double() - mu.double()) @ v.double()).float()


def _rbf_kernel(A, centers, sigma):
    """A: (n, d), centers: (B, d) -> K (n, B), K_ij = exp(-||a_i - c_j||^2 / (2 sigma^2))"""
    d2 = ((A.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2.0 * sigma ** 2))


def kliep_weights(cal_x: torch.Tensor, test_x: torch.Tensor, n_basis: int = 200,
                  n_cal: int = 20000, n_test: int = 20000, iters: int = 500,
                  seed: int = 42, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    """KLIEP (Sugiyama et al. 2008) 密度比权重 w = p_test/p_cal 的固定点估计。

    输入已 PCA 降维的 float32 特征。返回 (w_cal, w_test) 并归一化到均值 1。
    仅在子采样上做固定点迭代 (保持可控耗时), 在全量上评估权重。
    """
    rng = np.random.RandomState(seed)
    n_c = min(n_cal, len(cal_x)); n_t = min(n_test, len(test_x))
    ic = np.sort(rng.choice(len(cal_x), n_c, replace=False))
    it = np.sort(rng.choice(len(test_x), n_t, replace=False))
    cal_s, test_s = cal_x[ic].to(device), test_x[it].to(device)

    # 基中心采样自测试分布 (分子密度估计中心)
    b = min(n_basis, n_t)
    ib = np.sort(rng.choice(n_t, b, replace=False))
    centers = test_s[ib]

    # 带宽: 测试子样本对距离中位数
    sub = test_s[:: max(1, len(test_s) // 500)]
    pd = torch.cdist(sub, sub)
    sigma = float(torch.median(pd[pd > 0]).clamp(min=1e-6))

    A = _rbf_kernel(cal_s, centers, sigma)          # (n_cal_sub, B)
    B = _rbf_kernel(test_s, centers, sigma).mean(0)  # (B,)
    alpha = torch.ones(b, device=device) / b
    eps = 1e-10
    for _ in range(iters):
        g = A @ alpha                                  # (n_cal_sub,)
        denom = (A / (g.unsqueeze(1) + eps)).mean(0)   # (B,)
        alpha = alpha * (B / (denom + eps))
        # 约束: (1/n_cal) sum_i g(x_i) = 1
        alpha = alpha / ((A @ alpha).mean() + eps)
        alpha = alpha.clamp(min=0.0)
        alpha = alpha / alpha.sum().clamp(min=eps)
    # 全量评估权重
    def _eval(x):
        K = _rbf_kernel(x.to(device), centers, sigma)
        w = (K @ alpha)
        return w.cpu()
    w_cal, w_test = _eval(cal_x), _eval(test_x)
    w_cal = w_cal.clamp(min=0.05, max=20.0)
    w_test = w_test.clamp(min=0.05, max=20.0)
    w_cal = w_cal / w_cal.mean(); w_test = w_test / w_test.mean()
    info = {"sigma": sigma, "n_basis": b, "n_cal_sub": n_c, "n_test_sub": n_t}
    return w_cal, w_test, info


def logistic_dr_weights(cal_x: torch.Tensor, test_x: torch.Tensor, seed: int = 42,
                        max_test: int = 50000) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Logistic 密度比: 训练分类器 (cal 正常=0, test=1), w = p(test|x)/p(cal|x)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    X_cal = cal_x.numpy(); X_test = test_x.numpy()
    if len(X_test) > max_test:
        sel = np.sort(rng.choice(len(X_test), max_test, replace=False))
        X_test_sel = X_test[sel]
    else:
        X_test_sel = X_test
    X = np.vstack([X_cal, X_test_sel])
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test_sel))])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(sc.transform(X), y)
    # 密度比 w(x) = p(y=test|x) / p(y=cal|x) = odds(test | x)
    pt = clf.predict_proba(sc.transform(X_test))[:, 1]
    pc = clf.predict_proba(sc.transform(X_cal))[:, 1]
    w_test = torch.as_tensor(pt / np.clip(1.0 - pt, 1e-6, None)).float()
    w_cal = torch.as_tensor(pc / np.clip(1.0 - pc, 1e-6, None)).float()
    w_cal = w_cal.clamp(min=0.05, max=20.0) / w_cal.clamp(min=0.05, max=20.0).mean()
    w_test = w_test.clamp(min=0.05, max=20.0) / w_test.clamp(min=0.05, max=20.0).mean()
    return w_cal, w_test, {"n_cal": len(X_cal), "n_test_used": len(X_test_sel)}


def dr_weighted_cp(cal_scores, cal_y, cal_x, test_scores, test_x, w_cal, w_test,
                   alpha: float = 0.05, tau_quantile: float = 0.5,
                   score_quantile: float = 0.75, cap_ratio: float = 1.0):
    """密度比加权保形阈值: 与 generic_weighted_threshold 同构, 仅替换权重来源。
    伪正常筛选用特征密度比信号 (与 FeatWeightCP 一致, 保证公平)。
    内部统一在 CPU 上计算 (权重由 CPU 估计), 避免设备不一致。
    """
    cal_scores = cal_scores.cpu(); cal_y = cal_y.cpu(); cal_x = cal_x.cpu()
    test_scores = test_scores.cpu(); test_x = test_x.cpu()
    w_cal = w_cal.cpu(); w_test = w_test.cpu()
    normal = cal_y == 0
    s_cal = cal_scores[normal].flatten()
    x_cal = cal_x[normal]
    s_test = test_scores.flatten()
    mu = x_cal.mean(dim=0, keepdim=True)
    def _proxy(x):
        return torch.norm(x - mu, dim=1)
    proxy_cal = _proxy(x_cal)
    proxy_test = _proxy(test_x)
    # 伪正常筛选 (特征密度比代理)。注意: w_cal 已由调用方按校准正常样本计算。
    w_cal_n = w_cal.flatten()
    from adapters.conformal import _density_ratio
    dr_test = _density_ratio(proxy_test, proxy_cal)
    tau = torch.quantile(_density_ratio(proxy_cal, proxy_test), tau_quantile)
    gamma = 4.0 * (1.0 - score_quantile) if score_quantile is not None else 1.0
    s_upper = s_cal.max() + gamma * s_cal.std().clamp(min=1e-6)
    pseudo_mask = (dr_test >= tau) & (s_test <= s_upper)
    s_pseudo = s_test[pseudo_mask]
    w_pseudo = w_test[pseudo_mask]
    cap = int(len(s_cal) * cap_ratio)
    if s_pseudo.numel() > cap:
        keep = torch.topk(dr_test[pseudo_mask], cap).indices
        s_pseudo, w_pseudo = s_pseudo[keep], w_pseudo[keep]
    s_all = torch.cat([s_cal, s_pseudo])
    w_all = torch.cat([w_cal_n / w_cal_n.mean(), w_pseudo / w_pseudo.mean() * 0.5])
    from adapters.conformal import weighted_quantile
    n = s_all.numel()
    level = min(1.0, (1 - alpha) * (n + 1) / n)
    thr = weighted_quantile(s_all, w_all, level)
    info = {"n_pseudo_normal": int(s_pseudo.numel())}
    return thr, info


# ---------------------------------------------------------------------------
# 温度缩放 (评审 A4) —— 单调变换, FPR 不变的阴性对照
# ---------------------------------------------------------------------------
def temperature_scaling_threshold(s_cal: torch.Tensor, alpha: float = 0.05):
    """温度缩放 s' = s/T。拟合 T 使缩放后校准集均值=0.5 (概率校准惯例)。
    由于阈值同样基于缩放分数 (split-CP), 单调性保证 FPR 与 SplitCP 完全相同。
    返回 (T, thr_on_scaled, eval需用缩放分数)。
    """
    T = float(s_cal.mean().clamp(min=1e-4))
    thr = split_conformal_threshold(s_cal / T, alpha)
    return T, thr


def quantile_reg_threshold(cal_scores, cal_y, cal_x, test_x, alpha: float = 0.05,
                           seed: int = 42, max_cal: int = 20000):
    """分位数回归 (评审 A5): 学习 q_{1-alpha}(x) 作为逐节点自适应阈值。
    用校准正常样本拟合 GradientBoosting(loss='quantile')。
    """
    from sklearn.ensemble import GradientBoostingRegressor
    cal_scores = cal_scores.cpu(); cal_y = cal_y.cpu(); cal_x = cal_x.cpu()
    normal = cal_y == 0
    X = cal_x[normal].numpy(); s = cal_scores[normal].numpy()
    if len(X) > max_cal:
        rng = np.random.RandomState(seed)
        sel = np.sort(rng.choice(len(X), max_cal, replace=False))
        X, s = X[sel], s[sel]
    q = 1.0 - alpha
    gb = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=40,
                                   max_depth=3, learning_rate=0.1, random_state=seed)
    gb.fit(X, s)
    thr_test = torch.as_tensor(gb.predict(test_x.numpy())).float()
    return thr_test, {"max_cal": len(X)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_one(dataset: str, seed: int, device: str, alpha: float):
    torch.manual_seed(seed); np.random.seed(seed)
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model, ckpt_path = load_frozen_detector(dataset, "homo", root=_ROOT, device=str(device))
    cal_m, test_m = data.cal_mask, data.test_mask
    y_cal = data.y[cal_m]
    y_test = data.y[test_m].cpu().numpy()

    # ---- 分数 (raw 与 TTA) ----
    s_raw = _softmax_scores(model, data.x, data.edge_index)
    auroc_f = float(roc_auc_score(y_test, s_raw[test_m].cpu().numpy()))
    flip = auroc_f < 0.5
    if flip:
        s_raw = 1.0 - s_raw
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta = _softmax_scores(model, x_enh, data.edge_index)
    if flip:
        s_tta = 1.0 - s_tta
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    # StructCP 与主表 Tab:main 同口径: 对比一致性 (公式6-7, 乘积门控 σ(c_local)·σ(c_global))。
    # 其余基线不依赖一致性 (Frozen/SplitCP-TTA/TempScaling/QuantReg 无权重, FeatWeight/RandWeight/KLIEP/Logistic 用特征/密度比权重)。
    cal_normal = cal_bool & (data.y == 0)
    cons = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_normal)

    cal_n = cal_bool & (data.y == 0)
    s_cal_raw_n = s_raw[cal_n]
    s_cal_tta_n = s_tta[cal_n]
    s_test_raw = s_raw[test_m]
    s_test_tta = s_tta[test_m]

    out = {}

    # 1) Frozen (SplitCP on raw)
    thr = split_conformal_threshold(s_cal_raw_n, alpha)
    out["Frozen"] = evaluate_coverage(s_test_raw, data.y[test_m], thr)

    # 2) SplitCP-TTA
    thr = split_conformal_threshold(s_cal_tta_n, alpha)
    out["SplitCP-TTA"] = evaluate_coverage(s_test_tta, data.y[test_m], thr)

    # 3) StructCP (fixed alpha=0.05, 与主表及所有基线同一标称水平)
    thr, info = structcp_threshold(
        s_tta[cal_m], y_cal, cons[cal_m], s_tta[test_m], cons[test_m],
        alpha=alpha, tau_quantile=0.5, score_quantile=0.75,
        use_calib_stat=True, weight_mode="contrastive",
    )
    ev = evaluate_coverage(s_test_tta, data.y[test_m], thr)
    ev["n_pseudo"] = int(info["n_pseudo_normal"])
    out["StructCP"] = ev

    # 4) FeatWeightCP (feature 密度比加权, 通用加权保形)
    thr, info = generic_weighted_threshold(
        s_tta[cal_m], y_cal, x_enh[cal_m], s_tta[test_m], x_enh[test_m],
        alpha=alpha, tau_quantile=0.5, score_quantile=0.75, weight_source="feature",
    )
    ev = evaluate_coverage(s_test_tta, data.y[test_m], thr)
    ev["n_pseudo"] = int(info["n_pseudo_normal"])
    out["FeatWeightCP"] = ev

    # 5) RandWeightCP (随机权重阴性对照)
    thr, info = generic_weighted_threshold(
        s_tta[cal_m], y_cal, x_enh[cal_m], s_tta[test_m], x_enh[test_m],
        alpha=alpha, tau_quantile=0.5, score_quantile=0.75, weight_source="random",
    )
    ev = evaluate_coverage(s_test_tta, data.y[test_m], thr)
    ev["n_pseudo"] = int(info["n_pseudo_normal"])
    out["RandWeightCP"] = ev

    # 6) KLIEP-CP
    feats_all = torch.cat([x_enh[cal_n].detach().cpu(), x_enh[test_m].detach().cpu()], 0)
    x_low, fit = _pca(feats_all, n_comp=min(16, feats_all.shape[1]))
    x_cal_low = _pca_transform(x_enh[cal_n].detach().cpu(), fit)
    x_test_low = _pca_transform(x_enh[test_m].detach().cpu(), fit)
    w_cal, w_test, kinfo = kliep_weights(x_cal_low, x_test_low, seed=seed, device=device)
    thr, info = dr_weighted_cp(s_tta[cal_m], y_cal, x_enh[cal_m], s_tta[test_m],
                               x_enh[test_m], w_cal, w_test, alpha=alpha)
    ev = evaluate_coverage(s_test_tta, data.y[test_m], thr)
    ev["n_pseudo"] = int(info["n_pseudo_normal"])
    out["KLIEP-CP"] = ev

    # 7) Logistic-CP
    w_cal, w_test, linfo = logistic_dr_weights(x_cal_low, x_test_low, seed=seed)
    thr, info = dr_weighted_cp(s_tta[cal_m], y_cal, x_enh[cal_m], s_tta[test_m],
                               x_enh[test_m], w_cal, w_test, alpha=alpha)
    ev = evaluate_coverage(s_test_tta, data.y[test_m], thr)
    ev["n_pseudo"] = int(info["n_pseudo_normal"])
    out["Logistic-CP"] = ev

    # 8) TempScaling (阴性对照: 单调变换)
    T, thr = temperature_scaling_threshold(s_cal_tta_n, alpha)
    ev = evaluate_coverage(s_test_tta / T, data.y[test_m], thr)
    ev["T"] = T
    out["TempScaling"] = ev

    # 9) QuantReg (逐节点自适应阈值)
    thr_test, qinfo = quantile_reg_threshold(s_tta[cal_m], y_cal, x_enh[cal_m].cpu(),
                                             x_enh[test_m].cpu(), alpha=alpha, seed=seed)
    pred = (s_test_tta.cpu() > thr_test).long()
    y_test_t = data.y[test_m].cpu().long()
    tp = int(((pred == 1) & (y_test_t == 1)).sum()); fp = int(((pred == 1) & (y_test_t == 0)).sum())
    fn = int(((pred == 0) & (y_test_t == 1)).sum()); tn = int(((pred == 0) & (y_test_t == 0)).sum())
    out["QuantReg"] = {"FPR": fp / max(fp + tn, 1), "TPR": tp / max(tp + fn, 1),
                       "F1": 2 * tp / max(2 * tp + fp + fn, 1)}

    # AUROC (分数级, 与阈值无关)
    out["_auroc_raw"] = float(roc_auc_score(y_test, s_test_raw.cpu().numpy()))
    out["_auroc_tta"] = float(roc_auc_score(y_test, s_test_tta.cpu().numpy()))
    out["_flip"] = flip
    return out


def aggregate(per_seed, methods):
    agg = {}
    for m in methods + ["_auroc_raw", "_auroc_tta"]:
        agg[m] = {}
        for met in ["FPR", "TPR", "F1"]:
            vals = [per_seed[s][m][met] for s in per_seed
                    if m in per_seed[s] and isinstance(per_seed[s][m], dict)
                    and met in per_seed[s][m]]
            if vals:
                arr = np.array(vals, dtype=float)
                agg[m][f"{met}_mean"] = float(arr.mean())
                agg[m][f"{met}_std"] = float(arr.std(ddof=0))
                agg[m][f"{met}_min"] = float(arr.min())
                agg[m][f"{met}_max"] = float(arr.max())
        agg[m]["per_seed"] = {int(s): {met: per_seed[s][m][met] for met in ["FPR", "TPR", "F1"]}
                              for s in per_seed if m in per_seed[s]
                              and isinstance(per_seed[s][m], dict)}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance")
    ap.add_argument("--seeds", default="42,123,456,789,1011")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="calibration_baselines.json")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    summary = {}
    for ds in datasets:
        per_seed = {}
        for seed in seeds:
            print(f"=== {ds} seed={seed} ===", flush=True)
            r = run_one(ds, seed, device, args.alpha)
            per_seed[seed] = r
            row = "  ".join(f"{m}={r.get(m, {}).get('FPR', float('nan')):.3f}"
                            for m in ["Frozen", "SplitCP-TTA", "StructCP", "FeatWeightCP",
                                      "RandWeightCP", "KLIEP-CP", "Logistic-CP", "QuantReg"])
            print(f"  FPR: {row}", flush=True)
        agg = aggregate(per_seed, METHODS)
        summary[ds] = {"alpha": args.alpha, "seeds": seeds, "agg": agg, "per_seed": per_seed}
        print(f"\n--- {ds} mean±std ---", flush=True)
        hdr = f"{'Method':<12}" + "".join(f"{m:>18}" for m in ["FPR", "TPR", "F1"])
        print(hdr, flush=True)
        for m in METHODS + ["_auroc_raw", "_auroc_tta"]:
            if m.startswith("_"):
                continue
            line = f"{m:<12}"
            for met in ["FPR", "TPR", "F1"]:
                if f"{met}_mean" in agg[m]:
                    line += f"{agg[m][met+'_mean']:.4f}±{agg[m][met+'_std']:.4f}".rjust(18)
                else:
                    line += f"{'':>18}"
            print(line, flush=True)

    out = os.path.join(_ROOT, "outputs", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
