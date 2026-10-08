"""模块1 验证: 条件性 FPR 偏差上界 δ/2 + 1/(n+1) 的经验检验.

理论界 (Tibshirani et al. 2024, weighted conformal):
    |FPR - (1-α)| <= delta/2 + 1/(n+1)
其中 delta = Σ|w_true - ŵ| / Σ w_true 为权重估计归一化误差.

本脚本在合成设定下, 给定权重误差 δ 的多个水平, 蒙特卡洛估计实际 FPR 偏差,
并核对其是否被上界覆盖. 目的: 为论文 Sec.3 的条件性偏差界提供轻量实证,
而非训练模型 (StructCP 本身不要求此脚本运行).
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
import numpy as np

import torch
from adapters.conformal import weighted_fpr_bias_bound


def simulate(n: int, alpha: float, delta: float, n_trials: int = 2000):
    """在已知真实权重下, 用有界误差的估计权重做加权分位, 测实际 FPR 偏差."""
    rng = np.random.default_rng(0)
    errs = []
    bound = None
    for _ in range(n_trials):
        w_true = torch.from_numpy(rng.gamma(2.0, 1.0, size=n)).float()
        # 估计权重: 在真实权重上注入归一化误差 delta 的乘性扰动
        noise = torch.from_numpy(rng.uniform(1 - delta, 1 + delta, size=n)).float()
        w_hat = (w_true * noise).clamp(min=1e-3)
        # 合成 score: 正常类 s~U, 异常类 s 更大; 正常类占比 1-α 附近
        # 用估计权重做加权 (1-α) 分位阈值
        s = torch.from_numpy(rng.uniform(0, 1, size=n)).float()
        # 观测: 分数 < 阈值的加权比例
        thr = weighted_quantile_threshold(s, w_hat, alpha)
        fpr = float((w_true[s < thr].sum()) / w_true.sum())
        errs.append(abs(fpr - (1 - alpha)))
        if bound is None:
            bound = weighted_fpr_bias_bound(w_true, w_hat, n, alpha)
    return float(np.mean(errs)), float(np.max(errs)), bound


def weighted_quantile_threshold(s: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    order = torch.argsort(s)
    s_sorted, w_sorted = s[order], w[order]
    cw = torch.cumsum(w_sorted, dim=0)
    target = (1 - alpha) * cw[-1]
    idx = torch.searchsorted(cw, target)
    idx = min(int(idx), len(s) - 1)
    return s_sorted[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--deltas", type=str, default="0.05,0.15,0.30,0.50")
    ap.add_argument("--trials", type=int, default=1500)
    args = ap.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]
    print(f"{'delta':>6} | {'mean|FPR-(1-a)|':>16} | {'max':>8} | {'bound':>8} | covered")
    print("-" * 60)
    for d in deltas:
        mean_err, max_err, bound = simulate(args.n, args.alpha, d, args.trials)
        covered = "YES" if max_err <= bound + 1e-9 else "NO"
        print(f"{d:>6.2f} | {mean_err:>16.4f} | {max_err:>8.4f} | {bound:>8.4f} | {covered}")


if __name__ == "__main__":
    main()
