# -*- coding: utf-8 -*-
"""
真实 KS 诊断：验证理论分解 FPR <= alpha + rho_shift + eps_samp。

rho_shift 用真实 Kolmogorov-Smirnov 距离：
    KS = sup_t | F_w(t) - F_test_normal(t) |
其中 F_w 是加权池（校准正常 + 伪正常，权重 w_i）的加权经验 CDF，
F_test_normal 是测试正常分数的经验 CDF。

验证：FPR(seed=42, fixed alpha) <= alpha + KS + 1/sqrt(ESS_w)。

用法（fov 环境）:
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/compute_ks_diagnostics.py
"""
import io, json, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import (KNNHypergraph, hypergraph_propagation,  # noqa: E402
                                     contrastive_consistency, hyperedge_consistency)
from adapters.conformal import structcp_threshold  # noqa: E402
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHA = 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75


def weighted_ecdf(s_all: np.ndarray, w: np.ndarray, grid: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    # F_w(t) = sum_i w_i * 1[s_i <= t]
    return np.array([w[s_all <= t].sum() for t in grid])


def ecdf(s: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([(s <= t).mean() for t in grid])


def main():
    results = {}
    for ds in DATASETS:
        try:
            torch.manual_seed(42); np.random.seed(42)
            dev = torch.device("cpu")
            data = load_gad(ds, relation="homo", seed=42).to(dev)
            model, _ = load_frozen_detector(ds, "homo", root=_ROOT, device="cpu")
            hg = KNNHypergraph(data.x, k=K).to(dev)

            # --- StructCP 链路: 传播 -> 打分 -> 一致性 -> PPR gate -> 加权阈值(带中间量) ---
            x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
            from adapters.pipeline import get_scores
            s_sp = get_scores(model, x_enh, data.edge_index)

            cal_bool = data.cal_mask.bool()
            test_bool = data.test_mask.bool()
            cal_normal = cal_bool & (data.y == 0)
            # PPR 模式: 权重 = hyperedge 一致性; PPR 分数作伪正常准入门槛 (与 pipeline 一致)
            c_sp = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)
            c_ppr = ppr_structure_score(
                x_enh, data.edge_index, cal_normal,
                pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
            )
            # c_ppr 已是 tensor (与 x_enh 同 device); 直接按 test mask 取子集
            test_ppr_t = c_ppr[test_bool].float()

            t, info, inter = structcp_threshold(
                s_sp[cal_bool], data.y[cal_bool], c_sp[cal_bool],
                s_sp[test_bool], c_sp[test_bool],
                alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
                use_calib_stat=True, weight_mode="struct",  # 与 pipeline 的 thr_weight_mode 一致
                return_intermediates=True,
                y_test=data.y[test_bool],
                test_ppr=test_ppr_t, ppr_quantile=0.4,
            )

            s_all = inter["s_all"].numpy()
            w = inter["w"].numpy()
            s_test_normal = s_sp[test_bool & (data.y == 0)].numpy()

            # --- 真实 KS ---
            grid = np.unique(np.concatenate([s_all, s_test_normal]))
            Fw = weighted_ecdf(s_all, w, grid)
            Ft = ecdf(s_test_normal, grid)
            ks = float(np.max(np.abs(Fw - Ft)))

            ess_w = float(info.get("w_ess") or (w.sum() ** 2 / (w ** 2).sum()))
            fpr = float((s_test_normal > float(t)).mean())
            bound = ALPHA + ks + 1.0 / np.sqrt(ess_w)

            results[ds] = {
                "alpha": ALPHA, "n_pseudo": int(info.get("n_pseudo_normal") or 0),
                "n_cal_normal": int(info.get("n_cal_normal") or 0),
                "n_test_normal": int(len(s_test_normal)),
                "ESS_w": float(ess_w), "KS": ks,
                "FPR_seed42": fpr, "bound_alpha_KS_eps": float(bound),
                "bound_holds": bool(bound >= fpr),
                "pseudo_anomaly_frac": inter.get("pseudo_anomaly_frac"),
            }
            print(f"{ds:<10} KS={ks:.4f}  ESS_w={ess_w:.0f}  eps={1/np.sqrt(ess_w):.4f}  "
                  f"FPR={fpr:.4f}  bound={bound:.4f}  holds={'OK' if bound >= fpr else 'BREACH'}  "
                  f"poll={inter.get('pseudo_anomaly_frac')}")
        except Exception as e:
            print(f"{ds} ERROR: {e}")
            results[ds] = {"error": str(e)}

    out = os.path.join(_ROOT, "outputs", "ks_diagnostics.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
