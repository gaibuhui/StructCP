# -*- coding: utf-8 -*-
"""
SD (Stochastic Dominance) 验证: 把 FPR <= alpha + rho + eps 收紧为右尾 SD 下的
FPR <= alpha + eps_samp (rho 项消失)。

背景 (论文 Sec.method 4.6 升级版):
  设 t_hat 为加权池的加权 (1-alpha) 分位 (Eq. wquantile)。加权分位定义保证
      P_w(S > t_hat) <= alpha  (加权池上界)
  若右尾 SD 成立: 加权池得分分布在 t >= t_hat 区域随机占优于测试正常分布, 即
      sup_{t >= t_hat} (F_w(t) - F_T(t)) <= 0  <=>  F_w(t) <= F_T(t)  (t >= t_hat)
  则 FPR = P_T(S > t_hat) <= P_w(S > t_hat) <= alpha; 加上测试正常有限样本
  抽样误差, FPR <= alpha + eps_samp, eps_samp = O(ESS^{-1/2} + n_T^{-1/2})。

本脚本在 4 数据集 (seed 42, PPR 配置) 上:
  (1) 计算右尾 SD 统计量 D^-_tail = max_{t>=t_hat}(F_w - F_T), 并报告 SD 是否成立;
  (2) 计算右尾 KS (t >= t_hat 区域) 与全分布 KS 的对比 (预期右尾远小于全分布);
  (3) 验证收紧 bound: FPR <= alpha + 1/sqrt(ESS_w);
  (4) 对照原 bound: FPR <= alpha + KS + 1/sqrt(ESS_w)。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/sd_dominance_check.py
"""
import io, json, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import (KNNHypergraph, hypergraph_propagation,  # noqa: E402
                                     hyperedge_consistency)
from adapters.conformal import structcp_threshold  # noqa: E402
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHA = 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75


def weighted_ecdf(s_all, w, grid):
    w = w / w.sum()
    return np.array([w[s_all <= t].sum() for t in grid])


def ecdf(s, grid):
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

            x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
            from adapters.pipeline import get_scores
            s_sp = get_scores(model, x_enh, data.edge_index)

            cal_bool = data.cal_mask.bool()
            test_bool = data.test_mask.bool()
            cal_normal = cal_bool & (data.y == 0)
            c_sp = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)
            c_ppr = ppr_structure_score(
                x_enh, data.edge_index, cal_normal,
                pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
            )
            test_ppr_t = c_ppr[test_bool].float()

            t, info, inter = structcp_threshold(
                s_sp[cal_bool], data.y[cal_bool], c_sp[cal_bool],
                s_sp[test_bool], c_sp[test_bool],
                alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
                use_calib_stat=True, weight_mode="struct",
                return_intermediates=True,
                y_test=data.y[test_bool],
                test_ppr=test_ppr_t, ppr_quantile=0.4,
            )

            s_all = inter["s_all"].numpy()
            w = inter["w"].numpy()
            s_test_normal = s_sp[test_bool & (data.y == 0)].numpy()
            t_hat = float(t)

            # --- CDF 曲线 ---
            grid = np.unique(np.concatenate([s_all, s_test_normal, [t_hat]]))
            Fw = weighted_ecdf(s_all, w, grid)
            Ft = ecdf(s_test_normal, grid)

            d_full = Fw - Ft                      # 正 = 加权池在某点左移(违反右尾SD)
            ks_full = float(np.max(np.abs(d_full)))

            # 右尾区域: t >= t_hat
            tail_mask = grid >= t_hat - 1e-12
            d_tail = d_full[tail_mask]
            dmin_tail = float(np.max(d_tail)) if len(d_tail) else None   # <=0 => 右尾SD成立
            ks_tail = float(np.max(np.abs(d_tail))) if len(d_tail) else None

            # 全分布方向统计 (辅助): 最大违反
            dmin_all = float(np.max(d_full))

            ess_w = float(info.get("w_ess") or (w.sum() ** 2 / (w ** 2).sum()))
            n_T = int(len(s_test_normal))
            fpr = float((s_test_normal > t_hat).mean())
            eps = 1.0 / np.sqrt(ess_w)
            bound_sd = ALPHA + eps                      # SD 收紧 bound
            bound_orig = ALPHA + ks_full + eps          # 原 bound

            results[ds] = {
                "alpha": ALPHA,
                "t_hat": t_hat,
                "n_test_normal": n_T,
                "ESS_w": ess_w,
                "eps_samp": eps,
                "KS_full": ks_full,
                "KS_tail": ks_tail,
                "Dmin_tail": dmin_tail,        # max_{t>=t_hat}(F_w-F_T), <=tol => 右尾SD成立
                "tail_SD_holds": bool(dmin_tail is not None and dmin_tail <= 1e-4),
                "Dmin_all": dmin_all,
                "FPR_seed42": fpr,
                "bound_sd_alpha_eps": float(bound_sd),
                "sd_bound_holds": bool(bound_sd >= fpr),
                "bound_orig_alpha_KS_eps": float(bound_orig),
                "orig_bound_holds": bool(bound_orig >= fpr),
                "pseudo_anomaly_frac": inter.get("pseudo_anomaly_frac"),
            }
            print(f"{ds:<10} t_hat={t_hat:.4f} KS_full={ks_full:.4f} KS_tail={ks_tail:.4f} "
                  f"D^-_tail={dmin_tail:+.4f} SD_tail={'OK' if (dmin_tail or 1) <= 1e-4 else 'VIOL'} "
                  f"FPR={fpr:.4f}  alpha+eps={bound_sd:.4f} "
                  f"SD_bound={'OK' if bound_sd >= fpr else 'BREACH'}  "
                  f"alpha+KS+eps={bound_orig:.4f}  poll={inter.get('pseudo_anomaly_frac')}")
        except Exception as e:
            print(f"{ds} ERROR: {e}")
            results[ds] = {"error": str(e)}

    out = os.path.join(_ROOT, "outputs", "sd_dominance_check.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
