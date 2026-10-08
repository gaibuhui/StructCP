# -*- coding: utf-8 -*-
"""
验证 label-free monitor (Eq. sd_labelfree) 在 q_max 扫描下的判定稳定性。
对 4 数据集 (seed 42, PPR 配置): 计算 F_w(t) 与 (F_T_all(t) - q_max)/(1-q_max) 的
关系, 报告 monitor 是否通过 (F_w <= 阈值 对所有 t>=t_hat)。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/sd_labelfree_monitor.py
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
QMAX = [0.05, 0.10, 0.15]


def main():
    results = {}
    for ds in DATASETS:
        torch.manual_seed(42); np.random.seed(42)
        dev = torch.device("cpu")
        data = load_gad(ds, relation="homo", seed=42).to(dev)
        model, _ = load_frozen_detector(ds, "homo", root=_ROOT, device="cpu")
        hg = KNNHypergraph(data.x, k=K).to(dev)
        x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
        from adapters.pipeline import get_scores
        s_sp = get_scores(model, x_enh, data.edge_index)
        cal_bool = data.cal_mask.bool(); test_bool = data.test_mask.bool()
        cal_normal = cal_bool & (data.y == 0)
        c_sp = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)
        c_ppr = ppr_structure_score(x_enh, data.edge_index, cal_normal,
                                    pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60)
        test_ppr_t = c_ppr[test_bool].float()
        t, info, inter = structcp_threshold(
            s_sp[cal_bool], data.y[cal_bool], c_sp[cal_bool],
            s_sp[test_bool], c_sp[test_bool],
            alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
            use_calib_stat=True, weight_mode="struct", return_intermediates=True,
            y_test=data.y[test_bool], test_ppr=test_ppr_t, ppr_quantile=0.4)
        s_all = inter["s_all"].numpy(); w = inter["w"].numpy()
        s_test_all = s_sp[test_bool].numpy()   # 全测试 (含异常) —— monitor 只用这个
        t_hat = float(t)

        def wcdf(grid):
            ww = w / w.sum()
            return np.array([ww[s_all <= x].sum() for x in grid])

        grid = np.unique(np.concatenate([s_all, s_test_all, [t_hat]]))
        Fw = wcdf(grid)
        # F_T_all 经验 CDF (无标签)
        Ft_all = np.array([(s_test_all <= x).mean() for x in grid])
        tail = grid >= t_hat - 1e-12
        row = {}
        for q in QMAX:
            thr_cdf = np.clip((Ft_all - q) / (1 - q), 0, 1)
            viol = float(np.max(Fw[tail] - thr_cdf[tail]))  # <=0 => monitor 通过
            row[str(q)] = {"max_violation": viol, "monitor_passes": bool(viol <= 1e-9)}
        results[ds] = {"t_hat": t_hat, "qmax": row}
        print(f"{ds:<9} " + "  ".join(f"q={q}: pass={row[str(q)]['monitor_passes']} "
              f"(viol={row[str(q)]['max_violation']:+.4f})" for q in QMAX))

    out = os.path.join(_ROOT, "outputs", "sd_labelfree_monitor.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps({"alpha": ALPHA, "qmax_sweep": QMAX, "datasets": results},
                   indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
