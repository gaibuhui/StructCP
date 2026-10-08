# -*- coding: utf-8 -*-
"""
④ 伪正常池污染率压力测试 (回应审稿人 Major 3 / 问题 4).

对 4 数据集 (seed 42, 主配置 PPR+dual 的阈值构造) 人为控制伪正常池的
异常占比 r ∈ {0, 0.05, 0.10, 0.20, 0.30}:
  - r < 自然污染率: 从伪正常池移除部分真异常;
  - r > 自然污染率: 从测试异常补充真异常入池 (权重 = 一致性×0.5, 与伪正常口径一致)。
重算加权 (1-alpha) 分位阈值, 报告 FPR / TPR, 检验污染是否导致 FPR breach。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/pool_contamination_stress.py
"""
import io, json, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import (KNNHypergraph, hypergraph_propagation,  # noqa: E402
                                     hyperedge_consistency)
from adapters.conformal import structcp_threshold, weighted_quantile  # noqa: E402
from adapters.ppr_weight import ppr_structure_score  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHA = 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75
RATES = [0.0, 0.05, 0.10, 0.20, 0.30]


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

        cal_bool = data.cal_mask.bool()
        test_bool = data.test_mask.bool()
        cal_normal = cal_bool & (data.y == 0)
        c_sp = hyperedge_consistency(s_sp, hg, features=x_enh, cal_mask=cal_bool)
        c_ppr = ppr_structure_score(
            x_enh, data.edge_index, cal_normal,
            pca_dim=16, density_k=15, h=1.0, restart=0.15, K=60,
        )
        test_ppr_t = c_ppr[test_bool].float()
        y_cal = data.y[cal_bool]
        thr, info, inter = structcp_threshold(
            s_sp[cal_bool], y_cal, c_sp[cal_bool],
            s_sp[test_bool], c_sp[test_bool],
            alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
            use_calib_stat=True, weight_mode="struct",
            return_intermediates=True,
            y_test=data.y[test_bool],
            test_ppr=test_ppr_t, ppr_quantile=0.4,
        )

        n_cal = int(info["n_cal_normal"])
        s_all = inter["s_all"]; w = inter["w"]
        y_tf = data.y[test_bool]
        s_test = s_sp[test_bool]
        s_ps = s_all[n_cal:]          # 伪正常段分数
        w_ps = w[n_cal:]              # 伪正常段权重
        pidx = inter["pseudo_idx"]    # 伪正常对应的测试索引
        if pidx is None or pidx.numel() == 0:
            print(f"{ds}: no pseudo-normal pool; skip"); results[ds] = {"error": "no pool"}; continue
        ps_lab = y_tf[pidx]           # 伪正常段标签
        natural = float(ps_lab.float().mean())

        cal_s = s_all[:n_cal]; cal_w = w[:n_cal]
        norm_ps_mask = ps_lab == 0
        anom_ps_mask = ps_lab == 1
        s_norm_ps = s_ps[norm_ps_mask]; w_norm_ps = w_ps[norm_ps_mask]
        s_anom_ps = s_ps[anom_ps_mask]; w_anom_ps = w_ps[anom_ps_mask]

        # 可补充的测试异常候选 (不在池内的真异常测试节点)
        in_pool = torch.zeros_like(y_tf, dtype=torch.bool)
        in_pool[pidx] = True
        cand = (~in_pool) & (y_tf == 1)
        cand_idx = torch.where(cand)[0]
        s_cand = s_test[cand_idx]; c_cand = c_sp[test_bool][cand_idx]

        def make_pool(r):
            # 目标: 伪正常段异常占比 = r
            n_norm = int(s_norm_ps.numel())
            if n_norm == 0:
                return None
            n_anom_tgt = int(round(r * n_norm / max(1 - r, 1e-9)))
            # 正常段不变
            s_new = [s_norm_ps]; w_new = [w_norm_ps]
            n_anom_used = 0
            if n_anom_tgt > 0 and s_anom_ps.numel() > 0:
                take = min(n_anom_tgt, int(s_anom_ps.numel()))
                s_new.append(s_anom_ps[:take]); w_new.append(w_anom_ps[:take])
                n_anom_used += take
                n_anom_tgt -= take
            # 不够则从候选补充
            if n_anom_tgt > 0 and cand_idx.numel() > 0:
                take = min(n_anom_tgt, int(cand_idx.numel()))
                s_new.append(s_cand[:take])
                w_new.append((c_cand[:take] + 1e-8) * 0.5)  # 与伪正常折扣一致
                n_anom_used += take
            s_all2 = torch.cat([cal_s] + s_new)
            w_all2 = torch.cat([cal_w] + w_new)
            n2 = s_all2.numel()
            level = min(1.0, (1 - ALPHA) * (n2 + 1) / n2)
            t2 = weighted_quantile(s_all2, w_all2, level)
            fpr = float((s_test[y_tf == 0] > t2).float().mean())
            tpr = float((s_test[y_tf == 1] > t2).float().mean())
            n_pseudo2 = n2 - n_cal
            return {"target_pool_anom_frac": r,
                    "realized_pool_anom_frac": n_anom_used / max(n_pseudo2, 1),
                    "FPR": fpr, "TPR": tpr, "FPR_le_alpha": bool(fpr <= ALPHA + 1e-9)}

        rows = []
        for r in RATES:
            row = make_pool(r)
            # 计算实际池异常占比(诊断)
            rows.append(row)
            print(f"  {ds} r={r:.2f} FPR={row['FPR']:.4f} TPR={row['TPR']:.4f} "
                  f"FPR<=alpha={row['FPR_le_alpha']}")

        results[ds] = {"natural_pool_anomaly_frac": natural, "rows": rows}

    out = os.path.join(_ROOT, "outputs", "pool_contamination_stress.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps({"alpha": ALPHA, "rates": RATES, "datasets": results},
                   indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
