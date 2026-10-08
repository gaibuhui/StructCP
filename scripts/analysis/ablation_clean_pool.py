# -*- coding: utf-8 -*-
"""
A3 去污染消融: 量化伪正常池污染对 FPR/TPR 的影响。

对照 (seed=42, fixed alpha=0.05, 4 主数据集):
  * as-is    = 默认池 (含被一致性门误纳的异常, 污染率见 ks_diagnostics.json)
  * clean    = clean_pool=True, 用真实标签剔除池内异常 (仅诊断, 非部署)

回答审稿人: "19.4% 池污染是否系统性扭曲阈值/TPR?"

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/ablation_clean_pool.py
"""
import io, json, os, sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation, contrastive_consistency  # noqa: E402
from adapters.conformal import structcp_threshold  # noqa: E402
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHA = 0.05
K, LAYERS, ALPHA_RES = 10, 2, 0.5
TAU, SQ = 0.5, 0.75


def main():
    from adapters.pipeline import get_scores
    results = {}
    for ds in DATASETS:
        try:
            torch.manual_seed(42); np.random.seed(42)
            dev = torch.device("cpu")
            data = load_gad(ds, relation="homo", seed=42).to(dev)
            model, _ = load_frozen_detector(ds, "homo", root=_ROOT, device="cpu")
            hg = KNNHypergraph(data.x, k=K).to(dev)
            x_enh = hypergraph_propagation(data.x, hg, LAYERS, ALPHA_RES)
            s_sp = get_scores(model, x_enh, data.edge_index)

            cal_bool = data.cal_mask.bool()
            test_bool = data.test_mask.bool()
            cal_normal = cal_bool & (data.y == 0)
            c_sp = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_normal)

            y_test = data.y[test_bool]
            row = {}
            for tag, clean in [("as_is", False), ("clean", True)]:
                t, info, inter = structcp_threshold(
                    s_sp[cal_bool], data.y[cal_bool], c_sp[cal_bool],
                    s_sp[test_bool], c_sp[test_bool],
                    alpha=ALPHA, tau_quantile=TAU, mode="hard", score_quantile=SQ,
                    use_calib_stat=True, weight_mode="contrastive",
                    return_intermediates=True, y_test=y_test, clean_pool=clean,
                )
                y_te = data.y[test_bool]
                s_te = s_sp[test_bool]
                fpr = float((s_te[y_te == 0] > t).float().mean())
                tpr = float((s_te[y_te == 1] > t).float().mean())
                row[tag] = {
                    "FPR": round(fpr, 4), "TPR": round(tpr, 4),
                    "n_pseudo": int(info.get("n_pseudo_normal") or 0),
                    "pseudo_anomaly_frac": info.get("pseudo_anomaly_frac"),
                    "threshold": float(t),
                }
                print(f"[{ds} {tag}] FPR={fpr:.4f} TPR={tpr:.4f} "
                      f"n_pseudo={row[tag]['n_pseudo']} poll={info.get('pseudo_anomaly_frac')} "
                      f"thr={float(t):.5f}", flush=True)
            # 污染影响
            dfpr = row["clean"]["FPR"] - row["as_is"]["FPR"]
            dtpr = row["clean"]["TPR"] - row["as_is"]["TPR"]
            row["delta"] = {"FPR_clean_minus_as_is": round(dfpr, 4),
                            "TPR_clean_minus_as_is": round(dtpr, 4)}
            results[ds] = row
        except Exception as e:
            print(f"{ds} ERROR: {e}")
            results[ds] = {"error": str(e)}

    out = os.path.join(_ROOT, "outputs", "ablation_clean_pool.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
