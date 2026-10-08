# -*- coding: utf-8 -*-
"""
实验 D-2：在 TUNE 官方 few-shot/unseen-normal 协议下评估 Frozen / StructCP。

数据：TUNE DGL 图自带 train_mask / val_mask / test_mask / newnormal_masks。
协议（与 TUNE AAAI'26 一致）：
  * 校准池 = train_mask 中的正常节点（源域正常，few-shot）
  * 测试   = test_mask 节点（含 newnormal 新正常 + 异常）
  * FPR    = 测试正常（含 newnormal）被判定为异常的比例
  * TPR    = 测试异常被正确判出的比例

用法（fov 环境）:
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/tune_protocol_structcp.py --datasets reddit,photo --alpha 0.05
"""
import argparse, io, json, os, sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.loader import GADData, TUNE_GRAPH_ROOT  # noqa: E402
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation, contrastive_consistency  # noqa: E402
from adapters.conformal import structcp_threshold, split_conformal_threshold  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402


def load_tune_graph(name):
    import dgl
    from dgl.data.utils import load_graphs
    p = os.path.join(TUNE_GRAPH_ROOT, name)
    g = load_graphs(p)[0][0]
    feat = g.ndata["feature"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64).ravel()
    src, dst = g.edges()
    ei = np.vstack([src.numpy(), dst.numpy()]).astype(np.int64)
    ei = np.hstack([ei, np.flipud(ei)])
    ei_sorted = np.sort(ei, axis=0)
    _, uniq = np.unique(ei_sorted, axis=1, return_index=True)
    edge_index = ei[:, np.sort(uniq)]
    n = len(y)
    # 掩码（对齐到 n）
    def _mask(key):
        m = g.ndata.get(key)
        if m is None:
            return np.zeros(n, dtype=bool)
        m = m.numpy()
        if m.ndim == 2:
            m = m[:, 0]
        return m.astype(bool)
    train_mask = _mask("train_mask")
    test_mask = _mask("test_mask")
    novel_mask = _mask("newnormal_masks")
    return GADData(
        torch.tensor(feat), torch.tensor(edge_index), torch.tensor(y),
        torch.tensor(train_mask), torch.tensor(train_mask & (y == 0)),
        torch.tensor(test_mask), torch.tensor(novel_mask), name,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="reddit,photo")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from adapters.pipeline import get_scores, run_comparison, RunConfig, DEFAULT_DATASET_CFG

    results = {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        try:
            data = load_tune_graph(ds).to(args.device)
            model, _ = load_frozen_detector(ds, "homo", root=_ROOT, device=args.device)
            hg = KNNHypergraph(data.x, k=10).to(args.device)
            x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
            s_sp = get_scores(model, x_enh, data.edge_index)

            cal_bool = data.cal_mask.bool()
            test_bool = data.test_mask.bool()
            if int(cal_bool.sum()) < 20 or int(test_bool.sum()) < 20:
                print(f"[{ds}] cal/test too small ({int(cal_bool.sum())}/{int(test_bool.sum())}), skip")
                continue
            cal_normal = cal_bool & (data.y == 0)
            c_sp = contrastive_consistency(x_enh, hg, cal_normal_mask=cal_normal)

            # --- Frozen: split conformal on calibration-normal scores ---
            thr_frozen = split_conformal_threshold(s_sp[cal_bool], args.alpha)
            y_te = data.y[test_bool].cpu().numpy()
            s_te = s_sp[test_bool].cpu().numpy()
            fpr_frozen = float((s_te[y_te == 0] > float(thr_frozen)).mean())
            tpr_frozen = float((s_te[y_te == 1] > float(thr_frozen)).mean())

            # --- StructCP: weighted quantile + pseudo-normal ---
            t_s, info_s, inter = structcp_threshold(
                s_sp[cal_bool], data.y[cal_bool], c_sp[cal_bool],
                s_sp[test_bool], c_sp[test_bool],
                alpha=args.alpha, tau_quantile=0.5, mode="hard", score_quantile=0.75,
                use_calib_stat=True, weight_mode="contrastive",
                return_intermediates=True, y_test=data.y[test_bool],
            )
            thr_scp = float(t_s)
            fpr_scp = float((s_te[y_te == 0] > thr_scp).mean())
            tpr_scp = float((s_te[y_te == 1] > thr_scp).mean())

            # 分 novel-normal 的 FPR（TUNE 关心的新正常误报）
            novel_in_test = data.novel_mask[test_bool].cpu().numpy()
            if novel_in_test.any():
                fpr_scp_novel = float((s_te[novel_in_test] > thr_scp).mean())
                fpr_frozen_novel = float((s_te[novel_in_test] > float(thr_frozen)).mean())
            else:
                fpr_scp_novel = fpr_frozen_novel = float("nan")

            results[ds] = {
                "alpha": args.alpha,
                "n_cal_normal": int(cal_normal.sum()),
                "n_test": int(test_bool.sum()),
                "n_novel_in_test": int(novel_in_test.sum()),
                "Frozen": {"FPR": round(fpr_frozen, 4), "TPR": round(tpr_frozen, 4),
                           "FPR_novel": round(fpr_frozen_novel, 4)},
                "StructCP": {"FPR": round(fpr_scp, 4), "TPR": round(tpr_scp, 4),
                             "FPR_novel": round(fpr_scp_novel, 4),
                             "n_pseudo": int(info_s.get("n_pseudo_normal") or 0)},
            }
            print(f"[{ds}] Frozen FPR={fpr_frozen:.4f} TPR={tpr_frozen:.4f} | "
                  f"StructCP FPR={fpr_scp:.4f} TPR={tpr_scp:.4f} "
                  f"(novel-FPR {fpr_scp_novel:.4f}) npseudo={info_s.get('n_pseudo_normal')}",
                  flush=True)
        except Exception as e:
            print(f"[{ds}] ERROR: {e}", flush=True)
            results[ds] = {"error": str(e)}

    out = os.path.join(_ROOT, "outputs", "tune_protocol_structcp.json")
    io.open(out, "w", encoding="utf-8").write(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
