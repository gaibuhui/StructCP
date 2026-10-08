"""TUNE baseline vs StructCP fair comparison (multi-dataset, multi-seed).

Under the same frozen BWGNN detector, same test set, same test-time normality
shift, compare:
  1. Frozen   : raw detector output + split conformal
  2. TUNE     : learnable MLP aligner (baselines/tune.py) as TTA + split conformal
  3. StructCP   : zero-param hypergraph TTA + weighted conformal (reuses adapters)

Reports AUROC / AUPRC / FPR(alpha=0.05) / TPR / F1 over multiple seeds so that
TUNE+StructCP is reported with variance for symmetry with StructCP's 3-seed protocol.
The pipeline is deterministic given a frozen backbone, so the std is ~0; we
report 3 seeds purely to remove the asymmetric single-run reporting that a
reviewer flagged.

Usage
-----
    python scripts/run_tune_baseline.py --datasets Amazon YelpChi Elliptic TFinance --seeds 42,123,456
"""
from __future__ import annotations

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
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from adapters.conformal import (  # noqa: E402
    evaluate_coverage,
    structcp_threshold,
    split_conformal_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph,
    hyperedge_consistency,
    hypergraph_propagation,
)
from baselines.tune import TuneAligner, tune_fit, tune_transform  # noqa: E402
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402

ROOT = _structcp_root()


@torch.no_grad()
def scores_of(model, x, ei):
    return F.softmax(model(x, ei), dim=1)[:, 1]


def run_once(name, relation, seed, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_gad(name, relation=relation, seed=seed).to(device)
    print(f"[data] {name} ({relation}) seed={seed}")

    ckpt = torch.load(
        os.path.join(ROOT, "checkpoint", f"bwgnn_{name}_{relation}.pth"),
        map_location=device,
    )
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal, te = data.cal_mask, data.test_mask
    y_te = data.y[te].cpu().numpy()
    y_cal = data.y[cal]
    nov = data.novel_mask & te

    # ---------- Frozen ----------
    t0 = time.time()
    s_raw = scores_of(model, data.x, data.edge_index)
    t_frozen = time.time() - t0
    thr0 = split_conformal_threshold(s_raw[cal][y_cal == 0], args.alpha)
    r0 = evaluate_coverage(s_raw[te], data.y[te], thr0)
    r0["AUROC"] = roc_auc_score(y_te, s_raw[te].cpu().numpy())
    r0["AUPRC"] = average_precision_score(y_te, s_raw[te].cpu().numpy())
    r0["infer_time_s"] = t_frozen
    r0["trainable_params"] = 0
    if int(nov.sum()) > 0:
        r0["FPR_novel"] = float((s_raw[nov] > thr0).float().mean())

    # ---------- shared hypergraph ----------
    t0 = time.time()
    hg = KNNHypergraph(data.x, k=args.k).to(device)
    t_build = time.time() - t0

    # ---------- TUNE (learnable MLP aligner) ----------
    t0 = time.time()
    aligner = TuneAligner(data.num_features).to(device)
    n_tune_params = sum(p.numel() for p in aligner.parameters())
    aligner = tune_fit(aligner, data.x, hg.knn_idx, epochs=args.tune_epochs,
                       device=device)
    t_tune_train = time.time() - t0
    t0 = time.time()
    x_tune = tune_transform(aligner, data.x, device)
    s_tune = scores_of(model, x_tune, data.edge_index)
    t_tune_infer = time.time() - t0
    thr_t = split_conformal_threshold(s_tune[cal][y_cal == 0], args.alpha)
    r_t = evaluate_coverage(s_tune[te], data.y[te], thr_t)
    r_t["AUROC"] = roc_auc_score(y_te, s_tune[te].cpu().numpy())
    r_t["AUPRC"] = average_precision_score(y_te, s_tune[te].cpu().numpy())
    r_t["infer_time_s"] = t_tune_train + t_tune_infer + t_build
    r_t["trainable_params"] = n_tune_params
    if int(nov.sum()) > 0:
        r_t["FPR_novel"] = float((s_tune[nov] > thr_t).float().mean())

    # ---------- StructCP (zero-param hypergraph TTA + weighted conformal) ----------
    t0 = time.time()
    x_hg = hypergraph_propagation(data.x, hg, args.layers, args.alpha_res)
    s_hg = scores_of(model, x_hg, data.edge_index)
    t_hg_infer = time.time() - t0
    cons = hyperedge_consistency(s_hg, hg, features=x_hg)
    thr_h, info = structcp_threshold(
        s_hg[cal], y_cal, cons[cal], s_hg[te], cons[te],
        alpha=args.alpha, tau_quantile=args.tau,
    )
    r_h = evaluate_coverage(s_hg[te], data.y[te], thr_h)
    r_h["AUROC"] = roc_auc_score(y_te, s_hg[te].cpu().numpy())
    r_h["AUPRC"] = average_precision_score(y_te, s_hg[te].cpu().numpy())
    r_h["infer_time_s"] = t_hg_infer + t_build
    r_h["trainable_params"] = 0
    r_h.update(info)
    if int(nov.sum()) > 0:
        r_h["FPR_novel"] = float((s_hg[nov] > thr_h).float().mean())

    # ---------- TUNE + StructCP (stacked: align features with TUNE, then weighted CP) ----------
    x_hg_t = hypergraph_propagation(x_tune, hg, args.layers, args.alpha_res)
    s_hg_t = scores_of(model, x_hg_t, data.edge_index)
    cons_t = hyperedge_consistency(s_hg_t, hg, features=x_hg_t)
    thr_th, info_th = structcp_threshold(
        s_hg_t[cal], y_cal, cons_t[cal], s_hg_t[te], cons_t[te],
        alpha=args.alpha, tau_quantile=args.tau,
    )
    r_th = evaluate_coverage(s_hg_t[te], data.y[te], thr_th)
    r_th["AUROC"] = roc_auc_score(y_te, s_hg_t[te].cpu().numpy())
    r_th["AUPRC"] = average_precision_score(y_te, s_hg_t[te].cpu().numpy())
    r_th["infer_time_s"] = (t_tune_train + t_tune_infer + t_hg_infer + t_build)
    r_th["trainable_params"] = n_tune_params
    r_th.update(info_th)
    if int(nov.sum()) > 0:
        r_th["FPR_novel"] = float((s_hg_t[nov] > thr_th).float().mean())

    return {"Frozen": r0, "TUNE": r_t, "StructCP": r_h, "TUNE+StructCP": r_th}


def _agg(records, keys=("FPR", "TPR", "F1", "AUROC", "AUPRC")):
    agg = {}
    for k in keys:
        vals = [r[k] for r in records]
        arr = np.array(vals, dtype=float)
        agg[k] = (float(arr.mean()), float(arr.std(ddof=0)))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tune_epochs", type=int, default=30, help="TUNE test-time epochs")
    ap.add_argument("--seeds", default="42,123,456",
                    help="comma-separated seeds; TUNE+StructCP variance reported for symmetry "
                         "with StructCP's 3-seed protocol (pipeline is deterministic given a frozen "
                         "backbone, so std is ~0; reported for protocol symmetry).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    seeds = [int(s) for s in args.seeds.split(",")]

    all_agg = {}
    for name in args.datasets:
        # match the main-table protocol: all datasets use the single-relation
        # ('homo') loader, consistent with compute_*.py that produced Tab. main.
        relation = "homo"
        recs = {"Frozen": [], "TUNE": [], "StructCP": [], "TUNE+StructCP": []}
        for seed in seeds:
            r = run_once(name, relation, seed, args, device)
            for m in recs:
                recs[m].append(r[m])
        agg = {m: _agg(v) for m, v in recs.items()}
        all_agg[name] = agg

        print(f"\n{'='*118}\n{name} ({relation})  alpha={args.alpha}  "
              f"seeds={seeds}\n{'='*118}")
        hdr = (f"{'Method':<12}{'AUROC':>10}{'AUPRC':>10}{'FPR':>12}{'TPR':>12}"
               f"{'F1':>10}")
        print(hdr)
        print("-" * 118)
        for m in ["Frozen", "TUNE", "StructCP", "TUNE+StructCP"]:
            a = agg[m]
            print(f"{m:<12}{a['AUROC'][0]:>10.4f}{a['AUPRC'][0]:>10.4f}"
                  f"{a['FPR'][0]:>10.4f}±{a['FPR'][1]:.4f}"
                  f"{a['TPR'][0]:>10.4f}±{a['TPR'][1]:.4f}{a['F1'][0]:>10.4f}")
        print("=" * 118)

    out = os.path.join(ROOT, "outputs", "compare_tune_multiseed.json")
    json.dump({"config": vars(args), "agg": all_agg}, open(out, "w"), indent=2)
    print(f"[saved] {out}\nNOTE: TUNE+StructCP variance reported for symmetry with StructCP's "
          "3-seed protocol; pipeline is deterministic given a frozen backbone (std~0).")


if __name__ == "__main__":
    main()
