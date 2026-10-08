"""任务: AUTO no-guard 基线 (naive split conformal, 无结构加权守卫) 的 3-seed 稳定性复核.

对应主表 "StructCP w/o weighted CP" 行 —— 即关闭超图传播与一致性加权、
仅用 vanilla split conformal 做自动阈值 (auto-threshold) 的基线.
本脚本在 seed {42,123,456} 上聚合其 FPR / FPR_novel, 验证 naive split CP
在 neighborhood-shift 下 FPR_novel 失控是稳定的 (非单次偶然).

输出: outputs/noguard_baseline_3seed.json
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

import torch
import torch.nn.functional as F

sys.path.insert(0, _structcp_root())

from adapters.conformal import evaluate_coverage, split_conformal_threshold  # noqa: E402
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402

ROOT = _structcp_root()
SEEDS = [42, 123, 456]


def load_all(dataset, relation, device, seed):
    data = load_gad(dataset, relation=relation, seed=seed).to(device)
    ck = torch.load(os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth"),
                    map_location=device)
    a = ck["args"]
    model = BWGNN(ck["in_channels"], a["hidden"], 2, d=a["order"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.freeze()
    return data, model


@torch.no_grad()
def run_once(dataset, relation, alpha, device, seed):
    data, model = load_all(dataset, relation, device, seed)
    cal, te = data.cal_mask, data.test_mask
    y_cal, y_te = data.y[cal], data.y[te]
    nov = data.novel_mask & te
    s_raw = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    # naive split conformal (auto-threshold, no structural guard)
    t = split_conformal_threshold(s_raw[cal][y_cal == 0], alpha)
    r = evaluate_coverage(s_raw[te], y_te, t)
    fpr_novel = float((s_raw[nov] > t).float().mean()) if nov.any() else float("nan")
    return {"FPR": r["FPR"], "TPR": r["TPR"], "F1": r["F1"], "FPR_novel": fpr_novel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon", choices=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--relation", default="homo")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        per_seed.append(run_once(args.dataset, args.relation, args.alpha, device, seed))

    def agg(key):
        vals = [p[key] for p in per_seed if p[key] == p[key]]  # 跳过 nan
        if not vals:
            return None, None
        m = float(sum(vals) / len(vals))
        sd = float((sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5)
        return m, sd

    out = {
        "dataset": args.dataset, "relation": args.relation, "alpha": args.alpha,
        "seeds": SEEDS, "n": len(SEEDS),
        "base_no_HG_no_CP_weight": {
            "FPR_mean": agg("FPR")[0], "FPR_std": agg("FPR")[1],
            "TPR_mean": agg("TPR")[0], "TPR_std": agg("TPR")[1],
            "F1_mean": agg("F1")[0], "F1_std": agg("F1")[1],
            "FPR_novel_mean": agg("FPR_novel")[0], "FPR_novel_std": agg("FPR_novel")[1],
        },
        "per_seed": per_seed,
    }
    p = os.path.join(ROOT, "outputs", f"noguard_baseline_{args.dataset}_{args.relation}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    b = out["base_no_HG_no_CP_weight"]
    print(f"[{args.dataset}] naive split CP (AUTO no-guard) 3-seed:")
    print(f"  FPR       = {b['FPR_mean']:.4f} ± {b['FPR_std']:.4f}")
    print(f"  FPR_novel = {b['FPR_novel_mean']:.4f} ± {b['FPR_novel_std']:.4f}")
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
