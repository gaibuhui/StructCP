"""Aggregate multi-seed StructCP results for datasets lacking a seed sweep (Amazon, YelpChi).

重构说明（2026-08-23）：不再 monkeypatch 旧 `run_structcp.main()`，改为直接调用共享管线
`adapters.pipeline.evaluate_structcp`（单 seed 单数据集端到端推理），再聚合
Frozen / StructCP 的 FPR & TPR 的 mean±std。输出结构 `outputs/structcp_seeds_amazon_yelpchi.json`
与重构前一致。
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

import json
import os
import sys
from pathlib import Path

from adapters.pipeline import evaluate_structcp
from utils.results import save_json

ROOT = Path(_structcp_root())


def run_one(dataset, seed, alpha, score_quantile, device):
    """单 seed 运行，返回 {method: metrics}（共享管线 evaluate_structcp）。"""
    out = evaluate_structcp(
        dataset, seed=seed, alpha=alpha, device=device, root=str(ROOT),
        mode="hard", score_quantile=score_quantile, tau=0.5,
        k=10, layers=2, alpha_res=0.5,
    )
    return out["results"]


def aggregate(dataset, seeds, alpha, score_quantile, device):
    keys = [("Frozen", "FPR"), ("Frozen", "TPR"), ("StructCP", "FPR"), ("StructCP", "TPR")]
    acc = {f"{m}_{k}": [] for m, k in keys}
    for s in seeds:
        r = run_one(dataset, s, alpha, score_quantile, device)
        for m, k in keys:
            acc[f"{m}_{k}"].append(r[m][k])
    out = {"dataset": dataset, "alpha": alpha, "seeds": seeds, "n": len(seeds)}
    for m, k in keys:
        vals = acc[f"{m}_{k}"]
        out[f"{m}_{k}_mean"] = sum(vals) / len(vals)
        out[f"{m}_{k}_std"] = (sum((v - out[f"{m}_{k}_mean"]) ** 2 for v in vals) / len(vals)) ** 0.5
    return out


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["Amazon", "YelpChi"]
    seeds = [42, 123, 456, 789, 1011]
    summary = {}
    for ds in datasets:
        device = "cpu" if ds == "YelpChi" else "cuda"
        summary[ds] = aggregate(ds, seeds, 0.05, 0.75, device)
    dst = save_json(summary, ROOT / "outputs" / "structcp_seeds_amazon_yelpchi.json")
    print(f"[saved] {dst}")
    for ds, o in summary.items():
        print(ds, {k: round(v, 4) for k, v in o.items() if "mean" in k or "std" in k})
