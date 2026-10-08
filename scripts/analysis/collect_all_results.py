"""汇总 GraphSubspaceAD 全部实验结果 -> results_summary.json (论文填表用)。

包含:
  - main_table:        4 数据集同源最优 (AUC/AP/最优alpha)
  - ablation_var/tta/alpha/norm/encoder: 各数据集消融明细 (从 ablation_*_{ds}.json)
  - cross_matrix:     跨域源->目标 noTTA vs TTA (从 cross_matrix.json)
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


import json
import os

ROOT = _structcp_root()
OUT = os.path.join(ROOT, "outputs", "graph_subspace_ad")
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    summary = {}

    # 主表
    summary["main_table"] = load(os.path.join(OUT, "main_table.json")).get("table", {})

    # 各消融
    for atype in ["var", "tta", "alpha", "norm", "encoder"]:
        block = {}
        for ds in DATASETS:
            d = load(os.path.join(OUT, f"ablation_{atype}_{ds}.json"))
            if d:
                block[ds] = d.get("results", d)
        summary[f"ablation_{atype}"] = block

    # 跨域
    cm = load(os.path.join(OUT, "cross_matrix.json"))
    if cm:
        summary["cross_matrix"] = cm.get("results", cm)

    out_path = os.path.join(OUT, "results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_path}")
    print("main_table AUC:", {k: round(v["auc"], 4) for k, v in summary["main_table"].items()})


if __name__ == "__main__":
    main()
