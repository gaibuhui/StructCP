"""汇总 GraphSubspaceAD 同源主实验结果 -> 论文主结果表 main_table.json。

对每个数据集: 优先用 主脚本产物 {ds}_h128_a{a}_v0.92.json (α=0.8/1.0 等),
若缺档则从 消融 alpha 产物 ablation_alpha_{ds}.json 的 results 补齐,
自动选取 AUC 最高的 alpha 作为主表报告值 (方案 Step5 'alpha 可消融选最优')。
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
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]


def _load_main_json(ds, a):
    fp = os.path.join(OUT, f"{ds}_h128_a{a}_v0.92.json")
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    return None


def _load_alpha_ablation(ds):
    fp = os.path.join(OUT, f"ablation_alpha_{ds}.json")
    if not os.path.exists(fp):
        return {}
    with open(fp) as f:
        d = json.load(f)
    out = {}
    for r in d.get("results", []):
        a = r.get("alpha")
        if a is not None:
            out[a] = {"auc": r["auc"], "ap": r["ap"], "K": r["K"], "cum_var": r["cum_var"]}
    return out


def main():
    table = {}
    for ds in DATASETS:
        cands = {}
        # 主脚本格式
        for a in ALPHAS:
            r = _load_main_json(ds, a)
            if r:
                cands[a] = r
        # 消融 alpha 格式补齐
        for a, r in _load_alpha_ablation(ds).items():
            if a not in cands:
                cands[a] = r
        if not cands:
            print(f"[warn] {ds}: 无任何 alpha 结果, 跳过"); continue
        best_a = max(cands, key=lambda a: cands[a]["auc"])
        best = cands[best_a]
        table[ds] = {
            "auc": best["auc"], "ap": best["ap"],
            "best_alpha": best_a, "K": best.get("K"),
            "cum_var": best.get("cum_var"),
            "all_alpha_auc": {str(a): cands[a]["auc"] for a in sorted(cands)},
        }
    out_path = os.path.join(OUT, "main_table.json")
    with open(out_path, "w") as f:
        json.dump({"method": "GraphSubspaceAD", "table": table}, f, indent=2)
    print(json.dumps(table, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
