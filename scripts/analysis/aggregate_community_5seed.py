"""模块4: 从 community_mitigation.json 聚合 5 种子 FPR/TPR/F1/AUPRC 的 mean±std,
供论文 tab:community_full / tab:community_mitigation 回填 (含 95% CI).

用法: python scripts/aggregate_community_5seed.py
输出: outputs/community_5seed_agg.json  {group_key: {metric: [mean, std, ci95_low, ci95_high]}}
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
import numpy as np
from scipy import stats

ROOT = _structcp_root()
SRC = os.path.join(ROOT, "outputs", "community_mitigation.json")
DST = os.path.join(ROOT, "outputs", "community_5seed_agg.json")
METRICS = ["FPR", "TPR", "F1", "AUPRC", "alpha_eff", "ess"]
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


def main():
    data = json.load(open(SRC))
    groups = {}
    for k, v in data.items():
        parts = k.split("_")
        if len(parts) < 5:
            continue
        ds, seed, beta, gate, mode = parts[0], parts[1], parts[2], parts[3], parts[4]
        gk = f"{ds}_{beta}_{gate}_{mode}"
        groups.setdefault(gk, []).append((int(seed), v))

    out = {}
    for gk, items in sorted(groups.items()):
        items.sort(key=lambda x: x[0])
        seeds = [s for s, _ in items]
        out[gk] = {"seeds": seeds, "n": len(seeds)}
        for m in METRICS:
            vals = np.array([v.get(m, float("nan")) for _, v in items], dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            # 95% CI (t-distribution, n-1 dof)
            if len(vals) > 1 and std > 0:
                ci = stats.t.ppf(0.975, len(vals) - 1) * std / np.sqrt(len(vals))
            else:
                ci = 0.0
            out[gk][m] = [round(mean, 4), round(std, 4),
                          round(mean - ci, 4), round(mean + ci, 4)]

    json.dump(out, open(DST, "w"), indent=2)
    print(f"[saved] {DST}  groups={len(out)}")
    # 预览: 每数据集 default (beta=0, gate=False, fixed) 与 best (adaptive)
    for ds in DATASETS:
        for mode in ("fixed", "adaptive"):
            gk = f"{ds}_0.0_False_{mode}"
            if gk in out:
                r = out[gk]
                print(f"{gk}: TPR={r['TPR'][0]}±{r['TPR'][1]} FPR={r['FPR'][0]}±{r['FPR'][1]}")


if __name__ == "__main__":
    main()
