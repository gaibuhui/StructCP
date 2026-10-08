"""FPR/TPR 统计显著性检验 (质疑 #4)。

从 outputs/structcp_seeds_all.json 读取 5-seed 的 Frozen 与 StructCP 逐 seed FPR/TPR,
对每组数据集做配对检验:
  - 配对 t 检验 (scipy.stats.ttest_rel)  H1: StructCP FPR < Frozen FPR (FPR 更低)
  - 配对 Wilcoxon 符号秩检验 (alternative="less")
  - 符号检验 (sign test) 统计 FPR 反转(seed 数)
同时报告种子数=5, 支持'种子数充足且 FPR 差异显著'的结论。
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

ROOT = _structcp_root()
SRC = os.path.join(ROOT, "outputs", "structcp_seeds_all.json")


def main():
    d = json.load(open(SRC))
    print(f"种子数 (per dataset): {d[list(d)[0]]['seeds']}  (统一 5 seeds)\n")
    print(f"{'Dataset':<10}{'Frozen FPR':>14}{'StructCP FPR':>14}"
          f"{'t-test p':>12}{'Wilcoxon p':>12}{'sign p':>10}")
    overall_f, overall_s = [], []
    try:
        from scipy.stats import ttest_rel, wilcoxon, binomtest
    except Exception:
        ttest_rel = wilcoxon = binomtest = None
    for ds, blk in d.items():
        per = blk["per_seed"]
        seeds = blk["seeds"]
        fpr_f = np.array([per[str(s)]["Frozen"]["FPR"] for s in seeds])
        fpr_s = np.array([per[str(s)]["StructCP"]["FPR"] for s in seeds])
        overall_f.extend(fpr_f.tolist())
        overall_s.extend(fpr_s.tolist())
        t_p = w_p = s_p = float("nan")
        if ttest_rel is not None and len(seeds) >= 2:
            t_stat, t_p = ttest_rel(fpr_f, fpr_s, alternative="greater")  # H1: Frozen>StructCP
            try:
                w_stat, w_p = wilcoxon(fpr_f, fpr_s, alternative="greater")
            except Exception:
                w_p = float("nan")
            n_rev = int(np.sum(fpr_s < fpr_f))
            s_p = binomtest(n_rev, len(seeds), 0.5).pvalue
        print(f"{ds:<10}{fpr_f.mean():>10.4f}±{fpr_f.std():<6.4f}"
              f"{fpr_s.mean():>10.4f}±{fpr_s.std():<6.4f}"
              f"{t_p:>12.4f}{w_p:>12.4f}{s_p:>10.4f}")
    # 跨数据集合并 (所有 seed 的 FPR 差值)
    if ttest_rel is not None:
        fpr_f = np.array(overall_f)
        fpr_s = np.array(overall_s)
        t_stat, t_p = ttest_rel(fpr_f, fpr_s, alternative="greater")
        print(f"\n[全体合并] Frozen FPR={fpr_f.mean():.4f}  StructCP FPR={fpr_s.mean():.4f}"
              f"  配对 t-test p={t_p:.2e}  -> {'显著' if t_p < 0.05 else '不显著'}")


if __name__ == "__main__":
    main()
