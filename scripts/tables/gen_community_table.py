"""生成社区型异常补救济表 (质疑 #5): 让 TPR 下降有自适应改进方向。

数据源: outputs/community_5seed_agg.json (per-seed 数组)。
展示 Elliptic / T-Finance 上, 固定 alpha 的 StructCP 如何在
  (a) +自适应 FPR 目标 (alpha_max=0.10)
  (b) +高通一致性 beta
  (c) +超边度门控 deg_gate
下恢复 TPR (代价是 FPR 放宽到 <= alpha_max)。
输出 LaTeX tabular (tab:community)。
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
SRC = os.path.join(ROOT, "outputs", "community_5seed_agg.json")


def mean(arr):
    a = [x for x in arr if x == x]  # 去 NaN
    return float(np.mean(a)) if a else float("nan")


def main():
    d = json.load(open(SRC))
    print("% ===== tab:community (社区型异常补救, 质疑#5) =====")
    print("\\begin{tabular}{llrrrr}")
    print("\\toprule")
    print("Dataset & Variant & FPR & TPR & F1 & $\\alpha_\\mathrm{eff}$ \\\\")
    print("\\midrule")
    for ds in ["Elliptic", "TFinance"]:
        for label, key in [
            ("StructCP (fixed $\\alpha$)", f"{ds}_0.0_False_fixed"),
            ("+ adaptive $\\alpha\\!\\to\\!\\alpha_\\mathrm{max}$", f"{ds}_0.0_False_adaptive"),
            ("+ high-pass $\\beta$", f"{ds}_0.1_True_adaptive"),
            ("+ deg-gate", f"{ds}_0.1_False_adaptive"),
        ]:
            if key not in d:
                continue
            blk = d[key]
            fpr = mean(blk["FPR"])
            tpr = mean(blk["TPR"])
            f1 = mean(blk["F1"])
            ae = mean(blk["alpha_eff"])
            print(f"{ds if label.startswith('StructCP') else ''} & {label} & "
                  f"{fpr:.3f} & {tpr:.3f} & {f1:.3f} & {ae:.3f} \\\\")
        print("\\midrule")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
