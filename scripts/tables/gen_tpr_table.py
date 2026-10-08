"""生成各方法 TPR / F1 对比表 (质疑 #3): 让 YelpChi 等低 TPR 可横向对比基线。

数据源:
  - outputs/compare_tune_multiseed.json : Frozen / TUNE / TA-GGAD / GADT3 (3 seeds)
  - outputs/structcp_seeds_all.json      : StructCP (5 seeds, 校准集统计量硬编码)
输出: LaTeX tabular (TPR 与 F1 两张), 直接粘贴进 paper_iclr.tex 的 tab:tpr 扩展。
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

ROOT = _structcp_root()
TUNE = os.path.join(ROOT, "outputs", "compare_tune_multiseed.json")
SEEDS = os.path.join(ROOT, "outputs", "structcp_seeds_all.json")

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
# compare_tune_multiseed.json 键: Frozen / TUNE / HypeCP(=HG-TTA) / TUNE+HypeCP
METHODS = [("Frozen", "Frozen"), ("TUNE", "TUNE"), ("HypeCP", "HG-TTA"),
           ("TUNE+HypeCP", "TUNE+HG-TTA"), ("StructCP", "StructCP")]


def fmt(v):
    return f"{v[0]*100:.1f}$\\pm${v[1]*100:.1f}"


def main():
    tune = json.load(open(TUNE))["agg"]
    seeds = json.load(open(SEEDS))

    def get(method, ds, metric):
        if method == "StructCP":
            blk = seeds[ds]["agg"]["StructCP"]
            return [blk[f"{metric}_mean"], blk[f"{metric}_std"]]
        return tune[ds][method][metric]  # [mean, std]

    for metric in ["TPR", "F1"]:
        print(f"\n% ===== {metric} 表 (质疑#3: 各方法横向对比; 基线 3 seed, StructCP 5 seed) =====")
        head = " & ".join(DATASETS)
        print(f"\\begin{{tabular}}{{l|{'c'*len(DATASETS)}}}\n\\toprule")
        print(f"Method & {head} \\\\\n\\midrule")
        for key, label in METHODS:
            row = " & ".join(fmt(get(key, ds, metric)) for ds in DATASETS)
            print(f"{label} & {row} \\\\\n")
        print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
