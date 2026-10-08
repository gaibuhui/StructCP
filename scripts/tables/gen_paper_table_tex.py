"""按方法类型分组 + best 高亮, 从 comparison_table.json 生成论文 LaTeX table.

读 outputs/compare_all/comparison_table.json, 写 outputs/compare_all/tab_auc.tex.
支持方法分组 + 最佳值粗体高亮.
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

import argparse
import json
import os
import sys
from typing import Optional

# 按用户方案一分组 (Backbone / Self-Supervised / TTA)
METHOD_GROUPS = [
    ("Backbone Detectors (Frozen)", ["BWGNN", "GCN", "GAT"]),
    ("Self-Supervised / Unsupervised",
     ["DOMINANT", "GAD-NR", "CoLA", "ARC", "UNPrompt"]),
    ("Test-Time Adaptation", ["GADT3", "TA-GGAD"]),
]

DATASETS_MAIN = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
DATASETS_TUNE = ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]
DATASETS = DATASETS_MAIN + DATASETS_TUNE


def _fmt(value: Optional[float], bold: bool = False) -> str:
    """格式化 cell. best 用粗体, None -> N/A."""
    if value is None:
        return "N/A"
    s = f"{value:.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def _best_per_row(table, methods, datasets):
    """找出每行 (即 dataset) 上多个方法中的最大值."""
    best = {}
    for ds in datasets:
        vals = [(table[m][ds]["auc"], m)
                for m in methods if table[m][ds]["auc"] is not None]
        if not vals:
            continue
        m_val = max(v[0] for v in vals)
        best[ds] = m_val
    return best


def render(table, methods, datasets, out_path: str) -> None:
    n_ds = len(datasets)
    col_spec = "l" + "r" * n_ds
    # 用 \begin{tabular*} 让表格自动撑满页面宽
    lines = []
    # 注意: 仅生成 tabular 内部结构, 外层 \begin{table}...\end{table}
    # 由 paper_iclr.tex 通过 \input 提供, 避免嵌套浮动体.
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\caption{AUROC (larger is better) of anomaly-detection backbones "
                 "under the StructCP novel-normality shift protocol. Methods are grouped "
                 "by category (frozen backbones; self-supervised / unsupervised; test-time "
                 "adaptation); \\textbf{bold} marks the best result within each dataset "
                 "column. A per-method empirical direction check is applied "
                 "($\\mathrm{AUC}(-s)>\\mathrm{AUC}(s)+0.05 \\Rightarrow$ negate the score) "
                 "to avoid direction bugs. N/A: backbone unsupported on that graph "
                 "(8\\,GB memory OOM, missing weights, or training divergence). "
                 "Source: \\texttt{outputs/compare\\_all/comparison\\_table.json}.")
    lines.append(r"\label{tab:auc}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = "Method " + " & ".join(
        ds.replace("TFinance", "T-Fin") for ds in datasets
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for grp_name, grp_methods in METHOD_GROUPS:
        # 仅展示 METHODS 中实际存在的、属于本组的方法
        present = [m for m in grp_methods if m in methods]
        if not present:
            continue
        lines.append(
            r"\multicolumn{" + str(n_ds + 1) + r"}{c}{" +
            grp_name.replace("&", r"\&") + r"} \\"
        )
        # 每个 dataset 列在当前组内找 max
        best_per_col = _best_per_row(table, present, datasets)
        for m in present:
            row = [m]
            for ds in datasets:
                v = table[m][ds]["auc"]
                bold = (v is not None and ds in best_per_col
                        and abs(v - best_per_col[ds]) < 1e-6)
                row.append(_fmt(v, bold=bold))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    # 去掉最后一行的 midrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[written] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/compare_all/comparison_table.json")
    ap.add_argument("--out", default="outputs/compare_all/tab_auc.tex")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset; default = all in METHOD_GROUPS")
    args = ap.parse_args()

    with open(args.src) as f:
        full = json.load(f)
    table = full["table"]
    datasets = args.datasets.split(",")
    methods = (args.methods.split(",") if args.methods
               else [m for _, ms in METHOD_GROUPS for m in ms])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    render(table, methods, datasets, args.out)


if __name__ == "__main__":
    sys.exit(main())