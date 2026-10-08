# -*- coding: utf-8 -*-
"""
Aggregate structcp_tune6_fixedA_seeds3.json -> LaTeX table (FPR/TPR/F1)
for the KBS paper (TUNE benchmark graphs, fixed alpha=0.05, 3 seeds).

Usage: python scripts/aggregate_tune6_table.py
Output: prints the tabular block (copy into paper_kbs.tex) and writes
        outputs/tune6_agg.json
"""
import json, io

ROOT = r"<REPO_ROOT>"
SRC = ROOT + "/outputs/structcp_tune6_fixedA_seeds3.json"
DST = ROOT + "/outputs/tune6_agg.json"

data = json.load(io.open(SRC, encoding="utf-8"))

order = ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]
methods = ["Frozen", "HG-TTA", "StructCP"]

rows = []
for ds in order:
    if ds not in data:
        print(f"[skip] {ds} not in output")
        continue
    agg = data[ds]["agg"]
    row = {"dataset": ds}
    for m in methods:
        fpr = agg[m].get("FPR_mean")
        tpr = agg[m].get("TPR_mean")
        f1 = agg[m].get("F1_mean")
        fpr_s = agg[m].get("FPR_std")
        tpr_s = agg[m].get("TPR_std")
        f1_s = agg[m].get("F1_std")
        row[m] = {
            "FPR": None if fpr is None else round(float(fpr), 4),
            "FPR_std": None if fpr_s is None else round(float(fpr_s), 4),
            "TPR": None if tpr is None else round(float(tpr), 4),
            "TPR_std": None if tpr_s is None else round(float(tpr_s), 4),
            "F1": None if f1 is None else round(float(f1), 4),
            "F1_std": None if f1_s is None else round(float(f1_s), 4),
        }
    rows.append(row)

# ---------------- LaTeX table ----------------
def fmt(v, s):
    if v is None:
        return "--"
    if s is None or s == 0:
        return f"{v:.3f}"
    return f"{v:.3f}$\\pm${s:.3f}"

lines = []
lines.append("% TUNE benchmark graphs, fixed $\\alpha{=}0.05$, 3-seed mean$\\pm$std")
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\small")
lines.append("\\caption{Realized FPR / TPR / F1 on the six TUNE benchmark graphs ")
lines.append("(fixed $\\alpha{=}0.05$, 3-seed mean$\\pm$std, StructCP hard mode). ")
lines.append("StructCP holds FPR$\\le\\alpha$ on photo ($0.031$) and stays at or below ")
lines.append("the frozen baseline on every graph; on tolokers and questions the ")
lines.append("\\emph{frozen} baseline itself violates $\\alpha$ (weak-backbone regime, ")
lines.append("Sec.~\\ref{sec:limit}), and on computer/weibo/reddit the mild constructed ")
lines.append("shift leaves StructCP slightly above $\\alpha$. Full discussion in ")
lines.append("Appendix~\\ref{app:tune6}. Source: ")
lines.append("\\texttt{outputs/structcp\\_tune6\\_fixedA\\_seeds3.json}.}")
lines.append("\\label{tab:tune6}")
lines.append("\\resizebox{\\textwidth}{!}{%")
lines.append("\\begin{tabular}{lcccccccccc}")
lines.append("\\toprule")
lines.append("Dataset & \\multicolumn{3}{c}{Frozen} & \\multicolumn{3}{c}{HG-TTA} & \\multicolumn{3}{c}{StructCP} \\\\")
lines.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}")
lines.append(" & FPR & TPR & F1 & FPR & TPR & F1 & FPR & TPR & F1 \\\\")
lines.append("\\midrule")
for r in rows:
    ds = r["dataset"]
    cells = [ds]
    for m in methods:
        d = r[m]
        cells.append(fmt(d["FPR"], d["FPR_std"]))
        cells.append(fmt(d["TPR"], d["TPR_std"]))
        cells.append(fmt(d["F1"], d["F1_std"]))
    lines.append(" & ".join(cells) + r" \\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}}")
lines.append("\\end{table}")

out_tex = "\n".join(lines)
print(out_tex)

io.open(ROOT + "/outputs/tune6_table_block.tex", "w", encoding="utf-8").write(out_tex)
io.open(DST, "w", encoding="utf-8").write(json.dumps(rows, indent=1, ensure_ascii=False))
print("\n[saved] outputs/tune6_agg.json")
