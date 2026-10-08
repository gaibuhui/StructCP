"""把已有的 T-Social 100k 子采样结果 (outputs/hypecp_TSocial_a0.05.json) 转成
论文附录可引用的 LaTeX 表格行 + 汇总, 作为"最大图可扩展性实证"证据。
该图 146M 边, 远超主表 42.4M 边 (T-Finance), 直接正面回应审稿人对
"scalability claim 自相矛盾"的质疑。
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
PATH = os.path.join(ROOT, "outputs", "hypecp_TSocial_a0.05.json")

with open(PATH) as f:
    d = json.load(f)

rows = []
print(f"{'Method':<12}{'FPR':>9}{'TPR':>9}{'F1':>9}{'AUROC':>9}{'NovelFPR':>11}")
print("-" * 60)
for m, r in d["results"].items():
    fpr = r["FPR"]
    tpr = r["TPR"]
    f1 = r["F1"]
    auroc = r["AUROC"]
    nf = r.get("FPR_novel_normality", float("nan"))
    flag = "" if fpr <= d["config"]["alpha"] else "  [violates alpha]"
    print(f"{m:<12}{fpr:>9.4f}{tpr:>9.4f}{f1:>9.4f}{auroc:>9.4f}{nf:>11.4f}{flag}")
    rows.append((m, fpr, tpr, f1, auroc, nf))

# LaTeX 行 (appendix table)
print("\n--- LaTeX rows (appendix scalability) ---")
print("% T-Social (100k subsample of 146M edges), alpha=0.05")
for m, fpr, tpr, f1, auroc, nf in rows:
    mark = "\\checkmark" if fpr <= d["config"]["alpha"] else "\\textsf{x}"
    print(f"{m} & {mark} & {fpr:.4f} & {tpr:.4f} & {f1:.4f} & {auroc:.4f} & {nf:.4f} \\\\")

print(f"\nedges total = 146M (146,177,951); subsample = 100,000 nodes")
print(f"knn_backbone = {d['config']['knn_backbone']}, k = {d['config']['k']}")
