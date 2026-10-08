"""One-off cleanup: remove duplicated 'Reporting symmetry note' + (C1) criterion
blocks that were pasted after every table. Keep the FIRST occurrence only.
Run: python scripts/_dedup_notes.py
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

import re, sys, os

path = os.path.join(_structcp_root(),
                    "docs", "paper_iclr.tex")
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

# locate all start lines of the duplicated unit
start_pat = "\\noindent \\textbf{Reporting symmetry note.}"
starts = [i for i, l in enumerate(lines) if l.strip().startswith(start_pat)]
print(f"found {len(starts)} 'Reporting symmetry note' units at lines {[s+1 for s in starts]}")

# for each unit, the end is the line closing the (C1) criterion block.
end_marker = "applied uniformly across Tab."
to_delete = set()
keep_first = True
for s in starts:
    if keep_first:
        keep_first = False
        continue
    # find end line after s
    end = None
    for j in range(s, min(s + 40, len(lines))):
        if end_marker in lines[j]:
            end = j
            break
    if end is None:
        print(f"  [warn] could not find end for unit at line {s+1}; skipping")
        continue
    # delete from s to end inclusive, plus a trailing blank line if present
    for k in range(s, end + 1):
        to_delete.add(k)
    # also drop one blank line after end if present (avoid double blanks)
    if end + 1 < len(lines) and lines[end + 1].strip() == "":
        to_delete.add(end + 1)

new_lines = [l for i, l in enumerate(lines) if i not in to_delete]
with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print(f"deleted {len(to_delete)} lines; kept first unit at line {starts[0]+1}")
