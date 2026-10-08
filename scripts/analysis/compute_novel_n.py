"""统计各数据集 novel-normal (测试期新正常类别) 测试节点数 n.

用于附录 Tab:fpr_novel 的 novel-normal 测试集规模标注.
输出: outputs/novel_n.json
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
import sys

import torch

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


def main():
    out = {}
    for ds in DATASETS:
        d = load_gad(ds, relation="homo", seed=42)
        n_novel_test = int((d.novel_mask & d.test_mask).sum())
        n_test = int(d.test_mask.sum())
        out[ds] = {"novel_normal_test_n": n_novel_test, "test_n": n_test}
        print(f"{ds}: novel-normal test nodes n = {n_novel_test} (of test {n_test})")
    p = os.path.join(ROOT, "outputs", "novel_n.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
