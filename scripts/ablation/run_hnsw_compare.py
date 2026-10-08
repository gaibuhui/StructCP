"""HNSW 近似 k-NN 可扩展性验证脚本。

目的: 验证用 hnswlib 近似 k-NN (--approx_knn) 替代 sklearn 精确 k-NN 后,
StructCP 的 FPR/TPR 是否仍能维持, 缓解"大规模图精确 k-NN 不可扩展"的审稿质疑。

对比: 对四个核心数据集分别跑 精确k-NN / 近似k-NN (hnswlib), 记录
  - frozen / structcp_adaptive / structcp_fixed 的 FPR, TPR
  - k-NN 构建耗时 t_build (s)
输出: outputs/hnsw_compare.json (可直接喂给论文主实验补充段)
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

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


def run_once(dataset: str, approx: bool, extra: list[str]) -> dict:
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "run_structcp.py"),
        "--dataset", dataset,
    ]
    if approx:
        cmd.append("--approx_knn")
    cmd.extend(extra)
    subprocess.run(cmd, check=True, cwd=ROOT)
    # 读取 run_structcp.py 落盘的 json
    out = os.path.join(ROOT, "outputs", f"hypecp_{dataset}_a0.05{'_approxknn' if approx else ''}.json")
    with open(out) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="透传给 run_structcp.py 的额外参数, 如 --no-adaptive_alpha")
    args = ap.parse_args()
    extra = args.extra

    rows = []
    for ds in DATASETS:
        rec = {"dataset": ds}
        for approx in (False, True):
            r = run_once(ds, approx, extra)
            res = r["results"]
            tag = "exact" if not approx else "approx"
            rec[f"{tag}_build_s"] = res.get("StructCP", {}).get("knn_build_time_s")
            rec[f"{tag}_metrics"] = {
                "frozen": {"fpr": res["Frozen"]["FPR"], "tpr": res["Frozen"]["TPR"]},
                "structcp": {"fpr": res["StructCP"]["FPR"], "tpr": res["StructCP"]["TPR"]},
            }
        rows.append(rec)

    out_path = os.path.join(ROOT, "outputs", "hnsw_compare.json")
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"[saved] {out_path}")
    # 打印精简对比
    for row in rows:
        e, a = row["exact_metrics"], row["approx_metrics"]
        es, as_ = row["exact_build_s"], row["approx_build_s"]
        speed = (es / as_) if as_ else float("nan")
        print(f"\n=== {row['dataset']} ===  k-NN build: exact={es:.2f}s approx={as_:.2f}s  speedup={speed:.1f}x")
        print(f"  exact : frozen FPR={e['frozen']['fpr']:.3f} TPR={e['frozen']['tpr']:.3f} | "
              f"StructCP FPR={e['structcp']['fpr']:.3f} TPR={e['structcp']['tpr']:.3f}")
        print(f"  approx: frozen FPR={a['frozen']['fpr']:.3f} TPR={a['frozen']['tpr']:.3f} | "
              f"StructCP FPR={a['structcp']['fpr']:.3f} TPR={a['structcp']['tpr']:.3f}")


if __name__ == "__main__":
    main()
