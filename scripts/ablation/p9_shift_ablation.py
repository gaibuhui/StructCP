"""任务9(e): 偏移强度消融 (shift_strength = novel 簇占比 1/6~3/6)。

通过子进程调用 run_structcp.py --shift_strength X 复用完整推理, 解析 StructCP 行。
固定 n_normality_clusters=6, 故 1/6=1簇, 2/6=2簇, 3/6=3簇。
输出: outputs/p9_shift_ablation.json
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

import os
import re
import json
import subprocess
import argparse

ROOT = _structcp_root()
DATASETS = ["Amazon", "YelpChi", "Elliptic"]
SHIFT_STRENGTHS = [1 / 6, 2 / 6, 3 / 6]


def parse_structcp(line: str):
    m = re.match(
        r"StructCP\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        line.strip(),
    )
    if not m:
        return None
    return {
        "AUROC": float(m.group(1)), "AUPRC": float(m.group(2)),
        "FPR": float(m.group(3)), "TPR": float(m.group(4)),
        "F1": float(m.group(5)), "FPR_novel": float(m.group(6)),
        "time_s": float(m.group(7)),
    }


def run_one(dataset: str, shift: float, seed: int = 42) -> dict:
    out = subprocess.run(
        ["python", "scripts/evaluate/run_structcp.py", "--dataset", dataset,
         "--shift_strength", str(shift), "--seed", str(seed)],
        cwd=ROOT, capture_output=True, text=True,
    )
    rec = None
    for ln in out.stdout.splitlines():
        if ln.strip().startswith("StructCP"):
            rec = parse_structcp(ln)
            if rec:
                break
    if rec is None:
        print(f"[warn] {dataset} shift={shift:.3f} 未解析到 StructCP 行; stderr 末行:\n"
              + "\n".join(out.stderr.splitlines()[-3:]))
    return {"shift_strength": round(shift, 4), "n_novel_clusters": int(round(shift * 6)),
            **(rec or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--shifts", nargs="+", type=float, default=SHIFT_STRENGTHS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        print(f"\n##### shift-strength ablation: {ds} #####")
        rows = []
        for s in args.shifts:
            r = run_one(ds, s, seed=args.seed)
            rows.append(r)
            print(f"  shift={r.get('shift_strength'):<6} n_novel={r.get('n_novel_clusters')} -> {r}")
        results[ds] = rows

    out_path = os.path.join(ROOT, "outputs", "p9_shift_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
