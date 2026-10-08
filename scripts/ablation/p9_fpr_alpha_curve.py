"""任务9(b): 4 数据集 FPR-α 曲线 (StructCP 方法的 FPR / TPR 随目标 α 变化)。

通过子进程调用 run_structcp.py 复用其完整推理逻辑, 解析 'StructCP' 行提取 FPR/TPR/F1。
输出: outputs/p9_fpr_alpha_curve.json
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
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def parse_structcp(line: str):
    """解析 run_structcp.py 输出中的 StructCP 行:
    StructCP   0.8371   0.2513   0.0468   0.2080   0.2191   0.0212    12.345
       AUROC  AUPRC    FPR      TPR      F1       FPR_novel time
    """
    m = re.match(
        r"StructCP\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        line.strip(),
    )
    if not m:
        return None
    return {
        "AUROC": float(m.group(1)),
        "AUPRC": float(m.group(2)),
        "FPR": float(m.group(3)),
        "TPR": float(m.group(4)),
        "F1": float(m.group(5)),
        "FPR_novel": float(m.group(6)),
        "time_s": float(m.group(7)),
    }


def run_one(dataset: str, alpha: float, seed: int = 42) -> dict:
    out = subprocess.run(
        ["python", "scripts/run_structcp.py", "--dataset", dataset,
         "--alpha", str(alpha), "--seed", str(seed)],
        cwd=ROOT, capture_output=True, text=True,
    )
    rec = None
    for ln in out.stdout.splitlines():
        if ln.strip().startswith("StructCP"):
            rec = parse_structcp(ln)
            if rec:
                break
    if rec is None:
        print(f"[warn] {dataset} a={alpha} 未解析到 StructCP 行; stderr 末行:\n"
              + "\n".join(out.stderr.splitlines()[-3:]))
    return {"alpha": alpha, **(rec or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        print(f"\n##### FPR-α curve: {ds} #####")
        curve = []
        for a in args.alphas:
            r = run_one(ds, a, seed=args.seed)
            curve.append(r)
            print(f"  alpha={a:<5} -> {r}")
        results[ds] = curve

    out_path = os.path.join(ROOT, "outputs", "p9_fpr_alpha_curve.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
