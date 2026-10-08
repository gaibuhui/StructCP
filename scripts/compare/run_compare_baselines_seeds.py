"""P1-3: 为 TUNE 6 数据集补 3-seed 增强统计稳健性。

复用 run_compare_baselines.py 的 main（同一套统一口径：BWGNN 共享检测器 +
split_conformal alpha=0.05 定阈），但把每个 (dataset, seed) 的单文件 JSON 输出
重定向到带 seed 后缀的临时文件，最后聚合各方法指标的 mean±std，落盘到
outputs/compare_baselines_seeds_<dataset>.json。

这样主表 tab:main_tune 可从 single-seed 升级为 3-seed mean±std，回应审稿人
对"单 seed=42 是否运气好"的质疑。

运行 (后台, 实时 flush):
  cd /media/lixin/新加卷/数据集/test/StructCP && \\
  source <CONDA_PREFIX>/bin/activate CBP && \\
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
  nohup python -u scripts/run_compare_baselines_seeds.py \\
    --datasets photo computer reddit questions tolokers weibo \\
    --seeds 42,123,456 > outputs/log_compare_seeds.txt 2>&1 &
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
import sys
import tempfile

import numpy as np

ROOT = _structcp_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_compare_baselines as rcb  # noqa: E402

# 主表关心的 6 个方法（与 run_compare_baselines 一致）
METHOD_ORDER = ["Frozen", "TUNE", "TUNE+StructCP", "StructCP", "GADT3", "TA-GGAD"]
# 聚合指标：FPR/TPR/F1 受阈值影响；AUROC/AUPRC 阈值无关；FPR_novel 可选
AGG_METRICS = ["AUROC", "AUPRC", "FPR", "TPR", "F1", "FPR_novel"]


def run_one_seed(dataset, seed, base_args):
    """跑单个 (dataset, seed)，输出重定向到临时 json，返回该 seed 的 results dict。"""
    out_dir = tempfile.mkdtemp(prefix=f"cmp_seed_{dataset}_{seed}_")
    tmp = os.path.join(out_dir, "single.json")
    real_join = rcb.os.path.join

    def fake_join(*args):
        b = args[-1]
        if b.startswith("compare_baselines_") and b.endswith(".json"):
            return tmp
        return real_join(*args)

    rcb.os.path.join = fake_join
    try:
        # 构造 sys.argv 供 run_compare_baselines.main 解析
        argv = [
            "run_compare_baselines.py",
            "--dataset", dataset,
            "--relation", "homo",
            "--k", str(base_args.k),
            "--layers", str(base_args.layers),
            "--alpha_res", str(base_args.alpha_res),
            "--alpha", str(base_args.alpha),
            "--tau", str(base_args.tau),
            "--tune_epochs", str(base_args.tune_epochs),
            "--seed", str(seed),
            "--device", base_args.device,
        ]
        if base_args.no_train_baselines:
            argv.append("--no_train_baselines")
        if base_args.skip_arc:
            argv.append("--skip_arc")
        if base_args.skip_gadt3:
            argv.append("--skip_gadt3")
        sys.argv = argv
        rcb.main()
    finally:
        rcb.os.path.join = real_join

    with open(tmp) as f:
        return json.load(f)


def aggregate(results_per_seed, dataset, seeds):
    """聚合多 seed 的 methods 指标为 mean±std。"""
    methods = [m for m in METHOD_ORDER if any(m in r for r in results_per_seed)]
    agg = {"dataset": dataset, "seeds": seeds, "n": len(seeds), "alpha": None}
    agg["per_seed"] = results_per_seed
    agg["summary"] = {}
    for m in methods:
        agg["summary"][m] = {}
        for metric in AGG_METRICS:
            vals = []
            for r in results_per_seed:
                if m in r and metric in r[m]:
                    vals.append(r[m][metric])
            if vals:
                arr = np.array(vals, dtype=float)
                agg["summary"][m][f"{metric}_mean"] = float(arr.mean())
                agg["summary"][m][f"{metric}_std"] = float(arr.std(ddof=0))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["photo", "computer", "reddit", "questions", "tolokers", "weibo"])
    ap.add_argument("--seeds", default="42,123,456", help="逗号分隔 seed 列表")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha_res", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tune_epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_train_baselines", action="store_true")
    ap.add_argument("--skip_arc", action="store_true")
    ap.add_argument("--skip_gadt3", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    summary_all = {}
    for ds in args.datasets:
        print(f"\n{'#'*100}\n# TUNE multi-seed sweep: {ds}  seeds={seeds}\n{'#'*100}", flush=True)
        per_seed = []
        for s in seeds:
            r = run_one_seed(ds, s, args)
            per_seed.append(r)
        agg = aggregate(per_seed, ds, seeds)
        # 落盘（每数据集完成即落盘，支持中断后从输出读回，无需重跑）
        out = os.path.join(ROOT, "outputs", f"compare_baselines_seeds_{ds}.json")
        json.dump(agg, open(out, "w"), indent=2)
        print(f"[saved] {out}", flush=True)
        summary_all[ds] = agg["summary"]

        # 终端打印聚合表
        print(f"\n{'='*100}\n聚合 {ds}  (seeds={seeds}, alpha={args.alpha})\n{'='*100}", flush=True)
        for m in METHOD_ORDER:
            if m not in agg["summary"]:
                continue
            sm = agg["summary"][m]
            a = sm.get("AUROC_mean", float("nan"))
            a_s = sm.get("AUROC_std", float("nan"))
            f = sm.get("FPR_mean", float("nan"))
            f_s = sm.get("FPR_std", float("nan"))
            t = sm.get("TPR_mean", float("nan"))
            t_s = sm.get("TPR_std", float("nan"))
            print(f"  {m:<14} AUROC={a:.4f}±{a_s:.3f}  FPR={f:.4f}±{f_s:.3f}  "
                  f"TPR={t:.4f}±{t_s:.3f}", flush=True)

    out_all = os.path.join(ROOT, "outputs", "compare_baselines_seeds_all.json")
    json.dump(summary_all, open(out_all, "w"), indent=2)
    print(f"\n[all saved] {out_all}", flush=True)


if __name__ == "__main__":
    main()
