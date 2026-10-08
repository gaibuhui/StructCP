"""实验编排器：按阶段串行/批量调用子脚本，或直接调用共享管线。

设计动机（ARIS 工作流自动化）：旧流程需手工逐条执行
  pretrain -> evaluate -> seeds -> compare -> ablation -> tables -> manifest
本编排器用一个命令完成，并记录每个阶段产物路径。

用法示例:
  python scripts/orchestrate/run_experiments.py --stage evaluate --datasets Amazon --device cpu
  python scripts/orchestrate/run_experiments.py --stage seeds --datasets Elliptic,TFinance --seeds 42,123,456
  python scripts/orchestrate/run_experiments.py --all --datasets Amazon --device cpu --dry-run

阶段 -> 底层脚本/管线映射见 STAGES。dry-run 仅打印将要执行的命令。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys as _sys

# --- 项目根 bootstrap ---
_p = os.path.dirname(os.path.abspath(__file__))
_ROOT = None
for _ in range(10):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)
if _ROOT and _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

STAGES: dict[str, str] = {
    "data": "scripts/data_prep/convert_tfs.py",
    "pretrain": "scripts/train/pretrain_bwgnn.py",
    "evaluate": "scripts/evaluate/run_structcp.py",
    "seeds": "scripts/evaluate/run_structcp_seeds.py",
    "compare": "scripts/compare/run_compare_baselines.py",
    "ablation": "scripts/ablation/run_ablation.py",
    "tables": "scripts/tables/make_comparison_table.py",
    "manifest": "scripts/orchestrate/collect_manifest.py",
}

# 每阶段是否需要 dataset 参数（seeds/compare/ablation 传 --datasets，其余 --dataset）
MULTI_DATASET_FLAG = {"seeds", "compare", "ablation"}


def build_cmd(stage: str, datasets: str, args) -> list[str]:
    script = STAGES[stage]
    cmd = ["python", script]
    if stage == "data":
        # convert_tfs 按数据集逐个转换（--name），仅 TFinance/TSocial 有 DGL 源
        ds = datasets.split(",")[0]
        cmd += ["--name", ds] if ds in ("TFinance", "TSocial") else ["--help"]
    elif stage in MULTI_DATASET_FLAG:
        if stage == "ablation":
            cmd += ["--dataset", datasets.split(",")[0]]   # run_ablation 为单数据集参数
        else:
            cmd += ["--datasets", datasets]
    elif stage != "manifest" and stage != "tables":
        cmd += ["--dataset", datasets.split(",")[0]]
    if stage == "evaluate":
        cmd += ["--alpha", str(args.alpha), "--device", args.device]
    elif stage == "seeds":
        cmd += ["--seeds", args.seeds, "--device", args.device]
    elif stage in ("pretrain", "compare", "ablation"):
        cmd += ["--device", args.device]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="StructCP 实验编排器")
    ap.add_argument("--stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--all", action="store_true", help="运行全部阶段")
    ap.add_argument("--datasets", default="Amazon,YelpChi,Elliptic,TFinance,TSocial")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        order = ["data", "pretrain", "evaluate", "seeds", "compare", "ablation", "tables", "manifest"]
    elif args.stage == "all":
        order = ["data", "pretrain", "evaluate", "seeds", "compare", "ablation", "tables", "manifest"]
    elif args.stage:
        order = [args.stage]
    else:
        ap.error("需要 --stage <阶段> 或 --all")

    for stage in order:
        cmd = build_cmd(stage, args.datasets, args)
        print(f"\n[stage:{stage}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=_ROOT)
        if r.returncode != 0:
            print(f"[stage:{stage}] FAILED rc={r.returncode}", flush=True)
            raise SystemExit(r.returncode)
    print("\n[done] all requested stages completed", flush=True)


if __name__ == "__main__":
    main()
