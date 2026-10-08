"""全自动对比实验调度器 (StructCP vs 20+ 对比方法)。

功能:
  * 在统一数据集 (Amazon/YelpChi/Elliptic/TFinance, 可选 TUNE 系列) 上,
    对每个已注册方法自动串跑: fit -> fit_target -> score,
    用 test_mask + novel_mask 计算 AUC / AP (越大越异常),
    逐 (method,dataset) 落盘 outputs/compare_all/<M>__<D>.json,
    并聚合为 outputs/compare_all/compare_all.json (论文主表可直接用)。

评测协议: 严格对齐 StructCP 的 novel-normality 偏移 (data.loader.load_gad),
保证所有方法可比。FPR 类保形方法额外记录 a=0.1 经验 FPR (若 adapter 暴露)。

约束 (R7, 8GB 显存): 顺序执行、num_workers=0、不并行训练、关闭 output_attentions。

用法:
  python scripts/run_all_baselines.py --methods ALL --datasets main
  python scripts/run_all_baselines.py --methods BWGNN,ARC,GADT3,TA-GGAD --datasets main
  python scripts/run_all_baselines.py --methods ALL --datasets main,tune
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
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(_structcp_root())
sys.path.insert(0, str(ROOT))

from data.loader import load_gad  # noqa: E402
from adapters.registry import REGISTRY, ALL_METHODS, build  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402


MAIN_DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
TUNE_DATASETS = ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]


def _clean(v):
    """转成 JSON 安全标量。"""
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        v = int(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if v is None:
        return None
    return v


def eval_scores(score: np.ndarray, data) -> dict:
    """对全量 score 在 test 与 novel 子集上算 AUC/AP。

    经验方向校正: 协议要求 score 越大越异常。若取反后 AUC 显著提升
    (auc(-score) - auc(+score) > 0.05), 说明 adapter 内部方向反了,
    自动采用取反后的方向并标注 direction_flipped, 防止方向 bug 污染对比表。
    """
    y = data.y.detach().cpu().numpy().reshape(-1)
    test = data.test_mask.detach().cpu().numpy().reshape(-1).astype(bool)
    novel = data.novel_mask.detach().cpu().numpy().reshape(-1).astype(bool)

    def _auc(s, mask):
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            return None
        try:
            return float(roc_auc_score(y[mask], s[mask]))
        except ValueError:
            return None

    def _ap(s, mask):
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            return None
        try:
            return float(average_precision_score(y[mask], s[mask]))
        except ValueError:
            return None

    # 经验方向: 选 AUC(test) 更高的方向
    a_pos = _auc(score, test)
    a_neg = _auc(-score, test) if a_pos is not None else None
    flipped = (a_pos is not None and a_neg is not None and a_neg - a_pos > 0.05)
    eff = -score if flipped else score

    def _wrap(fn, mask):
        return _clean(fn(eff, mask))

    return {
        "auc_test": _wrap(_auc, test),
        "ap_test": _wrap(_ap, test),
        "auc_novel": _wrap(_auc, novel),
        "ap_novel": _wrap(_ap, novel),
        "n_test": int(test.sum()),
        "n_novel": int(novel.sum()),
        "direction_flipped": bool(flipped),
        "auc_test_raw": _clean(a_pos),
        "auc_test_flipped": _clean(a_neg),
    }


def run_one(method: str, ds: str, device: str, out_dir: Path) -> dict:
    t0 = time.time()
    rec = {"method": method, "dataset": ds, "status": "ok", "error": None}
    try:
        adapter = build(method, device=device)
        data = load_gad(ds)
        adapter.fit(data)
        adapter.fit_target(data)
        score = adapter.score(data)
        score = np.asarray(score).reshape(-1).astype(np.float64)
        rec.update(eval_scores(score, data))
        rec["elapsed_s"] = round(time.time() - t0, 1)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()
        rec["elapsed_s"] = round(time.time() - t0, 1)
    # 无论成败都落盘 (便于排查)
    try:
        fp = out_dir / f"{method}__{ds}.json"
        fp.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as we:
        print(f"  [warn] 落盘失败 {method}__{ds}: {we}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="ALL", help="逗号分隔或 ALL")
    ap.add_argument("--datasets", default="main", help="main / tune / main,tune")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/compare_all")
    args = ap.parse_args()

    methods = ALL_METHODS if args.methods.upper() == "ALL" else [m.strip() for m in args.methods.split(",")]
    # 只跑已注册 (能 build 的)
    usable = [m for m in methods if m in REGISTRY]
    missing = [m for m in methods if m not in REGISTRY]
    if missing:
        print(f"[warn] 以下方法未接入 REGISTRY, 跳过: {missing}")

    dss = []
    for part in args.datasets.split(","):
        if part == "main":
            dss += MAIN_DATASETS
        elif part == "tune":
            dss += TUNE_DATASETS
        else:
            dss.append(part)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] 方法 {len(usable)} 个, 数据集 {len(dss)} 个, 共 {len(usable)*len(dss)} 组")
    results = []
    for ds in dss:
        for m in usable:
            print(f"[run] {m} x {ds} ...", flush=True)
            r = run_one(m, ds, args.device, out_dir)
            print(f"      -> {r['status']} auc_novel={r.get('auc_novel')} ap_novel={r.get('ap_novel')}", flush=True)
            results.append(r)

    # 聚合
    agg = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "datasets": dss,
        "methods": usable,
        "missing_methods": missing,
        "detail": results,
        "summary": _summary(results),
    }
    (out_dir / "compare_all.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] 聚合 -> {out_dir / 'compare_all.json'}")


def _summary(results):
    """以 method 为行、dataset 为列的 AUC(novel) 表。"""
    table = {}
    for r in results:
        if r["status"] != "ok":
            continue
        table.setdefault(r["method"], {})[r["dataset"]] = r.get("auc_novel")
    return table


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
