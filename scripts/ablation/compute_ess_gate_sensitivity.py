"""ESS 门控阈值敏感性扫描 (论文 Table 3 注脚的 0.3 阈值依据)。

背景
----
Table 3 注脚称"高通门在伪正常池 ESS 低于 0.3×|A| 时激活", 但 0.3 此前既无实验
依据也无代码依据: 原实现位于 hypergraph_tta.py 内部, 是死代码 (无调用方传入
pseudo_normal_count; 且 if/else 两支均赋同一个值; 判据用样本数比而非 ESS)。
该死代码已移除, ESS 门控上移到 adapters/pipeline.py 的 run_comparison (两遍式)。

本脚本为该阈值提供**实测**依据: 在 4 个数据集 × 5 个种子上扫描
    ess_threshold ∈ {0.2, 0.25, 0.3, 0.35, 0.4}
并同时给出两个参照臂:
    off    —— 关闭门控 (ess_gate=False), 即标准 StructCP (beta=0)
    always —— 不做门控、无条件启用高通 (beta=0.1, deg_gate=True)
记录每档的 FPR / TPR / 是否触发 / 第一遍 ESS 占比, 用以判断 0.3 是经验值还是
敏感值, 以及门控相对"常开"是否有必要。

输出
----
outputs/ess_gate_sensitivity.json  每数据集完成即落盘 + 断点续跑

运行 (R6: 复用 conda CBP; R7: 8GB 显存用 CPU/单卡、num_workers=0)
----
cd /media/lixin/新加卷/数据集/test/StructCP && \\
source <CONDA_PREFIX>/bin/activate CBP && \\
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
python scripts/ablation/compute_ess_gate_sensitivity.py
"""
from __future__ import annotations

# --- StructCP 项目根定位（深度无关）---
import os as _os
import sys as _sys


def _structcp_root():
    _p = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(10):
        if _os.path.isfile(_os.path.join(_p, "configs", "default.yaml")):
            return _p
        _p = _os.path.dirname(_p)
    return _p


ROOT = _structcp_root()
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_sys.path.insert(0, ROOT)

from adapters.pipeline import DEFAULT_DATASET_CFG, RunConfig, run_comparison  # noqa: E402
from adapters.hypergraph_tta import KNNHypergraph  # noqa: E402
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
SEEDS = [42, 123, 456, 789, 1011]
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4]
ALPHA = 0.05

# 门控触发时启用的高通配置 (beta=0.1, deg_gate=True)
# 注意: 本臂用的是 beta=0.1, 与 Sec. community_mit 的 "+ high-pass β" 行并不同值 --
#       该行对应 community_5seed_agg.json 的 Elliptic_0.2_True_adaptive, 即 beta=0.2+adaptive-α,
#       Elliptic TPR 0.5577→0.8022; 本臂 beta=0.1 时 Elliptic TPR 仅 0.501→0.503 (无收益).
#       两结论均成立, 差异来自 β 取值, 不可互相否定.
BETA_ON_TRIGGER = 0.1
DEG_GATE_ON_TRIGGER = True


@torch.no_grad()
def run_one(dataset: str, seed: int, arm: str, device: torch.device) -> dict:
    """跑一个 (dataset, seed, arm) 组合, 返回指标字典。

    arm 取值:
      "off"            —— 门控关闭 (标准 StructCP)
      "always"         —— 无门控, 无条件启用高通
      "0.2"/"0.3"/...  —— 门控开启, ess_threshold = float(arm)
    """
    dcfg = dict(DEFAULT_DATASET_CFG.get(dataset, DEFAULT_DATASET_CFG["Amazon"]))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_gad(dataset, relation=dcfg.get("relation", "homo"), seed=seed).to(device)
    model, _ckpt_path = load_frozen_detector(dataset, dcfg.get("relation", "homo"),
                                             root=ROOT, device=device)

    t0 = time.time()
    hg = KNNHypergraph(data.x, k=dcfg.get("k", 10)).to(device)
    knn_build_time_s = time.time() - t0

    if arm == "off":
        override = dict(beta=0.0, deg_gate=False, ess_gate=False)
    elif arm == "always":
        override = dict(beta=BETA_ON_TRIGGER, deg_gate=DEG_GATE_ON_TRIGGER, ess_gate=False)
    else:
        override = dict(beta=0.0, deg_gate=False, ess_gate=True,
                        ess_threshold=float(arm),
                        beta_on_trigger=BETA_ON_TRIGGER,
                        deg_gate_on_trigger=DEG_GATE_ON_TRIGGER)

    cfg = RunConfig.from_dict({**dcfg, **override, "alpha": ALPHA, "seed": seed})
    results, extra = run_comparison(data, model, hg, cfg=cfg,
                                    knn_build_time_s=knn_build_time_s)
    name = extra["method_name"]
    r = results[name]

    return {
        "arm": arm,
        "FPR": float(r.get("FPR", float("nan"))),
        "TPR": float(r.get("TPR", float("nan"))),
        "F1": float(r.get("F1", float("nan"))),
        "AUROC": float(r.get("AUROC", float("nan"))),
        "threshold": float(r.get("threshold", float("nan"))),
        "ess_gate_triggered": bool(r.get("ess_gate_triggered", False)),
        "ess_gate_ratio": r.get("ess_gate_ratio", None),
        "n_pseudo_normal": int(r.get("n_pseudo_normal", 0)),
        "ess_pseudo_normal": r.get("ess_pseudo_normal", None),
        "alpha_eff": r.get("alpha_eff", None),
        "_time_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="限定单数据集 (默认全部)")
    ap.add_argument("--seeds", default=None, help="逗号分隔种子, 默认 5 标准种子")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    dsets = [args.dataset] if args.dataset else DATASETS
    arms = ["off", "always"] + [str(t) for t in THRESHOLDS]

    out_path = os.path.join(ROOT, "outputs", "ess_gate_sensitivity.json")
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}

    print(f"[cfg] device={device} datasets={dsets} seeds={seeds} arms={arms}")
    print(f"[out] {out_path}\n")

    for ds in dsets:
        cache.setdefault(ds, {})
        for seed in seeds:
            cache[ds].setdefault(str(seed), {})
            for arm in arms:
                key = arm
                if key in cache[ds][str(seed)]:
                    print(f"[skip] {ds} seed={seed} arm={arm}")
                    continue
                try:
                    r = run_one(ds, seed, arm, device)
                except Exception as e:
                    print(f"[err] {ds} seed={seed} arm={arm}: {type(e).__name__}: {e}")
                    continue
                cache[ds][str(seed)][key] = r
                print(f"[ok] {ds:9s} seed={seed:<5d} arm={arm:<6s} "
                      f"FPR={r['FPR']:.4f} TPR={r['TPR']:.4f} "
                      f"trig={int(r['ess_gate_triggered'])} "
                      f"ratio={r['ess_gate_ratio']} ({r['_time_s']}s)")
                # 每点即落盘, 支持断点续跑
                json.dump(cache, open(out_path, "w"), indent=2)
        print(f"[flush] {ds} done\n")

    json.dump(cache, open(out_path, "w"), indent=2)
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
