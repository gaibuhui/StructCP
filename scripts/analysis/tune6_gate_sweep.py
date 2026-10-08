# -*- coding: utf-8 -*-
"""
TUNE6 门控扫描：尝试用 (tau, score_quantile) 网格把 computer/weibo/reddit 的
StructCP FPR 拉回 <= alpha=0.05（弱 shift 构造下的结构性检查）。

同时输出伪正常池中真异常占比（污染率）——回应"伪正常池污染"质疑。

用法（fov 环境）:
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/tune6_gate_sweep.py --datasets computer,weibo,reddit --seeds 42,123,456
"""
import argparse, io, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp  # noqa: E402

DATASETS = ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]
GRID = [(0.4, 0.75), (0.5, 0.75), (0.6, 0.75), (0.5, 0.90), (0.6, 0.90), (0.7, 0.90)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="computer,weibo,reddit")
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    results = {}

    for ds in datasets:
        if ds not in DATASETS:
            print(f"[skip] {ds}"); continue
        results[ds] = {"configs": {}}
        for tau, sq in GRID:
            fprs, tp_frac = [], []
            try:
                for seed in seeds:
                    out = evaluate_structcp(
                        ds, seed=seed, alpha=args.alpha, device=args.device,
                        root=_ROOT, adaptive_alpha=False,
                        tau=tau, score_quantile=sq,
                    )
                    r = out["results"]["StructCP"]
                    fprs.append(r["FPR"])
                    info = out.get("extra", {}).get("info", {})
                    # 污染率：伪正常池中真异常占比（若 info 提供）
                    cont = info.get("pseudo_anomaly_frac")
                    if cont is not None:
                        tp_frac.append(cont)
            except Exception as e:
                print(f"[{ds}] tau={tau} sq={sq} ERROR {e}", flush=True)
                continue
            mean_fpr = sum(fprs) / len(fprs) if fprs else float("nan")
            ok = mean_fpr <= args.alpha
            results[ds]["configs"][f"tau{tau}_sq{sq}"] = {
                "fpr_mean": round(mean_fpr, 4),
                "fpr_per_seed": [round(x, 4) for x in fprs],
                "le_alpha": bool(ok),
                "pseudo_anomaly_frac": (round(sum(tp_frac)/len(tp_frac), 4)
                                        if tp_frac else None),
            }
            print(f"[{ds}] tau={tau} sq={sq}: FPR={mean_fpr:.4f} "
                  f"({'OK' if ok else 'X'})", flush=True)
        # 最优配置
        ok_cfgs = [c for c, v in results[ds]["configs"].items() if v["le_alpha"]]
        results[ds]["best_le_alpha"] = ok_cfgs
        print(f"[{ds}] best (FPR<=alpha): {ok_cfgs if ok_cfgs else 'NONE'}", flush=True)

    out = os.path.join(_ROOT, "outputs", "tune6_gate_sweep.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
