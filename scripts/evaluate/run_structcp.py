"""Step 7: StructCP 完整推理与对比实验（薄封装）。

重构说明（2026-08-23）：
  本脚本不再自行实现 Frozen / HG-TTA / StructCP 三段式推理逻辑，
  而是调用共享管线 `adapters/pipeline.evaluate_structcp` —— 该管线由本脚本
  与 `run_structcp_seeds.py`、`compute_ablation.py`、`run_calib_contamination*.py`、
  `run_backbone_yelpchi.py`、`aggregate_seeds.py` 共用，保证口径一致、修一处全局生效。
  CLI 参数与输出文件名保持向后兼容。

对比方法:
  1. Frozen        : 冻结 BWGNN + 标准 split conformal (无 TTA, 无重加权) —— CRC-SGAD 风格
  2. HG-TTA        : 超图免训练 TTA + 标准 split conformal (只有创新点一)
  3. StructCP (full) : 超图 TTA + 超边一致性加权保形 (创新点一 + 二 + 三)

汇报: AUROC / AUPRC / 目标 α 下实际 FPR / TPR / F1 / 推理耗时
"""
from __future__ import annotations

import argparse
import os
import sys as _sys

import torch

# 项目根 bootstrap（深度无关）——必须在任何项目内 import 之前执行
_ROOT = None
_p = os.path.dirname(os.path.abspath(__file__))
for _ in range(10):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)
if _ROOT and _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from adapters.pipeline import evaluate_structcp, METHOD_NAMES  # noqa: E402
from utils.config import inference_defaults, load_config_yaml  # noqa: E402
from utils.results import save_json  # noqa: E402


def main():
    cfg_yaml = load_config_yaml(_ROOT)
    defaults = inference_defaults(cfg_yaml)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon", choices=["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial"])
    ap.add_argument("--relation", default=defaults.get("relation", "homo"), choices=["homo", "multi"])
    ap.add_argument("--k", type=int, default=defaults.get("k", 10), help="k-NN 超边大小")
    ap.add_argument("--layers", type=int, default=defaults.get("layers", 2), help="超图传播层数")
    ap.add_argument("--alpha_res", type=float, default=defaults.get("alpha_res", 0.5))
    ap.add_argument("--alpha", type=float, default=defaults.get("alpha", 0.05), help="目标 FPR 上界")
    ap.add_argument("--tau", type=float, default=defaults.get("tau", 0.5),
                    help="伪正常筛选的一致性分位数")
    ap.add_argument("--mode", default=defaults.get("mode", "hard"), choices=["hard", "soft"],
                    help="hard=一致性硬截断; soft=全测试集连续软权重去污染")
    ap.add_argument("--score_quantile", type=float, default=defaults.get("score_quantile", None),
                    help="分数-一致性联合筛选上界 (如 0.75); None=纯一致性 (原逻辑)")
    ap.add_argument("--subsample", type=int, default=defaults.get("subsample", None),
                    help="TSocial 大规模图子采样节点数 (默认 100000)")
    ap.add_argument("--knn_backbone", type=int, default=defaults.get("knn_backbone", None),
                    help="TSocial 子采样后补充特征 k-NN 骨干边 (默认 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift_strength", type=float, default=None,
                    help="偏移强度 (novel 簇占比 1/6~3/6); None=默认")
    # ---- TPR 崩溃兜底 (默认开启的部署 safeguard, 见论文 Sec. safeguard) ----
    ap.add_argument("--no-adaptive_alpha", action="store_false", dest="adaptive_alpha", default=True,
                    help="禁用自适应 FPR 目标兜底(默认 ON)")
    ap.add_argument("--alpha_max", type=float, default=0.10,
                    help="adaptive_alpha 放宽上限 (论文推荐 0.10)")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="社区型高通行滤波强度, 默认 0.0 (维持 FPR 保证)")
    ap.add_argument("--deg_gate", action="store_true", default=False,
                    help="超边度自适应门控 (B 方案)")
    ap.add_argument("--tpr_tripwire", type=float, default=0.10,
                    help="TPR 不可用熔断阈值 (兼容旧 CLI；管线内未启用时仅记录)")
    ap.add_argument("--use_calib_stat", action="store_true", default=True,
                    help="权重归一化用校准集统计量 (默认开启)")
    ap.add_argument("--no-calib_stat", dest="use_calib_stat", action="store_false",
                    help="回退到测试批次统计量 (用于量化泄漏代价)")
    ap.add_argument("--robust_calibration", action="store_true", default=False,
                    help="污染鲁棒校准: 不信任 cal_y, 由超边一致性自清洗 CC")
    ap.add_argument("--weight_mode", type=str, default="contrastive",
                    choices=["contrastive", "struct", "simple_graph", "ppr"],
                    help="一致性权重来源: contrastive=对比一致性(默认,替换公式2/3); "
                         "struct=旧绝对一致性+exp/clip(消融对照); simple_graph=Simple-Graph-CP基线; "
                         "ppr=一致性×PPR全局结构门槛(论文主表配置)")
    ap.add_argument("--device", default=cfg_yaml.get("model", {}).get("device",
                    "cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--approx_knn", action="store_true", default=False,
                    help="使用 hnswlib 近似 k-NN (可扩展性验证)")
    ap.add_argument("--knn_metric", default="cosine", choices=["cosine", "euclidean"])
    args = ap.parse_args()

    out = evaluate_structcp(
        args.dataset, seed=args.seed, alpha=args.alpha,
        device=args.device, root=_ROOT,
        mode=args.mode, score_quantile=args.score_quantile, tau=args.tau,
        k=args.k, layers=args.layers, alpha_res=args.alpha_res,
        beta=args.beta, deg_gate=args.deg_gate,
        adaptive_alpha=args.adaptive_alpha, alpha_max=args.alpha_max,
        use_calib_stat=args.use_calib_stat, robust_calibration=args.robust_calibration,
        weight_mode=args.weight_mode, knn_metric=args.knn_metric,
        approx_knn=args.approx_knn,
        shift_strength=args.shift_strength,
    )

    results, extra = out["results"], out["extra"]
    method_name = extra["method_name"]
    print(f"\n{'='*100}")
    print(f"{method_name} on {args.dataset}  (target alpha = {args.alpha})")
    print(f"{'='*100}")
    hdr = f"{'Method':<16}{'AUROC':>9}{'AUPRC':>9}{'FPR':>9}{'TPR':>9}{'F1':>9}" \
          f"{'FPR_novel':>12}{'time(s)':>10}"
    print(hdr)
    print("-" * 100)
    for name in METHOD_NAMES:
        # ppr 模式下方法名为 "StructCP-PPR"（pipeline 按 weight_mode 命名），做键名归一
        r = results.get(name) or results.get(f"{name}-PPR") or results.get(f"{name}-Dual")
        if r is None:
            print(f"  {name:<12} (无结果)")
            continue
        print(f"{name:<12}{r['AUROC']:>9.4f}{r['AUPRC']:>9.4f}{r['FPR']:>9.4f}"
              f"{r['TPR']:>9.4f}{r['F1']:>9.4f}"
              f"{r.get('FPR_novel_normality', float('nan')):>12.4f}"
              f"{r['infer_time_s']:>10.3f}")
    print("=" * 100)

    wm_tag = "" if args.weight_mode in ("struct", "contrastive") else f"_{args.weight_mode}"
    ap_tag = "_approxknn" if args.approx_knn else ""
    out_path = save_json(out, os.path.join(_ROOT, "outputs",
                                           f"hypecp_{args.dataset}_a{args.alpha}{wm_tag}{ap_tag}.json"))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
