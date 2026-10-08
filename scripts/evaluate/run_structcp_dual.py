"""Step 7b: StructCP-Dual —— 双阈值预测集实验（薄封装）。

在 `adapters/pipeline.evaluate_structcp` 上开启 `dual=True`：
  - λ_anomaly : 复用 StructCP 加权保形阈值（控制 FPR ≤ α）
  - λ_normal  : 校准集内源异常分数的 β 分位（控制 FNR ≤ β）
  - 中间带输出 {正常,异常}（不确定，交人工复核），不再硬判为"正常"，
    从而在 FPR 保证不变下压缩 FNR / 缓解 TPR 坍缩。

汇报: AUROC / AUPRC / FPR / FNR / TPR / abstain(不确定率) / FPR_novel / 耗时
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

from adapters.pipeline import (  # noqa: E402
    DUAL_METHOD_NAME,
    METHOD_NAMES,
    evaluate_structcp,
)
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
    ap.add_argument("--beta_dual", type=float, default=0.05,
                    help="目标 FNR 上界 (λ_normal = 校准异常分数 β 分位)")
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
                         "ppr=StructCP-PPR(思路2)")
    # ---- PPR 结构权重超参 (GRAPHLCP, weight_mode="ppr" 时生效) ----
    ap.add_argument("--ppr_pca_dim", type=int, default=16, help="密化 PCA 降维维数")
    ap.add_argument("--ppr_density_k", type=int, default=15, help="密化 top-k 近邻边数")
    ap.add_argument("--ppr_h", type=float, default=1.0, help="各向异性高斯核带宽缩放")
    ap.add_argument("--ppr_restart", type=float, default=0.15, help="PPR 重启概率 α")
    ap.add_argument("--ppr_K", type=int, default=60, help="PPR 幂迭代最大步数")
    ap.add_argument("--ppr_max_edges", type=int, default=None,
                    help="密化新增边数上限 (None=不截断; 大图可设如 2_000_000)")
    ap.add_argument("--ppr_quantile", type=float, default=0.4,
                    help="PPR 伪正常门槛分位: 在一致性高子集内剔除 c_ppr 低于该分位的节点")
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
        dual=True, dual_beta=args.beta_dual,
        ppr_pca_dim=args.ppr_pca_dim, ppr_density_k=args.ppr_density_k,
        ppr_h=args.ppr_h, ppr_restart=args.ppr_restart, ppr_K=args.ppr_K,
        ppr_max_edges=args.ppr_max_edges, ppr_quantile=args.ppr_quantile,
    )

    results, extra = out["results"], out["extra"]
    method_name = extra["method_name"]
    # PPR 模式下双阈值方法名为 "StructCP-PPR-Dual"
    dual_name = ("StructCP-PPR-Dual" if args.weight_mode == "ppr" else DUAL_METHOD_NAME)
    print(f"\n{'='*110}")
    print(f"{method_name} (+ {dual_name}) on {args.dataset}  "
          f"(target alpha = {args.alpha}, target FNR = {args.beta_dual})")
    print(f"{'='*110}")
    hdr = f"{'Method':<16}{'AUROC':>9}{'AUPRC':>9}{'FPR':>9}{'FNR':>9}{'TPR':>9}" \
          f"{'abstain':>9}{'FPR_novel':>12}{'time(s)':>10}"
    print(hdr)
    print("-" * 110)
    for name in list(METHOD_NAMES) + [dual_name]:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<12}{r['AUROC']:>9.4f}{r['AUPRC']:>9.4f}{r['FPR']:>9.4f}"
              f"{r.get('FNR', float('nan')):>9.4f}{r['TPR']:>9.4f}"
              f"{r.get('abstain_rate', float('nan')):>9.4f}"
              f"{r.get('FPR_novel_normality', float('nan')):>12.4f}"
              f"{r['infer_time_s']:>10.3f}")
    print("=" * 110)

    wm_tag = "" if args.weight_mode in ("struct", "contrastive") else f"_{args.weight_mode}"
    ap_tag = "_approxknn" if args.approx_knn else ""
    out_path = save_json(out, os.path.join(_ROOT, "outputs",
                                           f"structcp_dual_{args.dataset}_a{args.alpha}"
                                           f"_b{args.beta_dual}{wm_tag}{ap_tag}.json"))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
