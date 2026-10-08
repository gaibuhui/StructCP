"""StructCP 核心推理编排（Frozen / HG-TTA / StructCP 三段式对比）。

【定位】
从旧 `scripts/run_structcp.py` 抽取的**可复用推理管线**。它只负责"编排"：
加载数据 → 构建 k-NN 超图 → 跑三段方法 → 汇总指标；**不含任何算法逻辑**。
算法本体仍在其归属模块（`adapters/hypergraph_tta.py` 的 dual-use k-NN graph 与
test-time propagation、`adapters/conformal.py` 的 weighted conformal prediction），
本模块对它们只做调用，逻辑保持不动。

【为什么存在】
旧代码中 `run_structcp.py` / `run_structcp_seeds.py` / `compute_ablation.py` /
`run_calib_contamination*.py` / `run_backbone_yelpchi.py` / `aggregate_seeds.py`
各自重复实现同一条三段式管线，或以 `import run_structcp` 互相耦合。抽出本模块后：
  - 主推理脚本变为薄封装；
  - 多 seed / 消融 / 鲁棒性脚本统一走 `evaluate_structcp` / `run_comparison`；
  - 修一处即全局生效，杜绝口径漂移。

【口径一致性】
三段式结果与重构前 `run_structcp.py` 逐字段一致（含 novel-normality FPR 使用
`s_tta` 而非 `s_tta_sp` 的原始行为），以保证重构前后结果可对齐。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from adapters.conformal import (
    adaptive_structcp_threshold,
    evaluate_coverage,
    evaluate_prediction_set,
    split_conformal_threshold,
    structcp_dual_threshold,
    structcp_threshold,
)
from adapters.hypergraph_tta import (
    KNNHypergraph,
    contrastive_consistency,
    hyperedge_consistency,
    hypergraph_propagation,
    simple_graph_consistency,
)
from adapters.ppr_weight import ppr_structure_score
from data.loader import load_gad
from utils.checkpoint import load_frozen_detector
from utils.metrics import auc_scores


# ---------------------------------------------------------------------------
# 每数据集对齐论文 Table 1 的运行配置（原 run_structcp_seeds.py::DATASET_CFG 迁入）
# ---------------------------------------------------------------------------
DEFAULT_DATASET_CFG: dict[str, dict[str, Any]] = {
    "Amazon": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                   k=10, layers=2, alpha_res=0.5),
    "YelpChi": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                    k=10, layers=2, alpha_res=0.5),
    "Elliptic": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                     k=10, layers=2, alpha_res=0.5),
    "TFinance": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                     k=10, layers=2, alpha_res=0.5),
    "TSocial": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                    k=10, layers=2, alpha_res=0.5),
    # TUNE (AAAI'26) benchmark graphs, same StructCP configuration as the four
    # main datasets; the frozen detector is bwgnn_{name}_homo.pth.
    "photo": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                  k=10, layers=2, alpha_res=0.5),
    "computer": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                     k=10, layers=2, alpha_res=0.5),
    "weibo": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                  k=10, layers=2, alpha_res=0.5),
    "tolokers": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                     k=10, layers=2, alpha_res=0.5),
    "questions": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                      k=10, layers=2, alpha_res=0.5),
    "reddit": dict(relation="homo", mode="hard", score_quantile=0.75, tau=0.5,
                   k=10, layers=2, alpha_res=0.5),
}

METHOD_NAMES = ("Frozen", "HG-TTA", "StructCP")
DUAL_METHOD_NAME = "StructCP-Dual"


@dataclass
class RunConfig:
    """`run_comparison` 的全部超参（默认与 run_structcp.py 一致）。"""
    alpha: float = 0.05
    mode: str = "hard"
    tau: float = 0.5
    score_quantile: float | None = None
    layers: int = 2
    alpha_res: float = 0.5
    beta: float = 0.0
    deg_gate: bool = False
    # ESS 门控 (两遍式, 见 run_comparison): 第一遍以 (beta, deg_gate) 传播并测量
    # 伪正常池 ESS; 若 ESS/|A| < ess_threshold 则第二遍改用 (beta_on_trigger,
    # deg_gate_on_trigger) 重传播。ess_gate=False 时完全退化为单遍, 行为不变。
    ess_gate: bool = False
    ess_threshold: float = 0.3
    beta_on_trigger: float = 0.1
    deg_gate_on_trigger: bool = True
    adaptive_alpha: bool = True
    alpha_max: float = 0.10
    use_calib_stat: bool = True
    robust_calibration: bool = False
    # 双阈值预测集 (StructCP-Dual): dual=True 时额外输出预测集, 用校准集内
    # 源异常样本控制 FNR≤dual_beta (λ_normal = 异常分数 β 分位), 中间带
    # {正常,异常} 交人工复核。不影响单阈值 StructCP 行。
    dual: bool = False
    dual_beta: float = 0.05
    # weight_mode:
    #   "contrastive" : 对比一致性 (默认, 替换公式2/3)。ci_struct = σ(ci_local)·σ(ci_global),
    #                   权重 wi = ci_struct + eps。乘积门控自动压制欺诈环 (社区异常), 无需
    #                   clip/exp/med/std 等冗余超参数, 亦无需 ESS/高通门控有条件触发。
    #   "struct"      : 原"绝对一致性" + exp/clip 权重 (旧公式2/3, 供消融对照)。
    #   "ppr"         : 一致性 × PPR 全局结构门槛 (思路2/GRAPHLCP)。
    #   "simple_graph": 极简图一致性 (Simple-Graph-CP 基线)。
    weight_mode: str = "contrastive"
    # PPR 结构权重 (weight_mode="ppr", 思路2/GRAPHLCP): 用个性化 PageRank 的全局
    # 结构得分调制局部一致性, 针对性缓解社区型异常 (Elliptic/TFinance) 的 TPR 坍缩。
    # c_sp = sqrt(c_consistency * c_ppr)。PPR 在 CPU (scipy.sparse) 计算, 不占显存。
    ppr_pca_dim: int = 16
    ppr_density_k: int = 15
    ppr_h: float = 1.0
    ppr_restart: float = 0.15
    ppr_K: int = 60
    ppr_max_edges: int | None = None
    # PPR 伪正常门槛分位: 在"一致性高"的测试子集内, 剔除 c_ppr 低于该分位的节点
    ppr_quantile: float = 0.4
    knn_metric: str = "cosine"
    approx_knn: bool = False
    k: int = 10
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


@torch.no_grad()
def get_scores(model: torch.nn.Module, x: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
    """冻结骨干异常概率：softmax 的 class-1 分量（越大越异常）。"""
    return F.softmax(model(x, edge_index), dim=1)[:, 1]


@torch.no_grad()
def run_comparison(
    data: Any,
    model: torch.nn.Module,
    hg: KNNHypergraph,
    cfg: RunConfig | None = None,
    knn_build_time_s: float = 0.0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """在一份数据 + 一个冻结骨干 + 一个 k-NN 超图上跑 Frozen / HG-TTA / StructCP。

    Args:
        data: `data.loader.GADData`（含 cal_mask / test_mask / novel_mask / y）。
        model: 已 `freeze()` 的检测器。
        hg: 已 `to(device)` 的 `KNNHypergraph`。
        cfg: 运行配置；None 用默认。
        knn_build_time_s: k-NN 超图构建耗时。为保持与旧 run_structcp.py 一致的
            `infer_time_s = t_propagate + t_build` 口径，会叠加到 HG-TTA/StructCP。

    Returns:
        (results, extra)
        results: {method: {AUROC, AUPRC, threshold, infer_time_s, FPR, TPR, ...}}
        extra:   {knn_build_time_s, method_name, info}
    """
    cfg = cfg or RunConfig()
    device = data.x.device
    cal_m, test_m = data.cal_mask, data.test_mask
    y_test = data.y[test_m].cpu().numpy()
    y_cal = data.y[cal_m]
    method_name = ("StructCP-PPR" if cfg.weight_mode == "ppr"
                   else "Simple-Graph-CP" if cfg.weight_mode == "simple_graph"
                   else "StructCP")
    results: dict[str, dict[str, Any]] = {}

    # ---------------- 1. Frozen baseline ----------------
    t0 = time.time()
    s_raw = get_scores(model, data.x, data.edge_index)
    t_frozen = time.time() - t0
    thr = split_conformal_threshold(s_raw[cal_m][y_cal == 0], cfg.alpha)
    results["Frozen"] = {
        **auc_scores(y_test, s_raw[test_m].cpu().numpy()),
        "threshold": float(thr),
        "infer_time_s": t_frozen,
        **evaluate_coverage(s_raw[test_m], data.y[test_m], thr),
    }

    # ---------------- 超图 TTA（方法层，零参数） ----------------
    t0 = time.time()
    # HG-TTA 基线：纯低通残差（beta=0, deg_gate=False）保持与 Table 1 口径一致
    x_enh = hypergraph_propagation(data.x, hg, cfg.layers, cfg.alpha_res,
                                   beta_highpass=0.0, deg_gate=False)
    s_tta = get_scores(model, x_enh, data.edge_index)
    t_tta = time.time() - t0

    # ---------------- 2. HG-TTA + 标准保形 ----------------
    thr2 = split_conformal_threshold(s_tta[cal_m][y_cal == 0], cfg.alpha)
    results["HG-TTA"] = {
        **auc_scores(y_test, s_tta[test_m].cpu().numpy()),
        "threshold": float(thr2),
        "infer_time_s": t_tta + knn_build_time_s,
        **evaluate_coverage(s_tta[test_m], data.y[test_m], thr2),
    }

    # ---------------- 3. StructCP (full) ----------------
    # 一致性尺度项 (d̄_s, d̄_cos) 仅从校准集 C 估计，不依赖测试批统计量（去泄漏）
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    sq = cfg.score_quantile if cfg.score_quantile is not None else 0.75
    # PPR 诊断（weight_mode="ppr" 时填充）：c_ppr / c_cons 在测试正常 vs 异常上的均值，
    # 用于验证"社区型异常局部同配(c_cons 高)但偏离正常骨架(c_ppr 低)"的机制。
    ppr_diag: dict[str, float] = {}
    _test_ppr = None  # weight_mode="ppr" 时填充: 测试节点 PPR 结构得分 (传给 threshold 作伪正常门槛)

    def _structcp_pass(beta: float, deg_gate: bool):
        """一趟完整的 StructCP：TTA 传播 → 打分 → 一致性 → 加权保形。

        抽成闭包是为了支持 ESS 门控的两遍调用（见下），避免两处重复实现导致口径漂移。
        返回 (thr, info, s_tta_sp)。
        """
        # _test_ppr 在 run_comparison 作用域初始化 (None)；闭包内 PPR 分支会重新绑定，
        # 必须 nonlocal 声明，否则 Python 将其视为闭包局部变量，非 PPR 模式引用即报错。
        nonlocal _test_ppr
        # StructCP 专用 TTA：可选高通 beta + 超边度门控（仅作用于 StructCP，不污染 HG-TTA）
        if beta > 0 or deg_gate:
            x_sp = hypergraph_propagation(data.x, hg, cfg.layers, cfg.alpha_res,
                                          beta_highpass=beta, deg_gate=deg_gate)
            s_sp = get_scores(model, x_sp, data.edge_index)
        else:
            x_sp, s_sp = x_enh, s_tta

        if cfg.weight_mode == "ppr":
            # 思路2 (GRAPHLCP): PPR 全局结构得分作为"伪正常准入"的额外门槛。
            # 社区型异常局部同配 (c_cons 高) 但 PPR 上偏离正常骨架 (c_ppr 低),
            # 在一致性高子集内按 c_ppr 分位剔除 → 不再污染校准池推高阈值 →
            # 修复 TPR 坍缩; 权重仍用 c_cons (尺度不变) → FPR 保证保持。
            # seed 用"校准正常节点" (cal_mask & y==0): c_ppr 衡量"从正常骨架出发
            # 的随机游走稳态落点概率", 越大越接近正常骨架。
            c_cons = hyperedge_consistency(s_sp, hg, features=x_sp, cal_mask=cal_bool)
            cal_normal = cal_bool & (data.y == 0)
            c_ppr = ppr_structure_score(
                x_sp, data.edge_index, cal_normal,
                pca_dim=cfg.ppr_pca_dim, density_k=cfg.ppr_density_k,
                h=cfg.ppr_h, restart=cfg.ppr_restart, K=cfg.ppr_K,
                max_edges=cfg.ppr_max_edges,
            )
            c_sp = c_cons
            _test_ppr = c_ppr[test_m]
            # 诊断统计（测试集正常 vs 异常）
            y_te = data.y[test_m].cpu().numpy()
            cppr_te = c_ppr[test_m].cpu().numpy()
            ccons_te = c_cons[test_m].cpu().numpy()
            ppr_diag["c_cons_test_normal"] = float(ccons_te[y_te == 0].mean())
            ppr_diag["c_cons_test_anomaly"] = float(ccons_te[y_te == 1].mean())
            ppr_diag["c_ppr_test_normal"] = float(cppr_te[y_te == 0].mean())
            ppr_diag["c_ppr_test_anomaly"] = float(cppr_te[y_te == 1].mean())
        elif cfg.weight_mode == "simple_graph":
            c_sp = simple_graph_consistency(s_sp, hg, cal_mask=cal_bool)
        elif cfg.weight_mode == "contrastive":
            # 对比一致性 (替换公式2/3): ci_struct = σ(ci_local)·σ(ci_global)。
            # cal_normal_mask 标记"校准集正常"节点用于估计常态原型 μC, 不依赖测试批统计。
            cal_normal = cal_bool & (data.y == 0)
            c_sp = contrastive_consistency(x_sp, hg, cal_normal_mask=cal_normal)
        else:
            c_sp = hyperedge_consistency(s_sp, hg, features=x_sp, cal_mask=cal_bool)

        # 对比一致性下权重直接用 wi = ci_struct + eps (公式3' 替换, 无 clip/exp/med/std)
        thr_weight_mode = "contrastive" if cfg.weight_mode == "contrastive" else "struct"
        if cfg.adaptive_alpha:
            t, i = adaptive_structcp_threshold(
                s_sp[cal_m], y_cal, c_sp[cal_m], s_sp[test_m], c_sp[test_m],
                alpha=cfg.alpha, alpha_max=cfg.alpha_max, tau_quantile=cfg.tau,
                score_quantile=sq, use_calib_stat=cfg.use_calib_stat,
                robust_calibration=cfg.robust_calibration,
                test_ppr=_test_ppr, ppr_quantile=cfg.ppr_quantile,
                weight_mode=thr_weight_mode,
            )
        else:
            t, i = structcp_threshold(
                s_sp[cal_m], y_cal, c_sp[cal_m], s_sp[test_m], c_sp[test_m],
                alpha=cfg.alpha, tau_quantile=cfg.tau, mode=cfg.mode,
                score_quantile=sq, use_calib_stat=cfg.use_calib_stat,
                robust_calibration=cfg.robust_calibration,
                test_ppr=_test_ppr, ppr_quantile=cfg.ppr_quantile,
                weight_mode=thr_weight_mode,
            )
        return t, i, s_sp, c_sp

    thr3, info, s_tta_sp, c_sp = _structcp_pass(cfg.beta, cfg.deg_gate)

    # ---- ESS 门控（两遍式）----
    # 判据: 伪正常池有效样本量占比 ESS/|A| 低于 ess_threshold 时, 认为权重过度集中、
    # 伪正常池退化, 此时改用高通传播 (beta_on_trigger) 重跑一趟。
    # 之所以必须放在 pipeline 层: 伪正常池的 ESS 是 *传播之后* 由加权保形算出的,
    # 放进 hypergraph_propagation 内部会造成循环依赖 (原实现即为死代码, 已移除)。
    if cfg.ess_gate:
        n_pseudo = int(info.get("n_pseudo_normal") or 0)
        ess_pseudo = float(info.get("ess_pseudo_normal") or 0.0)
        # n_pseudo == 0 (无伪正常入选) 时 ratio 取 1.0: 池为空谈不上"权重退化",
        # 且此时高通门无从改善, 保持不触发更符合语义。
        ess_ratio = (ess_pseudo / n_pseudo) if n_pseudo > 0 else 1.0
        if ess_ratio < cfg.ess_threshold:
            thr3, info, s_tta_sp, c_sp = _structcp_pass(cfg.beta_on_trigger,
                                                        cfg.deg_gate_on_trigger)
            info["ess_gate_triggered"] = True
        else:
            info["ess_gate_triggered"] = False
        # 记录的是 *第一遍* 的判据值 (即门控决策所依据的量),
        # 若已触发则与最终报告的 FPR 分属不同趟, 读表时需注意。
        info["ess_gate_ratio"] = round(float(ess_ratio), 4)
        info["ess_gate_p1_ratio"] = info["ess_gate_ratio"]
        info["ess_gate_threshold"] = float(cfg.ess_threshold)

    results[method_name] = {
        **auc_scores(y_test, s_tta_sp[test_m].cpu().numpy()),
        "threshold": float(thr3),
        "infer_time_s": t_tta + knn_build_time_s,
        **evaluate_coverage(s_tta_sp[test_m], data.y[test_m], thr3),
        **ppr_diag,
        **info,
    }

    # ---- StructCP-Dual: 双阈值预测集 (同时控 FPR≤α 与 FNR≤β) ----
    # λ_anomaly 直接复用单阈值 thr3 (与 StructCP 行口径一致), λ_normal 取校准
    # 集内源异常分数的 β 分位; 中间带输出 {正常,异常} 交人工复核 (abstention)。
    # 这使模糊样本不再被硬判为"正常", 从而在 FPR 保证不变下压缩 FNR/TPR 坍缩。
    if cfg.dual:
        # PPR 模式下的双阈值方法名为 "StructCP-PPR-Dual"（λ_anomaly 已由单阈值
        # thr3 带 PPR 伪正常门槛算出并复用，λ_normal 仍取校准异常 β 分位控 FNR）
        dual_name = ("StructCP-PPR-Dual" if cfg.weight_mode == "ppr"
                     else DUAL_METHOD_NAME)
        lb, la, dinfo = structcp_dual_threshold(
            s_tta_sp[cal_m], y_cal, c_sp[cal_m],
            s_tta_sp[test_m], c_sp[test_m],
            alpha=cfg.alpha, beta=cfg.dual_beta, lambda_anomaly=thr3,
            tau_quantile=cfg.tau, mode=cfg.mode, score_quantile=sq,
            use_calib_stat=cfg.use_calib_stat,
            robust_calibration=cfg.robust_calibration,
            test_ppr=_test_ppr, ppr_quantile=cfg.ppr_quantile,
            weight_mode=("contrastive" if cfg.weight_mode == "contrastive" else "struct"),
        )
        ps = evaluate_prediction_set(s_tta_sp[test_m], data.y[test_m], lb, la)
        results[dual_name] = {
            **auc_scores(y_test, s_tta_sp[test_m].cpu().numpy()),
            "threshold": float(la), "threshold_low": float(lb),
            "infer_time_s": t_tta + knn_build_time_s,
            **ps, **dinfo,
        }
        nov = data.novel_mask & test_m
        if int(nov.sum()) > 0:
            results[dual_name]["FPR_novel_normality"] = \
                float((s_tta_sp[nov] >= la).float().mean())

    # ---------------- novel normality 上的误报率 ----------------
    # 口径保持与 run_structcp.py 一致：StructCP 行用 s_tta（非 s_tta_sp）
    nov = data.novel_mask & test_m
    if int(nov.sum()) > 0:
        for name, s, t in [("Frozen", s_raw, thr), ("HG-TTA", s_tta, thr2),
                           (method_name, s_tta, thr3)]:
            results[name]["FPR_novel_normality"] = float((s[nov] > t).float().mean())

    return results, {"method_name": method_name, "info": info}


def evaluate_structcp(
    dataset: str,
    seed: int = 42,
    cfg: dict[str, Any] | None = None,
    alpha: float = 0.05,
    device: str | None = None,
    root: str | os.PathLike | None = None,
    save_output: bool = False,
    **override: Any,
) -> dict[str, Any]:
    """单数据集单 seed 的端到端推理（对齐论文 Table 1 口径）。

    Args:
        dataset: Amazon / YelpChi / Elliptic / TFinance / TSocial 等。
        seed: 随机种子（数据划分 + kNN）。
        cfg: 数据集运行配置；None 用 `DEFAULT_DATASET_CFG[dataset]`。
        alpha: 目标 FPR 上界。
        device: cuda/cpu；None 自动。
        root: 项目根；None 自动定位。
        save_output: True 时落盘 `outputs/hypecp_{dataset}_a{alpha}.json`。
        override: 额外覆盖 RunConfig 字段（如 mode/score_quantile/k...）。

    Returns:
        {"dataset", "seed", "alpha", "config", "results", "extra"}
    """
    if root is None:
        from utils.paths import project_root
        root = project_root()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg = dict(DEFAULT_DATASET_CFG.get(dataset, DEFAULT_DATASET_CFG["Amazon"]))
    # None 视为"未显式给定"，不覆盖数据集级默认（如 score_quantile=0.75）
    dcfg.update({k: v for k, v in (override or {}).items() if v is not None})

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    relation = dcfg.get("relation", "homo")
    if dataset in ("Elliptic", "TFinance", "TSocial"):
        relation = "homo"
    subsample = dcfg.get("subsample")
    knn_backbone = dcfg.get("knn_backbone")
    shift_strength = dcfg.get("shift_strength")
    if dataset == "TSocial":
        if subsample is None:
            subsample = 100000
        if knn_backbone is None:
            knn_backbone = 10

    data = load_gad(dataset, relation=relation, seed=seed,
                    shift_strength=shift_strength,
                    subsample=subsample, knn_backbone=knn_backbone).to(dev)
    model, ckpt_path = load_frozen_detector(dataset, relation, root=root, device=device)

    t0 = time.time()
    rcfg = RunConfig.from_dict(dcfg)
    rcfg.alpha = alpha
    hg = KNNHypergraph(data.x, k=rcfg.k, metric=rcfg.knn_metric,
                       approx=rcfg.approx_knn).to(dev)
    t_build = time.time() - t0

    results, extra = run_comparison(data, model, hg, rcfg, knn_build_time_s=t_build)
    # config 字段向后兼容：保留旧 run_structcp `config=vars(args)` 的关键键
    # （alpha/k/knn_backbone/...），下游 make_tsocial_table.py 等依赖 config["alpha"]。
    config_out = {
        **dcfg, "alpha": alpha, "seed": seed, "device": device,
        "subsample": subsample, "knn_backbone": knn_backbone,
        "shift_strength": shift_strength,
    }
    out = {
        "dataset": dataset, "seed": seed, "alpha": alpha,
        "config": config_out,
        "checkpoint": ckpt_path,
        "results": results,
        "extra": {"knn_build_time_s": t_build, **extra},
    }
    if save_output:
        from utils.results import save_json
        out_path = save_json(
            out, os.path.join(str(root), "outputs", f"hypecp_{dataset}_a{alpha}.json")
        )
        out["saved_to"] = out_path
    return out
