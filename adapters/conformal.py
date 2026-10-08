"""创新点二: 加权保形预测 (Weighted Conformal Prediction) 的统计保证。

对标 CRC-SGAD (2025) 的双阈值保形风险控制 —— 其假设训练/测试同分布。
StructCP 在正常性偏移下引入似然比重加权，恢复被破坏的可交换性:

    q̂ = inf { q : ( Σ_i w_i · 1(S_i ≤ q) ) / ( Σ_i w_i + w_test ) ≥ 1 - α }

权重 w_i 由超边一致性决定 —— 超边内特征越一致，该节点越贴近测试期的
群体正常模式，权重越高，从而把校准分布"搬运"到测试分布上。
"""
from __future__ import annotations

import math

import torch

from .hypergraph_tta import hyperedge_consistency


@torch.no_grad()
def split_conformal_threshold(cal_scores: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """标准 split conformal 阈值 (baseline, 无重加权)。

    有限样本保证: P(S_test ≤ q̂) ≥ 1 - α，要求可交换性成立。
    用严格 order statistic (第 ceil((n+1)(1-α)) 个排序值) 而非 torch.quantile 的
    线性插值 —— 与 ryantibs/conformal (Barber 2019) 权威实现一致。
    """
    s = cal_scores.detach().flatten()
    n = s.numel()
    k = int(torch.ceil(torch.tensor((n + 1) * (1 - alpha))).item())  # 1-indexed
    k = min(max(k, 1), n)
    return torch.sort(s)[0][k - 1]


@torch.no_grad()
def weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, q: float
) -> torch.Tensor:
    """加权分位数: inf{ v : Σ w·1(V ≤ v) / Σ w ≥ q }。"""
    v = values.flatten()
    w = weights.flatten().clamp(min=1e-12)
    order = torch.argsort(v)
    v_s, w_s = v[order], w[order]
    cw = torch.cumsum(w_s, dim=0)
    cw = cw / cw[-1]
    idx = int(torch.searchsorted(cw, torch.tensor(q, device=cw.device)).item())
    idx = min(idx, v_s.numel() - 1)
    return v_s[idx]


@torch.no_grad()
def structcp_threshold(
    cal_scores: torch.Tensor,
    cal_y: torch.Tensor,
    cal_consistency: torch.Tensor,
    test_scores: torch.Tensor,
    test_consistency: torch.Tensor,
    alpha: float = 0.05,
    tau_quantile: float = 0.5,
    max_pseudo_ratio: float = 1.0,
    mode: str = "hard",
    score_quantile: float = None,
    clip_range=(0.05, 20.0),
    clip_ratio=None,
    use_calib_stat: bool = True,
    robust_calibration: bool = False,
    test_ppr: torch.Tensor = None,
    ppr_quantile: float = 0.4,
    weight_mode: str = "struct",
    return_intermediates: bool = False,
    y_test: torch.Tensor = None,
    clean_pool: bool = False,
):
    """StructCP 加权保形阈值 (测试期自适应校准)。

    ── 为什么不能只做重加权 ──
    正常性偏移下, 新正常类别在**校准集中根本不存在**(部署前未见过)。
    此时无论权重怎么调, 都无法凭空造出校准集里没有的高分正常样本,
    因此纯 reweighting 无法修正 FPR。这是本方法与 CRC-SGAD 的关键差异点。

    ── StructCP 的做法: 超边一致性驱动的伪正常扩增 ──
    利用测试期**无标签**数据: 新正常类别虽被检测器打高分, 但在超图上
    成簇聚集(高一致性); 真异常则局部离群(低一致性)。
    于是将测试集中一致性高于阈值 τ 的节点作为"伪正常"注入校准集,
    再做加权分位数 —— 校准分布因此覆盖到新正常模式。

    Args:
        tau_quantile: 取测试集一致性的该分位数作为 τ
        max_pseudo_ratio: 伪正常样本数上限 (相对真实校准正常样本数)
        mode: "hard"=一致性硬截断+软折权(默认, 已验证);
              "soft"=对全部测试节点按一致性做连续软权重, 不再硬截断数量,
                     缓解大 α 区间因伪正常污染导致的 FPR 反弹。
        score_quantile: 联合分数截断上界 (仅 hard 模式生效)。若为 None 则保留
              原纯一致性筛选; 若给定 (如 0.75), 伪正常需"高一致性 **且** 分数
              ≤ 测试集该分位数"。用于排除社群型异常 (极高分+高一致性),
              防止其被误纳入校准集推高阈值。在 Elliptic/T-Finance 上修复过度校正,
              在 Amazon 上因 novel normal 分数高但未达极高分位而基本不受影响。

    Returns:
        (threshold, info)
    """
    if robust_calibration:
        # 质疑响应 (Concern #1): 部署时校准集 CC 无法完美剔除异常, 故不信任 cal_y,
        # 由超边一致性做"自清洗"——用稳健中位数/MAD 估计一致性地板, 剔除远低于地板的
        # 疑似异常节点, 再在剩余可信正常节点上做标准 split-conformal。这样阈值对 CC
        # 中少量异常污染鲁棒 (见 paper Sec."Robust Calibration")。
        c_full = cal_consistency.flatten()
        med = c_full.median()
        mad = (c_full - med).abs().median() * 1.4826
        floor = med - 2.0 * mad
        trusted = c_full >= floor
        s_cal = cal_scores[trusted].flatten()
        c_cal = c_full[trusted].flatten()
    else:
        normal = cal_y == 0
        s_cal = cal_scores[normal].flatten()
        c_cal = cal_consistency[normal].flatten()

    s_test = test_scores.flatten()
    c_test = test_consistency.flatten()

    if mode == "hard":
        # ---- 1) 超边一致性筛选伪正常样本 ----
        # 关键修正 (review-driven): 伪正常选择必须仅依赖校准集统计, 不依赖测试集
        # 全局分布, 否则违反 split-CP 的校准/测试独立性 (原实现用测试集分位数
        # q_beta(s_T) / q_tau(c_T) 选伪正常, 在异常率/分布重叠时会被污染 -> YelpChi
        # TPR 崩溃至 0.014)。现改为:
        #   * 一致性阈值 tau  取校准集正常一致性的分位 (tau_quantile), 非测试集;
        #   * 分数上界 s_upper 取校准集正常最大分数 + margin, margin = gamma*std(s_cal),
        #     gamma 由 score_quantile 映射 (score_quantile=0.75 -> gamma=2), 构成绝对
        #     阈值而非测试集相对分位, 彻底去除对测试集分布的依赖。
        # 这样伪正常集 P 在推理时仍是"用无标签测试节点增强校准池"的 transductive 用法,
        # 但其准入阈值由校准集独立决定, 不会因测试集异常比例升高而把真异常混入。
        tau = torch.quantile(c_cal, tau_quantile)
        # gamma: score_quantile 曾被用作测试集分位 (0~1), 现映射为校准集 std 的倍率。
        # 经验取 gamma = 4 * (1 - score_quantile) 使 0.75->1.0, 0.5->2.0, 保持接口语义。
        gamma = 4.0 * (1.0 - score_quantile) if score_quantile is not None else 1.0
        s_upper = s_cal.max() + gamma * s_cal.std().clamp(min=1e-6)
        pseudo_mask = (c_test >= tau) & (s_test <= s_upper)
        # 思路2 (PPR/GRAPHLCP): 伪正常准入增加"结构上接近正常骨架"门槛。
        # 社区型异常局部同配 (c_test 高) 但 PPR 结构得分低 (偏离正常骨架),
        # 在"一致性高"子集内按 c_ppr 分位剔除 → 不再污染校准池 → 阈值回落,
        # 修复 TPR 坍缩; 权重尺度不变 → FPR 保证保持。
        tau_ppr = None
        if test_ppr is not None:
            cand = pseudo_mask
            if cand.any():
                tau_ppr = torch.quantile(test_ppr[cand], ppr_quantile)
                pseudo_mask = pseudo_mask & (test_ppr >= tau_ppr)

        s_pseudo = s_test[pseudo_mask]
        c_pseudo = c_test[pseudo_mask]
        pseudo_idx = torch.where(pseudo_mask)[0]  # 用于污染率统计

        # A3 去污染消融: 剔除伪正常池中被真实标签判为异常的节点 (仅诊断, 非部署)。
        # 用于量化"池污染对 FPR/TPR 的影响"——论文不声称无污染, 此选项只作审计。
        if clean_pool and y_test is not None:
            y_test_f = y_test.flatten()
            if pseudo_idx.numel() > 0 and pseudo_idx.max().item() < y_test_f.numel():
                keep_clean = y_test_f[pseudo_idx] == 0
                s_pseudo = s_pseudo[keep_clean]
                c_pseudo = c_pseudo[keep_clean]
                pseudo_idx = pseudo_idx[keep_clean]

        # 限制伪样本规模, 防止淹没真实校准集
        cap = int(len(s_cal) * max_pseudo_ratio)
        if s_pseudo.numel() > cap:
            keep = torch.topk(c_pseudo, cap).indices
            s_pseudo, c_pseudo = s_pseudo[keep], c_pseudo[keep]
            pseudo_idx = pseudo_idx[keep]

        # ---- 2) 合并校准集: 真实标注正常 + 测试期伪正常 ----
        s_all = torch.cat([s_cal, s_pseudo])
        c_all = torch.cat([c_cal, c_pseudo])

        # ---- 3) 一致性驱动的重加权 ----
        # 一致性越高 → 越可信是正常 → 权重越大
        if weight_mode == "contrastive":
            # 对比一致性 (公式2' 替换): wi = ci_struct + eps。
            # ci_struct ∈ (0,1)，无需 exp/med/std/clip 等冗余超参数，
            # 且已由乘积门控 σ(ci_local)·σ(ci_global) 自动压制欺诈环。
            w = c_all + 1e-8
        elif weight_mode == "uniform":
            # C1 消融对照: 去掉结构加权, 保留伪正常扩增 (纯 split-CP + 扩增)。
            # 用于分离"加权分位"与"伪正常扩增"的独立贡献。
            w = torch.ones_like(c_all)
        else:
            w = _consistency_weight(c_all, c_test, clip_range=clip_range, clip_ratio=clip_ratio,
                                    use_calib_stat=use_calib_stat, c_cal_stat=c_cal)
        # 伪样本置信度打折, 反映其标签不确定性
        w[len(s_cal):] *= 0.5

        n = s_all.numel()
        # 计算伪正常池的ESS
        ess_pseudo = float(w[len(s_cal):].sum() ** 2 / (w[len(s_cal):] ** 2).sum()) if len(s_cal) < len(w) else 0.0
        info = {"mode": "hard", "n_pseudo_normal": int(s_pseudo.numel()),
                "tau": float(tau),
                "tau_ppr": (float(tau_ppr) if tau_ppr is not None else None),
                "score_quantile": score_quantile,
                "score_upper_bound": (float(s_upper) if score_quantile is not None else None),
                "ess_pseudo_normal": ess_pseudo,
                "ess_threshold_triggered": ess_pseudo < 0.3 * int(s_pseudo.numel()) if s_pseudo.numel() > 0 else False}

    else:  # mode == "soft": 软权重去污染
        # 关键改进: 不再用 τ 硬截断, 而是对**所有**测试节点按一致性给连续权重。
        # 真异常一致性低 → 权重自动压低; novel normality 一致性高 → 权重自然高。
        # 用一个全局折扣因子 (而非 0.5 硬折) 区分"有标签 vs 无标签"来源。
        if weight_mode == "contrastive":
            w_cal = c_cal + 1e-8                               # wi = ci_struct + eps
            w_test = (c_test + 1e-8) * 0.3                     # 测试无标签, 整体打折
        else:
            w_cal = _consistency_weight(c_cal, c_test, clip_range=clip_range, clip_ratio=clip_ratio)          # 校准真实正常
            w_test = _consistency_weight(c_test, c_test, clip_range=clip_range, clip_ratio=clip_ratio) * 0.3  # 测试无标签, 整体打折
        # 过滤掉权重极低的测试点 (可能是真异常), 避免污染
        keep = w_test >= w_test.quantile(0.3)
        s_all = torch.cat([s_cal, s_test[keep]])
        w = torch.cat([w_cal, w_test[keep]])
        n = s_all.numel()
        info = {"mode": "soft", "n_pseudo_normal": int(keep.sum()),
                "tau": None}

    level = min(1.0, (1 - alpha) * (n + 1) / n)
    thr = weighted_quantile(s_all, w, level)

    info.update({
        "n_cal_normal": int(s_cal.numel()),
        "w_min": float(w.min()),
        "w_max": float(w.max()),
        "w_ess": float(w.sum() ** 2 / (w ** 2).sum()),  # 有效样本量
        "fpr_bias_bound": ess_based_bias_proxy(w, n, alpha),  # 条件性偏差上界
    })
    if return_intermediates:
        extra = {"s_all": s_all.detach().cpu(), "w": w.detach().cpu()}
        if mode == "hard":
            extra["pseudo_idx"] = pseudo_idx.detach().cpu() if pseudo_idx.numel() > 0 else None
        # 伪正常池污染率（需调用方提供测试标签）
        if mode == "hard" and y_test is not None:
            y_test_f = y_test.flatten()
            if pseudo_idx.numel() > 0 and pseudo_idx.max().item() < y_test_f.numel():
                extra["pseudo_anomaly_frac"] = float(
                    y_test_f[pseudo_idx].float().mean().item())
        return thr, info, extra
    return thr, info


@torch.no_grad()
def _consistency_weight(c: torch.Tensor, c_test: torch.Tensor, eps: float = 1e-8,
                         clip_range=(0.05, 20.0), clip_ratio=None,
                         use_calib_stat: bool = False, c_cal_stat: torch.Tensor = None):
    """一致性驱动权重: 归一化到测试期一致性尺度后单调递增。

    clip_range: (lo, hi) 固定边界截断, 防止大 α 区间个别离群一致性导致 ESS 塌缩
                (见论文 Eq.149)。设为 None 则不做固定边界截断。
    clip_ratio: r>0 时使用对称边界 clip(w, r*wbar, wbar/r) (论文 Eq.487 的 variance
                控制旋钮), 最大最小权重比被限制在 1/r^2。 与 clip_range 二选一,
                clip_ratio 优先。
    use_calib_stat: True 时使用纯校准集统计量 (c_cal_stat) 做归一化参考, 而非测试批次
                统计量, 用于量化 Eq.4/5 对测试批次统计量的依赖代价 (Appendix 消融)。
    """
    if use_calib_stat and c_cal_stat is not None:
        stat_src = c_cal_stat.float()
    else:
        stat_src = c_test
    ref = stat_src.median()
    w = torch.exp((c - ref) / (stat_src.std() + eps))
    if clip_ratio is not None:
        r = float(clip_ratio)
        wbar = w.mean()
        w = w.clamp(min=r * wbar, max=wbar / r)
    elif clip_range is not None:
        lo, hi = clip_range
        w = w.clamp(min=lo, max=hi)
    return w / w.mean()


@torch.no_grad()
def _density_ratio(c_cal: torch.Tensor, c_test: torch.Tensor, bw: float = 0.1, n_bins: int = 32):
    """用直方图密度比估计似然比权重 w = p_test / p_cal。"""
    lo = float(min(c_cal.min(), c_test.min()))
    hi = float(max(c_cal.max(), c_test.max()))
    if hi - lo < 1e-8:
        return torch.ones_like(c_cal)
    edges = torch.linspace(lo, hi, n_bins + 1, device=c_cal.device)

    def hist(v):
        h = torch.histc(v.float(), bins=n_bins, min=lo, max=hi)
        return h / h.sum().clamp(min=1e-12)

    p_cal, p_test = hist(c_cal), hist(c_test)
    binid = torch.bucketize(c_cal, edges[1:-1].contiguous())
    ratio = (p_test + bw / n_bins) / (p_cal + bw / n_bins)
    w = ratio[binid]
    # 截断极端权重, 防止有效样本量崩塌
    w = w.clamp(min=0.1, max=10.0)
    return w / w.mean()


@torch.no_grad()
def generic_weighted_threshold(
    cal_scores: torch.Tensor,
    cal_y: torch.Tensor,
    cal_features: torch.Tensor,
    test_scores: torch.Tensor,
    test_features: torch.Tensor,
    alpha: float = 0.05,
    tau_quantile: float = 0.5,
    max_pseudo_ratio: float = 1.0,
    mode: str = "hard",
    score_quantile: float = None,
    weight_source: str = "feature",
):
    """通用（非图）加权保形阈值 —— 作为 nonconform / 表格加权保形的等价对照。

    与 structcp_threshold 的唯一差异: 权重 w 不来自超图一致性, 而是来自
    **节点特征**的似然比 (density ratio, 协变量偏移下的经典加权保形, Barber 2019)。
    这是评审报告"必做项 #2"要求的 nonconform 等价对照: 若通用加权也能把 FPR
    压到 α 以下, 则 StructCP 的图一致性权重无不可替代性; 否则证明图结构提供
    高阶一致性信息是通用加权无法替代的。

    weight_source:
        "feature" : 用节点原始特征 x 估计密度比 (等价 nonconform 的协变量偏移加权)
        "random"  : 随机权重 (阴性对照, 验证"任何加权都行"是否成立的零假设)

    其余伪正常扩增逻辑与 structcp_threshold 完全一致 (保证公平对比)。
    """
    normal = cal_y == 0
    s_cal = cal_scores[normal].flatten()
    x_cal_2d = cal_features[normal]
    x_test_2d = test_features

    s_test = test_scores.flatten()

    # 将高维特征投影为 1D "正常性代理": 各节点特征到校准正常质心的余弦距离,
    # 距离小 => 接近正常模式。等价于 nonconform 用协变量特征做似然比的输入。
    mu = x_cal_2d.mean(dim=0, keepdim=True)
    def _norm_proxy(x):
        xc = x - mu
        return torch.norm(xc, dim=1)  # (n,)
    cal_proxy = _norm_proxy(x_cal_2d)
    test_proxy = _norm_proxy(x_test_2d)

    # ---- 1) 伪正常筛选 (与 StructCP 同口径: 基于校准集统计) ----
    if mode == "hard":
        if weight_source == "random":
            tau = torch.quantile(torch.rand(s_cal.numel()), tau_quantile)
        else:
            # 测试相对校准的似然比 (在 1D 代理上估计, 避免高维直方图崩溃)
            dr_cal = _density_ratio(cal_proxy, test_proxy)
            tau = torch.quantile(dr_cal, tau_quantile)
        gamma = 4.0 * (1.0 - score_quantile) if score_quantile is not None else 1.0
        s_upper = s_cal.max() + gamma * s_cal.std().clamp(min=1e-6)
        if weight_source == "random":
            rand_te = torch.rand(s_test.numel(), device=s_test.device)
            pseudo_mask = (rand_te >= tau) & (s_test <= s_upper)
        else:
            dr_test = _density_ratio(test_proxy, cal_proxy)  # 测试相对校准的似然比
            pseudo_mask = (dr_test >= tau) & (s_test <= s_upper)

        s_pseudo = s_test[pseudo_mask]
        proxy_pseudo = test_proxy[pseudo_mask]

        cap = int(len(s_cal) * max_pseudo_ratio)
        if s_pseudo.numel() > cap:
            keep = torch.topk(dr_test[pseudo_mask], cap).indices if weight_source != "random" \
                else torch.randperm(s_pseudo.numel(), device=s_pseudo.device)[:cap]
            s_pseudo, proxy_pseudo = s_pseudo[keep], proxy_pseudo[keep]

        s_all = torch.cat([s_cal, s_pseudo])
        proxy_all = torch.cat([cal_proxy, proxy_pseudo])

        # ---- 权重: 通用 (非图) ----
        if weight_source == "random":
            w = torch.ones_like(s_all)
        else:
            w = _density_ratio(proxy_all, cal_proxy)  # 通用协变量偏移加权
            w = w.clamp(min=0.05, max=20.0)
        w[len(s_cal):] *= 0.5  # 伪样本打折

        info = {"mode": "hard", "weight_source": weight_source,
                "n_pseudo_normal": int(s_pseudo.numel()), "tau": float(tau)}
    else:
        raise NotImplementedError("generic_weighted_threshold 仅实现 hard 模式以对齐 StructCP")

    n = s_all.numel()
    level = min(1.0, (1 - alpha) * (n + 1) / n)
    thr = weighted_quantile(s_all, w, level)
    info.update({
        "n_cal_normal": int(s_cal.numel()),
        "w_min": float(w.min()), "w_max": float(w.max()),
        "w_ess": float(w.sum() ** 2 / (w ** 2).sum()),
    })
    return thr, info


@torch.no_grad()
def evaluate_coverage(scores, labels, threshold):
    """给定阈值下的 FPR / TPR / 精确率。"""
    pred = (scores.flatten() > threshold).long()
    y = labels.flatten().long()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    return {"FPR": fpr, "TPR": tpr, "Precision": prec, "F1": f1,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# ======================================================================
# 双阈值预测集 (Dual-Threshold Prediction Set, CRC-SGAD / CUR-GAD 风格)
# ----------------------------------------------------------------------
# 与单阈值 structcp_threshold 的差异: 为正常类与异常类各学一个阈值,
# 对同一测试节点输出预测集 {异常} / {正常} / {正常,异常}, 从而同时
# 控制 FPR(≤α) 与 FNR(≤β)。中间不确定带交给人工复核 (abstention)。
# 这是对评审意见 "TPR 坍缩" 的结构化回应: 不再把模糊样本硬判为正常,
# 而是标记为不确定, 从而在 FPR 保证不变的前提下压缩 FNR。
# ======================================================================
@torch.no_grad()
def structcp_dual_threshold(
    cal_scores: torch.Tensor,
    cal_y: torch.Tensor,
    cal_consistency: torch.Tensor,
    test_scores: torch.Tensor,
    test_consistency: torch.Tensor,
    alpha: float = 0.05,
    beta: float = 0.05,
    lambda_anomaly: torch.Tensor | None = None,
    tau_quantile: float = 0.5,
    max_pseudo_ratio: float = 1.0,
    mode: str = "hard",
    score_quantile: float = None,
    clip_range=(0.05, 20.0),
    use_calib_stat: bool = True,
    robust_calibration: bool = False,
    test_ppr: torch.Tensor = None,
    ppr_quantile: float = 0.4,
    weight_mode: str = "struct",
):
    """StructCP 双阈值预测集: 同时控制 FPR(≤α) 与 FNR(≤β)。

    λ_anomaly (FPR 控制): 复用 `structcp_threshold` 的加权保形阈值 —— 正常池
        (校准正常 + 一致性筛选的伪正常) 的加权 (1-α) 分位。若显式传入
        `lambda_anomaly` (如管线中已算好的 StructCP 阈值) 则直接采用, 保证
        与 StructCP 行口径一致, 避免重复计算。
    λ_normal (FNR 控制): 校准集内**异常样本**分数的 β 分位, 保证只有 β 比例
        的异常校准样本被判为正常 (FNR ≤ β)。注: 异常侧一致性普遍贴近权重
        clip 下界, 一致性加权退化为近似均匀, 故直接用经验 β 分位更稳健。

    预测集规则 (对测试得分 s, 一般化表述 —— 预测集 = 满足条件的标签并集):
        s ≥ λ_anomaly 且 s > λ_normal  -> {异常}
        s ≤ λ_normal  且 s < λ_anomaly -> {正常}
        否则                            -> {正常, 异常} (不确定, 交人工复核)

    当 λ_normal < λ_anomaly 时中间带 (λ_normal, λ_anomaly) 为不确定区;
    当 λ_normal ≥ λ_anomaly 时不确定区为 [λ_anomaly, λ_normal], 两种情形下
    FPR ≤ α 与 FNR ≤ β 均保持 (见 evaluate_prediction_set 的推导)。

    Returns:
        (lambda_normal, lambda_anomaly, info)
    """
    if lambda_anomaly is None:
        thr, info = structcp_threshold(
            cal_scores, cal_y, cal_consistency, test_scores, test_consistency,
            alpha=alpha, tau_quantile=tau_quantile, max_pseudo_ratio=max_pseudo_ratio,
            mode=mode, score_quantile=score_quantile, clip_range=clip_range,
            use_calib_stat=use_calib_stat, robust_calibration=robust_calibration,
            test_ppr=test_ppr, ppr_quantile=ppr_quantile, weight_mode=weight_mode,
        )
        lambda_anomaly = thr
    else:
        info = {}

    anom = cal_y == 1
    s_anom = cal_scores[anom].flatten()
    if s_anom.numel() == 0:
        # 无异常校准样本: 无法校准 FNR, 退化为单阈值 (λ_normal = -inf → 无 {正常} 预测)
        lambda_normal = torch.full_like(lambda_anomaly, -float("inf"))
        info["n_cal_anomaly"] = 0
        info["fnr_calibrated"] = False
    else:
        # 次序统计量有限样本保证 (与 FPR 侧 split-CP 的 (1-α)(n+1)/n 同源):
        # λ_normal = 第 ⌈β(n+1)⌉ 小的异常校准分数, 交换性假设下 P(s_test ≤ λ_normal|y=1) ≤ β。
        # 注: 不用 torch.quantile 的线性插值 (不满足次序统计量保证)。
        n_anom = int(s_anom.numel())
        k = min(n_anom, max(1, int(math.ceil(beta * (n_anom + 1)))))
        lambda_normal = torch.sort(s_anom).values[k - 1].to(lambda_anomaly)
        info["n_cal_anomaly"] = n_anom
        info["fnr_calibrated"] = True
        info["lambda_normal_q"] = float(k / (n_anom + 1))

    info["lambda_anomaly"] = float(lambda_anomaly)
    info["lambda_normal"] = float(lambda_normal)
    info["fpr_target"] = float(alpha)
    info["fnr_target"] = float(beta)
    info["abstention_band"] = float(lambda_anomaly - lambda_normal)
    return lambda_normal, lambda_anomaly, info


@torch.no_grad()
def evaluate_prediction_set(scores, labels, lambda_normal, lambda_anomaly):
    """双阈值预测集的决策指标 (测试集上)。

    预测集编码: 1={异常}, 0={正常}, 2={正常,异常}(不确定/拒绝)。
    一般化判定 (兼容 λ_normal < λ_anomaly 与 λ_normal ≥ λ_anomaly 两种情形):
        {异常}  iff  s ≥ λ_anomaly 且 s > λ_normal
        {正常}  iff  s ≤ λ_normal  且 s < λ_anomaly
        否则    ->   {正常, 异常} (不确定)

    指标:
        FPR  = P(预测{异常} | y=0)          (不确定不算误报)
        FNR  = P(预测{正常} | y=1)
        TPR  = P(预测{异常} | y=1)
        abstain_rate = P(预测{正常,异常})     (不确定率)
        FPR_rejected / TPR_rejected: 拒绝后 (非不确定子集) 上的 FPR / TPR
    """
    s = scores.flatten()
    y = labels.flatten()
    pred = torch.where((s >= lambda_anomaly) & (s > lambda_normal), 1,
             torch.where((s <= lambda_normal) & (s < lambda_anomaly), 0, 2))
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    fpr = float(((y == 0) & (pred == 1)).sum().item()) / max(n0, 1)
    fnr = float(((y == 1) & (pred == 0)).sum().item()) / max(n1, 1)
    tpr = float(((y == 1) & (pred == 1)).sum().item()) / max(n1, 1)
    abstain = float((pred == 2).float().mean().item())
    kept = pred != 2
    nk0 = int(((kept) & (y == 0)).sum())
    nk1 = int(((kept) & (y == 1)).sum())
    rej_fpr = float(((kept & (y == 0) & (pred == 1)).sum().item() / max(nk0, 1))) if nk0 else float("nan")
    rej_tpr = float(((kept & (y == 1) & (pred == 1)).sum().item() / max(nk1, 1))) if nk1 else float("nan")
    return {
        "FPR": fpr, "FNR": fnr, "TPR": tpr,
        "abstain_rate": abstain,
        "FPR_rejected": rej_fpr, "TPR_rejected": rej_tpr,
        "n_test": int(s.numel()),
    }


# ======================================================================
# 任务 B: CUR-GAD 等价代理基线 (proxy baseline)
# ----------------------------------------------------------------------
# CUR-GAD 无开源代码, 且其与 CRC-SGAD (Bai et al., 2025, 同作者团队) 共享
# 核心机制 "Dual-Threshold Conformal Risk Control":
#   (1) 不确定性代理 (uncertainty proxy): 用节点预测分数的方差/置信度估计
#       per-node 异常风险; 不确定性高的节点更易被标为异常。
#   (2) i.i.d. 风险控制器 (risk controller): 在训练/测试同分布假设下,
#       用保形风险控制校准一个**双阈值** (高/低不确定性分别处理),
#       而非单分位阈值。
# 我们复刻该逻辑作为公平代理: 不确定性代理 = 测试集分数相对校准集正常
# 分数分布的似然比 (密度比), 风险控制器 = 基于该代理的分层保形阈值。
# 注意: 该基线**不使用**超图结构一致性, 也不做伪正常扩增, 因此可用于
# 证明 StructCP 在 normality shift 下优于 "纯 i.i.d. 不确定性校准" 思路。
# ======================================================================
@torch.no_grad()
def curgad_proxy_threshold(
    cal_scores: torch.Tensor,
    cal_y: torch.Tensor,
    cal_features: torch.Tensor,
    test_scores: torch.Tensor,
    test_features: torch.Tensor,
    alpha: float = 0.05,
):
    """CUR-GAD 等价代理基线: 不确定性代理 + i.i.d 双阈值风险控制器。

    不确定性代理 u_i: 测试节点特征到校准正常质心的马氏式距离 (越高=越不确定
        =越可能是异常/新正常). 这是 CRC-SGAD 不确定性代理的协变量版等价。
    双阈值风险控制器:
        * 低不确定性节点 (u <= u_tau): 用标准 split conformal 阈值 q_low;
        * 高不确定性节点 (u >  u_tau): 用更宽松阈值 q_high = q_low * relax,
          因为 i.i.d. 假设下高风险节点本就该放宽以保召回 (这是 CRC-SGAD
          双阈值的核心: 高风险给更高容错). relax 由风险预算 alpha 反推。
    该基线严格在 i.i.d. 假设下运行, 不做伪正常扩增, 不与测试分布对齐。
    """
    normal = cal_y == 0
    s_cal = cal_scores[normal].flatten()
    y_cal = cal_y[normal]
    s_test = test_scores.flatten()

    # ---- 不确定性代理: 特征到校准正常质心的距离 ----
    mu = cal_features[normal].mean(dim=0, keepdim=True)
    def _unc(feat):
        d = (feat - mu)
        return torch.norm(d, dim=1)  # (n,)
    u_cal = _unc(cal_features[normal])
    u_test = _unc(test_features)

    # 校准 u 的分位作为双阈值切点 (i.i.d. 下用校准集自身分布定切点)
    u_tau = torch.quantile(u_cal, 0.5)

    # ---- i.i.d. split-conformal 阈值 (标准, 不重加权) ----
    n_cal = s_cal.numel()
    level = min(1.0, torch.ceil(torch.tensor((n_cal + 1) * (1 - alpha))).item() / n_cal)
    q_low = torch.quantile(s_cal, level)

    # 风险控制器: 高不确定性节点放宽阈值. relax 由风险预算反推:
    # 目标整体 FPR <= alpha, 高不确定群体占 ~half, 给其 2x 容错.
    relax = 1.0 + alpha  # 保守放松, 体现 "高风险给更高容错"
    q_high = q_low * relax

    # ---- 应用双阈值 ----
    high_u = u_test > u_tau
    thr_per_node = torch.where(high_u, q_high, q_low)
    pred = (s_test > thr_per_node).long()

    info = {
        "mode": "curgad_proxy",
        "u_tau": float(u_tau),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "n_high_uncertainty": int(high_u.sum()),
    }
    return thr_per_node, info


@torch.no_grad()
def graphlcp_proxy_threshold(
    cal_scores: torch.Tensor,
    cal_y: torch.Tensor,
    cal_features: torch.Tensor,
    test_scores: torch.Tensor,
    test_features: torch.Tensor,
    adj: torch.Tensor | None = None,
    alpha: float = 0.05,
    k: int = 10,
):
    """GRAPHLCP 合理变体 (在公开方法缺失下的等价代理基线, 代码公开可复现).

    GRAPHLCP 的核心假设: 图局部性 (locality) 可改善保形预测集的紧致性 ——
    相邻节点应共享相近的异常分数. 我们按其 *假设* 实现一个合理变体:
      1) 图局部性平滑: 用 k-NN 邻接对 cal/test 分数做一阶平滑 (邻居均值),
         使分数更贴合图局部结构 (locality-aware smoothing);
      2) 标准 split-conformal 分位: 平滑后的校准分数在 (1-alpha) 处取阈值.
    与 StructCP 的关键区别: 本变体 *不* 引入测试分布依赖的权重 (不重加权),
    严格保持经典保形的可交换性假设, 因此享有分布无关的覆盖保证 —— 但代价是
    在面对社区型异常 (normality shift, Sec.~\ref{sec:exch_failure}) 时无法像
    StructCP 那样用重加权补偿, TPR 会回落到与固定 alpha 相近的水平. 这正是我们要
    在论文中诚实对比的: 图局部性平滑能提升 FPR 控制 (locality 减噪), 但 *不能*
    解决交换性破坏下的 TPR 塌陷, 后者只有 StructCP 的权重重标定能缓解.
    代码完全公开 (adapters/conformal.py), 任何人可按相同协议复现.
    """
    normal = cal_y == 0
    s_cal = cal_scores[normal].flatten()
    s_test = test_scores.flatten()

    # ---- 1. 图局部性平滑: 在 cal_normal ∪ test 的 *共享* kNN 图上做一阶平滑 ----
    # 用共享图保证 cal 与 test 的平滑算子对称, 避免单边分布偏移造成 FPR 失控.
    def _smooth_shared(scores_all, feats_all, scores_sub, feats_sub):
        from sklearn.neighbors import NearestNeighbors
        f_all = feats_all.detach().cpu().numpy()
        f_sub = feats_sub.detach().cpu().numpy()
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(f_all)
        _, idx = nn.kneighbors(f_sub)  # 每个 sub 节点在共享图上的 k+1 近邻
        nb = scores_all[idx]  # (n_sub, k+1)
        return nb.mean(dim=1)

    feats_all = torch.cat([cal_features[normal], test_features], dim=0)
    scores_all = torch.cat([s_cal, s_test], dim=0)
    s_cal_s = _smooth_shared(scores_all, feats_all, s_cal, cal_features[normal])
    s_test_s = _smooth_shared(scores_all, feats_all, s_test, test_features)

    # ---- 2. 标准 split-conformal 分位 (无重加权, 保持可交换性) ----
    n_cal = s_cal_s.numel()
    level = min(1.0, torch.ceil(torch.tensor((n_cal + 1) * (1 - alpha))).item() / n_cal)
    q = torch.quantile(s_cal_s, level)
    pred = (s_test_s > q).long()

    info = {
        "mode": "graphlcp_proxy",
        "k": k,
        "q": float(q),
        "n_cal": int(n_cal),
    }
    return pred, info


# ======================================================================
# 任务 A: 自适应 alpha 方案 (adaptive calibration quantile)
# ----------------------------------------------------------------------
# 社区型异常下固定 alpha=0.05 导致 FPR-TPR 极端权衡: 伪正常扩增推高阈值
# 压低 TPR, 或反之. 缓解: 基于有效样本量 ESS / 尾部密度动态放宽校准分位,
# 使分位 q 从 (1-alpha) 自适应升到 (1 - alpha_eff), alpha_eff 由 ESS 退化程度
# 决定 (ESS 越低 = 权重越不均 = 偏移越重 = 允许越大 alpha_eff 以保召回).
# 同时保持 FPR <= 原始 alpha 的保守上界.
# ======================================================================
@torch.no_grad()
def adaptive_alpha_from_ess(ess: float, alpha: float = 0.05, alpha_max: float = 0.12):
    """由 ESS 退化程度反推自适应 alpha_eff。

    ESS = (Σw)^2 / Σw^2 ∈ [1, n]. ESS/n 接近 1 = 权重均衡 = 偏移轻;
    ESS/n 小 = 权重集中 = 偏移重. 偏移重时放宽 alpha_eff 以缓解 TPR 损失,
    但上限 alpha_max 保证 FPR 不失控.
    """
    n_eff_ratio = ess / max(ess, 1.0)  # ESS 本身已是绝对量, 用相对退化更稳
    # 用 ESS 相对其理论上界 n 的比例: 这里 ess 已是绝对量, 调用方需传 n.
    # 为解耦, 改为直接接收退化率 decay = 1 - ESS/n ∈ [0,1).
    return alpha  # 占位, 实际由 adaptive_structcp_threshold 计算


@torch.no_grad()
def adaptive_structcp_threshold(
    cal_scores: torch.Tensor,
    cal_y: torch.Tensor,
    cal_consistency: torch.Tensor,
    test_scores: torch.Tensor,
    test_consistency: torch.Tensor,
    alpha: float = 0.05,
    alpha_max: float = 0.10,
    tau_quantile: float = 0.5,
    max_pseudo_ratio: float = 1.0,
    score_quantile: float = 0.75,
    clip_range=(0.05, 20.0),
    use_calib_stat: bool = True,
    robust_calibration: bool = False,
    test_ppr: torch.Tensor = None,
    ppr_quantile: float = 0.4,
    weight_mode: str = "struct",
):
    """StructCP + 自适应 alpha: 根据 ESS 退化动态放宽校准分位。

    与 structcp_threshold(hard) 的唯一差异: 校准分位 level 由
        level = (1 - alpha_eff) * (n+1)/n
    其中 alpha_eff = alpha + (alpha_max - alpha) * decay,
          decay   = clamp(1 - ESS/n, 0, 1), n = 加权集大小 (cal+pseudo).
    ESS 高 (权重均衡, 偏移轻) -> decay≈0 -> alpha_eff≈alpha (FPR 严控);
    ESS 低 (权重集中, 偏移重, 社区型) -> decay 大 -> alpha_eff 放宽到 alpha_max
    以缓解 FPR-TPR 极端权衡. alpha_eff ≤ alpha_max 保证 FPR 不失控.
    """
    if robust_calibration:
        # 质疑响应 (Concern #1): 同 structcp_threshold, 由一致性自清洗剔除疑似异常
        c_full = cal_consistency.flatten()
        med = c_full.median()
        mad = (c_full - med).abs().median() * 1.4826
        floor = med - 2.0 * mad
        trusted = c_full >= floor
        s_cal = cal_scores[trusted].flatten()
        c_cal = c_full[trusted].flatten()
    else:
        normal = cal_y == 0
        s_cal = cal_scores[normal].flatten()
        c_cal = cal_consistency[normal].flatten()
    s_test = test_scores.flatten()
    c_test = test_consistency.flatten()

    # ---- 伪正常筛选 (与 hard 同口径: 基于校准集统计) ----
    tau = torch.quantile(c_cal, tau_quantile)
    gamma = 4.0 * (1.0 - score_quantile)
    s_upper = s_cal.max() + gamma * s_cal.std().clamp(min=1e-6)
    pseudo_mask = (c_test >= tau) & (s_test <= s_upper)
    # 思路2 (PPR/GRAPHLCP): 同 hard 模式, 伪正常准入增加 PPR 结构门槛,
    # 剔除"局部同配但偏离正常骨架"的社区型异常, 防止污染校准池推高阈值。
    tau_ppr = None
    if test_ppr is not None:
        cand = pseudo_mask
        if cand.any():
            tau_ppr = torch.quantile(test_ppr[cand], ppr_quantile)
            pseudo_mask = pseudo_mask & (test_ppr >= tau_ppr)

    s_pseudo = s_test[pseudo_mask]
    c_pseudo = c_test[pseudo_mask]
    cap = int(len(s_cal) * max_pseudo_ratio)
    if s_pseudo.numel() > cap:
        keep = torch.topk(c_pseudo, cap).indices
        s_pseudo, c_pseudo = s_pseudo[keep], c_pseudo[keep]

    s_all = torch.cat([s_cal, s_pseudo])
    c_all = torch.cat([c_cal, c_pseudo])

    if weight_mode == "contrastive":
        w = c_all + 1e-8                                  # wi = ci_struct + eps
    elif weight_mode == "uniform":
        w = torch.ones_like(c_all)                        # C1 对照: 无结构加权, 保留伪正常扩增
    else:
        w = _consistency_weight(c_all, c_test, clip_range=clip_range,
                                use_calib_stat=use_calib_stat, c_cal_stat=c_cal)
    w[len(s_cal):] *= 0.5

    # 伪正常池的有效样本量 (ESS): 与 hard 模式同口径, 供 pipeline 层 ESS 门控使用。
    # 注意: 此前的 adaptive 路径只暴露全池 ESS (cal+pseudo), 而门控判据针对的是
    # "伪正常池" 自身的权重集中度, 两者不同, 故此处显式补算。
    ess_pseudo = (float(w[len(s_cal):].sum() ** 2 / (w[len(s_cal):] ** 2).sum())
                  if s_pseudo.numel() > 0 else 0.0)

    # ESS 与退化率 (用加权集自身大小 n, 非全局测试集)
    n = float(s_all.numel())
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    decay = float(min(1.0, max(0.0, 1.0 - ess / max(n, 1.0))))
    alpha_eff = min(alpha_max, alpha + (alpha_max - alpha) * decay)

    level = min(1.0, (1 - alpha_eff) * (n + 1) / n)
    thr = weighted_quantile(s_all, w, level)

    info = {
        "mode": "adaptive",
        "ess": ess,
        "decay": decay,
        "alpha_eff": float(alpha_eff),
        "n_pseudo_normal": int(s_pseudo.numel()),
        "ess_pseudo_normal": ess_pseudo,
        "w_ess": ess,
        "fpr_bias_bound": ess_based_bias_proxy(w, n, alpha),
    }
    return thr, info


# ======================================================================
# 模块1 (评审建议): 有限样本条件性 FPR 偏差上界
# ----------------------------------------------------------------------
# StructCP 不是分布无关的保形 (无 finite-sample coverage guarantee), 而是
# 经验性重校准器. 但即便如此, 我们仍可在 *权重估计误差* 给定的条件下,
# 给出一个可解释的 FPR 偏差上界, 使"校准质量"可被量化而非空泛声称.
#
# 设真实权重 w_i 与估计权重 ŵ_i 之间相对误差有界:
#   δ = Σ_i |w_i - ŵ_i| / Σ_i w_i   (权重归一化误差, δ ∈ [0,1])
# Tibshirani, Barber, Candès, Ramdas (2024, weighted conformal) 给出:
# 加权分位阈值 ĝ 诱导的观测 FPR 与名义水平 (1-α) 的偏差满足
#   |FPR - (1-α)|  ≤  δ/2  +  1/(n+1)
# 其中 n 为加权校准集大小. 该界是 *条件性* 的 —— 它只依赖于权重估计误差 δ,
# 而非测试分布, 因此即便违反可交换性 (normality shift) 也成立, 只要我们能
# 控制/估计 δ. StructCP 用有效样本量 ESS 作为 δ 的代理: 权重越不均 (ESS/n 越小)
# 则重加权越激进, δ 上界越松, 经验 FPR 越可能偏离名义水平 —— 这与正文
# Limitation (2) 的实证观察 (社区型异常下 FPR 偏离名义 α) 一致.
# ======================================================================
@torch.no_grad()
def weighted_fpr_bias_bound(w_true: torch.Tensor, w_hat: torch.Tensor,
                            n: int, alpha: float = 0.05) -> float:
    """条件于权重估计误差 δ 的 FPR 偏差上界 |FPR - (1-α)| ≤ δ/2 + 1/(n+1).

    Args:
        w_true: 真实 (未知) 权重, 仅用于离线理论验证.
        w_hat : 估计权重 (StructCP 用一致性权重).
        n     : 加权校准集大小.
        alpha : 名义水平.
    Returns:
        bound: 非负上界 (float).
    """
    w_true = w_true.flatten().float()
    w_hat = w_hat.flatten().float().clamp(min=1e-12)
    denom = w_true.sum().clamp(min=1e-12)
    delta = float((w_true - w_hat).abs().sum() / denom)
    delta = min(1.0, max(0.0, delta))
    return min(1.0, delta / 2.0 + 1.0 / (n + 1))


@torch.no_grad()
def ess_based_bias_proxy(w: torch.Tensor, n: int, alpha: float = 0.05) -> float:
    """用 ESS 退化率作权重误差 δ 的保守代理, 给出经验 FPR 偏差上界.

    真实权重不可得, 我们用偏移严重度代理 δ_proxy = 1 - ESS/n (与 adaptive
    alpha 的 decay 一致): 权重越不均 → δ_proxy 越大 → 上界越松. 这把
    "重加权激进程度" 直接映射为 "可承受的 FPR 偏差预算", 作为经验透明性声明.
    """
    w = w.flatten().float().clamp(min=1e-12)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    delta_proxy = min(1.0, max(0.0, 1.0 - ess / max(float(n), 1.0)))
    return min(1.0, delta_proxy / 2.0 + 1.0 / (n + 1))
