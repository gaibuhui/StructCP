"""实验 1.1：一致性权重 w_i 与图拉普拉斯二次型 (Dirichlet 能量) 的相关性分析。

目的
----
将 Heuristic H1 从"观察到上尾边际有效"提升为"基于图信号处理"的解释：
一致性权重 w_i (∝ c_i) 若本质在捕捉图拉普拉斯的局部谱能量 —— 平滑(低频)区域
w_i 高、边界(高频)区域 w_i 低 —— 则 w_i 隐含了一次低通门控，加权分位天然偏向
平滑区域，从而拉高阈值 (保守上尾边际 Δ_tail > 0)。本实验直接检验该机制。

做法 (对应用户实验 1.1)
----
每个数据集 × 5 种子:
  1. 冻结 BWGNN，用原始特征 data.x 建 k-NN 超图 ("建一次，双重用途")；
  2. 默认 StructCP 协议 (β=0, 无高通) 超图传播 → x_enh；
  3. 算分数 s_tta 与一致性 c_i，按真实管线构造增广池 A 与权重 w_i；
  4. 在同一 k-NN 图上算**度归一化** Dirichlet 能量 (用户给定定义)
         E(i) = Σ_{j∈N(i)} ‖ x_i/√d_i − x_j/√d_j ‖²
     N(i)=k-NN 邻域(不含自身)，d_i=超图节点度 D_v(i) (节点出现在多少条超边中，
     即局部密度)；除以 √d 正是归一化拉普拉斯二次型的形式。
  5. 在三个信号上分别计算：x_enh (主)、x_raw (对照)、s_tta (1-D 分数信号)；
  6. 相关分析：Pearson / Spearman + 显著性，并额外给
       - 控制分数 s 的**偏相关** (排除"w-E 相关只是分数驱动"的替代解释)；
       - 按 E 四分位分组的 mean(w) 单调性检查 (非参数、审稿人易读)；
  7. 跨种子 mean±std + pooled Pearson (增量累积一二阶矩)。

保真性保证 (关键)
----
相关性必须针对**实际参与阈值计算的那些 w_i**，否则结论无意义。因此本脚本
复刻 `structcp_threshold` 内部的伪正常筛选与加权流程，并把复刻得到的阈值与
真实函数返回的阈值**逐位比对** (thr_match)；只有 match=True 的记录才纳入汇总。
注意两处易错点 (v1 版本即在此出错):
  * 权重统计量用的是**校准集正常节点** (y_cal==0)，不是整个 cal_mask (含异常)；
  * 伪正常权重额外乘 0.5 (标签不确定性折扣)，会引入组间常数偏移，故同时报告
    w_pre (折扣前) 与 w_post (折扣后，即实际使用值)，并在组内 (cal/pseudo 分开)
    给出不受该偏移影响的相关。

运行 (R6 复用 conda CBP；R7 8GB 显存 → CPU + 分块计算)
----
cd /media/lixin/新加卷/数据集/test/StructCP && \\
source <CONDA_PREFIX>/bin/activate CBP && \\
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
python scripts/analysis/weight_dirichlet_correlation.py
"""
from __future__ import annotations

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
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_sys.path.insert(0, ROOT)

from adapters.conformal import (  # noqa: E402
    _consistency_weight,
    structcp_threshold,
    weighted_quantile,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph,
    hyperedge_consistency,
    hypergraph_propagation,
)
from data.loader import load_gad  # noqa: E402
from utils.checkpoint import load_frozen_detector  # noqa: E402

# ----------------------------- 配置 -----------------------------
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
SEEDS = [42, 123, 456, 789, 1011]
ALPHA = 0.05
K = 10
LAYERS = 2
ALPHA_RES = 0.5
TAU = 0.5            # tau_quantile, 与 DEFAULT_DATASET_CFG 一致
SCORE_QUANTILE = 0.75  # 同上 (gamma = 4*(1-0.75) = 1.0)
CHUNK = 32768        # 分块大小, 控制 Elliptic(203k 节点) 的峰值内存

PRIMARY_SIGNAL = "E_enh"  # 主信号: 传播特征上的度归一化 Dirichlet 能量


# ----------------------------- 统计工具 -----------------------------
def _rankdata(v) -> np.ndarray:
    """平均秩 (ties 取平均)。放在模块层: 偏相关也依赖它,
    不能只定义在下面的 numpy 兜底分支里 (否则 scipy 可用时 NameError)。"""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    ranks_sorted = np.arange(1, len(v) + 1, dtype=float)
    sv = v[order]
    i = 0
    while i < len(v):                      # 处理并列
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


try:
    from scipy import stats as _stats

    def _pearson(a, b):
        r, p = _stats.pearsonr(a, b)
        return float(r), float(p)

    def _spearman(a, b):
        r, p = _stats.spearmanr(a, b)
        return float(r), float(p)

except Exception:  # 纯 numpy 兜底
    def _pearson(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        da, db = a - a.mean(), b - b.mean()
        denom = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
        if denom <= 0:
            return 0.0, 1.0
        r = float((da * db).sum()) / denom
        n = len(a)
        if n <= 2 or abs(r) >= 1.0:
            return r, 0.0
        t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return r, float(p)

    def _spearman(a, b):
        return _pearson(_rankdata(a), _rankdata(b))


def _residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """x 对 z 做一元线性回归后的残差 (用于偏相关)。"""
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    A = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _partial_pearson(x, y, z) -> float:
    """控制 z 后 x 与 y 的 Pearson 偏相关。"""
    try:
        rx, ry = _residualize(np.asarray(x, float), z), _residualize(np.asarray(y, float), z)
        if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
            return float("nan")
        return float(_pearson(rx, ry)[0])
    except Exception:
        return float("nan")


def _partial_spearman(x, y, z) -> float:
    """控制 z 后 x 与 y 的**秩**偏相关。

    必须与主报告的 Spearman 同度量: 若用 Pearson 偏相关去和 Spearman 并排展示,
    在重尾变量 (如 Dirichlet 能量) 上会给出不可比的数值甚至误导性结论。
    """
    try:
        return _partial_pearson(_rankdata(np.asarray(x, float)),
                                _rankdata(np.asarray(y, float)),
                                _rankdata(np.asarray(z, float)))
    except Exception:
        return float("nan")


# ----------------------------- 能量 -----------------------------
@torch.no_grad()
def dirichlet_energy(x: torch.Tensor, knn_idx: torch.Tensor,
                     dv_inv_sqrt: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """度归一化 Dirichlet 能量: E(i) = Σ_{j∈N(i)} ‖ x_i/√d_i − x_j/√d_j ‖²。

    x: (n, d) 信号; knn_idx: (n, k+1) 含自身, 取 [:, 1:] 作为 N(i);
    dv_inv_sqrt: D_v^{-1/2} = 1/√d, 故 x/√d = x * dv_inv_sqrt (广播到列)。

    分块计算以避免 Elliptic 上 (n×k×d) 中间张量一次性驻留内存。
    """
    xn = x * dv_inv_sqrt.unsqueeze(1)          # (n, d), 已按 1/√d 缩放
    n = xn.shape[0]
    out = torch.empty(n, dtype=xn.dtype, device=xn.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        idx = knn_idx[s:e, 1:]                 # (bs, k)
        diff = xn[s:e].unsqueeze(1) - xn[idx]  # (bs, k, d)
        out[s:e] = (diff * diff).sum(dim=-1).sum(dim=-1)
    return out


# ----------------------------- 一致性分量 -----------------------------
@torch.no_grad()
def consistency_components(scores: torch.Tensor, hypergraph: KNNHypergraph,
                           features: torch.Tensor, cal_mask: torch.Tensor,
                           eps: float = 1e-8):
    """拆分 hyperedge_consistency 的两个分量，与真实实现逐行对齐。

    返回 (c_score, c_feat, dev, dist):
      dev  = |s_i - median(s_{N(i)})|              分数邻域偏离 (分数粗糙度)
      dist = 1 - cos(f_i, centroid(f_{N(i)}))      到超边质心的余弦距离 (特征粗糙度)
      c_score = exp(-dev  / scale_s), scale_s = mean(dev[cal_mask])
      c_feat  = exp(-dist / scale_f), scale_f = mean(dist[cal_mask])
    真实 c = sqrt(c_score * c_feat)。

    为什么必须拆分: c_feat 的底层量 dist 本身就是一种"局部平滑度"统计量
    (到邻域质心的距离)，因此 c 与 Dirichlet 能量的相关**部分由构造决定**。
    只有分别看两个分量，才能回答"哪个分量在真正感知谱能量"。
    """
    s = scores.detach().flatten()
    idx = hypergraph.knn_idx                       # (n, k+1) 含自身

    # --- 分数分量 ---
    nb = s[idx]
    med = nb.median(dim=1).values
    dev = (s - med).abs()
    scale_s = dev[cal_mask].mean()
    c_score = torch.exp(-dev / (scale_s + eps))

    # --- 特征分量 ---
    f = features.detach()
    f = f / (f.norm(dim=1, keepdim=True) + eps)
    centroid = f[idx].mean(dim=1)
    centroid = centroid / (centroid.norm(dim=1, keepdim=True) + eps)
    dist = 1.0 - (f * centroid).sum(dim=1)         # cosine 距离
    scale_f = dist[cal_mask].mean()
    c_feat = torch.exp(-dist / (scale_f + eps))

    return c_score, c_feat, dev, dist


def _corr_pair(x: np.ndarray, E: np.ndarray, z: np.ndarray | None = None) -> dict:
    """x 与 E 的 Pearson/Spearman + (可选) 控制 z 的偏相关。"""
    out = {}
    if len(x) < 3 or np.std(E) < 1e-12 or np.std(x) < 1e-12:
        return {"degenerate": True}
    pr, pp = _pearson(x, E)
    sr, sp = _spearman(x, E)
    out["pearson"] = round(pr, 4)
    out["p_pearson"] = float(f"{pp:.3e}")
    out["spearman"] = round(sr, 4)
    out["p_spearman"] = float(f"{sp:.3e}")
    if z is not None:
        # 秩偏相关, 与上面的 spearman 同度量
        out["partial_given_other"] = round(_partial_spearman(x, E, z), 4)
    return out


# ----------------------------- 相关 -----------------------------
def corr_block(w: np.ndarray, c: np.ndarray, E: np.ndarray, s: np.ndarray) -> dict:
    """给定节点子集, 算 w/c 与 E 的 Pearson/Spearman/p, 及控制分数的偏相关。"""
    out = {}
    if len(w) < 3 or np.std(E) < 1e-12 or np.std(w) < 1e-12:
        return {"n": int(len(w)), "degenerate": True}
    pr, pp = _pearson(w, E)
    sr, sp = _spearman(w, E)
    cr, cp = _pearson(c, E)
    out.update(
        n=int(len(w)),
        pearson=round(pr, 4), p_pearson=float(f"{pp:.3e}"),
        spearman=round(sr, 4), p_spearman=float(f"{sp:.3e}"),
        pearson_c=round(cr, 4),
        partial_given_score=round(_partial_pearson(w, E, s), 4),
    )
    # 非参数单调性: 按 E 四分位分组的 mean(w)
    try:
        q = np.quantile(E, [0.25, 0.5, 0.75])
        grp = np.digitize(E, q)  # 0..3 低能量->高能量
        out["mean_w_by_E_quartile"] = [round(float(w[grp == g].mean()), 4)
                                       if int((grp == g).sum()) > 0 else None
                                       for g in range(4)]
    except Exception:
        out["mean_w_by_E_quartile"] = None
    # Pearson 矩, 供跨种子 pooled 累积
    out["_mom"] = [int(len(w)), float(w.sum()), float(E.sum()),
                   float((w * E).sum()), float((w * w).sum()), float((E * E).sum())]
    return out


# ----------------------------- 主分析 -----------------------------
@torch.no_grad()
def analyze_one(dataset: str, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model, _ = load_frozen_detector(dataset, "homo", root=ROOT, device=device)

    cal_m = torch.as_tensor(data.cal_mask, dtype=torch.bool, device=device)
    test_m = torch.as_tensor(data.test_mask, dtype=torch.bool, device=device)
    y_cal = data.y[cal_m]

    x_raw = data.x
    hg = KNNHypergraph(x_raw, k=K).to(device)
    x_enh = hypergraph_propagation(x_raw, hg, LAYERS, ALPHA_RES,
                                   beta_highpass=0.0, deg_gate=False)
    s_tta = F.softmax(model(x_enh, data.edge_index), dim=1)[:, 1]
    c = hyperedge_consistency(s_tta, hg, features=x_enh, cal_mask=cal_m)

    # ---- 复刻 structcp_threshold 的池构造与加权 (保真) ----
    normal = (y_cal == 0)
    cal_idx = torch.where(cal_m)[0]
    test_idx = torch.where(test_m)[0]
    cal_normal_idx = cal_idx[normal]

    s_cal, c_cal = s_tta[cal_normal_idx], c[cal_normal_idx]
    s_test, c_test = s_tta[test_idx], c[test_idx]

    tau = torch.quantile(c_cal.float(), TAU)
    gamma = 4.0 * (1.0 - SCORE_QUANTILE)
    s_upper = s_cal.max() + gamma * s_cal.std().clamp(min=1e-6)
    pseudo_mask = (c_test >= tau) & (s_test <= s_upper)
    pseudo_idx = test_idx[pseudo_mask]
    c_pseudo = c[pseudo_idx]
    cap = int(s_cal.numel())            # max_pseudo_ratio = 1.0
    if pseudo_idx.numel() > cap:
        keep = torch.topk(c_pseudo, cap).indices
        pseudo_idx = pseudo_idx[keep]
        c_pseudo = c_pseudo[keep]

    pool_idx = torch.cat([cal_normal_idx, pseudo_idx])
    s_all = torch.cat([s_cal, s_tta[pseudo_idx]])
    c_all = torch.cat([c_cal, c_pseudo])

    w_pre = _consistency_weight(c_all, c_test, clip_range=(0.05, 20.0),
                                use_calib_stat=True, c_cal_stat=c_cal)
    w_post = w_pre.clone()
    w_post[s_cal.numel():] *= 0.5       # 伪正常标签不确定性折扣 (真实管线行为)

    n_pool = int(s_all.numel())
    level = min(1.0, (1 - ALPHA) * (n_pool + 1) / n_pool)
    thr_rep = float(weighted_quantile(s_all, w_post, level))

    # ---- 分量拆分 (实验 1.1 追加) ----
    c_score, c_feat, dev_n, dist_n = consistency_components(s_tta, hg, x_enh, cal_m)
    # 内部一致性: sqrt(c_score*c_feat) 必须复现真实的 c (否则分量拆分口径错误)
    c_recon = torch.sqrt(c_score * c_feat).clamp(min=1e-8, max=1.0)
    comp_match = bool(torch.allclose(c_recon, c, atol=1e-6))

    # ---- 保真校验: 与真实 structcp_threshold 比对 ----
    try:
        thr_ref, info_ref = structcp_threshold(
            s_tta[cal_m], y_cal, c[cal_m], s_tta[test_m], c[test_m],
            alpha=ALPHA, tau_quantile=TAU, mode="hard",
            score_quantile=SCORE_QUANTILE, use_calib_stat=True,
            robust_calibration=False,
        )
        thr_ref = float(thr_ref)
        thr_match = abs(thr_rep - thr_ref) <= 1e-12
    except Exception as e:
        thr_ref, thr_match = None, False
        info_ref = {"_err": f"{type(e).__name__}: {e}"}

    # ---- Dirichlet 能量 (三个信号) ----
    E_enh = dirichlet_energy(x_enh, hg.knn_idx, hg.Dv_inv_sqrt)
    E_raw = dirichlet_energy(x_raw, hg.knn_idx, hg.Dv_inv_sqrt)
    E_s = dirichlet_energy(s_tta.reshape(-1, 1), hg.knn_idx, hg.Dv_inv_sqrt)

    def to_np(t):
        return t.detach().cpu().numpy().astype(float)

    E_enh_np, E_raw_np, E_s_np = to_np(E_enh), to_np(E_raw), to_np(E_s)
    cs_np, cf_np = to_np(c_score), to_np(c_feat)
    dev_np, dist_np = to_np(dev_n), to_np(dist_n)
    c_np, s_np = to_np(c), to_np(s_tta)
    w_pre_np, w_post_np = to_np(w_pre), to_np(w_post)
    pool_np = pool_idx.detach().cpu().numpy()
    n_cal_n = int(s_cal.numel())

    def block(idx_np, w_np, tag):
        """对节点索引子集算三个信号的相关。"""
        out = {}
        for ename, E_np in (("E_enh", E_enh_np), ("E_raw", E_raw_np), ("E_s", E_s_np)):
            out[ename] = corr_block(w_np, c_np[idx_np], E_np[idx_np], s_np[idx_np])
        return out

    # ---- 分量相关 (在增广池上, 与主分析同一批节点) ----
    idx = pool_np
    E_pool, s_pool = E_enh_np[idx], s_np[idx]
    comps = {
        "c_score": _corr_pair(cs_np[idx], E_pool, z=cf_np[idx]),
        "c_feat": _corr_pair(cf_np[idx], E_pool, z=cs_np[idx]),
        # 底层粗糙度 (取负后与 c_* 同向: 越粗糙 dev/dist 越大、c_* 越小)
        "dev_raw": _corr_pair(dev_np[idx], E_pool),
        "dist_raw": _corr_pair(dist_np[idx], E_pool),
        # 分量间相关, 用于判断二者是否冗余
        "c_score_vs_c_feat": {
            "spearman": round(float(_spearman(cs_np[idx], cf_np[idx])[0]), 4)},
        "component_reconstruction_match": comp_match,
    }

    res = {
        "dataset": dataset, "seed": seed,
        "components": comps,
        "fidelity": {
            "thr_replicated": thr_rep,
            "thr_structcp_threshold": thr_ref,
            "thr_match": bool(thr_match),
            "n_pool": n_pool,
            "n_cal_normal": n_cal_n,
            "n_pseudo": int(pseudo_idx.numel()),
            "info_mode": info_ref.get("mode") if isinstance(info_ref, dict) else None,
        },
        # 主分析在增广池 A 上 (真正参与阈值计算的节点)
        "pool_w_post": block(pool_np, w_post_np, "pool_post"),
        "pool_w_pre": block(pool_np, w_pre_np, "pool_pre"),
        # 组内拆分: 不受 0.5 组间偏移影响
        "pool_calnormals": block(cal_normal_idx.detach().cpu().numpy(),
                                 w_pre_np[:n_cal_n], "cal"),
        "pool_pseudonormals": block(pseudo_idx.detach().cpu().numpy(),
                                    w_pre_np[n_cal_n:], "pseudo"),
    }
    if not thr_match:
        res["fidelity"]["_info_err"] = info_ref.get("_err") if isinstance(info_ref, dict) else None
    return res


# ----------------------------- 汇总 -----------------------------
def _merge_mom(m_list):
    if not m_list:
        return None
    n, sw, se, swe, sw2, se2 = (sum(x[i] for x in m_list) for i in range(6))
    if n <= 1:
        return None
    cov = swe - sw * se / n
    denom = math.sqrt(max(0.0, (sw2 - sw * sw / n)) * max(0.0, (se2 - se * se / n)))
    return {"n": int(n), "pearson_pooled": round(cov / denom, 4) if denom > 0 else 0.0}


def _summarize(cache: dict, out_path: str):
    per_ds = {}
    for ds in DATASETS:
        keys = [f"{ds}_{s}" for s in SEEDS if f"{ds}_{s}" in cache]
        keys = [k for k in keys if cache[k].get("fidelity", {}).get("thr_match")]
        if not keys:
            continue
        d = {}
        for grp, tag in (("pool_w_post", "pool"), ("pool_calnormals", "calN"),
                         ("pool_pseudonormals", "pseudo")):
            for ename in ("E_enh", "E_raw", "E_s"):
                for stat in ("pearson", "spearman", "partial_given_score"):
                    vals = [cache[k][grp][ename][stat] for k in keys
                            if stat in cache[k][grp][ename]]
                    if vals:
                        d[f"{grp}.{ename}.{stat}"] = [round(float(np.mean(vals)), 4),
                                                      round(float(np.std(vals)), 4)]
        per_ds[ds] = d

    mom = {}
    for grp in ("pool_w_post", "pool_calnormals", "pool_pseudonormals"):
        for ename in ("E_enh", "E_raw", "E_s"):
            m = [cache[k][grp][ename]["_mom"] for k in cache
                 if isinstance(cache[k], dict) and grp in cache[k]
                 and "_mom" in cache[k][grp][ename]]
            r = _merge_mom(m)
            if r:
                mom[f"{grp}.{ename}"] = r

    runs = [k for k in cache if isinstance(cache[k], dict) and k != "_summary"]
    n_match = sum(1 for k in runs
                  if cache[k].get("fidelity", {}).get("thr_match")
                  and cache[k].get("components", {}).get("component_reconstruction_match"))

    # ---- 分量分解汇总 ----
    per_ds_comp = {}
    for ds in DATASETS:
        keys = [f"{ds}_{s}" for s in SEEDS if f"{ds}_{s}" in runs]
        keys = [k for k in keys
                if cache[k].get("fidelity", {}).get("thr_match")
                and cache[k].get("components", {}).get("component_reconstruction_match")]
        if not keys:
            continue
        d = {}
        for comp in ("c_score", "c_feat", "dev_raw", "dist_raw"):
            for stat in ("pearson", "spearman", "partial_given_other"):
                vals = [cache[k]["components"][comp][stat] for k in keys
                        if stat in cache[k]["components"][comp]]
                if vals:
                    d[f"{comp}.{stat}"] = [round(float(np.mean(vals)), 4),
                                           round(float(np.std(vals)), 4)]
        d["c_score_vs_c_feat.spearman"] = [
            round(float(np.mean([cache[k]["components"]["c_score_vs_c_feat"]["spearman"]
                                 for k in keys])), 4),
            round(float(np.std([cache[k]["components"]["c_score_vs_c_feat"]["spearman"]
                                for k in keys])), 4)]
        per_ds_comp[ds] = d
    summary = {
        "config": dict(K=K, LAYERS=LAYERS, ALPHA_RES=ALPHA_RES, beta_highpass=0.0,
                       deg_gate=False, alpha=ALPHA, tau_quantile=TAU,
                       score_quantile=SCORE_QUANTILE,
                       energy="sum_{j in kNN(i)} || x_i/sqrt(d_i) - x_j/sqrt(d_j) ||^2",
                       degree="hypergraph node degree D_v (count of hyperedges containing v)"),
        "n_runs": len(runs),
        "n_thr_match": n_match,
        "per_dataset_mean_std": per_ds,
        "components_per_dataset_mean_std": per_ds_comp,
        "pooled_pearson": mom,
    }
    cache["_summary"] = summary
    json.dump(cache, open(out_path, "w"), indent=2)

    print(f"\n[summary] 约束: 仅纳入 thr_match=True 的运行 ({n_match}/{len(runs)})")
    for grp, ename in (("pool_w_post", "E_enh"), ("pool_w_post", "E_raw"),
                       ("pool_w_post", "E_s")):
        print(f"\n--- {grp} vs {ename} (w 与度归一化 Dirichlet 能量) ---")
        print(f"{'dataset':9s} {'pearson':>16s} {'spearman':>16s} {'partial|s':>16s}")
        for ds, d in per_ds.items():
            def g(s):
                v = d.get(f"{grp}.{ename}.{s}")
                return f"{v[0]:+.3f}±{v[1]:.3f}" if v else "--"
            print(f"{ds:9s} {g('pearson'):>16s} {g('spearman'):>16s} "
                  f"{g('partial_given_score'):>16s}")
    print("\n[pooled Pearson]", {k: v for k, v in mom.items() if "pool_w_post" in k})

    # ---- 分量分解打印 ----
    print("\n=== 分量分解: 哪个分量在感知谱能量? (E_enh, 增广池, mean±std over 5 seeds) ===")
    print(f"{'dataset':9s} {'c_score·sp':>12s} {'c_feat·sp':>12s} "
          f"{'c_score·partial':>16s} {'c_feat·partial':>15s} "
          f"{'dev·sp':>9s} {'dist·sp':>9s} {'cs~cf':>8s}")
    for ds, d in per_ds_comp.items():
        def g(k):
            v = d.get(k)
            return f"{v[0]:+.3f}" if v else "--"
        print(f"{ds:9s} {g('c_score.spearman'):>12s} {g('c_feat.spearman'):>12s} "
              f"{g('c_score.partial_given_other'):>16s} "
              f"{g('c_feat.partial_given_other'):>15s} "
              f"{g('dev_raw.spearman'):>9s} {g('dist_raw.spearman'):>9s} "
              f"{g('c_score_vs_c_feat.spearman'):>8s}")
    print("      (sp=spearman; partial=控制另一分量后的**秩**偏相关, 与 sp 同度量; "
          "dev/dist 为底层粗糙度, 正号=越粗糙能量越高)")

    print(f"[done] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_path = os.path.join(ROOT, "outputs", "weight_dirichlet_correlation.json")
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}
    cache.pop("_summary", None)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    dsets = [args.dataset] if args.dataset else DATASETS

    n_ok = n_bad = 0
    for ds in dsets:
        for seed in seeds:
            key = f"{ds}_{seed}"
            # 需同时具备 thr_match 与 components 字段 (后者为实验 1.1 追加)
            if (key in cache
                    and cache[key].get("fidelity", {}).get("thr_match")
                    and "components" in cache[key]):
                print(f"[skip] {key}")
                continue
            t0 = time.time()
            try:
                r = analyze_one(ds, seed, device)
            except Exception as e:
                print(f"[err] {key}: {type(e).__name__}: {e}")
                n_bad += 1
                continue
            r["_time_s"] = round(time.time() - t0, 2)
            cache[key] = r
            json.dump(cache, open(out_path, "w"), indent=2)
            f = r["fidelity"]
            p = r["pool_w_post"]["E_enh"]
            cp = r["components"]
            flag = "OK " if (f["thr_match"] and cp["component_reconstruction_match"]) else "MISMATCH"
            print(f"[{flag}] {key:16s} n_pool={f['n_pool']:>7d} "
                  f"w|E={p.get('pearson', float('nan')):+.3f}/"
                  f"{p.get('spearman', float('nan')):+.3f}  "
                  f"c_score|E={cp['c_score'].get('spearman', float('nan')):+.3f}  "
                  f"c_feat|E={cp['c_feat'].get('spearman', float('nan')):+.3f}  "
                  f"({r['_time_s']}s)")
            if not cp["component_reconstruction_match"]:
                print(f"       [warn] 分量重构与真实 c 不一致, 该运行被排除")
            n_ok += f["thr_match"]
            n_bad += (not f["thr_match"])
        print(f"[flush] {ds} done")

    print(f"\n[fidelity] thr_match={n_ok}  mismatch/err={n_bad}")
    _summarize(cache, out_path)


if __name__ == "__main__":
    main()
