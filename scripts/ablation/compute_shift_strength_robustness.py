"""论文 Sec 4.4 回填: 启发式重加权在不同偏移强度下的韧性 (大规模消融)。

【背景】
论文理论声明已从"恢复交换性 + Assumption H 证明"重塑为:
  "基于局部同质性的似然比重加权, 经验上最小化校准-测试分布间的 1-Wasserstein 距离"
并将其定位为一种**启发式 (Heuristic) 重加权策略**, 而非由假设 H 证明的定理。
本脚本为该启发式提供大规模经验证据:
  - 逐数据集 × 逐偏移强度 (shift_strength = novel 簇占比 1/6, 2/6, 3/6) × 逐权重方案
    (uniform / graph / feature) 测量校准池→测试分布的 1-Wasserstein 距离 W1 与 FPR。
  - 核心断言: 在全部偏移强度下, 基于局部同质性的图权重 (graph) 相比 feature/uniform
    取得**更小或相当的 W1** 并保持 FPR <= alpha —— 展示韧性, 而非证明上界。

【W1 定义】
  W1( 重加权校准池分布, 测试分布 ) = wasserstein_distance(A_scores, test_scores,
                                          u_weights=A_weights)
  其中 A = C_norm ∪ P (伪正常池), A_weights 按权重方案给出, 伪正常按 0.5 降权。

【权重方案】
  uniform : w_i = 1                              (无重加权, 即标准 split conformal)
  graph   : w_i ∝ 局部同质性 edge-consistency c_i  (StructCP 的似然比权重)
  feature : w_i ∝ 纯特征余弦一致性 (图权重去掉了分数一致性项) —— 对照"仅特征"重加权

【用法】 (CBP 环境, 固定前缀见 memory)
  cd /media/lixin/新加卷/数据集/test/StructCP && \
  source <CONDA_PREFIX>/bin/activate CBP && \
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \
  python scripts/ablation/compute_shift_strength_robustness.py \
      --datasets Amazon YelpChi Elliptic TFinance
输出: outputs/shift_strength_robustness.json (每数据集完成即落盘, 支持断点续跑)
"""
from __future__ import annotations

# --- StructCP 项目根定位（深度无关） ---
import os as _os, sys as _sys
def _structcp_root():
    _p = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(10):
        if _os.path.isfile(_os.path.join(_p, "configs", "default.yaml")):
            return _p
        _p = _os.path.dirname(_p)
    return _p
if _structcp_root() not in _sys.path:
    _sys.path.insert(0, _structcp_root())

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from data.loader import load_gad
from utils.checkpoint import load_frozen_detector
from adapters.hypergraph_tta import (
    KNNHypergraph,
    hyperedge_consistency,
    hypergraph_propagation,
)
from adapters.conformal import evaluate_coverage, structcp_threshold

RESULT_DIR = os.path.join(_structcp_root(), "outputs")
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
SHIFTS = [1 / 6, 2 / 6, 3 / 6]
SEED = 42
ALPHA = 0.05
OUT = os.path.join(RESULT_DIR, "shift_strength_robustness.json")


@torch.no_grad()
def run_dataset(name, shift, seed=SEED):
    """单数据集 × 单偏移强度: 计算三种权重方案的 W1 与 FPR。"""
    data = load_gad(name, relation="homo", seed=seed, shift_strength=shift)
    model, _ = load_frozen_detector(name, "homo", root=_structcp_root())
    # R7 (8GB 显存约束): 统一 CPU 推理, 既消除 cuda/cpu 设备不一致, 也避免
    # Elliptic(203k 节点) k-NN 全对全距离在 GPU 上 OOM。
    model = model.to("cpu")
    x, ei = data.x, data.edge_index
    cal_m, test_m = data.cal_mask, data.test_mask
    # 掩码/标签统一取 numpy 视图 (numpy 侧索引与布尔运算), torch 侧仍用原 mask
    def _np(v):
        return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)

    cal_np, test_np, y_np = _np(cal_m), _np(test_m), _np(data.y)

    # 超图 TTA 增强分数 (与主实验口径一致)
    hg = KNNHypergraph(x, k=10)
    feat = hypergraph_propagation(x, hg, num_layers=2)
    s = F.softmax(model(feat, ei), dim=1)[:, 1]          # (n,)
    s_np = s.cpu().numpy()

    # 局部同质性 edge-consistency (全图)
    c = hyperedge_consistency(s, hg, x)                    # (n,)
    c_np = c.cpu().numpy()
    # 纯特征余弦一致性 (feature 权重的基)
    dcos = _feature_cosine_dev(x, hg)

    # 校准"正常"子集 (与 structcp_threshold 口径一致)
    cal_norm = np.where(cal_np & (y_np == 0))[0]
    # W1 的对齐对象: **测试正常节点**的分数分布。
    # 理由: FPR 控制只关心正常类分数的分位数对齐; 若把异常节点也算进"测试分布",
    # W1 将主要由异常占比决定, 而非由校准-测试的正常分布偏移决定, 度量即失真。
    # (诊断用途: 仅用于事后度量, 阈值计算本身不使用测试标签。)
    test_idx = np.where(test_np & (y_np == 0))[0]
    test_all = np.where(test_np)[0]

    # 校准集刻度 (source-only, 无测试泄漏)
    med_c, std_c = float(np.median(c_np[cal_norm])), float(np.std(c_np[cal_norm])) + 1e-12
    med_d, std_d = float(np.median(dcos[cal_norm])), float(np.std(dcos[cal_norm])) + 1e-12

    # 伪正常池: 高一致性 + 低分数 (门槛全部来自校准正常子集)
    tau = float(np.quantile(c_np[cal_norm], 0.5))
    s_up = float(s_np[cal_norm].max()) + 1.0 * float(np.std(s_np[cal_norm]))
    pseudo = (c_np >= tau) & (s_np <= s_up) & test_np
    p_idx = np.where(pseudo)[0]

    # 各权重方案在"校准池 ∪ 伪正常池"上的权重向量
    a_idx = np.concatenate([cal_norm, p_idx])
    is_pseudo = np.concatenate([np.zeros(len(cal_norm), dtype=bool),
                                np.ones(len(p_idx), dtype=bool)])
    a_scores = s_np[a_idx]

    def graph_w():
        w = np.clip(np.exp((c_np[a_idx] - med_c) / std_c), 0.05, 20.0)
        w[is_pseudo] *= 0.5
        return w / w.mean()

    def feature_w():
        w = np.clip(np.exp((dcos[a_idx] - med_d) / std_d), 0.05, 20.0)
        w[is_pseudo] *= 0.5
        return w / w.mean()

    def uniform_w():
        w = np.ones(len(a_idx))
        w[is_pseudo] = 0.5
        return w / w.mean()

    schemes = {
        "uniform": uniform_w(),
        "feature": feature_w(),
        "graph": graph_w(),
    }

    # 主口径: 校准池 vs 测试**正常**分布
    w1 = {}
    for k, w in schemes.items():
        w1[k] = round(float(wasserstein_distance(
            a_scores, s_np[test_idx], u_weights=w)), 4)

    # 参考口径: 校准池 vs 全测试分布 (含异常) —— 度量会被异常占比主导, 仅作对照
    w1_all = {}
    for k, w in schemes.items():
        w1_all[k] = round(float(wasserstein_distance(
            a_scores, s_np[test_all], u_weights=w)), 4)

    # ---- 分位数对齐误差 (直接对应 FPR 控制目标) ----
    # 保形 FPR 控制只要求**上尾分位数**对齐, 而非全分布匹配: 阈值 thr 需落在
    # 测试正常分布的 (1-alpha) 分位数上。故用加权分位数与目标分位数的偏差
    # 作为比全分布 W1 更贴合任务目标的度量。
    # 有符号误差: thr - q_target。
    #   < 0 => 阈值偏低(过松) => FPR 倾向 > alpha;
    #   > 0 => 阈值偏高(保守) => FPR 倾向 <= alpha。
    # 方向比绝对值更关键: 偏保守只损失一点 TPR, 偏低则直接违反 FPR 保证。
    level = min(1.0, (1 - ALPHA) * (len(a_scores) + 1) / len(a_scores))
    q_target = float(np.quantile(s_np[test_idx], 1 - ALPHA))
    dq = {}
    for k, w in schemes.items():
        dq[k] = round(float(_weighted_quantile_np(a_scores, w, level)) - q_target, 4)

    # FPR —— 与 quantile_align_err **严格同口径** (同一 thr = 同一加权分位数),
    # 否则阈值来源不同会使两个度量互相矛盾。FPR = P(s_test_normal > thr)。
    fpr_by_scheme = {}
    for k, w in schemes.items():
        thr_k = _weighted_quantile_np(a_scores, w, level)
        fpr_by_scheme[k] = round(float((s_np[test_idx] > thr_k).mean()), 4)

    # 另记录: 无伪正常扩增的**标准 split conformal** (仅 cal_norm, 均匀权重),
    # 作为"pseudo-normal augmentation 消融"的参考点。
    thr_scp = float(np.quantile(s_np[cal_norm], 1 - ALPHA))
    fpr_scp_no_aug = round(float((s_np[test_idx] > thr_scp).mean()), 4)

    # 完整管线口径的 graph 阈值 (含 adaptive_alpha 等主实验设置), 仅作参考:
    # 论文报告的 graph-FPR 以 p9_shift_ablation.json 为准 (main 中合并覆盖)。
    thr_g, _ = structcp_threshold(
        s[cal_m], data.y[cal_m], c[cal_m], s[test_m], c[test_m],
        alpha=ALPHA, mode="hard",
        score_quantile=0.75 if name in ("Elliptic", "TFinance") else None,
    )
    cov_g = evaluate_coverage(s[test_m], data.y[test_m], thr_g)

    return {
        "shift_strength": round(shift, 4),
        "n_novel_clusters": int(round(shift * 6)),
        "W1": w1,                       # vs 测试**正常**分布 (诊断)
        "W1_vs_all_test": w1_all,       # vs 全测试分布 (含异常, 参考)
        "quantile_align_err": dq,       # 有符号: thr - q* (对应 FPR 控制目标)
        "FPR": {
            "uniform": fpr_by_scheme["uniform"],
            "graph": fpr_by_scheme["graph"],        # 将被 p9 覆盖
        },
        "FPR_feature": fpr_by_scheme["feature"],
        "FPR_graph_local_scheme": fpr_by_scheme["graph"],
        "FPR_graph_pipeline": round(float(cov_g["FPR"]), 4),
        "FPR_scp_no_aug": fpr_scp_no_aug,
        "q_target": round(q_target, 4),
        "n_cal_norm": int(len(cal_norm)),
        "n_pseudo": int(len(p_idx)),
        "n_test_normal": int(len(test_idx)),
    }


def _weighted_quantile_np(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """加权分位数 inf{v : sum w*1(V<=v) / sum w >= q} (与 weighted_quantile 一致)。"""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    idx = int(np.searchsorted(cw, q))
    return float(v[min(idx, len(v) - 1)])


def _feature_cosine_dev(x, hg):
    """节点到其 k-NN 邻居均值的特征余弦偏离 (纯特征一致性, 不含分数)。"""
    knn = hg.knn_idx.cpu().numpy()
    x_np = x.cpu().numpy()
    nbr_feat = x_np[knn[:, 1:]]                       # (n, k, d) 排除自身
    mu = nbr_feat.mean(axis=1)                         # (n, d)
    mu_n = np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12
    x_n = np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-12
    cos = (x_np * mu).sum(axis=1, keepdims=True) / (x_n * mu_n)
    return (1.0 - cos[:, 0])


def _p9_fpr(name):
    """从 p9 偏移强度消融读取 graph-FPR: {n_novel_clusters: FPR}。

    p9 由 run_structcp.py **完整管线**产出 (含默认开启的 adaptive_alpha 兜底),
    是论文主表报告的权威口径, 但与本脚本的 Δ_tail / 方案 FPR **不同源**
    (伪正常池构造规则与兜底逻辑不同), 故仅作参考列存放, **不覆盖**同口径的
    FPR.graph, 否则 Δ_tail 与 FPR 会互相矛盾 (如 T-Finance 出现
    "Δ_tail 最保守却 FPR 超标" 的假象)。
    """
    p9_path = os.path.join(RESULT_DIR, "p9_shift_ablation.json")
    if not os.path.exists(p9_path):
        return {}
    with open(p9_path) as f:
        p9 = json.load(f)
    out = {}
    for r in p9.get(name, []):
        k, v = r.get("n_novel_clusters"), r.get("FPR")
        if k is not None and v is not None:
            out[int(k)] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--shifts", nargs="+", type=float, default=SHIFTS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    results = {}
    if os.environ.get("FORCE") and os.path.exists(OUT):
        results = {}
    elif os.path.exists(OUT):                            # 断点续跑
        with open(OUT) as f:
            results = json.load(f)

    print(f"{'Dataset':<9}{'shift':>6}{'W1_uni':>8}{'W1_feat':>8}{'W1_graph':>9}"
          f"{'FPR_uni':>8}{'FPR_graph':>9}", flush=True)
    for name in args.datasets:
        if name in results and results[name]:             # 空列表视为未跑
            print(f"  [cached] {name}", flush=True)
            continue
        rows = []
        p9 = _p9_fpr(name)
        print(f"\n##### {name} #####", flush=True)
        for s in args.shifts:
            try:
                r = run_dataset(name, s, seed=args.seed)
            except Exception as e:
                print(f"  [skip] {name} shift={s:.3f}: {e}", flush=True)
                continue
            # graph-FPR: 主列采用**同口径**值 (与 Δ_tail 同源), 保证"Δ_tail 最大
            # ↔ FPR 最低"的论断可验证; p9 完整管线值另存参考列并标注口径。
            if r["n_novel_clusters"] in p9:
                r["FPR_graph_p9_pipeline"] = round(p9[r["n_novel_clusters"]], 4)
                r["FPR_graph_p9_source"] = "p9_shift_ablation (full pipeline, adaptive_alpha ON)"
            r["FPR_graph_source"] = "same-scheme weighted quantile (matches Δ_tail)"
            rows.append(r)
            print(f"{name:<9}{r['shift_strength']:<6.3f}{r['W1']['uniform']:>8.4f}"
                  f"{r['W1']['feature']:>8.4f}{r['W1']['graph']:>9.4f}"
                  f"{r['FPR']['uniform']:>8.4f}{r['FPR']['graph']:>9.4f}", flush=True)
        results[name] = rows
        with open(OUT, "w") as f:                        # 每数据集完成即落盘
            json.dump(results, f, indent=2)
    print(f"\n[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
