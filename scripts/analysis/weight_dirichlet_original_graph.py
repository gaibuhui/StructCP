"""实验 1.1c（最终检验）：用**原始图** edge_index 计算 Dirichlet 能量，
彻底切断 c_feat 与 E 之间的构造联系。

为什么这是"最终检验"
--------------------
1.1b 已把 c_feat 与 E 在"特征视图"和"k"两个轴上错开，但两个轴都不独立：
  * x_enh 是 x_raw 的平滑结果（同源）；
  * G20 的邻域包含 G10 的邻域（嵌套）。
因此 1.1b 只能**削弱**同义反复，无法**排除**它。

本脚本改用**数据集自带的原始图 edge_index** 计算 E，而 c_feat 仍固定使用
部署管线真实的 k-NN 图 (k=10, cosine, 由 raw 特征构建)。二者的信息来源
**完全独立**：
  * k-NN 图  ← 由特征余弦相似度派生（几何/特征空间）
  * 原始图   ← 数据集真实拓扑（交易关系 / 评价关系 / 用户-商品关系）
若 c_feat 与"原始图上的 E"依然强相关，则 c_feat 确实在感知**真实的图平滑性**，
而非复述自己在 k-NN 图上的定义。

前置核查 (已做)
--------------
确认 edge_index 不是从特征派生的（否则检验失效）：
  * Amazon / YelpChi: 由 .mat 的邻接矩阵 `_to_edge_index(adj)` 得到, 真实拓扑;
  * Elliptic / TFinance: 数据集原始边; TFinance 的 `knn_backbone=None`
    (DEFAULT_DATASET_CFG["TFinance"] 未设置该键), 故未注入特征 k-NN 边。

能量定义与高效实现
------------------
E(i) = Σ_{j∈N_orig(i)} ‖ f_i/√d_i − f_j/√d_j ‖²
     = ‖f_i‖² + Σ_j ‖f_j‖²/d_j − 2·(f_i · Σ_j f_j/√d_j)/√d_i
展开后只需对边做 index_add_，内存 O(E) 而非 O(n·k·d)。
这对 TFinance (21.2M 无向边 / 42.4M 有向对) 是必须的。

节点子集
--------
原始图度分布极不均，且 Elliptic 有约 23% 孤立节点（度=0，E 无定义）。
故同时报告三个子集，避免"孤立节点 E=0"稀释或伪造相关：
  all / non_isolated(deg≥1) / deg_ge5(deg≥5, 能量估计较可靠)

运行 (R6 conda CBP; R7 8GB → CPU)
---------------------------------
cd /media/lixin/新加卷/数据集/test/StructCP && \\
source <CONDA_PREFIX>/bin/activate CBP && \\
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
python scripts/analysis/weight_dirichlet_original_graph.py --dataset Amazon
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
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_sys.path.insert(0, ROOT)

from scripts.analysis.weight_dirichlet_correlation import (  # noqa: E402
    CHUNK,
    DATASETS,
    K,
    LAYERS,
    ALPHA_RES,
    ALPHA,
    SEEDS,
    SCORE_QUANTILE,
    TAU,
    _partial_spearman,
    _spearman,
    dirichlet_energy,
)
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

EDGE_CHUNK = 2_000_000  # 边分块大小, 控制 TFinance (42M 有向对) 峰值内存


@torch.no_grad()
def original_graph_degrees(edge_index: torch.Tensor, n: int):
    """原始图 -> 无向去重边 (u,v) 双向列表 + 度。返回 (u2, v2, deg)。"""
    ei = edge_index
    src, dst = ei[0].long(), ei[1].long()
    keep = src != dst                       # 去自环
    src, dst = src[keep], dst[keep]
    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    key = a.to(torch.int64) * n + b.to(torch.int64)
    key = torch.unique(key)                 # 无向去重
    a, b = (key // n).long(), (key % n).long()
    u2 = torch.cat([a, b])                  # 双向, 便于 index_add_
    v2 = torch.cat([b, a])
    deg = torch.zeros(n, dtype=torch.float32, device=ei.device)
    deg.index_add_(0, u2, torch.ones_like(u2, dtype=torch.float32))
    return u2, v2, deg


@torch.no_grad()
def dirichlet_energy_original(features: torch.Tensor, u2: torch.Tensor,
                              v2: torch.Tensor, deg: torch.Tensor,
                              chunk: int = EDGE_CHUNK) -> torch.Tensor:
    """原始图上的度归一化 Dirichlet 能量, 用展开式 + 边分块。

    E(i) = ‖f_i‖² + Σ_{j∈N(i)} ‖f_j‖²/d_j − 2·(f_i · Σ_{j∈N(i)} f_j/√d_j)/√d_i
    """
    f = features.detach().float()
    n, d = f.shape
    dev = f.device
    d_safe = deg.clamp(min=1.0)
    inv_sqrt_d = d_safe.pow(-0.5)

    sum_terms = torch.zeros(n, dtype=torch.float32, device=dev)  # Σ_j ‖f_j‖²/d_j
    sum_f = torch.zeros(n, d, dtype=torch.float32, device=dev)   # Σ_j f_j/√d_j

    m = u2.numel()
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        ub, vb = u2[s:e], v2[s:e]
        fv = f[vb]
        # Σ_j ‖f_j‖²/d_j = Σ_j (f_j/√d_j · f_j/√d_j)
        fv_s = fv * inv_sqrt_d[vb].unsqueeze(1)
        sum_terms.index_add_(0, ub, (fv_s * fv_s).sum(dim=1))
        sum_f.index_add_(0, ub, fv_s)

    sq_norm = (f * f).sum(dim=1)                    # ‖f_i‖²
    cross = (f * sum_f).sum(dim=1)                  # f_i · Σ_j f_j/√d_j
    E = sq_norm + sum_terms - 2.0 * cross * inv_sqrt_d
    E = E * (deg > 0).float()                       # 孤立节点置 0
    return E


def _to_np(t):
    return t.detach().cpu().numpy().astype(float)


def _safe_spearman(a, b):
    try:
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return round(float(_spearman(a, b)[0]), 4)
    except Exception:
        return float("nan")


def _subset_stats(cf: np.ndarray, E: np.ndarray, logdeg: np.ndarray,
                  mask: np.ndarray) -> dict:
    """给定子集, 算 cf~E 的 Spearman + 控制 log(度) 的秩偏相关。"""
    if int(mask.sum()) < 3:
        return {"n": int(mask.sum()), "spearman": float("nan"),
                "partial_given_logdeg": float("nan")}
    c, e, ld = cf[mask], E[mask], logdeg[mask]
    sp = _safe_spearman(c, e)
    par = float("nan")
    try:
        if np.std(ld) >= 1e-12 and np.std(c) >= 1e-12 and np.std(e) >= 1e-12:
            par = round(float(_partial_spearman(c, e, ld)), 4)
    except Exception:
        pass
    return {"n": int(mask.sum()), "spearman": sp, "partial_given_logdeg": par}


@torch.no_grad()
def analyze_one(dataset: str, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    model, _ = load_frozen_detector(dataset, "homo", root=ROOT, device=device)

    cal_m = torch.as_tensor(data.cal_mask, dtype=torch.bool, device=device)
    test_m = torch.as_tensor(data.test_mask, dtype=torch.bool, device=device)
    y_cal = data.y[cal_m]
    n = data.x.shape[0]

    x_raw = data.x
    G10 = KNNHypergraph(x_raw, k=K).to(device)          # 部署 k-NN 图
    x_enh = hypergraph_propagation(x_raw, G10, LAYERS, ALPHA_RES,
                                   beta_highpass=0.0, deg_gate=False)
    s_tta = F.softmax(model(x_enh, data.edge_index), dim=1)[:, 1]
    c = hyperedge_consistency(s_tta, G10, features=x_enh, cal_mask=cal_m)

    # ---- 复刻部署管线的池 (与 1.1a/1.1b 同口径) ----
    cal_idx = torch.where(cal_m)[0]
    test_idx = torch.where(test_m)[0]
    cal_normal_idx = cal_idx[y_cal == 0]
    s_cal, c_cal = s_tta[cal_normal_idx], c[cal_normal_idx]
    s_test, c_test = s_tta[test_idx], c[test_idx]

    tau = torch.quantile(c_cal.float(), TAU)
    gamma = 4.0 * (1.0 - SCORE_QUANTILE)
    s_upper = s_cal.max() + gamma * s_cal.std().clamp(min=1e-6)
    pseudo_mask = (c_test >= tau) & (s_test <= s_upper)
    pseudo_idx = test_idx[pseudo_mask]
    c_pseudo = c[pseudo_idx]
    cap = int(s_cal.numel())
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
    w_post[s_cal.numel():] *= 0.5
    n_pool = int(s_all.numel())
    level = min(1.0, (1 - ALPHA) * (n_pool + 1) / n_pool)
    thr_rep = float(weighted_quantile(s_all, w_post, level))
    try:
        thr_ref, _ = structcp_threshold(
            s_tta[cal_m], y_cal, c[cal_m], s_tta[test_m], c[test_m],
            alpha=ALPHA, tau_quantile=TAU, mode="hard",
            score_quantile=SCORE_QUANTILE, use_calib_stat=True,
            robust_calibration=False)
        thr_match = abs(thr_rep - float(thr_ref)) <= 1e-12
    except Exception:
        thr_match = False

    # ---- c_feat: 固定在部署的 k-NN 图 G10 上 (部署管线真实使用) ----
    f_n = x_enh / (x_enh.norm(dim=1, keepdim=True) + 1e-8)
    centroid = f_n[G10.knn_idx].mean(dim=1)
    centroid = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-8)
    dist_cf = 1.0 - (f_n * centroid).sum(dim=1)
    cf = torch.exp(-dist_cf / (dist_cf[cal_m].mean() + 1e-8))

    # ---- E: 原始图 (独立信息源) ----
    u2, v2, deg_orig = original_graph_degrees(data.edge_index, n)
    E_enh_orig = dirichlet_energy_original(x_enh, u2, v2, deg_orig)
    E_raw_orig = dirichlet_energy_original(x_raw, u2, v2, deg_orig)
    # 对照: 部署 k-NN 图上的 E (1.1a 的上界)
    E_enh_knn = dirichlet_energy(x_enh, G10.knn_idx, G10.Dv_inv_sqrt, chunk=CHUNK)

    pool_np = pool_idx.detach().cpu().numpy()
    cf_np = _to_np(cf)[pool_np]
    Eo_np = _to_np(E_enh_orig)[pool_np]
    Eor_np = _to_np(E_raw_orig)[pool_np]
    Ek_np = _to_np(E_enh_knn)[pool_np]
    deg_np = _to_np(deg_orig)[pool_np]
    logdeg = np.log(np.maximum(deg_np, 1.0))

    m_all = np.ones(len(pool_np), dtype=bool)
    m_ni = deg_np >= 1
    m_d5 = deg_np >= 5

    out = {
        "dataset": dataset, "seed": seed,
        "fidelity": {"thr_match": bool(thr_match), "n_pool": n_pool},
        "graph_stats": {
            "n_nodes": int(n),
            "undirected_unique_edges": int(u2.numel() // 2),
            "mean_degree": round(float(deg_orig.mean()), 2),
            "median_degree": round(float(deg_orig.median()), 2),
            "frac_isolated_all": round(float((deg_orig == 0).float().mean()), 4),
            "frac_isolated_in_pool": round(float((deg_np == 0).mean()), 4),
        },
        # 主结果: c_feat(kNN图) vs E(原始图) —— 完全独立的信息源
        "cf_knn_vs_E_orig": {
            "all": _subset_stats(cf_np, Eo_np, logdeg, m_all),
            "non_isolated": _subset_stats(cf_np, Eo_np, logdeg, m_ni),
            "deg_ge5": _subset_stats(cf_np, Eo_np, logdeg, m_d5),
        },
        # 对照1: 两侧都用原始图特征/图无关性较弱的对 (E 用 raw 特征)
        "cf_knn_vs_E_orig_rawfeat": {
            "all": _subset_stats(cf_np, Eor_np, logdeg, m_all),
            "non_isolated": _subset_stats(cf_np, Eor_np, logdeg, m_ni),
            "deg_ge5": _subset_stats(cf_np, Eor_np, logdeg, m_d5),
        },
        # 对照2: 同 k-NN 图 (1.1a 的上界, 含同义反复)
        "cf_knn_vs_E_knn_bound": {
            "all": _subset_stats(cf_np, Ek_np, logdeg, m_all),
        },
        # 密度混杂: 原始图度 vs c_feat / E
        "degree_confound": {
            "spearman_cf_logdeg": _safe_spearman(cf_np, logdeg),
            "spearman_Eorig_logdeg": _safe_spearman(Eo_np, logdeg),
        },
    }
    return out


def _mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]


def _summarize(cache: dict, out_path: str):
    runs = [k for k in cache if isinstance(cache[k], dict) and k != "_summary"]
    per_ds = {}
    for ds in DATASETS:
        keys = [f"{ds}_{s}" for s in SEEDS
                if f"{ds}_{s}" in runs and cache[f"{ds}_{s}"]["fidelity"]["thr_match"]]
        if not keys:
            continue
        d = {}
        for block in ("cf_knn_vs_E_orig", "cf_knn_vs_E_orig_rawfeat",
                      "cf_knn_vs_E_knn_bound"):
            for sub in cache[keys[0]][block]:
                for stat in ("spearman", "partial_given_logdeg"):
                    d[f"{block}.{sub}.{stat}"] = _mean_std(
                        [cache[k][block][sub][stat] for k in keys])
                d[f"{block}.{sub}.n"] = _mean_std(
                    [float(cache[k][block][sub]["n"]) for k in keys])
        for stat in ("spearman_cf_logdeg", "spearman_Eorig_logdeg"):
            d[f"degree_confound.{stat}"] = _mean_std(
                [cache[k]["degree_confound"][stat] for k in keys])
        gs = {}
        for stat in cache[keys[0]]["graph_stats"]:
            gs[stat] = round(float(np.mean([cache[k]["graph_stats"][stat] for k in keys])), 4)
        per_ds[ds] = {"stats": d, "graph_stats": gs}

    cache["_summary"] = {
        "config": dict(k_deployed=K, energy_definition=
                       "sum_{j in N_ORIGINAL(i)} ||f_i/sqrt(d_i) - f_j/sqrt(d_j)||^2",
                       cf_source="k-NN graph (k=10, cosine) - deployed",
                       independence="c_feat uses feature-derived kNN graph; "
                                    "E uses dataset original topology - fully independent"),
        "n_runs": len(runs),
        "n_thr_match": sum(1 for k in runs if cache[k]["fidelity"]["thr_match"]),
        "per_dataset": per_ds,
    }
    json.dump(cache, open(out_path, "w"), indent=2)

    print("\n" + "=" * 78)
    print("最终检验: c_feat (k-NN 图, 部署) vs Dirichlet 能量 (原始图, 独立信息源)")
    print("=" * 78)
    print(f"{'dataset':9s} {'孤立%':>7s} {'均度':>8s} "
          f"{'all':>9s} {'非孤立':>9s} {'度≥5':>9s} {'控制度':>9s} {'kNN上界':>9s} {'保留%':>7s}")
    for ds, dd in per_ds.items():
        s, g = dd["stats"], dd["graph_stats"]

        def v(key):
            x = s.get(key)
            return f"{x[0]:+.3f}" if x else "--"
        # 保留比例以"非孤立"子集对比 kNN 上界
        ori = s.get("cf_knn_vs_E_orig.non_isolated.spearman")
        bnd = s.get("cf_knn_vs_E_knn_bound.all.spearman")
        ratio = f"{ori[0] / bnd[0] * 100:6.0f}%" if (ori and bnd and abs(bnd[0]) > 1e-9) else "--"
        print(f"{ds:9s} {g['frac_isolated_in_pool'] * 100:6.1f}% "
              f"{g['mean_degree']:8.1f} "
              f"{v('cf_knn_vs_E_orig.all.spearman'):>9s} "
              f"{v('cf_knn_vs_E_orig.non_isolated.spearman'):>9s} "
              f"{v('cf_knn_vs_E_orig.deg_ge5.spearman'):>9s} "
              f"{v('cf_knn_vs_E_orig.non_isolated.partial_given_logdeg'):>9s} "
              f"{v('cf_knn_vs_E_knn_bound.all.spearman'):>9s} "
              f"{ratio:>7s}")

    print("\n[密度混杂] (原始图度 log d)")
    print(f"{'dataset':9s} {'cf~logdeg':>11s} {'E_orig~logdeg':>15s}")
    for ds, dd in per_ds.items():
        s = dd["stats"]
        a = s.get("degree_confound.spearman_cf_logdeg")
        b = s.get("degree_confound.spearman_Eorig_logdeg")
        print(f"{ds:9s} {a[0]:+11.3f} {b[0]:+15.3f}")
    print(f"[done] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_path = os.path.join(ROOT, "outputs", "weight_dirichlet_original_graph.json")
    cache = json.load(open(out_path)) if os.path.exists(out_path) else {}
    cache.pop("_summary", None)

    dsets = [args.dataset] if args.dataset else DATASETS
    for ds in dsets:
        for seed in SEEDS:
            key = f"{ds}_{seed}"
            if key in cache and cache[key].get("fidelity", {}).get("thr_match"):
                print(f"[skip] {key}")
                continue
            t0 = time.time()
            try:
                r = analyze_one(ds, seed, device)
            except Exception as e:
                print(f"[err] {key}: {type(e).__name__}: {e}")
                continue
            r["_time_s"] = round(time.time() - t0, 2)
            cache[key] = r
            json.dump(cache, open(out_path, "w"), indent=2)
            ni = r["cf_knn_vs_E_orig"]["non_isolated"]
            print(f"[{'OK ' if r['fidelity']['thr_match'] else 'MIS'}] {key:16s} "
                  f"cf~E_orig(非孤立)={ni['spearman']:+.3f} "
                  f"partial={ni['partial_given_logdeg']:+.3f} "
                  f"({r['_time_s']}s)")
        print(f"[flush] {ds} done")
    _summarize(cache, out_path)


if __name__ == "__main__":
    main()
