"""实验 1.1b：去同义反复检验 (de-tautology check) —— c_feat 与 Dirichlet 能量的相关
是否只是"同一信号、同一张图上的两个局部平滑度统计量"的代数同义反复？

背景与问题
----------
实验 1.1a 的分量分解显示：与谱能量相关的几乎全是 `c_feat`，其底层量
    dist(i) = 1 - cos(f_i, centroid(f_{N(i)}))
与度归一化 Dirichlet 能量
    E(i)    = Σ_{j∈N(i)} ‖ f_i/√d_i − f_j/√d_j ‖²
的 Spearman 高达 +0.61 ~ +0.78。但二者是**在同一张 k-NN 图 (k=10, cosine,
由 raw 特征构建)、同一组传播特征 x_enh 上**计算的，均为局部平滑度度量，
因此该相关含有显著的构造性 (同义反复) 成分 —— 是机制确认，不构成独立证据。

此外还有一个**未被排除的混杂**：局部密度。二者都随节点度 d_i 变化
(c_feat 未做度归一化；E 虽除以 √d 仍可能与密度相关)，故"相关"可能只是
"密集区域既平滑又被判为正常"的密度效应。

设计 (2×2 交叉 + 混杂控制)
--------------------------
在两个轴上让 c_feat 与 E **错开**，若相关在错开条件下依然强，则机制为真：

  特征视图轴:  x_enh (传播后, 部署所用)  vs  x_raw (传播前原始特征)
  图结构轴:    G10 (k=10, cosine, 部署所用)  vs  G20 (k=20, cosine)

  c_feat 变体:  cf(x_enh,G10)[部署] / cf(x_raw,G10) / cf(x_enh,G20)
  E 变体:       E(x_enh,G10)[部署]  / E(x_raw,G10)  / E(x_enh,G20) / E(x_raw,G20)

  共 3×4 交叉相关。"对角线"(完全同条件) 给出含同义反复的上界；
  "非对角线"(错开特征视图 / 错开图 / 两者皆错开) 若仍强 → 机制独立于构造。

  混杂控制: 报告 c_feat 与 log(度) 的相关，以及**控制 log(度) 后** c_feat 与 E
  的秩偏相关 —— 用于判定相关是否由局部密度单独驱动。

保真性
------
相关性仍在**部署管线实际使用的增广池 A** 上计算 (与 1.1a 相同)，并复用
1.1a 的保真校验 (复刻阈值 vs 真实 structcp_threshold 逐位比对)，
仅 thr_match=True 的运行纳入汇总。

运行 (R6 复用 conda CBP; R7 8GB → CPU + 分块)
---------------------------------------------
cd /media/lixin/新加卷/数据集/test/StructCP && \\
source <CONDA_PREFIX>/bin/activate CBP && \\
export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
python scripts/analysis/weight_dirichlet_identity_check.py --dataset Amazon
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

# 复用 1.1a 已校验的工具, 避免口径漂移 (DRY)
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
    _rankdata,
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

K_ALT = 20  # 错开图结构的 k

CF_VARIANTS = ("cf_enh_G10", "cf_raw_G10", "cf_enh_G20")
E_VARIANTS = ("E_enh_G10", "E_raw_G10", "E_enh_G20", "E_raw_G20")
DEPLOYED_CF, DEPLOYED_E = "cf_enh_G10", "E_enh_G10"


@torch.no_grad()
def feat_component(features: torch.Tensor, hg: KNNHypergraph,
                   cal_mask: torch.Tensor, eps: float = 1e-8):
    """c_feat 分量 (可换特征视图 / 换图): dist = 1-cos(f_i, centroid(f_N(i))),
    c_feat = exp(-dist/scale_f), scale_f = mean(dist[cal_mask])。

    与 hyperedge_consistency 逐行对齐: 质心计算**包含自身** (knn_idx 列0)。
    """
    f = features.detach()
    f = f / (f.norm(dim=1, keepdim=True) + eps)
    idx = hg.knn_idx                                # (n, k+1) 含自身
    centroid = f[idx].mean(dim=1)
    centroid = centroid / (centroid.norm(dim=1, keepdim=True) + eps)
    dist = 1.0 - (f * centroid).sum(dim=1)
    scale_f = dist[cal_mask].mean()
    c_feat = torch.exp(-dist / (scale_f + eps))
    return dist, c_feat


def _to_np(t):
    return t.detach().cpu().numpy().astype(float)


def _safe_spearman(a, b):
    """退化 (常量/过短) 时返回 nan 而非抛错。"""
    try:
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return round(float(_spearman(a, b)[0]), 4)
    except Exception:
        return float("nan")


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
    G10 = KNNHypergraph(x_raw, k=K).to(device)          # 部署图 (k=10, cosine)
    x_enh = hypergraph_propagation(x_raw, G10, LAYERS, ALPHA_RES,
                                   beta_highpass=0.0, deg_gate=False)
    s_tta = F.softmax(model(x_enh, data.edge_index), dim=1)[:, 1]
    c = hyperedge_consistency(s_tta, G10, features=x_enh, cal_mask=cal_m)

    # ---- 复刻部署管线的池构造与加权 (与 1.1a 同口径) ----
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

    # ---- 变体: 特征视图 × 图结构 ----
    G20 = KNNHypergraph(x_raw, k=K_ALT).to(device)
    cf_variants = {}
    for name, feats, hg in (("cf_enh_G10", x_enh, G10),
                            ("cf_raw_G10", x_raw, G10),
                            ("cf_enh_G20", x_enh, G20)):
        _d, cf = feat_component(feats, hg, cal_m)
        cf_variants[name] = cf

    E_variants = {}
    for name, feats, hg in (("E_enh_G10", x_enh, G10),
                            ("E_raw_G10", x_raw, G10),
                            ("E_enh_G20", x_enh, G20),
                            ("E_raw_G20", x_raw, G20)):
        E_variants[name] = dirichlet_energy(feats, hg.knn_idx,
                                            hg.Dv_inv_sqrt, chunk=CHUNK)

    # 部署 c_feat 的内部一致性: 必须与真实 c 的 feat 分量一致 (由 1.1a 已校验, 此处复检)
    dist_dep, cf_dep = feat_component(x_enh, G10, cal_m)

    # ---- 在分析用的池上做交叉相关 ----
    pool_np = pool_idx.detach().cpu().numpy()
    cf_np = {k: _to_np(v)[pool_np] for k, v in cf_variants.items()}
    E_np = {k: _to_np(v)[pool_np] for k, v in E_variants.items()}

    cross = {}
    for ck in CF_VARIANTS:
        for ek in E_VARIANTS:
            cross[f"{ck}|{ek}"] = _safe_spearman(cf_np[ck], E_np[ek])

    # ---- 局部密度混杂控制 ----
    # 度 d_i = Dv_inv_sqrt^{-2} (超图节点度 = 含该节点的超边数)
    deg_np = _to_np(G10.Dv_inv_sqrt.pow(-2.0))[pool_np]
    logdeg = np.log(np.maximum(deg_np, 1.0))
    cf_dep_np, E_dep_np = cf_np[DEPLOYED_CF], E_np[DEPLOYED_E]
    degree_control = {
        "spearman_cf_E": _safe_spearman(cf_dep_np, E_dep_np),
        "spearman_cf_logdeg": _safe_spearman(cf_dep_np, logdeg),
        "spearman_E_logdeg": _safe_spearman(E_dep_np, logdeg),
        "partial_cf_E_given_logdeg": round(
            float(_partial_spearman(cf_dep_np, E_dep_np, logdeg)), 4),
        "mean_degree": round(float(deg_np.mean()), 3),
    }

    return {
        "dataset": dataset, "seed": seed,
        "fidelity": {"thr_match": bool(thr_match), "n_pool": n_pool},
        "cross": cross,
        "degree_control": degree_control,
    }


def _summarize(cache: dict, out_path: str):
    runs = [k for k in cache if isinstance(cache[k], dict) and k != "_summary"]
    per_ds = {}
    for ds in DATASETS:
        keys = [f"{ds}_{s}" for s in SEEDS
                if f"{ds}_{s}" in runs and cache[f"{ds}_{s}"]["fidelity"]["thr_match"]]
        if not keys:
            continue
        d = {}
        for pair in cache[keys[0]]["cross"]:
            vals = [cache[k]["cross"][pair] for k in keys]
            d[pair] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
        dc = {}
        for stat in cache[keys[0]]["degree_control"]:
            vals = [cache[k]["degree_control"][stat] for k in keys]
            dc[stat] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
        per_ds[ds] = {"cross": d, "degree_control": dc}

    cache["_summary"] = {
        "config": dict(k_deployed=K, k_alt=K_ALT, metric="cosine",
                       cf_variants=list(CF_VARIANTS), e_variants=list(E_VARIANTS)),
        "n_runs": len(runs),
        "n_thr_match": sum(1 for k in runs if cache[k]["fidelity"]["thr_match"]),
        "per_dataset": per_ds,
    }
    json.dump(cache, open(out_path, "w"), indent=2)

    print("\n=== 去同义反复检验: Spearman(c_feat, E) 交叉矩阵 (mean over 5 seeds) ===")
    print("行=c_feat 变体, 列=E 变体; G10/k=10 为部署图, G20/k=20 为错开图")
    print(f"{'':13s}" + "".join(f"{e:>14s}" for e in E_VARIANTS))
    for ds, d in per_ds.items():
        print(f"--- {ds} ---")
        for ck in CF_VARIANTS:
            row = f"{ck:13s}"
            for ek in E_VARIANTS:
                v = d["cross"].get(f"{ck}|{ek}")
                row += f"{v[0]:+14.3f}" if v else f"{'--':>14s}"
            print(row)

    print("\n=== 局部密度混杂控制 (部署对 cf_enh_G10 | E_enh_G10) ===")
    print(f"{'dataset':9s} {'cf~E':>9s} {'cf~logdeg':>11s} {'E~logdeg':>10s} "
          f"{'partial|logdeg':>15s}")
    for ds, d in per_ds.items():
        dc = d["degree_control"]
        print(f"{ds:9s} {dc['spearman_cf_E'][0]:+9.3f} "
              f"{dc['spearman_cf_logdeg'][0]:+11.3f} "
              f"{dc['spearman_E_logdeg'][0]:+10.3f} "
              f"{dc['partial_cf_E_given_logdeg'][0]:+15.3f}")

    # 关键对比: 对角线 vs 最错开格
    print("\n=== 关键对比 ===")
    print(f"{'dataset':9s} {'同条件(上界)':>14s} {'错开图+视图':>14s} {'保留比例':>10s}")
    for ds, d in per_ds.items():
        diag = d["cross"].get(f"{DEPLOYED_CF}|{DEPLOYED_E}")
        far = d["cross"].get("cf_raw_G10|E_enh_G20")  # 视图与图皆不同(相对 c_feat)
        if diag and far and abs(diag[0]) > 1e-9:
            print(f"{ds:9s} {diag[0]:+14.3f} {far[0]:+14.3f} {far[0] / diag[0]:>10.2f}")
    print(f"[done] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_path = os.path.join(ROOT, "outputs", "weight_dirichlet_identity_check.json")
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
            dc = r["degree_control"]
            print(f"[{'OK ' if r['fidelity']['thr_match'] else 'MIS'}] {key:16s} "
                  f"cf~E={dc['spearman_cf_E']:+.3f} "
                  f"partial|logdeg={dc['partial_cf_E_given_logdeg']:+.3f} "
                  f"({r['_time_s']}s)")
        print(f"[flush] {ds} done")
    _summarize(cache, out_path)


if __name__ == "__main__":
    main()
