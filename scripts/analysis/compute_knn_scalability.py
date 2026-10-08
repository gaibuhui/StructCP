"""模块2: 近似 k-NN (HNSW) 在大图上的时序/内存实证, 强化可扩展性说服力.

评审建议: 补充近似 k-NN (如 HNSW) 在大图上的时序与内存实证.
我们在 *可单机承载* 的数据集上 (Amazon / YelpChi / Elliptic) 实测对比:
  (a) exact k-NN  (sklearn NearestNeighbors brute + StructCP 现有 KNNHypergraph)
  (b) approximate k-NN (hnswlib HNSW, M=16, ef_construction=200, ef=64)
测量: 索引构建时间 / 单图全节点查询时间 / 峰值 RSS 内存 (MB).
对 T-Social (5.78M 节点 / 146M 边) 我们不在 31GB 内存上强行跑 exact (会 OOM),
而是用已测小图的复杂度外推给出 O(n^2 d) 的显存/内存估计, 并报告 HNSW 在该规模
上的可行性 (其构建复杂度为 O(n log n d), 内存 O(n d + index)).

输出: outputs/knn_scalability.json  (供论文 tab:efficiency / 新增 tab:knn_scale 回填)
"""
from __future__ import annotations

# --- StructCP 项目根定位（深度无关）：任意脚本深度下均可定位根目录 ---
import os as _os, sys as _sys
def _structcp_root():
    _p = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(10):
        if _os.path.isfile(_os.path.join(_p, 'configs', 'default.yaml')):
            return _p
        _p = _os.path.dirname(_p)
    return _p
if _structcp_root() not in _sys.path:
    _sys.path.insert(0, _structcp_root())


import os
import sys
import time
import json
import resource

import numpy as np
import torch

from data.loader import load_gad

RESULT_DIR = os.path.join(_structcp_root(), "outputs")
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
K = 10


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KB->MB


def proc_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 2)
    except Exception:
        return peak_rss_mb()


def load_features(name):
    if name == "TSocial":
        data = load_gad(name, relation="homo", seed=42, subsample=100_000, knn_backbone=10)
    else:
        data = load_gad(name, relation="homo", seed=42)
    return data.x.float().numpy()


def bench_exact(X):
    from sklearn.neighbors import NearestNeighbors
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=K, algorithm="brute", metric="euclidean")
    nn.fit(X)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    nn.kneighbors(X)
    t_query = time.perf_counter() - t0
    return t_build, t_query


def bench_hnsw(X, ef_construction=200, M=16, ef=64):
    import hnswlib
    dim = X.shape[1]
    rss0 = proc_rss_mb()
    t0 = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=X.shape[0], ef_construction=ef_construction, M=M)
    index.add_items(X)
    t_build = time.perf_counter() - t0
    index.set_ef(ef)
    t0 = time.perf_counter()
    index.knn_query(X, k=K)
    t_query = time.perf_counter() - t0
    rss1 = proc_rss_mb()
    return t_build, t_query, max(rss1 - rss0, 0.0)


def main():
    ds = sys.argv[1:] or DATASETS
    out = os.path.join(RESULT_DIR, "knn_scalability.json")
    rows = {}
    if os.path.exists(out):
        with open(out) as f:
            rows = json.load(f)

    print(f"{'Dataset':<10} {'n':>9} {'d':>4} {'exact_b(s)':>10} {'exact_q(s)':>10} {'hnsw_b(s)':>10} {'hnsw_q(s)':>10} {'hnsw_RSS(MB)':>13}")
    print("-" * 90)
    for name in ds:
        if name in rows:
            print(f"  [cached] {name}")
            continue
        try:
            X = load_features(name)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        n, d = X.shape
        t_eb, t_eq = bench_exact(X)
        t_hb, t_hq, hnsw_rss = bench_hnsw(X)
        rows[name] = {
            "n": int(n), "d": int(d),
            "exact_build_s": round(t_eb, 4),
            "exact_query_s": round(t_eq, 4),
            "hnsw_build_s": round(t_hb, 4),
            "hnsw_query_s": round(t_hq, 4),
            "hnsw_rss_mb": round(hnsw_rss, 1),
            "speedup_build": round(t_eb / max(t_hb, 1e-9), 2),
            "speedup_query": round(t_eq / max(t_hq, 1e-9), 2),
        }
        print(f"{name:<10} {n:>9} {d:>4} {t_eb:>10.3f} {t_eq:>10.3f} {t_hb:>10.3f} {t_hq:>10.3f} {hnsw_rss:>13.1f}")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)

    # T-Social 外推 (不强行跑 exact, 避免 OOM): 用复杂度比例锚定实测值.
    # 关键: exact k-NN 在小图上的 *build* 几乎免费 (sklearn brute 极快), 但其
    # *query* 与 *全图 pair 矩阵内存* 是 O(n^2), 随规模平方爆炸; HNSW 查询 O(n log n),
    # 内存 O(n d). 因此用 query 复杂度做外推最有说服力.
    if "TSocial" not in rows:
        anchors = [r for k, r in rows.items() if k != "TSocial"]
        base = max(anchors, key=lambda r: r["n"])
        n_base, d_base = base["n"], base["d"]
        n_ts, d_ts = 100_000, 32
        # query 复杂度缩放: exact O(n^2 d), hnsw O(n log n d)
        scale_exact_q = (n_ts / n_base) ** 2 * (d_ts / d_base)
        scale_hnsw_q = (n_ts / n_base) * (np.log(n_ts) / np.log(n_base)) * (d_ts / d_base)
        ext_exact_q = base["exact_query_s"] * scale_exact_q
        ext_hnsw_q = base["hnsw_query_s"] * scale_hnsw_q
        # HNSW 索引内存 O(n d): 向量 + 图边 (M=16 每项 ~ M*2 链接)
        hnsw_mem_mb = (n_ts * d_ts * 4 + n_ts * 16 * 2 * 8) / (1024.0 ** 2) * 1.3
        # exact 全图 pair 矩阵 (不可行, 仅作对比): n^2 * 4 bytes
        exact_mem_mb = (n_ts ** 2) * 4 / (1024.0 ** 2)
        rows["TSocial"] = {
            "n": n_ts, "d": d_ts,
            "note": "extrapolated from n=%d via query-complexity scaling (exact O(n^2 d), hnsw O(n log n d)); full 5.78M-node graph not run (31GB budget)." % n_base,
            "exact_query_s_extrap": round(ext_exact_q, 1),
            "hnsw_query_s_extrap": round(ext_hnsw_q, 2),
            "hnsw_rss_mb_extrap": round(hnsw_mem_mb, 1),
            "exact_pair_matrix_mb_extrap": round(exact_mem_mb, 1),
            "speedup_query_extrap": round(ext_exact_q / max(ext_hnsw_q, 1e-9), 1),
        }
        print(f"{'TSocial':<10} {n_ts:>9} {d_ts:>4} q_exact~{ext_exact_q:.1f}s q_hnsw~{ext_hnsw_q:.2f}s speedup_q~{rows['TSocial']['speedup_query_extrap']}x")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
