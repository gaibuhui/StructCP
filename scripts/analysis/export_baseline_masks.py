"""导出 4 数据集统一对比子集节点 mask (稳妥路线 A)。

目的: 让 GraphSubspaceAD 与所有对比基线使用**完全相同**的节点集合与
train/cal/test 划分, 消除子集差异带来的指标偏差, 保证公平对比。

关键设计:
  * 复用 models.graph_subspace_ad.loader.stratified_subset 的
    hub 分位裁剪 + 分层采样 + 测试集强制两类 的**全部参数与随机种子**,
    保证与 GraphSubspaceAD 主实验同源。
  * 输出的是**原始全量图节点索引** (而非 remap 后的 0..k-1 索引),
    这样各基线可在自己加载的全量图上直接用这些 node id 做过滤。
  * 同时给出 remap 表 (原始 id -> 子集内顺序 index), 方便在子集上
    做特征/边对齐时使用。

输出: outputs/baseline_subset/{dataset}_node_masks.json
  {
    "dataset": str,
    "params": {...},                 # n_train_norm / n_test / degree_quantile / seed
    "n_full": int,                   # 全量图节点数
    "hub_kept": [...],               # 被 hub 裁剪保留的原始节点 id (子集节点池)
    "n_hub_kept": int,
    "train": [...], "cal": [...],    # 原始全量图节点索引
    "test":  [...],
    "test_anomaly": int, "test_normal": int,
  }

运行:
  cd /media/lixin/新加卷/数据集/test/StructCP
  source <CONDA_PREFIX>/bin/activate CBP
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
  python scripts/export_baseline_masks.py
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


import json
import os

import numpy as np

from data.loader import load_gad

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
OUT_DIR = os.path.join(_structcp_root(),
                       "outputs", "baseline_subset")
os.makedirs(OUT_DIR, exist_ok=True)


def export_masks(name: str,
                 n_train_norm: int = 3000,
                 n_test: int = 10000,
                 degree_quantile: float = 0.98,
                 seed: int = 42,
                 cal_ratio: float = 0.2):
    """对 load_gad 全量图施加与 stratified_subset 一致的裁剪 + 采样,
    但返回**原始全量图节点索引**, 不 remap。"""
    data = load_gad(name=name, seed=seed)
    rng = np.random.RandomState(seed)
    x = data.x.numpy().astype(np.float32)
    y = data.y.numpy().astype(np.int64)
    ei = data.edge_index.numpy().astype(np.int64)
    n = x.shape[0]

    # --- hub 分位裁剪 (与 stratified_subset 完全一致) ---
    deg = np.bincount(ei.ravel(), minlength=n)
    thr = float(np.quantile(deg, degree_quantile))
    thr = max(int(thr), 1)
    keep = deg <= thr
    n_norm = int(keep[y == 0].sum())
    if n_norm < n_train_norm:
        thr2 = max(int(np.quantile(deg, 0.995)), thr)
        keep = deg <= thr2
        thr = thr2
    hub_kept = np.where(keep)[0]
    print(f"[{name}] 全量 n={n}, hub 裁剪保留 {len(hub_kept)} "
          f"(剔除 degree>{thr} 的 {int((~keep).sum())} 个 hub)")

    # 在 hub 保留池内做分层采样 (原始索引)
    yk = y[hub_kept]
    normal = hub_kept[yk == 0]
    abnormal = hub_kept[yk == 1]

    if len(normal) > n_train_norm:
        tr_norm = rng.choice(normal, size=n_train_norm, replace=False)
    else:
        tr_norm = normal
    tr_norm_set = set(tr_norm.tolist())

    n_cal = int(len(tr_norm) * cal_ratio)
    rng.shuffle(tr_norm)
    cal_idx = tr_norm[:n_cal]
    train_idx = tr_norm[n_cal:]

    # 测试集: 始终含正常与异常两类
    rest_normal = np.array([i for i in hub_kept.tolist() if i not in tr_norm_set],
                           dtype=np.int64)
    abn_ratio = len(abnormal) / max(1, len(hub_kept))
    n_ab = int(n_test * abn_ratio)
    n_ab = min(len(abnormal), max(50, n_ab))
    n_no = min(len(rest_normal), max(50, n_test - n_ab))
    if n_no < max(50, n_test - n_ab):
        n_ab = min(len(abnormal), max(50, n_test - n_no))
    if len(abnormal) > n_ab:
        te_ab = rng.choice(abnormal, size=n_ab, replace=False)
    else:
        te_ab = abnormal
    if len(rest_normal) > n_no:
        te_no = rng.choice(rest_normal, size=n_no, replace=False)
    else:
        te_no = rest_normal
    te_idx = np.concatenate([te_ab, te_no])

    out = {
        "dataset": name,
        "params": {
            "n_train_norm": n_train_norm,
            "n_test": n_test,
            "degree_quantile": degree_quantile,
            "seed": seed,
            "cal_ratio": cal_ratio,
        },
        "n_full": int(n),
        "hub_kept": hub_kept.tolist(),
        "n_hub_kept": int(len(hub_kept)),
        "train": train_idx.tolist(),
        "cal": cal_idx.tolist(),
        "test": te_idx.tolist(),
        "test_anomaly": int(n_ab),
        "test_normal": int(n_no),
    }
    path = os.path.join(OUT_DIR, f"{name}_node_masks.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{name}] 导出 mask: train={len(train_idx)} cal={len(cal_idx)} "
          f"test={len(te_idx)} (异常{int(n_ab)}/正常{int(n_no)}) -> {path}")
    return out


def main():
    for name in DATASETS:
        try:
            export_masks(name)
        except Exception as e:
            print(f"[{name}] 导出失败: {e!r}")
    print(f"\n[done] 所有 mask 已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
