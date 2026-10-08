"""统一对比基线评测入口 (稳妥路线 A: 半自动回填, 不做全自动串跑)。

职责 (严格遵守用户路线 A, 不自作主张串跑基线):
  ① 导出统一子集 mask  (可调用 scripts/export_baseline_masks.py, 产物在
     outputs/baseline_subset/{dataset}_node_masks.json, 节点为**原始全量图索引**)
  ② 用统一 mask 在**同一节点集合**上重跑 GraphSubspaceAD (ours), 保证可比。
  ③ 读取各基线手动回填的指标 json (由人工在固定子集上跑出后放入指定目录),
     与 ours 合并, 输出可直接粘贴论文主表的 compare.json。

基线回填文件约定:
  每个基线一个 json, 形如 {"auc": 0.xx, "ap": 0.xx} 或 {"AUROC":, "AUPRC":},
  放在 outputs/baseline_subset/fill/{dataset}__{baseline}.json
  例: outputs/baseline_subset/fill/Amazon__TUNE.json

注意: 本脚本**不臆造**任何基线数字, 只做聚合与对比打印。

运行:
  cd /media/lixin/新加卷/数据集/test/StructCP
  source <CONDA_PREFIX>/bin/activate CBP
  export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP"
  python scripts/run_compare_baselines.py                # 仅聚合并打印
  python scripts/run_compare_baselines.py --rerun_ours   # 同时用统一 mask 重跑 ours
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


import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from data.loader import GADData, load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from models.graph_subspace_ad.pipeline import GraphSubspaceAD  # noqa: E402

ROOT = _structcp_root()
OUT = os.path.join(ROOT, "outputs", "graph_subspace_ad")
MASK_DIR = os.path.join(ROOT, "outputs", "baseline_subset")
FILL_DIR = os.path.join(MASK_DIR, "fill")

DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
# 各数据集 ours 的最优 alpha (来自主实验记录)
OURS_ALPHA = {"Amazon": 0.8, "YelpChi": 0.8, "Elliptic": 1.0, "TFinance": 1.0}


def load_mask(dataset: str) -> dict:
    path = os.path.join(MASK_DIR, f"{dataset}_node_masks.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺失统一 mask: {path} (先跑 export_baseline_masks.py)")
    with open(path) as f:
        return json.load(f)


def apply_mask_to_gad(data: GADData, mask: dict, device: str) -> GADData:
    """用原始索引 mask 在全量图上筛选, 不做 remap (节点池 = hub_kept 上的采样)。"""
    idx = np.array(mask["hub_kept"], dtype=np.int64)
    remap = -np.ones(data.x.shape[0], dtype=np.int64)
    remap[idx] = np.arange(idx.shape[0])

    def pick(key):
        arr = np.array(mask[key], dtype=np.int64)
        # 转成子集内 (remap 后) 顺序 index, 以便对齐 GADData
        return remap[arr]

    ei = data.edge_index.numpy().astype(np.int64)
    keep = np.isin(ei[0], idx) & np.isin(ei[1], idx)
    sub_ei = ei[:, keep]
    sub_ei = np.vstack([remap[sub_ei[0]], remap[sub_ei[1]]]).astype(np.int64)

    x = data.x.numpy().astype(np.float32)[idx]
    y = data.y.numpy().astype(np.int64)[idx]
    n = x.shape[0]

    def _mask(arr):
        m = np.zeros(n, dtype=bool)
        m[arr] = True
        return m

    train = _mask(pick("train"))
    cal = _mask(pick("cal"))
    test = _mask(pick("test"))

    return GADData(
        x=torch.tensor(x),
        edge_index=torch.tensor(sub_ei).long(),
        y=torch.tensor(y),
        train_mask=torch.tensor(train),
        cal_mask=torch.tensor(cal),
        test_mask=torch.tensor(test),
        novel_mask=torch.zeros(n, dtype=torch.bool),
        name=data.name + "_masked",
    )


def run_ours(dataset: str, hidden: int, device: str, alpha: float,
             auto_alpha: bool = False) -> dict:
    data = load_gad(name=dataset, seed=42)
    mask = load_mask(dataset)
    sub = apply_mask_to_gad(data, mask, device).to(device)
    ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h128.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}.pth")
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model = BWGNN(sd.get("in_channels") or 100, hidden, 2, d=2).to(device)
    model.load_state_dict(sd["state_dict"])
    model.freeze()
    g = GraphSubspaceAD(model, device=device, alpha=alpha,
                        n_components_max=hidden, auto_alpha=auto_alpha)
    parts = g.fit_predict(sub.x, sub.edge_index, sub.train_mask, sub.test_mask,
                          return_parts=True)
    y = sub.y[sub.test_mask].cpu().numpy().astype(np.int64)
    return {"auc": float(roc_auc_score(y, parts["score"])),
            "ap": float(average_precision_score(y, parts["score"]))}


def read_baseline_fill(dataset: str) -> dict:
    """读取 outputs/baseline_subset/fill/{dataset}__{baseline}.json 回填指标。"""
    out = {}
    if not os.path.isdir(FILL_DIR):
        return out
    for fn in os.listdir(FILL_DIR):
        if not fn.startswith(f"{dataset}__") or not fn.endswith(".json"):
            continue
        base = fn[len(f"{dataset}__"):-len(".json")]
        path = os.path.join(FILL_DIR, fn)
        try:
            with open(path) as f:
                d = json.load(f)
            if "auc" in d:
                out[base] = {"auc": d["auc"], "ap": d.get("ap")}
            elif "AUROC" in d:
                out[base] = {"auc": d["AUROC"], "ap": d.get("AUPRC")}
            elif "results" in d and d["results"]:
                last = d["results"][-1]
                out[base] = {"auc": last.get("auc"), "ap": last.get("ap")}
        except Exception as e:
            print(f"[skip] 读取回填 {fn} 失败: {e!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rerun_ours", action="store_true",
                    help="用统一 mask 重跑 ours (默认 False, 直接读 main_table.json)")
    ap.add_argument("--auto_alpha", action="store_true",
                    help="ours 用自适应 alpha (无标签, 仅训练正常节点定权) 替代手调 OURS_ALPHA")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    summary = {}

    for ds in args.datasets:
        print(f"\n===== {ds} =====")
        table = {}
        if args.rerun_ours:
            a = OURS_ALPHA.get(ds, 1.0)
            print(f"[ours] GraphSubspaceAD (alpha={a}" +
                  (", auto" if args.auto_alpha else "") + ") ...")
            try:
                table["GraphSubspaceAD"] = run_ours(
                    ds, args.hidden, args.device, a, args.auto_alpha)
            except Exception as e:
                print(f"  ours 重跑失败: {e!r}")
        else:
            # 从 main_table.json 取 ours 最优结果
            mt = os.path.join(OUT, "main_table.json")
            if os.path.exists(mt):
                with open(mt) as f:
                    mtd = json.load(f)
                ours = mtd.get("table", {}).get(ds)
                if ours:
                    table["GraphSubspaceAD"] = {"auc": ours["auc"], "ap": ours.get("ap")}
        if "GraphSubspaceAD" in table:
            print(f"  GraphSubspaceAD: {table['GraphSubspaceAD']}")

        # 基线回填
        fills = read_baseline_fill(ds)
        for name, met in fills.items():
            print(f"  {name}: {met}")
        table.update(fills)
        summary[ds] = table

    out_path = os.path.join(OUT, "compare.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")

    # 控制台打印论文可直接粘贴的主表
    print("\n=========== 论文主表 (ours + baselines) ===========")
    hdr = ["Dataset"] + [m for m in next(iter(summary.values())).keys()]
    # 用固定顺序: GraphSubspaceAD 优先
    methods = ["GraphSubspaceAD"] + [m for m in next(iter(summary.values()))
                                     if m != "GraphSubspaceAD"]
    print("Dataset\t" + "\t".join(methods))
    for ds in args.datasets:
        row = [ds]
        for m in methods:
            met = summary.get(ds, {}).get(m)
            if met and met.get("auc") is not None:
                row.append(f"{met['auc']:.3f}")
            else:
                row.append("-")
        print("\t".join(row))


if __name__ == "__main__":
    main()
