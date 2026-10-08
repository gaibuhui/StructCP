"""校准集污染鲁棒性实验 v2 (Rebuttal: 亮点3 抗污染, 支撑数据)。

在 run_calib_contamination.py 基础上扩展:
  - 注入模式 low : 骨干漏分的"低分异常"被误当正常 (原协议, 分数取校准正常分布低位)
  - 注入模式 high: 真实异常节点被误标为正常混入校准集 (分数取测试集真实异常分布)
  - 基线: Frozen (split-conformal) / TUNE (learnable aligner + split-conformal) /
         StructCP (一致性重加权) / StructCP-robust (一致性自清洗地板)

注意: TUNE 的 aligner 只用全图特征拟合, 不读校准分数/标签,
      故在 low/high 两种分数注入下其分数不变, 仅阈值估计受注入校准集影响。
"""

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

import numpy as np
import torch
from torch.nn.functional import softmax

from adapters.conformal import (  # noqa: E402
    split_conformal_threshold,
    structcp_threshold,
)
from adapters.hypergraph_tta import (  # noqa: E402
    KNNHypergraph,
    hyperedge_consistency,
)
from adapters.pipeline import get_scores  # noqa: E402
from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from baselines.tune import TuneAligner, tune_fit, tune_transform  # noqa: E402

MAIN_DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]
CONTAM = [0.0, 0.01, 0.05, 0.10]
MODES = ["low", "high"]


@torch.no_grad()
def scores_of(model, x, ei):
    return softmax(model(x, ei), dim=1)[:, 1]


def load_and_score(ds, device, seed, tune_epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_gad(ds, relation="homo", seed=seed).to(device)
    ckpt_path = os.path.join(_structcp_root(), "checkpoint", f"bwgnn_{ds}_homo.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    cal_m, test_m = data.cal_mask, data.test_mask
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_enh = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_tta = get_scores(model, x_enh, data.edge_index).to(device)
    cal_bool = torch.as_tensor(cal_m, dtype=torch.bool, device=device)
    cons = hyperedge_consistency(s_tta, hg, features=x_enh, cal_mask=cal_bool)

    # TUNE: aligner 仅用全图特征拟合
    aligner = TuneAligner(data.num_features).to(device)
    aligner = tune_fit(aligner, data.x, hg.knn_idx, epochs=tune_epochs, device=device)
    x_tune = tune_transform(aligner, data.x, device)
    s_tune = scores_of(model, x_tune, data.edge_index).to(device)
    return data, s_tta, cons, s_tune, cal_m, test_m


def inject(s_cal, y_cal, p, rng, mode, anom_pool=None):
    """把 p 比例校准正常节点标记为'被误当正常的异常'。

    low : 分数取校准正常分布低位 (骨干漏分的低分新异异常)
    high: 分数取测试集真实异常分布 (真实异常被误标为正常混入校准集)
    标签一律保持 0 (部署者不知其为异常)。
    """
    s = s_cal.clone()
    y = y_cal.clone()
    n = s.numel()
    k = int(round(p * n))
    if k <= 0:
        return s, y
    if mode == "high":
        k = min(k, int(anom_pool.numel()))
    if k <= 0:
        return s, y
    normal_idx = torch.where(y == 0)[0]
    if normal_idx.numel() < k:
        k = int(normal_idx.numel())
    pick = normal_idx[rng.choice(normal_idx.numel(), size=k, replace=False)]
    if mode == "low":
        norm_scores = s[y == 0]
        qs = torch.linspace(0.0, 0.15, max(k, 1), device=s.device)
        vals = torch.quantile(norm_scores, qs)
    else:
        vals = anom_pool[rng.choice(anom_pool.numel(), size=k, replace=False)]
    s[pick] = vals[:k].to(s.dtype)
    return s, y


def fpr_tpr(s_test, y_test, thr):
    y = y_test.cpu().numpy()
    pred = (s_test.cpu().numpy() > float(thr)).astype(int)
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else float("nan")
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    return fpr, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_epochs", type=int, default=30)
    ap.add_argument("--datasets", nargs="+", default=MAIN_DATASETS)
    args = ap.parse_args()
    
    # 支持多种子运行（Rebuttal: 增强统计显著性）
    SEED_LIST = [42, 123, 456]
    if args.seed not in SEED_LIST:
        print("WARNING: Running single seed", args.seed, "instead of multi-seed list.")
        SEED_LIST = [args.seed]
    
    out = {}
    for seed in SEED_LIST:
        print(f"\n=== SEED {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        device = torch.device(args.device)
        
        for ds in args.datasets:
            # 如果该数据集已有该种子，跳过
            if ds in out and str(seed) in out[ds]:
                print(f"  Skipping {ds} (seed {seed} already exists)")
                continue
            print(f"\n=== {ds} ===", flush=True)
            data, s_tta, cons, s_tune, cal_m, test_m = load_and_score(
                ds, device, seed, args.tune_epochs
            )
        y_cal = data.y[cal_m].to(device)
        y_test = data.y[test_m].to(device)
        s_test, cons_cal, cons_test = s_tta[test_m], cons[cal_m], cons[test_m]
        # 真实异常分数池 (供 high 注入; 两种分数流各用各的池)
        anom_tta = s_tta[test_m][y_test == 1]
        anom_tune = s_tune[test_m][y_test == 1]
        rec = {}
        for mode in MODES:
            for p in CONTAM:
                s_cal_tta, y_cal_inj = inject(s_tta[cal_m].to(device), y_cal, p, rng,
                                              mode, anom_pool=anom_tta)
                s_cal_tune, _ = inject(s_tune[cal_m].to(device), y_cal, p, rng,
                                       mode, anom_pool=anom_tune)
                # Frozen (split-conformal, 完全信任 cal_y)
                thr_f = split_conformal_threshold(s_cal_tta[y_cal_inj == 0], args.alpha)
                fpr_f, tpr_f = fpr_tpr(s_test, y_test, thr_f)
                # StructCP 标准 (一致性重加权 + 校准集统计量)
                thr_s, _ = structcp_threshold(s_cal_tta, y_cal_inj, cons_cal, s_test,
                                              cons_test, alpha=args.alpha,
                                              tau_quantile=0.5, score_quantile=0.75,
                                              use_calib_stat=True,
                                              robust_calibration=False)
                fpr_s, tpr_s = fpr_tpr(s_test, y_test, thr_s)
                # StructCP-robust (一致性自清洗地板)
                thr_r, _ = structcp_threshold(s_cal_tta, y_cal_inj, cons_cal, s_test,
                                              cons_test, alpha=args.alpha,
                                              tau_quantile=0.5, score_quantile=0.75,
                                              use_calib_stat=True,
                                              robust_calibration=True)
                fpr_r, tpr_r = fpr_tpr(s_test, y_test, thr_r)
                # TUNE (split-conformal 阈值来自注入后的 s_tune 校准集)
                thr_t = split_conformal_threshold(s_cal_tune[y_cal_inj == 0], args.alpha)
                fpr_t, tpr_t = fpr_tpr(s_tune[test_m], y_test, thr_t)

                key = f"{mode}_p{p:.2f}"
                rec[key] = {
                    "frozen_fpr": fpr_f, "frozen_tpr": tpr_f,
                    "tune_fpr": fpr_t, "tune_tpr": tpr_t,
                    "struct_fpr": fpr_s, "struct_tpr": tpr_s,
                    "robust_fpr": fpr_r, "robust_tpr": tpr_r,
                }
                print(f"  {key}  Frozen {fpr_f:.3f}/{tpr_f:.3f}  TUNE {fpr_t:.3f}/{tpr_t:.3f}"
                      f"  StructCP {fpr_s:.3f}/{tpr_s:.3f}  Robust {fpr_r:.3f}/{tpr_r:.3f}",
                      flush=True)
        
        # 按数据集组织，每个数据集包含多个种子
            if ds not in out:
                out[ds] = {}
            if seed not in out[ds]:
                out[ds][seed] = rec
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/calib_contamination_v2.json"
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        existing.update(out)
        out = existing
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] -> outputs/calib_contamination_v2.json", flush=True)


if __name__ == "__main__":
    main()
