"""诊断 StructCP 在 photo/computer/weibo 上的"第二类失效边界"。

核心假设 (P0-1)：StructCP 的伪正常扩增机制 (structcp_threshold, hard 模式) 依赖
"novel-normal 节点在结构一致性空间与 seen-normal 可分离" 这一隐含前提 —— 即 novel
节点一致性应**显著低于** seen-normal, 才能被正确排除在伪正常池之外。

当 novel-normal 模式规模大且与 seen-normal 在一致性空间高度纠缠时:
  (1) novel 节点一致性偏高 → 被误纳入伪正常池;
  (2) novel 本身是高分正常 → 注入后推高加权分位数阈值;
  (3) 本来 Frozen 已建立的 FPR 控制被加权 CP 反向破坏。

本脚本对每个数据集量化 4 个指标, 直接验证上述机制:
  - novel_ratio        : 测试集正常节点中 novel 占比
  - pseudo_novel_rate  : 伪正常池中 novel 节点的污染率 (越高越糟)
  - auroc_sep          : 用一致性权重区分 novel/seen 的 AUROC (<0.6 = 纠缠)
  - cal_vs_pseudo_gap  : 伪正常池与真实校准正常池的平均分数差 (推高阈值幅度)

成功案例 (reddit/questions/tolokers) 应表现为 pseudo_novel_rate 低 + auroc_sep 高;
失败案例 (photo/computer/weibo) 应表现为 pseudo_novel_rate 高 + auroc_sep 低。

输出: 终端表格 + outputs/diagnose_type2_boundary.json
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

import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

from data.loader import load_gad
from models.bwgnn import BWGNN
from adapters.hypergraph_tta import KNNHypergraph, hypergraph_propagation, hyperedge_consistency
from adapters.conformal import structcp_threshold

ROOT = _structcp_root()


def build_model(dataset, relation, device):
    ckpt = torch.load(
        os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_{relation}.pth"),
        map_location=device,
    )
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    return model


def evaluate(dataset, device, n_nc, n_nv, alpha=0.05, tau=0.5):
    data = load_gad(dataset, seed=42, n_normality_clusters=n_nc,
                    n_novel_clusters=n_nv).to(device)
    model = build_model(dataset, "homo", device).eval()

    cal, te = data.cal_mask, data.test_mask
    y_cal = data.y[cal].to(device)
    nov = data.novel_mask.to(device)
    y_te = data.y[te].to(device)
    nov_te = nov[te]

    # 结构传播 + 分数 + 一致性
    t0 = time.time()
    hg = KNNHypergraph(data.x, k=10).to(device)
    x_hg = hypergraph_propagation(data.x, hg, 2, 0.5)
    s_hg = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    cons = hyperedge_consistency(s_hg, hg, features=x_hg)
    t_infer = time.time() - t0

    s_cal = s_hg[cal]; c_cal = cons[cal]
    s_te = s_hg[te];  c_te = cons[te]

    # ---- 复现 StructCP hard 模式伪正常选择 ----
    tau_thr = torch.quantile(c_cal, tau)
    pseudo_mask = (c_te >= tau_thr)          # 简化: 不依赖 score_upper (与 hard 模式同义)
    s_pseudo = s_te[pseudo_mask]
    c_pseudo = c_te[pseudo_mask]

    # ---- 指标 1: novel 占比 (测试正常节点中) ----
    te_normal = y_te == 0
    novel_ratio = float(nov_te[te_normal].float().mean()) if te_normal.any() else 0.0

    # ---- 指标 2: 伪正常池中的 novel 污染率 ----
    # 注意: 伪正常来自测试集; novel 标签仅对正常节点有意义
    pseudo_is_normal = (y_te[pseudo_mask] == 0)
    pseudo_novel_rate = float(
        nov_te[pseudo_mask][pseudo_is_normal].float().mean()
    ) if pseudo_is_normal.any() else 0.0

    # ---- 指标 3: 用一致性区分 novel/seen 的 AUROC ----
    # novel 节点一致性应低; seen 应高. AUROC 低 = 不可分 = 纠缠
    seen_mask = te_normal & (~nov_te.bool())
    nov_c = c_te[nov_te.bool()].cpu().numpy()
    seen_c = c_te[seen_mask].cpu().numpy()
    from sklearn.metrics import roc_auc_score
    try:
        auroc_sep = float(roc_auc_score(
            np.r_[np.ones(len(seen_c)), np.zeros(len(nov_c))],
            np.r_[seen_c, nov_c]))
    except Exception:
        auroc_sep = float("nan")

    # ---- 指标 4: 伪正常池 vs 真实校准正常池的平均分数差 ----
    cal_mean = float(s_cal.mean())
    pseudo_mean = float(s_pseudo.mean()) if s_pseudo.numel() else float("nan")
    cal_vs_pseudo_gap = (pseudo_mean - cal_mean) if s_pseudo.numel() else float("nan")

    # ---- 真实 StructCP 结果 (复跑加权分位数) ----
    thr, info = structcp_threshold(
        s_hg[cal], y_cal, cons[cal], s_hg[te], cons[te],
        alpha=alpha, tau_quantile=tau,
    )
    pred = (s_hg[te] > thr).float()
    fpr = float(pred[te_normal].mean())
    fpr_novel = float(pred[nov_te.bool()].mean()) if nov_te.any() else float("nan")

    return {
        "dataset": dataset,
        "novel_ratio": round(novel_ratio, 4),
        "pseudo_novel_rate": round(pseudo_novel_rate, 4),
        "auroc_sep_novel_seen": round(auroc_sep, 4),
        "cal_mean_score": round(cal_mean, 4),
        "pseudo_mean_score": round(pseudo_mean, 4) if s_pseudo.numel() else None,
        "cal_vs_pseudo_gap": round(cal_vs_pseudo_gap, 4) if s_pseudo.numel() else None,
        "structcp_FPR": round(fpr, 4),
        "structcp_FPR_novel": round(fpr_novel, 4) if nov_te.any() else None,
        "n_pseudo_normal": info.get("n_pseudo_normal"),
        "w_ess": round(info.get("w_ess", float("nan")), 1),
        "infer_time_s": round(t_infer, 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    # 与 run_compare_baselines 一致的默认簇配置
    ap.add_argument("--photo_nc", type=int, default=12); ap.add_argument("--photo_nv", type=int, default=3)
    ap.add_argument("--weibo_nc", type=int, default=40); ap.add_argument("--weibo_nv", type=int, default=10)
    args = ap.parse_args()

    # (dataset, n_nc, n_nv)
    cfgs = [
        ("photo", args.photo_nc, args.photo_nv),
        ("computer", 12, 3),
        ("weibo", args.weibo_nc, args.weibo_nv),
        ("reddit", 12, 3),
        ("questions", 12, 3),
        ("tolokers", 12, 3),
    ]
    rows = []
    for ds, nc, nv in cfgs:
        print(f"[diag] {ds} (nc={nc}, nv={nv}) ...", flush=True)
        rows.append(evaluate(ds, args.device, nc, nv))

    print("\n" + "=" * 108)
    print("StructCP 第二类失效边界诊断 (novel 与 seen 在一致性空间纠缠 → 伪正常污染)")
    print("=" * 108)
    hdr = (f"{'dataset':<10}{'novel%':>8}{'pseudo_nov%':>12}{'sep_AUROC':>12}"
           f"{'cal→pseudo_gap':>16}{'StructCP_FPR':>14}{'FPR_nov':>10}")
    print(hdr); print("-" * 108)
    for r in rows:
        print(f"{r['dataset']:<10}{r['novel_ratio']*100:>7.1f}%{r['pseudo_novel_rate']*100:>11.1f}%"
              f"{r['auroc_sep_novel_seen']:>12.3f}{r['cal_vs_pseudo_gap']:>16.3f}"
              f"{r['structcp_FPR']:>14.4f}{str(r['structcp_FPR_novel']):>10}")
    print("=" * 108)
    print("解读: 失败案例 (photo/computer/weibo) 应 pseudo_nov% 高 + sep_AUROC 低 (≤0.6);")
    print("      成功案例 (reddit/questions/tolokers) 应 pseudo_nov% 低 + sep_AUROC 高 (≥0.65)。")

    out = os.path.join(ROOT, "outputs", "diagnose_type2_boundary.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"[saved] {out}")
