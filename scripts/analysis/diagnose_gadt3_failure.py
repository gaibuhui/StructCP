"""P0-2: GADT3 / TA-GGAD 失败归因消融。

论文主表观察到 GADT3/TA-GGAD 在 6 个 TUNE 数据集上全部退化
(AUROC<=0.65, TPR=0, 退化成 all-negative 检测器)。本脚本做归因,
核心假设是: 这种失败是 **backbone 对特定图(低维特征 / 弱同配 / 稀疏) 的
不适配**, 而非方法本质无效 —— 证据应是它们在 GADBench (Amazon/YelpChi)
上能正常工作。

对每个数据集量化:
  (1) GADT3 source 训练集 AUROC (是否学到正常/异常区分)
  (2) GADT3 测试分数 std / range (是否塌缩)
  (3) GADT3 测试 AUROC / TPR
  (4) TA-GGAD 同 (1)(2)(3)
  (5) 图特征维度 dim, 同配系数 homophily (校验 backbone 适配假设)

对照: Amazon/YelpChi 应 (1) 高 + (3) 高; TUNE 数据集应 (1) 低或 (2) 塌缩 + (3) 低。

输出: 终端表 + outputs/diagnose_gadt3_failure.json
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

import os, json, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

_BASE = os.path.join(_structcp_root(), "baselines")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from data.loader import load_gad
from gadt3_pyg.adapter import GADT3Adapter
from ta_ggad_pyg.adapter import ARCAdapter


def homophily(y, edge_index):
    """边同配系数: 两端标签相同的边占比 (y: 0/1 二值)。"""
    ei = edge_index.cpu().numpy()
    ys = y.cpu().numpy()
    same = (ys[ei[0]] == ys[ei[1]]).mean()
    return float(same)


def gadt3_run(name, data, device):
    t0 = time.time()
    g = GADT3Adapter(ndim_s=data.num_features, device="cpu")
    g.fit_source(data)
    # source 训练集 AUROC
    with torch.no_grad():
        pred, h = g.model(data.x.to("cpu"), data.edge_index.to("cpu"), extract_features=True)
        s_train = F.softmax(pred, 1)[:, 1].cpu().numpy()
    tr_auc = float(roc_auc_score(data.y[data.train_mask].numpy(), s_train[data.train_mask.cpu().numpy()]))
    g.fit_target(data)
    s = g.score(data).numpy()
    t = time.time() - t0
    test = data.test_mask.cpu().numpy()
    auc = float(roc_auc_score(data.y[test], s[test]))
    thr = np.quantile(s[data.cal_mask.cpu().numpy() & (data.y.cpu().numpy() == 0)], 0.05)
    tpr = float(((s[test] > thr) & (data.y[test].numpy() == 1)).sum() /
                max(1, (data.y[test].numpy() == 1).sum()))
    return {
        "source_train_AUROC": round(tr_auc, 4),
        "score_std": round(float(s.std()), 4),
        "score_min": round(float(s.min()), 4),
        "score_max": round(float(s.max()), 4),
        "test_AUROC": round(auc, 4),
        "test_TPR": round(tpr, 4),
        "time_s": round(t, 2),
    }


def arc_run(name, data, device):
    t0 = time.time()
    a = ARCAdapter(dim=data.num_features, device="cpu")
    a.fit(data)
    # source 训练集区分度: 用 fit 后的 support code 与 train 正常/异常距离
    a.fit_target(data)
    s = a.score(data).numpy()
    t = time.time() - t0
    test = data.test_mask.cpu().numpy()
    auc = float(roc_auc_score(data.y[test], s[test]))
    cal_norm = data.cal_mask.cpu().numpy() & (data.y.cpu().numpy() == 0)
    thr = np.quantile(s[cal_norm], 0.05)
    tpr = float(((s[test] > thr) & (data.y[test].numpy() == 1)).sum() /
                max(1, (data.y[test].numpy() == 1).sum()))
    return {
        "score_std": round(float(s.std()), 4),
        "score_min": round(float(s.min()), 4),
        "score_max": round(float(s.max()), 4),
        "test_AUROC": round(auc, 4),
        "test_TPR": round(tpr, 4),
        "time_s": round(t, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Amazon/YelpChi = GADBench 对照 (主表 GADT3 有效); 后 6 个 = TUNE 退化
    cfgs = [
        ("Amazon", "homo", 8, 2),
        ("YelpChi", "homo", 8, 2),
        ("photo", "homo", 12, 3),
        ("computer", "homo", 12, 3),
        ("reddit", "homo", 12, 3),
        ("questions", "homo", 12, 3),
        ("tolokers", "homo", 12, 3),
        ("weibo", "homo", 40, 10),
    ]
    rows = []
    for name, rel, nc, nv in cfgs:
        print(f"[gadt3-diag] {name} (nc={nc}, nv={nv}) ...", flush=True)
        data = load_gad(name, relation=rel, seed=42,
                        n_normality_clusters=nc, n_novel_clusters=nv).to("cpu")
        hom = homophily(data.y, data.edge_index)
        r = {"dataset": name, "dim": int(data.num_features),
             "homophily": round(hom, 4),
             "n_nodes": int(data.num_nodes),
             "anom_ratio": round(float((data.y == 1).float().mean()), 4)}
        r["GADT3"] = gadt3_run(name, data, args.device)
        r["TA-GGAD"] = arc_run(name, data, args.device)
        rows.append(r)

    print("\n" + "=" * 120)
    print("GADT3 / TA-GGAD 失败归因消融  (对照: GADBench 应高, TUNE 数据集应退化)")
    print("=" * 120)
    hdr = (f"{'dataset':<10}{'dim':>5}{'homo':>7}{'G3_srcAUROC':>13}{'G3_scoreStd':>13}"
           f"{'G3_testAUROC':>13}{'G3_TPR':>8}{'ARC_testAUROC':>15}{'ARC_TPR':>8}")
    print(hdr); print("-" * 120)
    for r in rows:
        g, a = r["GADT3"], r["TA-GGAD"]
        print(f"{r['dataset']:<10}{r['dim']:>5}{r['homophily']:>7.2f}"
              f"{g['source_train_AUROC']:>13.3f}{g['score_std']:>13.3f}"
              f"{g['test_AUROC']:>13.3f}{g['test_TPR']:>8.3f}"
              f"{a['test_AUROC']:>15.3f}{a['test_TPR']:>8.3f}")
    print("=" * 120)
    print("解读: GADBench(Amazon/YelpChi) GADT3 srcAUROC 高 + testAUROC 高 => 方法本质有效;")
    print("      TUNE 数据集若 srcAUROC 低 或 score_std 塌缩 => backbone 对低维/弱同配图失效 (非方法无效)。")

    out = os.path.join(_structcp_root(),
                       "outputs", "diagnose_gadt3_failure.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"[saved] {out}")
