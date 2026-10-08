"""合并 StructCP/outputs/compare_all 的单文件 JSON, 生成初步对比表 (JSON + PDF)。

读取规则:
  - 每个 <Method>__<Dataset>.json 单文件; ok 取 auc_test (标准异常检测 AUROC),
    failed / 缺失取 None (输出表标 N/A)。
  - GAD-NR 已重训补齐 4 数据集权重 (2026-08-18), 纳入对比表。
  - 2026-09-20 补齐: GCN/GAT x TUNE 六数据集共 12 格。此前表格生成于 08-24 07:19,
    而对应 checkpoint 于同日 07:27-07:32 才训练完成, 属"表格先生成、权重后练完"的
    时序错配, 已用 run_all_baselines.py --methods GCN,GAT --datasets tune 重跑补齐。
  - 仍 N/A 的格子: DOMINANT/ARC/GADT3/TA-GGAD x TFinance (全图 OOM);
    GADT3 x YelpChi (全图 OOM); GAD-NR x TUNE 六集 (loss=nan 发散)。

用法:
  python scripts/make_comparison_table.py --indir outputs/compare_all --outdir outputs/compare_all
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

METHODS = ["BWGNN", "GCN", "GAT", "DOMINANT", "GAD-NR", "CoLA", "ARC", "UNPrompt", "GADT3", "TA-GGAD"]
# 主数据集 + TUNE 复现数据集 (novel-normality 统一协议)。TSocial 578万节点/1.46亿边
# 仅 BWGNN 有 homo checkpoint 但 CPU 上 146M 边稠密化不可行, 未纳入对比表。
DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance",
            "photo", "computer", "weibo", "tolokers", "questions", "reddit"]


# GAD-NR 在 TUNE 数据集上 loss=nan 发散 (骨干不支持), 即使残留 JSON 显示
# auc_test≈0.5 (无判别力) 也强制标 N/A, 避免伪 ok。
GADNR_TUNE_UNSUPPORTED = {"photo", "computer", "weibo", "tolokers", "questions", "reddit"}


def load_cell(indir, method, ds):
    if method == "GAD-NR" and ds in GADNR_TUNE_UNSUPPORTED:
        return None, "backbone_unsupported"
    path = os.path.join(indir, f"{method}__{ds}.json")
    if not os.path.exists(path):
        return None, "missing"
    try:
        r = json.load(open(path))
    except Exception:
        return None, "missing"
    if r.get("status") != "ok":
        return None, r.get("status", "failed")
    v = r.get("auc_test")
    if v is None:
        return None, "no_auc"
    return float(v), "ok"


# N/A / 特殊状态原因标注
# 算力限制 (全图 OOM): 历史全图方法在 TFinance 4200万边 8GB 环境 OOM
# 骨干不支持: GCN/GAT 无 TUNE checkpoint (冻结权重缺失); GAD-NR 在 TUNE 上 loss=nan 发散 (数值不稳定)。
NA_REASON = {
    ("DOMINANT", "TFinance"): "算力限制 (TFinance 4200万边全图 OOM)",
    ("ARC", "TFinance"): "算力限制 (TFinance 4200万边全图 OOM, 8GB GPU/CPU 内存不足)",
    ("GADT3", "YelpChi"): "算力限制 (YelpChi 全图 OOM)",
    ("GADT3", "TFinance"): "算力限制 (TFinance 全图 OOM)",
    ("TA-GGAD", "TFinance"): "算力限制 (TFinance 全图 OOM)",
    # TUNE 数据集: GAD-NR loss=nan 发散 (骨干不支持)
    # 注: GCN/GAT 的 TUNE 权重于 2026-08-24 07:27-07:32 训练完成, 对应 12 格已由
    #     scripts/compare/run_all_baselines.py --methods GCN,GAT --datasets tune
    #     于 2026-09-20 补齐落盘, 故此处不再标 N/A。
    ("GAD-NR", "photo"): "骨干不支持 (TUNE 上 loss=nan 发散)",
    ("GAD-NR", "computer"): "骨干不支持 (TUNE 上 loss=nan 发散)",
    ("GAD-NR", "weibo"): "骨干不支持 (TUNE 上 loss=nan 发散)",
    ("GAD-NR", "tolokers"): "骨干不支持 (TUNE 上 loss=nan 发散)",
    ("GAD-NR", "questions"): "骨干不支持 (TUNE 上 loss=nan 发散)",
    ("GAD-NR", "reddit"): "骨干不支持 (TUNE 上 loss=nan 发散)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs/compare_all")
    ap.add_argument("--outdir", default="outputs/compare_all")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    table = {}
    for m in METHODS:
        table[m] = {}
        for ds in DATASETS:
            v, st = load_cell(args.indir, m, ds)
            reason = NA_REASON.get((m, ds), "未跑通" if st != "ok" else "")
            table[m][ds] = {"auc": v, "status": st, "reason": reason if v is None else ""}

    # 汇总 JSON
    na_count = sum(1 for m in METHODS for ds in DATASETS if table[m][ds]["auc"] is None)
    from collections import Counter
    reason_counter = Counter(
        table[m][ds]["reason"]
        for m in METHODS for ds in DATASETS
        if table[m][ds]["auc"] is None
    )
    oom_count = sum(c for r, c in reason_counter.items() if r.startswith("算力限制"))
    backbone_count = sum(c for r, c in reason_counter.items() if r.startswith("骨干不支持"))
    summary = {
        "methods": METHODS,
        "datasets": DATASETS,
        "metric": "AUROC (test set, raw anomaly vs normal)",
        "na_total": na_count,
        "na_breakdown": {
            "算力限制 (全图 OOM)": oom_count,
            "骨干不支持 (无 checkpoint / loss=nan 发散)": backbone_count,
            **{f"其他: {r[:40]}": c for r, c in reason_counter.items()
               if not r.startswith(("算力限制", "骨干不支持"))},
        },
        "note": "全部数值来自 GitHub 官方代码真实运行 (2026-08-18/19, GCN/GAT x TUNE 于 2026-09-20 补跑), 并经经验方向校正 (若 auc(-score)>auc(+score)+0.05 则取反, 记录 direction_flipped)。TA-GGAD 官方检测器核心即 ARC 模型, 二者共用同一实现, 数值一致 (对比表并列注明等价)。数据集: 主数据集 Amazon/YelpChi/Elliptic/TFinance + TUNE novel-normality 6 集 (photo/computer/weibo/tolokers/questions/reddit)。TFinance 上 DOMINANT/ARC/GADT3/TA-GGAD 全图 OOM -> OOM; GAD-NR 在 TUNE 上 loss=nan 发散 -> 骨干不支持 N/A。注: GCN/GAT 在 TUNE 上 AUROC 普遍 < 0.6 (TUNE 为异质图, 同质 GCN/GAT 先天不适配), 属方法特性而非实现缺陷。",
        "table": table,
    }
    json_path = os.path.join(args.outdir, "comparison_table.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[written] {json_path}")

    # 简易文本矩阵
    print("\n=== 初步对比表 (AUROC, 越大越好) ===")
    hdr = f"{'Method':<10}" + "".join(f"{ds:>12}" for ds in DATASETS)
    print(hdr)
    print("-" * len(hdr))
    for m in METHODS:
        cells = []
        for ds in DATASETS:
            v = table[m][ds]["auc"]
            cells.append(f"{v:>12.4f}" if v is not None else f"{'N/A':>12}")
        print(f"{m:<10}" + "".join(cells))
    print("\nN/A 原因:")
    for m in METHODS:
        for ds in DATASETS:
            if table[m][ds]["auc"] is None and table[m][ds]["reason"]:
                print(f"  {m} x {ds}: {table[m][ds]['reason']}")

    # 文本矩阵 (总是更新, 无论 PDF 成败)
    txt_path = os.path.join(args.outdir, "comparison_table.txt")
    # 方法分组: Backbone(Frozen) / Self-Supervised / Test-Time Adaptation
    METHOD_GROUPS = [
        ("Backbone Detectors (Frozen)", ["BWGNN", "GCN", "GAT"]),
        ("Self-Supervised / Unsupervised", ["DOMINANT", "GAD-NR", "CoLA", "ARC", "UNPrompt"]),
        ("Test-Time Adaptation", ["GADT3", "TA-GGAD"]),
    ]
    with open(txt_path, "w") as f:
        f.write("# StructCP baselines comparison (AUROC, test set, raw anomaly vs normal)\n")
        f.write("# Method" + "\t" + "\t".join(DATASETS) + "\n")
        for grp_name, grp_methods in METHOD_GROUPS:
            f.write(f"# --- {grp_name} ---\n")
            for m in grp_methods:
                f.write(m + "\t" + "\t".join(
                    (f"{table[m][ds]['auc']:.4f}" if table[m][ds]['auc'] else "N/A")
                    for ds in DATASETS) + "\n")
        f.write("# N/A breakdown:\n")
        for r, c in reason_counter.most_common():
            f.write(f"#   {r}: {c}\n")
    print(f"[written] {txt_path}")

    # PDF (若 matplotlib 可用则用, 否则退化为文本)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        im = []
        for i, m in enumerate(METHODS):
            row = []
            for ds in DATASETS:
                v = table[m][ds]["auc"]
                row.append(v if v is not None else float("nan"))
            im.append(row)
        im = [[0.0 if v != v else v for v in r] for r in im]  # nan->0 for color
        ax.imshow(im, cmap="YlGnBu", aspect="auto", vmin=0.3, vmax=1.0)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS)
        for i, m in enumerate(METHODS):
            for j, ds in enumerate(DATASETS):
                v = table[m][ds]["auc"]
                txt = f"{v:.3f}" if v is not None else "N/A"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black")
        ax.set_title("StructCP Baselines Comparison (AUROC, test set)")
        plt.tight_layout()
        pdf_path = os.path.join(args.outdir, "comparison_table.pdf")
        fig.savefig(pdf_path)
        print(f"[written] {pdf_path}")
    except Exception as e:
        print(f"[WARN] PDF 生成跳过 (matplotlib 不可用或错误): {e}")


if __name__ == "__main__":
    main()
