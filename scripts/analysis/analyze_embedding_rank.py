"""分析冻结 BWGNN 嵌入空间中"正常节点流形"的低秩性 (几何事实验证)。

核心问题 (审稿人切入点): 正常节点嵌入是否位于低维线性子空间?
若是, 则邻域一致性 c_i 可解释为对该低秩流形的正交残差的局部代理,
为 StructCP 的图结构权重提供几何依据 (paper Sec.3.1 / tab:r90 / fig:pca_curve)。

输出:
  - outputs/rank_analysis.json : 每数据集 r90、前 k 主成分累积方差、正常/异常投影残差
  - docs/figs/pca_curve.pdf    : 累积方差曲线 (4 数据集, 标 90% 线 + r90)
  - docs/figs/pca_residual.png : 正常 vs 异常节点投影残差直方图

口径:
  - 嵌入 = model.embed(data.x, edge_index)  # (N, hidden)
  - 正常流形 = PCA 在 cal_mask & (y==0) (源域正常, 去泄漏) 上拟合
  - 异常 = test_mask & (y==1); 投影残差 = ||x - P_r x|| (r=r90)
  - r90 = 累积解释方差达 90% 所需最少主成分数
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
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, _structcp_root())
ROOT = _structcp_root()

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402

MAIN_DATASETS = ["Amazon", "YelpChi", "Elliptic", "TFinance"]


def load_model(dataset, device):
    ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_h128.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT, "checkpoint", f"bwgnn_{dataset}_homo.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    return model


@torch.no_grad()
def get_embed(model, data, device):
    return model.embed(data.x.to(device), data.edge_index.to(device)).cpu().numpy()


def analyze(dataset, device, seed=42, var_thresh=0.90, model_name="bwgnn", model_cls=None, ckpt_pattern=None):
    data = load_gad(dataset, relation="homo", seed=seed).to(device)
    # 加载模型
    if model_name == "bwgnn":
        model = load_model(dataset, device)
    else:
        if not model_cls or not ckpt_pattern:
            raise ValueError("model_cls and ckpt_pattern must be provided for non-bwgnn models")
        ckpt_path = os.path.join(ROOT, "checkpoint", ckpt_pattern.format(dataset=dataset))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ROOT, "checkpoint", ckpt_pattern.format(dataset=dataset).replace("h128", "homo"))
        ckpt = torch.load(ckpt_path, map_location=device)
        ca = ckpt["args"]
        model = model_cls(ckpt["in_channels"], ca["hidden"], 2, d=ca["order"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.freeze()
    emb = get_embed(model, data, device)
    emb = emb.astype(np.float64)
    # 基线1: 原始节点特征
    raw_emb = data.x.cpu().numpy().astype(np.float64)
    # 基线2: 随机投影特征 (Gaussian random matrix)
    np.random.seed(seed)
    rand_proj = np.random.randn(emb.shape[1], 128)  # 投影到128维，和嵌入维度一致
    rand_emb = emb @ rand_proj

    # 正常流形: 源域正常 (cal_mask & y==0, 去泄漏口径)
    cal = data.cal_mask.cpu().numpy()
    y = data.y.cpu().numpy()
    test = data.test_mask.cpu().numpy()
    normal_idx = np.where(cal & (y == 0))[0]
    anomaly_idx = np.where(test & (y == 1))[0]
    normal_emb = emb[normal_idx]

    # PCA (中心化后)
    pca = PCA()
    pca.fit(normal_emb)
    cum = np.cumsum(pca.explained_variance_ratio_)
    r90 = int(np.argmax(cum >= var_thresh)) + 1
    cum2 = float(cum[1]) if len(cum) > 1 else float(cum[0])
    pc1 = float(pca.explained_variance_ratio_[0])
    pc2 = float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0

    # 两种残差度量 (正交子空间能量 vs 前 r90 维重建损失)
    # 1) 正交残差能量: ||z[:, r90:]||  (z = PCA scores) —— 直接度量"偏离低秩流形的量",
    #    不受保留主成分稀释, 是最稳健的流形偏离指标
    def orth_residual(X):
        z = pca.transform(X)
        return np.linalg.norm(z[:, r90:], axis=1)

    # 2) 重建损失 (保持向后兼容)
    def recon_residual(X):
        z = pca.transform(X)
        zr = np.zeros_like(z)
        zr[:, :r90] = z[:, :r90]
        return np.linalg.norm(X - pca.inverse_transform(zr), axis=1)

    z_normal = pca.transform(normal_emb)
    res_normal_orth = np.linalg.norm(z_normal[:, r90:], axis=1)
    res_normal_recon = recon_residual(normal_emb)

    anomaly_idx = np.where(test & (y == 1))[0]
    novel_idx = np.where(test & (y == 0))[0]
    res_anom_orth = orth_residual(emb[anomaly_idx])
    res_novel_orth = orth_residual(emb[novel_idx])

    # 计算基线的r90
    def compute_r90(embedding, normal_idx, var_thresh=0.90):
        normal_emb = embedding[normal_idx]
        pca = PCA()
        pca.fit(normal_emb)
        cum = np.cumsum(pca.explained_variance_ratio_)
        r90 = int(np.argmax(cum >= var_thresh)) + 1
        return r90, cum

    # 基线1: 原始节点特征
    raw_r90, raw_cum = compute_r90(raw_emb, normal_idx, var_thresh=var_thresh)
    # 基线2: 随机投影特征 (Gaussian random matrix)
    np.random.seed(seed)
    rand_proj = np.random.randn(emb.shape[1], 128)  # 投影到128维，和嵌入维度一致
    rand_emb = emb @ rand_proj
    rand_r90, rand_cum = compute_r90(rand_emb, normal_idx, var_thresh=var_thresh)

    rec = {
        "dataset": dataset, "seed": seed, "model": model_name, "hidden_dim": emb.shape[1],
        "n_normal_cal": int(len(normal_idx)), "n_anomaly_test": int(len(anomaly_idx)),
        "r90": r90, "cum_var_at_r90": float(cum[r90 - 1]),
        "raw_r90": raw_r90, "raw_cum_var_at_r90": float(raw_cum[raw_r90 - 1]),
        "rand_r90": rand_r90, "rand_cum_var_at_r90": float(rand_cum[rand_r90 - 1]),
        "pc1_var": pc1, "pc2_var": pc2, "cum_var_top2": cum2 + pc1 if r90 >= 2 else pc1,
        # 正交残差能量 (主指标): 异常应显著 > 正常
        "orth_residual_normal_mean": float(res_normal_orth.mean()),
        "orth_residual_normal_std": float(res_normal_orth.std()),
        "orth_residual_anomaly_mean": float(res_anom_orth.mean()) if len(res_anom_orth) else float("nan"),
        "orth_residual_anomaly_std": float(res_anom_orth.std()) if len(res_anom_orth) else float("nan"),
        "orth_residual_novel_mean": float(res_novel_orth.mean()) if len(res_novel_orth) else float("nan"),
        "orth_residual_novel_std": float(res_novel_orth.std()) if len(res_novel_orth) else float("nan"),
        "orth_ratio_anom_over_normal": float(res_anom_orth.mean() / max(res_normal_orth.mean(), 1e-12)) if len(res_anom_orth) else float("nan"),
        # 重建损失 (次指标, 兼容旧版)
        "residual_normal_mean": float(res_normal_recon.mean()),
        "residual_ratio_anom_over_normal": float(res_anom_orth.mean() / max(res_normal_orth.mean(), 1e-12)) if len(res_anom_orth) else float("nan"),
        "cumvar_full": [float(x) for x in cum],
        "raw_cumvar_full": [float(x) for x in raw_cum],
        "rand_cumvar_full": [float(x) for x in rand_cum],
    }
    return rec


def plot_curves(recs, out_pdf):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Amazon": "#1f77b4", "YelpChi": "#ff7f0e", "Elliptic": "#2ca02c", "TFinance": "#d62728"}
    for r in recs:
        cum = r["cumvar_full"]
        ax.plot(range(1, len(cum) + 1), cum, label=f"{r['dataset']} (r90={r['r90']})",
                color=colors.get(r["dataset"], None), linewidth=2)
        ax.axvline(r["r90"], color=colors.get(r["dataset"], "gray"), linestyle=":", alpha=0.6)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="90% variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Low-rank structure of source-normal embeddings (BWGNN, h128)")
    ax.set_xlim(0, min(16, max(len(r["cumvar_full"]) for r in recs)))
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[saved] {out_pdf}")


def plot_residual(recs, out_png):
    # 按数据集分组，每个数据集一个子图
    datasets = list(set(r["dataset"] for r in recs))
    n_plots = len(datasets)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 3.6))
    if n_plots == 1:
        axes = [axes]
    
    for ax, ds in zip(axes, datasets):
        ds_recs = [r for r in recs if r["dataset"] == ds]
        if not ds_recs:
            continue
        # 取平均的残差
        labels = ["normal", "novel", "anomaly"]
        means = [
            np.mean([r["orth_residual_normal_mean"] for r in ds_recs]),
            np.mean([r["orth_residual_novel_mean"] for r in ds_recs]),
            np.mean([r["orth_residual_anomaly_mean"] for r in ds_recs])
        ]
        stds = [
            np.mean([r["orth_residual_normal_std"] for r in ds_recs]),
            np.mean([r["orth_residual_novel_std"] for r in ds_recs]),
            np.mean([r["orth_residual_anomaly_std"] for r in ds_recs])
        ]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#d62728"])
        ax.set_title(f"{ds} (avg r90={np.mean([r['r90'] for r in ds_recs]):.1f})")
        ax.set_ylabel("orth. residual energy")
        for i, (b, m) in enumerate(zip(bars, means)):
            ax.text(b.get_x() + b.get_width() / 2, m + stds[i], f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_r90_distribution(recs, out_pdf):
    import seaborn as sns
    import pandas as pd
    
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="dataset", y="r90", hue="model", ax=ax)
    ax.set_title("r90 distribution across seeds and models")
    ax.set_ylabel("Number of principal components (r90)")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[saved] {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(MAIN_DATASETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--models", type=str, default="bwgnn")
    ap.add_argument("--var_thresh", type=float, default=0.90)
    args = ap.parse_args()
    device = torch.device(args.device)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    # 加载模型
    from models import gcn_encoder, gat_encoder, bwgnn
    model_classes = {
        "bwgnn": (bwgnn.BWGNN, "bwgnn_{dataset}_h128.pth"),
        "gcn": (gcn_encoder.GCNEncoder, "gcn_{dataset}_h128.pth"),
        "gat": (gat_encoder.GATEncoder, "gat_{dataset}_h128.pth"),
    }
    # 检查请求的模型是否都支持
    for m in model_list:
        if m not in model_classes:
            raise ValueError(f"Unsupported model: {m}, available: {list(model_classes.keys())}")

    all_recs = []
    for model_name in model_list:
        model_cls, ckpt_pattern = model_classes[model_name]
        for ds in datasets:
            print(f"=== {model_name} | {ds} ===", flush=True)
            seed_recs = []
            for seed in seed_list:
                r = analyze(ds, device, seed=seed, var_thresh=args.var_thresh, model_name=model_name, model_cls=model_cls, ckpt_pattern=ckpt_pattern)
                print(f"  seed={seed}: r90={r['r90']}  PC1={r['pc1_var']:.3f}  PC2={r['pc2_var']:.3f}  "
                      f"cum@r90={r['cum_var_at_r90']:.3f}")
                print(f"  orth-residual mean: normal={r['orth_residual_normal_mean']:.4f} "
                      f"novel={r['orth_residual_novel_mean']:.4f} anomaly={r['orth_residual_anomaly_mean']:.4f} "
                      f"(anom/normal={r['orth_ratio_anom_over_normal']:.2f}x)")
                seed_recs.append(r)
            all_recs.extend(seed_recs)

    # 保存所有结果
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out_json = os.path.join(ROOT, "outputs", "rank_analysis_multi_seed_multi_model.json")
    with open(out_json, "w") as f:
        json.dump({"config": {"var_thresh": args.var_thresh, "seeds": seed_list, "models": model_list}, "datasets": all_recs}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_json}")

    # 绘制累积方差曲线（按模型和数据集）
    for model_name in model_list:
        model_recs = [r for r in all_recs if r["model"] == model_name]
        if not model_recs:
            continue
        for ds in datasets:
            ds_recs = [r for r in model_recs if r["dataset"] == ds]
            if not ds_recs:
                continue
            plot_curves(ds_recs, os.path.join(ROOT, "docs", "figs", f"pca_curve_{model_name}_{ds}.pdf"))

    # 绘制r90分布箱线图
    plot_r90_distribution(all_recs, os.path.join(ROOT, "docs", "figs", "r90_distribution.pdf"))

    # 绘制残差对比图
    for model_name in model_list:
        model_recs = [r for r in all_recs if r["model"] == model_name]
        if not model_recs:
            continue
        for ds in datasets:
            ds_recs = [r for r in model_recs if r["dataset"] == ds]
            if not ds_recs:
                continue
            # 取第一个种子的残差？或者取平均？
            plot_residual(ds_recs, os.path.join(ROOT, "docs", "figs", f"pca_residual_{model_name}_{ds}.pdf"))

    # 绘制基线对比表
    with open(os.path.join(ROOT, "outputs", "rank_analysis_summary.md"), "w") as f:
        f.write("# r90 分析结果汇总\n\n")
        f.write("| 模型 | 数据集 | 种子 | r90 (嵌入) | r90 (原始特征) | r90 (随机投影) |\n")
        f.write("|------|--------|------|------------|----------------|----------------|\n")
        for r in all_recs:
            f.write(f"| {r['model']} | {r['dataset']} | {r['seed']} | {r['r90']} | {r['raw_r90']} | {r['rand_r90']} |\n")
    print(f"\n[saved] 汇总结果到 {os.path.join(ROOT, 'outputs', 'rank_analysis_summary.md')}")


if __name__ == "__main__":
    main()
