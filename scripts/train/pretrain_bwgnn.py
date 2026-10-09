"""Step 2: 预训练 BWGNN 检测器 (仅在"已见正常性"分布上训练)。

训练完成后权重固定，TTA 阶段完全冻结。
显存约束(R7): 单图全量前向, num_workers=0, 不开并行。
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
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402
from models.bwgnn import BWGNN  # noqa: E402
from models.gcn_encoder import GCNEncoder  # noqa: E402
from models.gat_encoder import GATEncoder  # noqa: E402

ROOT = _structcp_root()

ENCODERS = {"bwgnn": BWGNN, "gcn": GCNEncoder, "gat": GATEncoder}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Amazon", choices=["Amazon", "YelpChi", "Elliptic", "TFinance", "TSocial",
                             "photo", "computer", "weibo", "tolokers", "questions", "reddit"])
    ap.add_argument("--relation", default="homo", choices=["homo", "multi"],
                    help="homo=单图; multi=融合全部 relation (仅 YelpChi 有意义)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--order", type=int, default=2, help="Beta 小波阶数 d")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-5)
    ap.add_argument("--subsample", type=int, default=None,
                    help="TSocial 子采样节点数 (默认 100000)")
    ap.add_argument("--knn_backbone", type=int, default=None,
                    help="TSocial 子采样后补充特征 k-NN 骨干边 (默认 10)")
    ap.add_argument("--n_normality_clusters", type=int, default=8,
                    help="KMeans novel-normality 簇数 (默认 8)")
    ap.add_argument("--n_novel_clusters", type=int, default=2,
                    help="取最小的若干簇作为 novel normality (默认 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--encoder", default="bwgnn", choices=["bwgnn", "gcn", "gat"],
                    help="A4 消融: 切换 backbone (bwgnn/gcn/gat), 其余训练配置完全一致")
    ap.add_argument("--resume", action="store_true",
                    help="从续跑点 ({encoder}_{ds}_h{hidden}_resume.pth) 继续训练")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[data] loading {args.dataset} ...")
    relation = args.relation if args.dataset not in ("Elliptic", "TFinance", "TSocial") else "homo"
    subsample = args.subsample
    knn_backbone = args.knn_backbone
    if args.dataset == "TSocial":
        if subsample is None:
            subsample = 100000
        if knn_backbone is None:
            knn_backbone = 10
    # TUNE 复现涉及的额外数据集用稍高的簇数, 与 run_compare_baselines 保持一致
    if args.dataset in ("photo", "computer", "reddit", "questions", "tolokers"):
        n_nc, n_nv = 12, 3
    elif args.dataset == "weibo":
        n_nc, n_nv = 40, 10
    else:
        n_nc, n_nv = args.n_normality_clusters, args.n_novel_clusters
    data = load_gad(args.dataset, relation=relation, seed=args.seed,
                    subsample=subsample, knn_backbone=knn_backbone,
                    n_normality_clusters=n_nc, n_novel_clusters=n_nv).to(device)
    print("[data]", data)

    enc_cls = ENCODERS[args.encoder]
    if args.encoder == "bwgnn":
        model = BWGNN(data.num_features, args.hidden, 2, d=args.order,
                      dropout=0.0).to(device)
    elif args.encoder == "gcn":
        model = GCNEncoder(data.num_features, args.hidden, 2,
                           d=args.order, dropout=0.0).to(device)
    else:  # gat
        model = GATEncoder(data.num_features, args.hidden, 2,
                           d=args.order, dropout=0.0, heads=8, att_out=16).to(device)
    print(f"[model] {args.encoder} 参数量={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 类别不平衡加权
    y_tr = data.y[data.train_mask]
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    print(f"[train] pos={n_pos} neg={n_neg} class_weight={weight.tolist()}")

    best_auc, best_state = 0.0, None
    start_ep = 1
    # 续跑支持: 若 --resume 且中间文件存在, 从断点继续 (防长任务空闲超时中断)
    resume_path = os.path.join(
        ROOT, "checkpoint",
        f"{args.encoder}_{args.dataset}_h{args.hidden}_resume.pth")
    if args.resume and os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rs["model"])
        opt.load_state_dict(rs["opt"])
        start_ep = rs["epoch"] + 1
        best_auc = rs.get("best_auc", 0.0)
        best_state = rs.get("best_state")
        print(f"[resume] 从 epoch {start_ep} 续跑, 历史 best_auc={best_auc:.4f}")

    # 提前建目录（续跑点/最终 checkpoint 都可能写入；避免首次运行时父目录缺失）
    os.makedirs(os.path.join(ROOT, "checkpoint"), exist_ok=True)
    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        opt.step()

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
            m = data.cal_mask
            auc = roc_auc_score(data.y[m].cpu(), probs[m].cpu())
            ap_ = average_precision_score(data.y[m].cpu(), probs[m].cpu())
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep:3d} | loss {loss.item():.4f} | cal AUROC {auc:.4f} AUPRC {ap_:.4f}", flush=True)

        # 每 20 epoch 落盘续跑点, 防止长任务被环境中断丢失进度
        if ep % 20 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "best_auc": best_auc, "best_state": best_state,
                        "args": vars(args)}, resume_path)

    dur = time.time() - t0
    os.makedirs(os.path.join(ROOT, "checkpoint"), exist_ok=True)
    if os.path.exists(resume_path):
        os.remove(resume_path)  # 正常结束清续跑点
    # 文件名省略 relation 段 (A4 仅 homo 单图; BWGNN 与现有 bwgnn_{ds}_h128.pth 同名对齐)
    ckpt = os.path.join(ROOT, "checkpoint",
                        f"{args.encoder}_{args.dataset}_h{args.hidden}.pth")
    torch.save({"state_dict": best_state, "args": vars(args),
                "in_channels": data.num_features, "encoder": args.encoder}, ckpt)
    print(f"[done] best cal AUROC={best_auc:.4f} | {dur:.1f}s | saved -> {ckpt}")

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", f"pretrain_{args.encoder}_{args.dataset}.json"), "w") as f:
        json.dump({"dataset": args.dataset, "encoder": args.encoder,
                   "best_cal_auroc": best_auc, "train_time_s": dur,
                   **vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
