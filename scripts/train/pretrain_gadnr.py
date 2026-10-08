"""训练 GAD-NR 骨干 (YelpChi) 用于 StructCP 强骨干恢复实验 (任务9d)。

设定: 仅在 train_mask (已见正常性) 节点上无监督优化邻域重建损失,
训练后冻结, 用于全图异常打分。显存约束(R7): 单图全量, num_workers=0。
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

sys.path.insert(0, _structcp_root())

from data.loader import load_gad  # noqa: E402
from models.gadnr import GADNR  # noqa: E402

ROOT = _structcp_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YelpChi", choices=["YelpChi", "Amazon", "Elliptic", "TFinance"])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lambda_n", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    help="GAD-NR YelpChi 无监督重建在 CPU 即可 (46k 节点), 避免大边表 GPU OOM; 可选 cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[data] loading {args.dataset} ...")
    data = load_gad(args.dataset, relation="homo", seed=args.seed).to(device)
    print("[data]", data)

    model = GADNR(data.num_features, args.hidden, lambda_n=args.lambda_n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_mask = data.train_mask
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        l_u, parts = model(data.x, data.edge_index)
        loss = l_u[train_mask].mean()
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                full, _ = model(data.x, data.edge_index)
            print(f"  ep {ep:3d} | train loss {loss.item():.4f} | full mean {full.mean().item():.4f}")

    dur = time.time() - t0
    os.makedirs(os.path.join(ROOT, "checkpoint"), exist_ok=True)
    ckpt = os.path.join(ROOT, "checkpoint", f"gadnr_{args.dataset}_homo.pth")
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "in_channels": data.num_features}, ckpt)
    print(f"[done] {dur:.1f}s | saved -> {ckpt}")

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", f"pretrain_gadnr_{args.dataset}.json"), "w") as f:
        json.dump({"dataset": args.dataset, "train_time_s": dur, **vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
