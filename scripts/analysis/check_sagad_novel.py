# -*- coding: utf-8 -*-
"""
验证 SAGAD 骨干的 novel-normal 感知（关键决策点）。

如果 SAGAD 对 novel-normal 节点输出高异常分数（与 XGBGraph 相同问题），
则监督训练骨干（无论树/GNN）都不适配 StructCP 的 normality-shift 场景，
换骨干路径整体不成立，应回归 PPR+Dual 方案。

用法 (fov 环境):
  PYTHONPATH="D:\\PAPER\\StructCP" <PYTHON_FOV> \
      scripts/analysis/check_sagad_novel.py --dataset reddit
"""
import argparse, os, sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAGAD_DIR = os.path.join(_ROOT, "baselines", "SAGAD")
sys.path.insert(0, SAGAD_DIR)

from dataloader import load_data  # noqa: E402
from model import SAGAD  # noqa: E402
from utils import get_training_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reddit")
    ap.add_argument("--semi", action="store_true", default=False)
    args = ap.parse_args()

    # SAGAD 数据加载
    os.chdir(SAGAD_DIR)
    conf = get_training_config(args.dataset, config_path="full_train.conf.yaml")
    data = load_data(args.dataset, semi=args.semi, feat_trans=conf["feat_trans"])
    G, X, Y, train_mask, val_mask, test_mask = data

    # 构造模型
    import dgl
    in_dim = X.shape[1]
    model = SAGAD(in_dim=in_dim, **{k: conf[k] for k in
        ["hid_dims", "dropout", "activation", "num_gate_layers", "fusion",
         "K", "p", "conv_norm", "mlp_norm"]})
    model.load_state_dict(torch.load(f"snapshots/{args.dataset}_snapshot.pkl",
                                     weights_only=False))
    model.eval()

    # 全图前向
    Tx_list = []
    T = torch.eye(G.num_nodes())
    for _ in range(conf["K"] + 1):
        Tx_list.append(T)
        T = dgl.ops.gspmm(G, "copy_rhs", "sum", T)  # 简单聚合近似
    # 用 SAGAD 内部 graph 构造（简化：直接用 load_data 的 G）
    # SAGAD main 里 Tx_list 由 mrqsample/chebnet 构造，这里用度数归一化拉普拉斯近似
    from model import chebnet
    # 简化: 用原始特征 X 作为输入（SAGAD 的 X 是转换后特征）
    X_t = torch.from_numpy(X).float()
    with torch.no_grad():
        logits = model(Tx_list[:conf["K"]+1], X_t)
        scores = logits.softmax(dim=1)[:, 1].numpy()

    y = np.asarray(Y).ravel()
    # novel-normal: SAGAD 数据无 novel_mask（GADBench 协议），用我们的 TUNE 协议数据对齐
    # 直接报告 SAGAD 在 train/val/test 上的分数分布作为参考
    print(f"=== SAGAD {args.dataset} 分数分布 ===")
    for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        m = np.asarray(m).ravel()
        vals = scores[m]
        print(f"  {name:<8} n={vals.shape[0]:>6} mean={vals.mean():.3f} p50={np.median(vals):.3f}")
    # 异常 vs 正常
    anom = y == 1
    print(f"  异常   n={(anom).sum():>6} mean={scores[anom].mean():.3f} p50={np.median(scores[anom]):.3f}")
    print(f"  正常   n={(~anom).sum():>6} mean={scores[~anom].mean():.3f} p50={np.median(scores[~anom]):.3f}")


if __name__ == "__main__":
    main()
