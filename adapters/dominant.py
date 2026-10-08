"""DOMINANT (SDM'19) 无监督图异常检测 adapter。

原仓库: baselines/DOMINANT/{run,model,utils,layers}.py
核心: 属性-结构双重构自编码器 (AE), 异常分 = alpha*属性重构误差 + (1-alpha)*结构重构误差。

复现对齐 (见 _dominant_sparse.py 说明):
- 原仓库用 dense adj 训练 (run.py 的 loss_func: 逐节点 L2 重构误差, lr=5e-3, dropout=0.3)。
- 为服从 R7 8GB 约束, 本 adapter 改用 PyG 稀疏 GCN 实现同一算法;
  结构重构 (原 Z@Z^T) 在 [真实边+随机负边] 上近似, 语义等价。
- 训练用全部节点 (与原 run.py 一致, 不区分 train_mask)。
- 异常分对全部节点计算 (越大越异常)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# 让原仓库模块可被 import (用于 normalize_adj 等工具)
_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baselines", "DOMINANT")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from adapters.baseline_base import BaselineAdapter  # noqa: E402
from adapters._dominant_sparse import (  # noqa: E402
    _SparseDominant,
    _struct_recon_loss,
    _sample_neg_edges,
)


class DOMINANTAdapter(BaselineAdapter):
    method_name = "DOMINANT"

    def __init__(self, device="cpu", alpha=0.8, hid_units=64,
                 epochs=100, lr=5e-3, dropout=0.3, weight_decay=0.0,
                 neg_ratio=1.0, pos_edge_sample=1000000, seed=42):
        super().__init__(device=device)
        self.alpha = alpha
        self.hid_units = hid_units
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.neg_ratio = neg_ratio
        # 大图 (如 TFinance 4200万边) 全边点积过慢, 每 epoch 随机采样正边算结构误差
        self.pos_edge_sample = pos_edge_sample
        self.seed = seed
        self.model = None
        self._data = None

    def fit(self, data):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._data = data

        x = data.x.detach().cpu().float()
        edge_index = data.edge_index.detach().cpu().long()
        num_nodes = x.shape[0]
        rng = np.random.RandomState(self.seed)

        # 负边数量上限, 避免大图 (如 Amazon 880万边) 点积过慢
        n_neg = min(int(edge_index.shape[1] * self.neg_ratio), 200000)
        if n_neg <= 0:
            n_neg = int(edge_index.shape[1] * self.neg_ratio)
        neg_edge_index = _sample_neg_edges(edge_index, num_nodes, n_neg, rng)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        self.model = _SparseDominant(
            nfeat=x.shape[1], nhid=self.hid_units, dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.model.train()
        for ep in range(self.epochs):
            optimizer.zero_grad()
            x_hat, z, _ = self.model(x, edge_index)
            feat_err = torch.sqrt(torch.sum(torch.pow(x_hat - x, 2), 1))
            # 结构误差: 在 (采样正边+负边) 上计算, scatter 回两端节点
            ei = edge_index.to(self.device)
            ne = neg_edge_index.to(self.device)
            # 大图正边采样, 避免 4200万边点积过慢
            if ei.shape[1] > self.pos_edge_sample:
                perm = torch.randperm(ei.shape[1], device=self.device)[:self.pos_edge_sample]
                sei = ei[:, perm]
            else:
                sei = ei
            pos_score = (z[sei[0]] * z[sei[1]]).sum(dim=1)
            neg_score = (z[ne[0]] * z[ne[1]]).sum(dim=1) if ne.shape[1] > 0 else torch.empty(0, device=self.device)
            pos_err = torch.sqrt((pos_score - 1.0).pow(2) + 1e-8)
            neg_err = torch.sqrt((neg_score - 0.0).pow(2) + 1e-8)
            struct_per_edge = torch.cat([pos_err, neg_err])
            edge_all = torch.cat([sei, ne], dim=1) if ne.shape[1] > 0 else sei
            # 误差累加到两端节点 (与 score 一致)
            struct_node = torch.zeros(num_nodes, device=self.device)
            cnt = torch.zeros(num_nodes, device=self.device)
            for ends in (edge_all[0], edge_all[1]):
                struct_node.index_add_(0, ends, struct_per_edge)
                cnt.index_add_(0, ends, torch.ones_like(ends, dtype=torch.float))
            struct_node = struct_node / cnt.clamp(min=1)
            loss_vec = self.alpha * feat_err + (1 - self.alpha) * struct_node
            l = torch.mean(loss_vec)
            l.backward()
            optimizer.step()
            if (ep + 1) % 5 == 0:
                print(f"    [DOMINANT {data.name}] epoch {ep+1}/{self.epochs} loss={l.item():.2f}", flush=True)
        self.model.eval()
        return self

    def score(self, data):
        if self.model is None:
            self.fit(data)
        x = data.x.detach().cpu().float().to(self.device)
        edge_index = data.edge_index.detach().cpu().long().to(self.device)
        num_nodes = x.shape[0]
        rng = np.random.RandomState(self.seed + 1)
        n_neg = min(int(edge_index.shape[1] * self.neg_ratio), 200000)
        if n_neg <= 0:
            n_neg = int(edge_index.shape[1] * self.neg_ratio)
        neg_edge_index = _sample_neg_edges(edge_index.cpu(), num_nodes, n_neg, rng).to(self.device)

        self.model.eval()
        with torch.no_grad():
            x_hat, z, _ = self.model(x, edge_index)
        feat_err = torch.sqrt(torch.sum(torch.pow(x_hat - x, 2), 1))
        # 与训练一致: 结构误差在 (真实边+负边) 上计算, scatter 回节点
        ei = edge_index.to(self.device)
        ne = neg_edge_index.to(self.device)
        pos_score = (z[ei[0]] * z[ei[1]]).sum(dim=1)
        neg_score = (z[ne[0]] * z[ne[1]]).sum(dim=1) if ne.shape[1] > 0 else torch.empty(0, device=self.device)
        pos_err = torch.sqrt((pos_score - 1.0).pow(2) + 1e-8)
        neg_err = torch.sqrt((neg_score - 0.0).pow(2) + 1e-8)
        struct_per_edge = torch.cat([pos_err, neg_err])
        edge_all = torch.cat([ei, ne], dim=1) if ne.shape[1] > 0 else ei
        # 每条边误差累加到两端节点
        struct_node = torch.zeros(num_nodes, device=self.device)
        cnt = torch.zeros(num_nodes, device=self.device)
        for ends in (edge_all[0], edge_all[1]):
            struct_node.index_add_(0, ends, struct_per_edge)
            cnt.index_add_(0, ends, torch.ones_like(ends, dtype=torch.float))
        struct_node = struct_node / cnt.clamp(min=1)
        score = self.alpha * feat_err + (1 - self.alpha) * struct_node
        return score.detach().cpu().numpy().astype(np.float32)
