"""DOMINANT 的 PyG 稀疏实现 (8GB 显存/内存约束适配)。

原仓库 (baselines/DOMINANT) 用 dense adj 训练, 在 YelpChi(46k)/Elliptic(203k)/TFinance(39k)
上 dense 邻接矩阵会 OOM 且极慢。本实现保持 DOMINANT 核心算法:
  - 共享 GCN 编码器 (2 层)
  - 属性解码器 (重构 X)
  - 结构解码器 (重构邻接, 原仓库用 Z@Z^T 低秩近似)
  - 异常分 = alpha*属性重构 L2 误差 + (1-alpha)*结构重构 L2 误差

稀疏化关键: 结构重构误差原是 dense (Z@Z^T - A), 这里改为只在
  [真实边 (A_ij=1) + 随机负采样边 (A_ij=0)] 上计算点积误差,
  与 dense 版本语义一致且复杂度 O(E + neg)。

注意: 与原仓库数值不完全相等 (稀疏采样近似 + PyG GCN 归一化), 但算法等价,
  属 R7 8GB 约束下的必要复现适配。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class _SparseDominant(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.enc1 = GCNConv(nfeat, nhid)
        self.enc2 = GCNConv(nhid, nhid)
        self.attr1 = GCNConv(nhid, nhid)
        self.attr2 = GCNConv(nhid, nfeat)
        self.struct1 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.relu(self.enc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.enc2(x, edge_index))
        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        # 属性解码
        h = F.relu(self.attr1(z, edge_index))
        h = F.dropout(h, self.dropout, training=self.training)
        x_hat = self.attr2(h, edge_index)
        # 结构解码 (低秩 Z@Z^T 近似在 score 中计算)
        s = F.relu(self.struct1(z, edge_index))
        return x_hat, z, s


def _struct_recon_loss(z, edge_index, neg_edge_index, device):
    """在 真实边 + 负采样边 上计算 Z@Z^T 与 A 的 L2 重构误差。"""
    ei = edge_index.to(device)
    ne = neg_edge_index.to(device)
    pos_score = (z[ei[0]] * z[ei[1]]).sum(dim=1)          # 应为 1
    neg_score = (z[ne[0]] * z[ne[1]]).sum(dim=1)          # 应为 0
    pos_err = torch.sqrt((pos_score - 1.0).pow(2) + 1e-8)
    neg_err = torch.sqrt((neg_score - 0.0).pow(2) + 1e-8)
    # 与 dense 版本对齐: 每条边一个误差, 求均值
    return torch.cat([pos_err, neg_err])


def _sample_neg_edges(edge_index, num_nodes, n_neg, rng):
    """用 PyG 的快速负采样 (基于训练边的稀疏排斥, 向量化, 避免 Python 循环瓶颈)。"""
    if n_neg <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    try:
        from torch_geometric.utils import negative_sampling
        neg = negative_sampling(
            edge_index, num_nodes=num_nodes, num_neg_samples=n_neg,
            method="sparse", force_undirected=False,
        )
        return neg.long()
    except Exception:
        # 退化为纯随机向量化采样 (不查重, 接受少量重叠)
        a = rng.randint(0, num_nodes, size=n_neg)
        b = rng.randint(0, num_nodes, size=n_neg)
        mask = a != b
        a, b = a[mask], b[mask]
        return torch.stack([torch.from_numpy(a), torch.from_numpy(b)], dim=0).long()
