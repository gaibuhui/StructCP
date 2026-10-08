"""GCN 编码器 (稀疏矩阵乘法实现, 内存对齐 BWGNN), 用于 A4 消融。

为什么不用 PyG GCNConv: GCNConv 的 message 会把全部边源特征 gather 成
(E, hidden) 张量, TFinance (E=42M) 下约 21GB 直接 OOM。本实现改用
归一化邻接稀疏矩阵 A_hat = D^{-1/2} A D^{-1/2} (含自环) 的 sparse.mm 聚合,
内存 O(N*hidden), 与 BWGNN 的 NormalizedLaplacian 完全一致, 可跑通超大图。

接口契约 (与 models.bwgnn.BWGNN 一致):
  * __init__(in_channels, hidden_channels=64, out_channels=2, d=2, dropout=0.0)
  * forward(x, edge_index) -> logits (N, out_channels)
  * embed(x, edge_index) -> (N, hidden_channels)
  * freeze() -> self
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

from models.bwgnn import NormalizedLaplacian


class GCNEncoder(nn.Module):
    """两层 GCN (稀疏聚合), embed 维度 = hidden_channels (默认 128)。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        d: int = 2,               # 占位兼容 BWGNN 签名
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.dropout = dropout
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.lin_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.lin_out1 = nn.Linear(hidden_channels, hidden_channels)  # embed 取这里
        self.lin_out2 = nn.Linear(hidden_channels, out_channels)
        self._lap = None
        self._lap_key = None

    def _get_lap(self, edge_index, num_nodes, device):
        key = (int(edge_index.shape[1]), num_nodes)
        if self._lap is None or self._lap_key != key:
            self._lap = NormalizedLaplacian(edge_index, num_nodes, device)
            self._lap_key = key
        return self._lap

    def _gcn_layer(self, x, w, lap):
        # A_hat @ (x @ w^T)  含自环 (NormalizedLaplacian 构造时已 add_self_loops)
        h = x @ w.weight.T
        if w.bias is not None:
            h = h + w.bias
        return torch.sparse.mm(lap.A_hat, h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        lap = self._get_lap(edge_index, x.shape[0], x.device)
        h = F.relu(self._gcn_layer(x, self.lin_in, lap))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self._gcn_layer(h, self.lin_hidden, lap))
        h = F.relu(self.lin_out1(h))          # (N, hidden_channels)
        h = F.dropout(h, self.dropout, self.training)
        return self.lin_out2(h)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        lap = self._get_lap(edge_index, x.shape[0], x.device)
        h = F.relu(self._gcn_layer(x, self.lin_in, lap))
        h = F.relu(self._gcn_layer(h, self.lin_hidden, lap))
        h = F.relu(self.lin_out1(h))          # (N, hidden_channels)
        return h

    @torch.no_grad()
    def anomaly_score(self, x, edge_index):
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
