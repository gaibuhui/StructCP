"""GAT 编码器 (分块稀疏边注意力实现), 用于 A4 消融。

显存约束 (R7, 8GB): GAT 不能用 PyG 原生 GATConv / MessagePassing.propagate,
因为其内部会对全部边做 index_select 把节点特征 gather 成 (E, heads, att_out),
TFinance (E=42M) / Amazon (E=8.8M) / YelpChi (E=7.7M) 下直接 OOM (>3.6GB)。

本实现:
  * 边分数 e (E, heads) 分块计算, 不 materialize (E, heads, att_out)
  * 加权消息 alpha*Wh_j 分块聚合 (scatter_add), 每块仅 (chunk, heads, att_out)
  * 反向时 autograd 按块重算, 峰值显存 O(chunk * heads * att_out) 可控

注意力 (标准 GAT, 多头, 含自环):
  Wh   = x @ W^T                      (N, heads, att_out)
  e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )   (E, heads)
  α_ij = softmax_j(e_ij)             按目标节点 i 归一 (E, heads)
  h'_i = ||_k ( Σ_j α_ij Wh_j )       (N, heads*att_out)

接口契约 (与 models.bwgnn.BWGNN 一致):
  * __init__(in_channels, hidden_channels=64, out_channels=2, d=2, dropout=0.0,
             heads=8, att_out=16)
  * forward(x, edge_index) -> logits (N, out_channels)
  * embed(x, edge_index) -> (N, hidden_channels)
  * freeze() -> self
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax as scatter_softmax, scatter as _scatter


def _scatter_add(src, index, dim=0, dim_size=None):
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="add")


class GATEncoder(nn.Module):
    """两层 GAT (分块稀疏边注意力), embed 维度 = hidden_channels (默认 128)。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        d: int = 2,
        dropout: float = 0.0,
        heads: int = 8,
        att_out: int = 16,
        chunk: int = 200_000,
    ):
        super().__init__()
        self.d = d
        self.dropout = dropout
        self.heads = heads
        self.att_out = att_out
        self.hid = heads * att_out
        self.chunk = chunk

        self.W1 = nn.Linear(in_channels, self.hid, bias=False)
        self.W2 = nn.Linear(self.hid, self.hid, bias=False)
        self.att1 = nn.Parameter(torch.empty(2 * att_out, heads))
        self.att2 = nn.Parameter(torch.empty(2 * att_out, heads))
        nn.init.xavier_uniform_(self.att1)
        nn.init.xavier_uniform_(self.att2)

        self.lin_mid = nn.Linear(self.hid, hidden_channels)
        self.lin_out1 = nn.Linear(hidden_channels, hidden_channels)  # embed 取这里
        self.lin_out2 = nn.Linear(hidden_channels, out_channels)

    def _gat_layer(self, x, edge_index, W, att):
        """单层 GAT, 返回 (N, hid). 完全流式分块: e/alpha 不预分配全量张量,
        逐 chunk 计算 e->alpha->scatter_add, 峰值显存 O(chunk*heads*att_out),
        TFinance (E=42M) 下每 chunk=100K 仅 ~6MB, 可在 8GB 内跑通。
        """
        N = x.shape[0]
        row, col = edge_index[0], edge_index[1]
        E = row.shape[0]
        H, A = self.heads, self.att_out
        Wh = W(x).view(N, H, A)                     # (N, H, A) 小
        a_head = att.t()                            # (H, 2A)
        out = torch.zeros(N, H, A, device=x.device, dtype=x.dtype)
        for s in range(0, E, self.chunk):
            en = min(s + self.chunk, E)
            rs, cs = row[s:en], col[s:en]
            wi = Wh[rs]                             # (c, H, A)
            wj = Wh[cs]
            e_c = F.leaky_relu(
                (wi * a_head[:, :A] + wj * a_head[:, A:]).sum(dim=-1), 0.2)  # (c, H)
            alpha_c = scatter_softmax(e_c, rs)      # (c, H) 仅本块
            msg = alpha_c.unsqueeze(-1) * Wh[cs]    # (c, H, A)
            out = _scatter_add(msg, rs, dim=0, dim_size=N)
        return F.elu(out.reshape(N, self.hid))

    def _encode(self, x, edge_index):
        ei = add_self_loops(edge_index, num_nodes=x.shape[0])[0]
        h = self._gat_layer(x, ei, self.W1, self.att1)
        h = F.dropout(h, self.dropout, self.training)
        h = self._gat_layer(h, ei, self.W2, self.att2)
        h = F.relu(self.lin_mid(h))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.lin_out1(h))               # (N, hidden_channels)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self._encode(x, edge_index)
        return self.lin_out2(h)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self._encode(x, edge_index)

    @torch.no_grad()
    def anomaly_score(self, x, edge_index):
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
