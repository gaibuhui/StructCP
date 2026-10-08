"""BWGNN (Beta Wavelet Graph Neural Network, ICML 2022) — PyG 实现。

原官方实现依赖 DGL 0.8.1，与本环境 torch 2.3.1 不兼容，
故基于 torch_geometric 的稀疏拉普拉斯重写，数学上等价：

    Beta 小波:  W_{p,q} = (1/(2^{p+q+1} B(p+1,q+1))) * L^p (2I - L)^q
    其中 L = I - D^{-1/2} A D^{-1/2} 为归一化拉普拉斯, 谱位于 [0, 2]。

参考: Tang et al., "Rethinking Graph Neural Networks for Anomaly Detection", ICML 2022.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


def beta_wavelet_coeffs(d: int):
    """返回 d+1 组 (p, q) 的 Beta 小波参数。p + q = d。"""
    return [(p, d - p) for p in range(d + 1)]


class NormalizedLaplacian:
    """缓存 L = I - D^{-1/2} A D^{-1/2} 的稀疏算子，支持 L @ X。"""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device):
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float32).clamp(min=1)
        dinv = deg.pow(-0.5)
        # \hat{A} = D^{-1/2} A D^{-1/2}
        vals = dinv[row] * dinv[col]
        self.A_hat = torch.sparse_coo_tensor(
            edge_index, vals, (num_nodes, num_nodes), device=device
        ).coalesce()
        self.n = num_nodes

    def L(self, x: torch.Tensor) -> torch.Tensor:
        """L @ x = x - A_hat @ x"""
        return x - torch.sparse.mm(self.A_hat, x)

    def two_minus_L(self, x: torch.Tensor) -> torch.Tensor:
        """(2I - L) @ x = x + A_hat @ x"""
        return x + torch.sparse.mm(self.A_hat, x)


class BWGNN(nn.Module):
    """Beta Wavelet GNN。

    Args:
        in_channels:  输入特征维度
        hidden_channels: 隐藏维度
        out_channels: 输出维度 (二分类用 2)
        d: 小波阶数, 产生 d+1 个滤波器
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        d: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.dropout = dropout
        self.coeffs = beta_wavelet_coeffs(d)

        self.lin_in1 = nn.Linear(in_channels, hidden_channels)
        self.lin_in2 = nn.Linear(hidden_channels, hidden_channels)
        # 每个 Beta 滤波器一个独立线性层
        self.filters = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in self.coeffs]
        )
        self.lin_out1 = nn.Linear(hidden_channels * len(self.coeffs), hidden_channels)
        self.lin_out2 = nn.Linear(hidden_channels, out_channels)

        self._lap: NormalizedLaplacian | None = None
        self._lap_key = None

    # ------------------------------------------------------------------
    def _get_lap(self, edge_index, num_nodes, device):
        key = (int(edge_index.shape[1]), num_nodes)
        if self._lap is None or self._lap_key != key:
            self._lap = NormalizedLaplacian(edge_index, num_nodes, device)
            self._lap_key = key
        return self._lap

    def _beta_filter(self, h: torch.Tensor, p: int, q: int, lap: NormalizedLaplacian):
        """计算 W_{p,q} h。"""
        out = h
        for _ in range(q):
            out = lap.two_minus_L(out)
        for _ in range(p):
            out = lap.L(out)
        scale = 1.0 / (2 ** (p + q + 1) * math.exp(_log_beta(p + 1, q + 1)))
        return out * scale

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        lap = self._get_lap(edge_index, x.shape[0], x.device)

        h = F.relu(self.lin_in1(x))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.lin_in2(h))

        outs = []
        for (p, q), lin in zip(self.coeffs, self.filters):
            hw = self._beta_filter(h, p, q, lap)
            outs.append(F.relu(lin(hw)))
        h = torch.cat(outs, dim=1)

        h = F.relu(self.lin_out1(h))
        h = F.dropout(h, self.dropout, self.training)
        return self.lin_out2(h)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """返回 BWGNN 倒数第二层隐藏 (维度 = hidden_channels, 即子空间输入 d)。

        冻结特征提取模式专用: 不取分类 logits, 取 lin_out1 后的表征。
        """
        lap = self._get_lap(edge_index, x.shape[0], x.device)
        h = F.relu(self.lin_in1(x))
        h = F.relu(self.lin_in2(h))
        outs = []
        for (p, q), lin in zip(self.coeffs, self.filters):
            hw = self._beta_filter(h, p, q, lap)
            outs.append(F.relu(lin(hw)))
        h = torch.cat(outs, dim=1)
        h = F.relu(self.lin_out1(h))          # (N, hidden_channels)
        return h

    @torch.no_grad()
    def anomaly_score(self, x, edge_index):
        """返回异常概率 (softmax 的 class-1 分量)。"""
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]

    def freeze(self):
        """TTA 设定：冻结全部参数，检测器不更新。"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


def _log_beta(a: int, b: int) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
