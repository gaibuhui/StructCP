"""GAD-NR (WSDM-2024): Graph Anomaly Detection via Neighborhood Reconstruction.

复现核心思想: 图自编码器, 编码器(GCN 一跳聚合)把节点及其邻居压缩为表示 h_u,
解码器从 h_u 重构 (1)自身属性 x_u (MSE), (2)节点度 d_u (MSE), (3)邻居特征分布
(预测高斯, 与真实邻居属性分布 KL 散度)。异常分数 = 总体重建损失 L_u, 越高越异常。
无监督训练, 不依赖异常标签。

参考: Roy et al., "GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction", WSDM 2024.
默认超参 lambda_x=0.8, lambda_d=0.5, lambda_n=0.001, hidden=16。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, add_self_loops, scatter


class GADNR(nn.Module):
    """Neighborhood-Reconstruction 图自编码器骨干。

    用法: 无监督训练后冻结, anomaly_score(x, edge_index) 返回每个节点的
    重建损失(异常分数, 越高越异常)。StructCP 以该分数排序 + 加权共形分位数。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        dropout: float = 0.0,
        lambda_x: float = 0.8,
        lambda_d: float = 0.5,
        lambda_n: float = 0.001,
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.dropout = dropout
        self.lambda_x = lambda_x
        self.lambda_d = lambda_d
        self.lambda_n = lambda_n

        # 高维稀疏 -> 密集低维 (随机投影, 不训练)
        self.proj = nn.Linear(in_channels, hidden_channels, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, std=0.1)
            self.proj.weight.requires_grad = False

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # 解码器: 自身属性 / 度 / 邻居特征分布
        self.dec_x = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dec_d = nn.Linear(hidden_channels, 1)
        self.dec_n_mu = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dec_n_logvar = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.conv2(h, edge_index))
        return h

    def _neighbor_feat_mean_std(self, x, edge_index, num_nodes):
        """返回每个节点邻居(不含自身)特征的均值与对数方差 (真实邻域分布)。

        用 torch_geometric scatter 高效聚合, 避免大边表 gather 导致的显存爆炸。
        """
        row, col = edge_index
        # 仅用无自环的原始边
        mask = row != col
        row, col = row[mask], col[mask]
        h = self.proj(x)  # 用投影后特征近似邻居分布 (num_nodes, hidden)
        # 邻居均值: 对每个节点聚合其邻居 h[row]
        deg = degree(col, num_nodes, dtype=torch.float32).clamp(min=1)
        sum_h = scatter(h[row], col, dim=0, dim_size=num_nodes, reduce="sum")
        mu = sum_h / deg.unsqueeze(1)
        # 邻居二阶矩 -> 方差
        sum_h2 = scatter(h[row] ** 2, col, dim=0, dim_size=num_nodes, reduce="sum")
        var = (sum_h2 / deg.unsqueeze(1)) - mu ** 2
        logvar = torch.log(var.clamp(min=1e-6))
        return mu, logvar

    def reconstruction_loss(self, x, edge_index, h):
        num_nodes = x.shape[0]
        # (1) 自身属性 MSE
        x_rec = self.dec_x(h)
        feat = self.proj(x)
        l_x = F.mse_loss(x_rec, feat, reduction="none").mean(dim=1)  # [N]

        # (2) 度 MSE (归一化 log1p)
        deg = degree(edge_index[0], num_nodes, dtype=torch.float32).clamp(min=1)
        deg_norm = torch.log1p(deg)  # [N]
        d_rec = self.dec_d(h).squeeze(-1)  # [N]
        l_d = F.mse_loss(d_rec, deg_norm, reduction="none")  # [N]

        # (3) 邻居特征分布 KL: N(mu_pred, var_pred) || N(mu_true, var_true)
        mu_true, logvar_true = self._neighbor_feat_mean_std(x, edge_index, num_nodes)
        mu_pred = self.dec_n_mu(h)
        logvar_pred = self.dec_n_logvar(h)
        var_pred = logvar_pred.exp()
        var_true = logvar_true.exp()
        # KL(N0||N1) 封闭形式, 逐特征求和
        l_n = 0.5 * (
            (logvar_true - logvar_pred)
            + (var_pred + (mu_pred - mu_true) ** 2) / var_true
            - 1.0
        ).sum(dim=1)  # [N]

        l_u = self.lambda_x * l_x + self.lambda_d * l_d + self.lambda_n * l_n  # [N]
        return l_u, {"lx": l_x, "ld": l_d, "ln": l_n}

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        l_u, parts = self.reconstruction_loss(x, edge_index, h)
        return l_u, parts  # [N] 异常分数, 以及分项字典

    @torch.no_grad()
    def anomaly_score(self, x, edge_index):
        """返回异常分数 (总体重建损失 L_u)。已为越高越异常, 无需 softmax。"""
        self.eval()
        l_u, _ = self.forward(x, edge_index)
        return l_u

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
