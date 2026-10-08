"""UNPrompt (IJCAI 2025) PyG adapter for StructCP.

Reference: https://github.com/mala-lab/UNPrompt
Paper: UNPrompt: Zero-shot Graph Anomaly Detection with a Pre-trained Prompt Model.

适配到 novel-normality 协议:
  原 UNPrompt 是 zero-shot 跨数据集 (source=其他数据集训练 GCN+prompt,
  target=目标数据集推理)。本项目无独立 source dataset, 把 train_mask 节点
  作为 "source" (有监督), test_mask+novel_mask 当 "target", 与 ARC/CoLA/GADT3
  在同一可比协议上 (BWGNN frozen 仍是无监督, 作 backbone 基线)。

忠实复现的组件 (PyG 重写, 8GB 显存约束 R7):
  * GPF: residual additive prompts (in_dim 大小), sigmoid 加权组合
  * 2-layer GCN encoder (sparse matmul)
  * AttPool: attention-weighted prompt pool -> (n, emb_dim)
  * ProjHead: linear -> PReLU -> linear
  * linear+spmm 邻居平滑 (UNPrompt completion loss 的核心: MLP vs 图传播)

训练目标 (in-domain 实用化):
  原 UNPrompt 仅在 source 域做 CL + target normal 节点 completion, 跨数据集迁移
  才有效。在 in-domain 设置下, completion loss 收敛到 cos≈1, score (1-cos)
  区分度几乎为零。我们采用 UNPrompt 组件 + 监督判别 (BCE on train_mask):
    loss = BCE(Linear(h), y, mask=train_mask) + alpha * completion
  score = sigmoid(logit) (越大越异常).

  此版本在诚实标注 "UNPrompt (PyG, in-domain supervised)" 前提下提供稳定的 4-seed 复现.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_base import BaselineAdapter


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _normalize_adj(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """构造 D^-1/2 A D^-1/2 的稀疏 coo tensor (假设 edge_index 已无向化, 由 load_gad 给出)."""
    ei = edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, src, 1.0)
    np.add.at(deg, dst, 1.0)  # 无向图两方向都计数
    d_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0).astype(np.float32)
    val = d_inv_sqrt[src] * d_inv_sqrt[dst]
    indices = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    values = torch.tensor(val, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


# ---------------------------------------------------------------------------
# UNPrompt 模型组件 (PyG 重写, 与官方一致)
# ---------------------------------------------------------------------------

class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(A, self.lin(x))


class _GCN(nn.Module):
    """2-layer GCN encoder (中间层带 ReLU + Dropout)."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = [_GCNLayer(in_dim, hid_dim)]
        for _ in range(n_layers - 2):
            layers.append(_GCNLayer(hid_dim, hid_dim))
        layers.append(_GCNLayer(hid_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, A)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class _GPFPrompt(nn.Module):
    """GPF: x' = x + sigmoid(MLP(x)) @ P, P: (p_num, in_dim) learnable prompts."""

    def __init__(self, in_dim: int, p_num: int = 10):
        super().__init__()
        self.p_num = p_num
        self.prompts = nn.Parameter(torch.empty(p_num, in_dim))
        nn.init.xavier_uniform_(self.prompts)
        self.mlp = nn.Linear(in_dim, p_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.sigmoid(self.mlp(x))  # (n, p_num)
        return x + v @ self.prompts     # residual additive


class _AttPromptPool(nn.Module):
    """Attention-based prompt pool: 输入 x -> softmax(MLP(x)) @ prompt_emb."""

    def __init__(self, in_dim: int, emb_dim: int, p_num: int = 10):
        super().__init__()
        self.p_num = p_num
        self.prompt_emb = nn.Parameter(torch.empty(p_num, emb_dim))
        nn.init.xavier_uniform_(self.prompt_emb)
        self.mlp = nn.Sequential(nn.Linear(in_dim, p_num), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.mlp(x)              # (n, p_num)
        return att @ self.prompt_emb   # (n, emb_dim)


class _ProjHead(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(emb_dim, hid_dim)
        self.act = nn.PReLU()
        self.lin2 = nn.Linear(hid_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class _UNPromptNet(nn.Module):
    """UNPrompt 网络: GPF + GCN encoder + AttPool + ProjHead + Classifier.

    - h     = GCN(GPF(x), A)        # (n, emb_dim) 图编码
    - h_pool = AttPool(GPF(x))       # (n, emb_dim) 提示池化 (UNPrompt 官方结构)
    - h_proj = ProjHead(h)           # (n, emb_dim) MLP 投影 (completion 视角)
    - h_nei = spmm(A, Linear(h_pool))# (n, emb_dim) 邻居平滑 (图传播视角)
    - logit = Classifier(h_pool)     # (n,) 异常判别 (in-domain 监督, 取代官方 CL)
    """

    def __init__(self, in_dim: int, hid_dim: int = 64, emb_dim: int = 64,
                 p_num: int = 10, dropout: float = 0.0):
        super().__init__()
        self.gcn = _GCN(in_dim, hid_dim, emb_dim, n_layers=2, dropout=dropout)
        self.gpf = _GPFPrompt(in_dim, p_num)
        self.pool = _AttPromptPool(in_dim, emb_dim, p_num)
        self.proj = _ProjHead(emb_dim, hid_dim)
        self.lin_nei = nn.Linear(emb_dim, emb_dim, bias=False)
        self.classifier = nn.Linear(emb_dim, 1)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class UNPromptAdapter(BaselineAdapter):
    name = "UNPrompt"

    def __init__(self, device: str = "cpu",
                 p_num: int = 10, hid_dim: int = 64, emb_dim: int = 64,
                 n_layers: int = 2, epochs: int = 30, lr: float = 1e-3,
                 weight_decay: float = 5e-4, dropout: float = 0.0,
                 max_feat_dim: int = 64, completion_weight: float = 0.05,
                 pos_weight: float = 5.0, **kwargs):
        super().__init__(device=device)
        self.p_num = p_num
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_feat_dim = max_feat_dim
        self.completion_weight = completion_weight  # completion loss 权重 (UNPrompt 风格)
        self.pos_weight = pos_weight  # 异常/正常比例调整
        self.net: _UNPromptNet | None = None
        self.A: torch.Tensor | None = None
        self._svd = None

    # ----- 内部 -----
    def _featurize(self, x: torch.Tensor) -> torch.Tensor:
        """特征 > max_feat_dim 时用 TruncatedSVD 降到 max_feat_dim。"""
        d = x.shape[1]
        if d <= self.max_feat_dim:
            return x.float()
        from sklearn.decomposition import TruncatedSVD
        np_x = x.detach().cpu().numpy()
        if self._svd is None:
            self._svd = TruncatedSVD(n_components=self.max_feat_dim, random_state=42)
            np_red = self._svd.fit_transform(np_x).astype(np.float32)
        else:
            np_red = self._svd.transform(np_x).astype(np.float32)
        return torch.from_numpy(np_red).to(x.device)

    # ----- 接口 -----
    def fit(self, data) -> "UNPromptAdapter":
        d = self._to(data, self.device)
        n = d.num_nodes
        # 稀疏对称归一化邻接
        self.A = _normalize_adj(d.edge_index, n).to(self.device)
        # 特征 (必要时 SVD 降维)
        x = self._featurize(d.x).to(self.device)
        y = d.y.float().to(self.device)
        train_mask = d.train_mask.to(self.device)

        self.net = _UNPromptNet(
            in_dim=x.shape[1], hid_dim=self.hid_dim, emb_dim=self.emb_dim,
            p_num=self.p_num, dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        train_idx = train_mask.nonzero(as_tuple=True)[0]
        train_normal = (train_mask & (d.y == 0)).nonzero(as_tuple=True)[0]
        if train_normal.numel() == 0:
            raise RuntimeError("train_mask & (y==0) 为空, 无法训练 UNPrompt")

        # 自动估计正例权重 (异常在 train 中通常 < 10%)
        n_anomaly = (train_mask & (d.y == 1)).sum().item()
        n_normal = (train_mask & (d.y == 0)).sum().item()
        pw = max(self.pos_weight, n_normal / max(n_anomaly, 1))
        pos_weight_t = torch.tensor([pw], device=self.device)

        for ep in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            # 主路径: GPF + GCN
            x_gpf = self.net.gpf(x)
            h = self.net.gcn(x_gpf, self.A)
            h_pool = self.net.pool(x_gpf)
            logit = self.net.classifier(h_pool).squeeze(-1)

            # 监督 BCE on train_mask
            bce = F.binary_cross_entropy_with_logits(
                logit[train_idx], y[train_idx], pos_weight=pos_weight_t,
            )

            # UNPrompt 风格 completion loss (辅助)
            h_proj = self.net.proj(h)
            h_nei = torch.sparse.mm(self.A, self.net.lin_nei(h_pool))
            cos = F.cosine_similarity(h_proj, h_nei, dim=1)
            completion = -cos[train_normal].mean()

            loss = bce + self.completion_weight * completion
            loss.backward()
            opt.step()
            if ep % 10 == 0 or ep == self.epochs - 1:
                with torch.no_grad():
                    sig = torch.sigmoid(logit)
                    pred_anom = (sig > 0.5).float()
                    acc = (pred_anom[train_idx] == y[train_idx]).float().mean().item()
                print(
                    f"  [UNPrompt train] epoch {ep}/{self.epochs} "
                    f"bce={bce.item():.4f} comp={completion.item():+.4f} "
                    f"train_acc={acc:.3f}",
                    flush=True,
                )
        return self

    def fit_target(self, data) -> "UNPromptAdapter":
        # in-domain supervised, 不做 TTA 泄漏
        return self

    @torch.no_grad()
    def score(self, data) -> np.ndarray:
        assert self.net is not None and self.A is not None, "请先调用 fit"
        d = self._to(data, self.device)
        x = self._featurize(d.x).to(self.device)
        self.net.eval()
        x_gpf = self.net.gpf(x)
        h_pool = self.net.pool(x_gpf)
        logit = self.net.classifier(h_pool).squeeze(-1)
        # anomaly = sigmoid(logit), 越大越异常
        return torch.sigmoid(logit).cpu().numpy()


def _make_unprompt(**kw):
    """工厂: 供 registry 统一调用 (kw 由 build() 传入, 例如 device=...)."""
    kw.setdefault("device", "cpu")
    return UNPromptAdapter(**kw)