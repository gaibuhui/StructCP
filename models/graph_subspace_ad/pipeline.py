"""Graph SubspaceAD 端到端推理流水线 (零训练 / 零反向传播)。

整合:
  Step2 冻结 BWGNN 分批次特征提取 -> Z (N, d)
  Step3 自适应 PCA 子空间 -> 残差打分 r_i
  Step4 轻量 TTA 对齐 -> 对齐残差 r_i^align
  Step5 拓扑融合 -> s_final = alpha * Norm(r^align) + (1-alpha) * s_topo
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .subspace import AdaptiveSubspacePCA
from .topo import topology_anomaly_score
from .auto_alpha import select_alpha_adaptive


class GraphSubspaceAD:
    def __init__(
        self,
        bwgnn,
        device: str = "cuda",
        batch_size: int = 512,
        var_threshold: float = 0.92,
        tta_iters: int = 20,
        tta_enabled: bool = False,
        alpha: float = 1.0,
        n_components_max: int = 128,
        random_state: int = 42,
        auto_alpha: bool = False,
    ):
        self.bwgnn = bwgnn.to(device).eval()
        for p in self.bwgnn.parameters():
            p.requires_grad = False
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.var_threshold = var_threshold
        self.tta_iters = tta_iters
        self.tta_enabled = tta_enabled
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.n_components_max = n_components_max
        self.random_state = random_state

    @torch.no_grad()
    def _extract_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """单次全图前向取 encoder lin_out1 隐藏 (d 维 embedding), 避免逐 batch 重复全图前向。

        注意: GAT/GCN 的全图前向本身已较重, 逐 batch 重复会指数级拖慢, 故只前向一次。
        """
        h = self.bwgnn.embed(x, edge_index)  # (N, d) 单次全图前向
        return h.cpu().numpy().astype(np.float32)

    def fit_predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> np.ndarray | dict:
        """训练/校准用源域正常节点 (train_mask) 构建子空间;
        对 test_mask (默认全图) 打分。"""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x_np = x.cpu().numpy().astype(np.float32)

        # Step2: 冻结特征提取
        Z = self._extract_embeddings(x, edge_index)          # (N, d)
        Z_norm = Z[train_mask.cpu().numpy().astype(bool)]     # 仅源域正常

        # Step3: 自适应 PCA 子空间
        self.subspace = AdaptiveSubspacePCA(
            var_threshold=self.var_threshold,
            n_components_max=self.n_components_max,
            random_state=self.random_state,
        ).fit(Z_norm)

        # 测试集范围
        if test_mask is None:
            test_idx = np.arange(Z.shape[0])
        else:
            test_idx = np.where(test_mask.cpu().numpy().astype(bool))[0]
        Z_test = Z[test_idx]

        # Step4: 轻量 TTA 对齐
        if self.tta_enabled:
            Z_test = lightweight_tta_align(
                Z_test, Z_norm, iters=self.tta_iters, verbose=False
            )

        # 对齐后残差打分
        r_align = self.subspace.reconstruct_residual(Z_test)
        r_norm = MinMaxScaler().fit_transform(r_align.reshape(-1, 1)).ravel()

        # Step5: 拓扑得分 (基于有判别力的嵌入 Z, 而非原始特征 x)
        s_topo = topology_anomaly_score(Z, edge_index.cpu().numpy().astype(np.int64))
        s_topo_test = s_topo[test_idx]

        # 自适应 alpha: 仅用训练正常节点 (无标签) 定权, 消除手调 alpha 的塌陷风险
        if self.auto_alpha:
            r_norm_all = MinMaxScaler().fit_transform(
                self.subspace.reconstruct_residual(Z).reshape(-1, 1)).ravel()
            s_topo_all = MinMaxScaler().fit_transform(s_topo.reshape(-1, 1)).ravel()
            train_idx = np.where(train_mask.cpu().numpy().astype(bool))[0]
            self.alpha = select_alpha_adaptive(r_norm_all, s_topo_all, train_idx)
            print(f"[auto_alpha] selected alpha={self.alpha:.4f} "
                  f"(train nodes={len(train_idx)})", flush=True)

        # 最终融合
        s_final = self.alpha * r_norm + (1 - self.alpha) * s_topo_test
        s_final = MinMaxScaler().fit_transform(s_final.reshape(-1, 1)).ravel()

        if return_parts:
            return {
                "score": s_final.astype(np.float32),
                "residual": r_align.astype(np.float32),
                "topo": s_topo_test.astype(np.float32),
                "index": test_idx,
                "cum_var": self.subspace.cum_explained_variance,
                "k": self.subspace.n_components_,
            }
        return s_final.astype(np.float32)


def run_pipeline(model, data, **kwargs) -> np.ndarray:
    """便捷入口。data: StructCP.data.loader.GADData。"""
    g = GraphSubspaceAD(model, **{k: v for k, v in kwargs.items()
                                  if k in GraphSubspaceAD.__init__.__code__.co_varnames})
    return g.fit_predict(data.x, data.edge_index, data.train_mask, data.test_mask)
