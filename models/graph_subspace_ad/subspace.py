"""Step 3: 自适应 PCA 正态子空间构建 (本文核心创新模块)。

对源域正常节点嵌入 Z_norm 标准化后, 使用 随机 SVD (randomized SVD) 估计
主成分, 保留累计解释方差 >= var_threshold 的投影矩阵 P。

打分: 对测试节点特征 z_i
   1) 中心化:  z_tilde = z_i - mu
   2) 投影:    z_hat   = P P^T z_tilde
   3) 残差:    r_i     = ||z_tilde - z_hat||_2^2
正常节点紧贴子空间 -> 残差极小; 异常节点脱离流形 -> 残差巨大。
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class AdaptiveSubspacePCA:
    """正态特征子空间流形估计器 (training-free)。"""

    def __init__(
        self,
        var_threshold: float = 0.92,
        n_components_max: int = 128,
        random_state: int = 42,
    ):
        """
        Args:
            var_threshold:   累计解释方差阈值 (默认 0.92, 可消融 {0.8,0.85,0.9,0.92,0.95})
            n_components_max: 随机 SVD 最大秩 (<= 特征维度, 显存核心约束)
            random_state:   随机 SVD 固定种子, 保证可复现
        """
        self.var_threshold = var_threshold
        self.n_components_max = n_components_max
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.P: np.ndarray | None = None          # (d, K) 投影矩阵
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int = 0

    def fit(self, Z_norm: np.ndarray):
        """Z_norm: (N_norm, d) 仅源域正常节点嵌入。"""
        Z_norm = np.asarray(Z_norm, dtype=np.float32)
        self.scaler = StandardScaler()
        Zc = self.scaler.fit_transform(Z_norm)             # 中心化 + 标准化

        d = Zc.shape[1]
        k = min(self.n_components_max, d - 1, Zc.shape[0] - 1)
        # 随机 SVD: 高维低秩近似, 显存极低 (<0.5GB), 适合 128 维嵌入
        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        svd.fit(Zc)

        cum = np.cumsum(svd.explained_variance_ratio_)
        # 保留首个累计方差 >= 阈值的主成分数
        keep = int(np.searchsorted(cum, self.var_threshold) + 1)
        keep = max(1, min(keep, k))
        self.n_components_ = keep
        self.explained_variance_ratio_ = svd.explained_variance_ratio_

        # 投影矩阵 P = V[:, :keep], 使 P P^T 为重投影算子
        self.P = svd.components_[:keep].T.astype(np.float32)   # (d, keep)
        self._cum_var = float(cum[keep - 1]) if keep <= len(cum) else 1.0
        return self

    def reconstruct_residual(self, Z: np.ndarray) -> np.ndarray:
        """返回每个节点的重构残差 r_i = ||z_tilde - P P^T z_tilde||^2。

        Z: (N, d) 任意节点嵌入 (测试/校准/全图)。
        """
        Z = np.asarray(Z, dtype=np.float32)
        Zc = self.scaler.transform(Z)
        Zhat = Zc @ self.P @ self.P.T          # 子空间投影
        resid = np.sum((Zc - Zhat) ** 2, axis=1)
        return resid.astype(np.float32)

    def component_contributions(self, Z: np.ndarray) -> np.ndarray:
        """返回每个节点在各主成分上的投影能量 (用于异常归因/可解释性)。

        c_i = Zc_i @ P   -> (N, K) 每节点在主成分轴方向的投影幅度。
        高异常节点的 |c_i| 会显著偏离训练正常的投影分布, 揭示其"脱离
        子空间的方向", 对应论文的异常归因分析 (潜力 3)。

        Z: (N, d) 节点嵌入。
        """
        Z = np.asarray(Z, dtype=np.float32)
        Zc = self.scaler.transform(Z)
        contrib = Zc @ self.P                  # (N, K)
        return contrib.astype(np.float32)

    @property
    def cum_explained_variance(self) -> float:
        return getattr(self, "_cum_var", 0.0)
