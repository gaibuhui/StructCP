"""Graph SubspaceAD — 视觉训练免子空间异常检测范式引入图异常检测。

论文方法章节落地实现 (零训练 / 零反向传播 / 消费级显卡友好):
  Step1  分层采样 + hub 裁剪 (datasets 固定目录, 软链引用, 不拷贝)
  Step2  冻结 BWGNN 特征提取 (分批次前向, 显存友好)
  Step3  自适应 PCA 正态子空间 + 重构残差异常打分 (核心创新)
  Step4  嵌入级轻量 TTA 跨域对齐 (无梯度, 纯矩阵优化)
  Step5  拓扑结构得分融合 (邻域相似度 + 度偏差 + 局部结构熵)
"""
from .pipeline import GraphSubspaceAD, run_pipeline
from .subspace import AdaptiveSubspacePCA
from .topo import topology_anomaly_score
from .auto_alpha import select_alpha_adaptive

__all__ = [
    "GraphSubspaceAD",
    "run_pipeline",
    "AdaptiveSubspacePCA",
    "topology_anomaly_score",
    "select_alpha_adaptive",
]
