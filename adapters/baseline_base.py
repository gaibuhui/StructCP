"""统一基线适配器抽象基类 (全自动对比实验框架)。

设计目标:
  * 让所有对比方法 (GCN/GAT/BWGNN/DOMINANT/CoLA/TAM/SpaceGNN/GCTAM/
    CONSISGAD/UNPrompt/AnomalyGFM/ARC/GADT3/TA-GGAD/...) 共用同一接口,
    由 scripts/run_all_baselines.py 在同数据集、同 mask 上自动串跑、
    统一算 AUC/AP, 产出可直接粘贴论文主表的 compare_all.json。

接口约定 (对齐已有 ARC/GADT3 adapter):
  * fit(data):        用 train_mask 节点训练/拟合 (无标签方法用 train 正常节点)
  * fit_target(data): 测试时自适应 (可选; 无则空实现)
  * score(data):      返回连续异常分, **越大越异常**, shape=(n_nodes,)
  * 输入/输出均为 data.loader.GADData

所有适配器必须实现 __name__ 类属性 (方法简称, 与 baselines/<NAME>/ 对应)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaselineAdapter(ABC):
    """对比方法统一接口。子类实现 fit / fit_target / score。"""

    #: 方法简称, 用于落盘文件名与论文主表 (如 "BWGNN", "GCN", "TAM")
    name: str = "UNNAMED"

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.extra = kwargs

    @abstractmethod
    def fit(self, data) -> "BaselineAdapter":
        """用 train_mask 节点拟合/训练。返回 self 便于链式调用。"""
        raise NotImplementedError

    def fit_target(self, data) -> "BaselineAdapter":
        """测试时自适应 (默认空实现, 无 TTA 的方法直接跳过)。"""
        return self

    @abstractmethod
    @torch.no_grad()
    def score(self, data) -> torch.Tensor:
        """返回连续异常分, 越大越异常, shape=(n_nodes,)。"""
        raise NotImplementedError

    # -------- 通用工具 --------
    @staticmethod
    def _to(data, device):
        return data.to(device)

    @staticmethod
    def _normal_train_mask(data) -> torch.Tensor:
        """train 中的正常节点 (label==0) 掩码。"""
        return data.train_mask & (data.y == 0)

    @staticmethod
    def _safe_score_shape(score, n: int) -> np.ndarray:
        s = np.asarray(score).reshape(-1)
        if s.shape[0] != n:
            # 部分方法只给 test 子集分数, 这里要求全量; 不一致则报错提示子类
            raise ValueError(
                f"score 长度 {s.shape[0]} != 节点数 {n}, 子类需返回全量分数"
            )
        return s.astype(np.float64)
