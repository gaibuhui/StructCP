"""统一随机种子设置（可复现性基础设施）。

各脚本此前自行 `torch.manual_seed`，口径不一。统一入口保证
`python -m seed=42` 与论文 3-seed / 5-seed 口径一致。
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置 python / numpy / torch / cuda 全部随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
