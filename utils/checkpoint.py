"""冻结检测器加载（TTA 设定：检测器全程不更新）。

语义与旧 `scripts/run_structcp.py` 的加载块一致：
  - checkpoint 路径 = `{root}/checkpoint/bwgnn_{dataset}_{relation}.pth`
  - 从 ckpt["args"] 读 hidden / order（Beta 小波阶数）
  - 加载后 `model.freeze()`（requires_grad=False + eval）
权重文件名兼容候补（_h128 后缀等）由 registry 自行处理，此处保持主路径行为。
"""
from __future__ import annotations

import os

import torch

from models.bwgnn import BWGNN


def load_frozen_detector(
    dataset: str,
    relation: str = "homo",
    root: str | os.PathLike | None = None,
    device: str | None = None,
) -> tuple[BWGNN, str]:
    """加载并冻结 BWGNN 检测器。

    Returns:
        (model, ckpt_path): model 已 `freeze()`。
    """
    if root is None:
        from .paths import project_root
        root = project_root()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(str(root), "checkpoint", f"bwgnn_{dataset}_{relation}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"checkpoint 不存在: {ckpt_path}；请先运行 scripts/train/pretrain_bwgnn.py "
            f"--dataset {dataset}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    ca = ckpt["args"]
    model = BWGNN(int(ckpt["in_channels"]), int(ca["hidden"]), 2, d=int(ca["order"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    return model, ckpt_path
