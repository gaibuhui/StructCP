"""路径工具：深度无关地定位 StructCP 项目根目录。

设计动机（重构后 scripts/ 分层为 train/evaluate/compare/ablation/analysis/
tables/figures/data_prep/orchestrate/archive 等子目录），脚本深度不再固定，
旧式 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 失效。
这里统一以"含 `configs/default.yaml` 的祖先目录"作为项目根（marker 定位），
对任意脚本深度均成立。
"""
from __future__ import annotations

import os

#: 用于判定"这是 StructCP 项目根"的标记文件（任选其一即可，优先级从前到后）
ROOT_MARKERS = (
    os.path.join("configs", "default.yaml"),
    "pyproject.toml",
)


def project_root(start: str | os.PathLike | None = None) -> str:
    """返回 StructCP 项目根目录（绝对路径）。

    Args:
        start: 起始文件/目录路径；为 None 时用调用方文件 (`__file__`) 所在目录。
               若传入的是文件路径，则从其所在目录开始向上查找。

    Returns:
        第一个同时包含 `configs/default.yaml` 的祖先目录；若 10 层内找不到，
        返回起点所在目录（通常不可能发生，marker 恒在根目录）。
    """
    p = os.path.abspath(start) if start is not None else os.getcwd()
    if os.path.isfile(p):
        p = os.path.dirname(p)
    for _ in range(12):
        if any(os.path.isfile(os.path.join(p, m)) for m in ROOT_MARKERS):
            return p
        parent = os.path.dirname(p)
        if parent == p:  # 到达文件系统根
            break
        p = parent
    return p


def resolve_root(anchor: str | os.PathLike) -> str:
    """便捷别名：`project_root`。供需要显式锚点的调用方使用。"""
    return project_root(anchor)
