"""配置加载：`configs/default.yaml` 的单一事实源（SSOT）读取。

语义与旧 `scripts/run_structcp.py::load_config()` 保持一致并泛化：
  - yaml 不存在 / 无 PyYAML 时静默返回空 dict（不阻断老用法）；
  - 返回 dict 供 argparse default 填充，CLI 显式传参优先级最高。
"""
from __future__ import annotations

import os
from typing import Any

from .paths import project_root

DEFAULT_CONFIG_NAME = "configs/default.yaml"


def _safe_load(path: str) -> dict:
    try:
        import yaml  # 软依赖
    except Exception:
        return {}
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def load_config_yaml(root: str | os.PathLike | None = None,
                     name: str = DEFAULT_CONFIG_NAME) -> dict:
    """读取项目根下 `configs/default.yaml`（或指定 name）为 dict。

    Args:
        root: 项目根；None 时自动定位。
        name: 相对根目录的配置文件路径。

    Returns:
        dict；文件缺失/解析失败时返回 {}（不抛异常）。
    """
    root = str(root) if root is not None else project_root()
    return _safe_load(os.path.join(root, name))


def inference_defaults(cfg: dict | None = None) -> dict:
    """从 yaml 的 `inference:` 段抽取 run_structcp 风格默认值（SSOT 桥接）。

    返回可直接作为 argparse 默认值的 dict；键与 `run_structcp.py` 的
    `--k/--layers/--alpha_res/--alpha/--tau/--mode/--score_quantile/
     --subsample/--knn_backbone/--relation` 一一对应。
    """
    cfg = cfg or {}
    inf = cfg.get("inference", {}) or {}
    out = {}
    for k in ("relation", "k", "layers", "alpha_res", "alpha", "tau", "mode",
              "score_quantile", "subsample", "knn_backbone"):
        if k in inf and inf[k] is not None:
            out[k] = inf[k]
    mdl = cfg.get("model", {}) or {}
    if mdl.get("device"):
        out["device"] = mdl["device"]
    return out


def merge_defaults(defaults: dict[str, Any], **override: Any) -> dict[str, Any]:
    """把 override 中非 None 的值覆盖到 defaults 上（None 表示"未显式给定"）。"""
    merged = dict(defaults)
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged
