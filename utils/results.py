"""结果读写 / 聚合 / 实验清单（manifest）。

统一 JSON 落盘与多种子聚合口径，替代各脚本自行 `open().write()` 的重复样板；
`ResultManifest` 提供"配置 → 产物文件"的可审计映射，供 `collect_manifest.py` 使用。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable

import numpy as np


def save_json(obj: Any, path: str | os.PathLike, ensure_ascii: bool = False,
              indent: int = 2) -> str:
    """把对象写为 JSON，自动建目录；返回绝对路径。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    return path


def load_json(path: str | os.PathLike) -> Any:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metric(values: Iterable[float], name: str = "") -> dict[str, float]:
    """多种子某指标 → mean±std/min/max（std 用总体标准差，与旧口径一致）。"""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"{name}_mean": float("nan"), f"{name}_std": float("nan"),
                f"{name}_min": float("nan"), f"{name}_max": float("nan")}
    return {
        f"{name}_mean": float(arr.mean()),
        f"{name}_std": float(arr.std(ddof=0)),
        f"{name}_min": float(arr.min()),
        f"{name}_max": float(arr.max()),
    }


class ResultManifest:
    """实验清单：一行 = 一次实验产物的元信息。

    用法:
        m = ResultManifest()
        m.add(output="outputs/hypecp_Amazon_a0.05.json", method="StructCP",
              dataset="Amazon", seed=42, alpha=0.05, **info)
        m.save("outputs/MANIFEST.json")
    """

    def __init__(self, entries: list[dict] | None = None):
        self.entries = list(entries) if entries else []

    def add(self, **kwargs: Any) -> None:
        self.entries.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs})

    def filter(self, **query: Any) -> list[dict]:
        def _match(e):
            return all(e.get(k) == v for k, v in query.items())
        return [e for e in self.entries if _match(e)]

    def save(self, path: str | os.PathLike) -> str:
        return save_json({"count": len(self.entries), "entries": self.entries}, path)
