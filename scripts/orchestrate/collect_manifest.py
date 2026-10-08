"""结果清单生成器：扫描 outputs/*.json，构建 配置→产物 可审计映射。

产出 `outputs/MANIFEST.json`：
  {
    "count": N,
    "generated_at": "...",
    "entries": [
      {"file": "outputs/hypecp_Amazon_a0.05.json",
       "dataset": "Amazon", "alpha": 0.05, "methods": ["Frozen","HG-TTA","StructCP"],
       "keys": [...], "n_results": ...},
      ...
    ]
  }

支持多类产物命名（hypecp_* / structcp_seeds_* / compare_* / graph_subspace_ad/* 等）。
运行:
  python scripts/orchestrate/collect_manifest.py [--out outputs/MANIFEST.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys as _sys
import time

_p = os.path.dirname(os.path.abspath(__file__))
_ROOT = None
for _ in range(10):
    if os.path.isfile(os.path.join(_p, "configs", "default.yaml")):
        _ROOT = _p
        break
    _p = os.path.dirname(_p)
if _ROOT and _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)


def _safe_load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_meta(fname: str, data) -> dict:
    """从文件名 + 内容推断实验元信息（尽力而为，缺省字段为 None）。"""
    meta: dict = {"file": fname}
    if not isinstance(data, dict):
        return meta

    m = re.match(r"^hypecp_([A-Za-z]+)_a(\d+(?:\.\d+)?)", os.path.basename(fname))
    if m:
        meta["dataset"] = m.group(1)
        meta["alpha"] = float(m.group(2))
    if "results" in data and isinstance(data["results"], dict):
        meta["methods"] = list(data["results"].keys())
    if "config" in data and isinstance(data["config"], dict):
        cfg = data["config"]
        for k in ("dataset", "alpha", "seed", "mode", "relation", "k", "score_quantile"):
            if k in cfg:
                meta[k] = cfg[k]
    if "dataset" in data:
        meta["dataset"] = meta.get("dataset") or data["dataset"]
    if "seed" in data:
        meta["seed"] = meta.get("seed") or data["seed"]
    if "alpha" in data:
        meta["alpha"] = meta.get("alpha") or data["alpha"]
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "MANIFEST.json"))
    ap.add_argument("--patterns", nargs="*",
                    default=["hypecp_*.json", "structcp_seeds_*.json",
                             "compare_*.json", "ablation_*.json",
                             "graph_subspace_ad/*.json"])
    args = ap.parse_args()

    entries = []
    for pat in args.patterns:
        for path in sorted(glob.glob(os.path.join(_ROOT, "outputs", pat))):
            data = _safe_load(path)
            if data is None:
                continue
            rel = os.path.relpath(path, _ROOT)
            entries.append(infer_meta(rel, data))

    manifest = {
        "count": len(entries),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patterns": args.patterns,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
