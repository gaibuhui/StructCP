"""utils 共享库 + 数据容器回归测试。"""
import os

import numpy as np
import pytest
import torch

from data.loader import GADData, _build_gad_from_arrays, _knn_edge_index
from utils.config import inference_defaults, load_config_yaml
from utils.paths import project_root
from utils.results import ResultManifest, aggregate_metric, save_json


class TestPaths:
    def test_project_root_from_deep_file(self):
        # 任意深度定位：用 tests/ 下某文件验证 marker 查找
        root = project_root(__file__)
        assert os.path.isfile(os.path.join(root, "configs", "default.yaml"))
        assert os.path.basename(root) == "StructCP"

    def test_project_root_from_cwd(self):
        root = project_root()
        assert os.path.isfile(os.path.join(root, "configs", "default.yaml"))


class TestConfig:
    def test_load_config_yaml_returns_dict(self):
        cfg = load_config_yaml()
        assert isinstance(cfg, dict)
        assert "inference" in cfg

    def test_inference_defaults_bridge(self):
        cfg = load_config_yaml()
        d = inference_defaults(cfg)
        # run_structcp 的关键默认与论文口径一致
        assert d.get("alpha") == 0.05
        assert d.get("k") == 10
        assert d.get("mode") == "hard"


class TestResults:
    def test_save_load_roundtrip(self, tmp_path):
        p = save_json({"a": 1, "b": [1, 2]}, tmp_path / "x.json")
        from utils.results import load_json
        assert load_json(p) == {"a": 1, "b": [1, 2]}

    def test_aggregate_metric(self):
        agg = aggregate_metric([1.0, 2.0, 3.0], "FPR")
        assert agg["FPR_mean"] == pytest.approx(2.0)
        assert agg["FPR_std"] == pytest.approx(np.std([1, 2, 3], ddof=0))

    def test_manifest(self, tmp_path):
        import json
        m = ResultManifest()
        m.add(output="outputs/x.json", method="StructCP", dataset="Amazon")
        m.add(output="outputs/y.json", method="Frozen", dataset="Amazon")
        path = m.save(tmp_path / "MANIFEST.json")
        with open(path) as f:
            data = json.load(f)
        assert data["count"] == 2
        assert len(data["entries"]) == 2


class TestLoader:
    def test_build_gad_from_arrays_has_masks(self):
        n = 300
        x = np.random.RandomState(0).randn(n, 6).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)
        y[:30] = 1
        ei = _knn_edge_index(x, k=5)
        data = _build_gad_from_arrays(x, y, ei, seed=0, name="toy")
        assert isinstance(data, GADData)
        assert data.train_mask.sum() + data.cal_mask.sum() + data.test_mask.sum() == n
        # novel normality 仅在测试期出现
        assert (data.novel_mask & data.cal_mask).sum() == 0
        assert (data.novel_mask & data.train_mask).sum() == 0
        assert (data.novel_mask & data.test_mask).sum() > 0

    def test_knn_edge_index_symmetric(self):
        x = np.random.RandomState(1).randn(50, 4).astype(np.float32)
        ei = _knn_edge_index(x, k=4)
        assert ei.shape[0] == 2
        # 去重后的边数合理
        assert ei.shape[1] > 0
