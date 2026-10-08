"""共享管线（adapters/pipeline.py）回归测试。

验证 Frozen / HG-TTA / StructCP 三段式推理在合成小图上可端到端运行、
且"同一输入 + 同一冻结骨干"下结果确定（可复现性）。
"""
import numpy as np
import pytest
import torch

from adapters.hypergraph_tta import KNNHypergraph
from adapters.pipeline import RunConfig, evaluate_structcp, get_scores, run_comparison
from data.loader import GADData
from models.bwgnn import BWGNN


def _make_data(n=200, d=8, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.randn(n, d)
    ei = torch.randint(0, n, (2, 1500))
    y = torch.zeros(n)
    y[torch.arange(0, n, 20)] = 1.0
    train = torch.zeros(n, dtype=torch.bool)
    train[:120] = True
    cal = torch.zeros(n, dtype=torch.bool)
    cal[120:160] = True
    test = torch.zeros(n, dtype=torch.bool)
    test[160:] = True
    novel = torch.zeros(n, dtype=torch.bool)
    novel[170:180] = True  # 部分 novel-normal
    return GADData(x, ei, y, train, cal, test, novel, "synth")


def _make_model(d=8):
    model = BWGNN(d, 16, 2, d=2).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class TestPipeline:
    def test_run_comparison_three_methods(self):
        data = _make_data()
        model = _make_model()
        hg = KNNHypergraph(data.x, k=8)
        cfg = RunConfig(alpha=0.05, score_quantile=0.75)
        results, extra = run_comparison(data, model, hg, cfg)
        for name in ("Frozen", "HG-TTA", "StructCP"):
            assert name in results
            for k in ("FPR", "TPR", "F1", "AUROC", "infer_time_s"):
                assert k in results[name], f"{name}.{k} missing"
        assert extra["method_name"] == "StructCP"

    def test_run_comparison_deterministic(self):
        data = _make_data()
        model = _make_model()
        hg = KNNHypergraph(data.x, k=8)
        cfg = RunConfig(alpha=0.05, score_quantile=0.75)
        r1, _ = run_comparison(data, model, hg, cfg)
        # 重新构造（同一 seed）应完全一致
        data2 = _make_data()
        hg2 = KNNHypergraph(data2.x, k=8)
        r2, _ = run_comparison(data2, model, hg2, cfg)
        for name in ("Frozen", "HG-TTA", "StructCP"):
            for k in ("FPR", "TPR", "F1", "AUROC"):
                assert r1[name][k] == pytest.approx(r2[name][k]), f"{name}.{k}"

    def test_get_scores_shape_and_range(self):
        data = _make_data()
        model = _make_model()
        s = get_scores(model, data.x, data.edge_index)
        assert s.shape == (200,)
        assert (s >= 0).all() and (s <= 1).all()

    def test_default_dataset_cfg_has_alpha_aligned_keys(self):
        from adapters.pipeline import DEFAULT_DATASET_CFG
        for ds, cfg in DEFAULT_DATASET_CFG.items():
            assert cfg["mode"] in ("hard", "soft")
            assert cfg["k"] >= 1


class TestEvaluateStructcp:
    def test_evaluate_missing_checkpoint_raises(self, tmp_path):
        # 不存在的数据集/checkpoint 应报清晰错误（不静默）
        from utils.paths import project_root
        with pytest.raises(FileNotFoundError):
            evaluate_structcp("NoSuchDataset", seed=42, root=project_root(),
                              device="cpu", save_output=False)
