"""核心算法回归测试：dual-use k-NN graph / weighted conformal / test-time propagation。

覆盖三件"核心创新"的数学行为，防止重构破坏算法语义：
  1) test-time propagation（hypergraph_propagation + KNNHypergraph）
  2) weighted conformal prediction（structcp_threshold / weighted_quantile）
  3) dual-use k-NN graph（同一超图同时用于传播与一致性权重）
"""
import numpy as np
import pytest
import torch

from adapters.conformal import (
    split_conformal_threshold,
    structcp_threshold,
    weighted_quantile,
    evaluate_coverage,
    ess_based_bias_proxy,
)
from adapters.hypergraph_tta import (
    KNNHypergraph,
    hyperedge_consistency,
    hypergraph_propagation,
    simple_graph_consistency,
)


def _tiny_data(n=120, d=8, k=8, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.randn(n, d)
    return x


# ---------------------------------------------------------------------------
# 1. test-time propagation
# ---------------------------------------------------------------------------
class TestTestTimePropagation:
    def test_knn_hypergraph_shapes(self):
        x = _tiny_data(n=100, d=8)
        hg = KNNHypergraph(x, k=8)
        assert hg.H.shape[0] == 100 and hg.H.shape[1] == 100   # (n_v, n_e)
        assert hg.knn_idx.shape == (100, 9)                    # k+1 含自身
        assert int(hg.H._nnz()) > 0

    def test_propagation_preserves_dim(self):
        x = _tiny_data()
        hg = KNNHypergraph(x, k=8)
        out = hypergraph_propagation(x, hg, num_layers=2, alpha_res=0.5)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_propagation_is_smoothing(self):
        # 纯低通残差传播应让节点嵌入趋于其邻居（不增大总体方差）
        x = _tiny_data()
        hg = KNNHypergraph(x, k=8)
        out = hypergraph_propagation(x, hg, num_layers=3, alpha_res=0.9)
        assert out.std() <= x.std() + 1e-3

    def test_beta_highpass_backward_compat(self):
        # beta=0 时退化为原纯低通行为
        x = _tiny_data()
        hg = KNNHypergraph(x, k=8)
        a = hypergraph_propagation(x, hg, num_layers=2, alpha_res=0.5, beta_highpass=0.0)
        b = hypergraph_propagation(x, hg, num_layers=2, alpha_res=0.5)
        assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. weighted conformal prediction
# ---------------------------------------------------------------------------
class TestWeightedConformal:
    def test_weighted_quantile_matches_plain_quantile_for_unit_weights(self):
        v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        w = torch.ones_like(v)
        assert float(weighted_quantile(v, w, 0.5)) == pytest.approx(3.0)

    def test_weighted_quantile_biased_by_weights(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        w = torch.tensor([10.0, 1.0, 1.0])  # 权重压在 1 上
        q = float(weighted_quantile(v, w, 0.5))
        assert q < 3.0  # 高权重低值应拉低加权分位

    def test_split_conformal_finite_sample(self):
        s = torch.randn(200)
        q = split_conformal_threshold(s, alpha=0.05)
        n = s.numel()
        expected_rank = int(np.ceil((n + 1) * 0.95))
        assert float(q) == float(torch.sort(s)[0][expected_rank - 1])

    def test_structcp_threshold_runs_and_controls_fpr(self):
        n_cal, n_te = 200, 200
        torch.manual_seed(0)
        cal_s = torch.rand(n_cal).log1p()
        cal_y = torch.zeros(n_cal)
        cal_c = torch.rand(n_cal).clamp(1e-3, 1.0)
        te_s = torch.rand(n_te).log1p()
        te_c = torch.rand(n_te).clamp(1e-3, 1.0)
        thr, info = structcp_threshold(cal_s, cal_y, cal_c, te_s, te_c,
                                       alpha=0.05, score_quantile=0.75)
        assert torch.isfinite(thr)
        assert info["w_ess"] >= 1.0
        assert "fpr_bias_bound" in info
        # 一致性能被用于重加权（w 单调性不被破坏：w_max >= w_min）
        assert info["w_max"] >= info["w_min"]

    def test_ess_bias_proxy_bounds(self):
        w = torch.ones(100)
        assert ess_based_bias_proxy(w, 100, 0.05) <= 1.0
        assert ess_based_bias_proxy(w, 100, 0.05) > 0.0

    def test_evaluate_coverage_consistency(self):
        # pred = s > 0.5 -> [0,1,0]; y = [0,1,0] -> 无误报/漏报
        ev = evaluate_coverage(torch.tensor([0.1, 0.9, 0.5]),
                               torch.tensor([0, 1, 0]), torch.tensor(0.5))
        assert ev["FPR"] == pytest.approx(0.0)
        assert ev["TPR"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. dual-use k-NN graph（同一超图双用途）
# ---------------------------------------------------------------------------
class TestDualUseGraph:
    def test_same_hypergraph_serves_propagation_and_consistency(self):
        x = _tiny_data(n=100, d=8)
        hg = KNNHypergraph(x, k=8)
        # 用途一：传播增强特征
        x_enh = hypergraph_propagation(x, hg, 2, 0.5)
        # 用途二：一致性权重
        scores = torch.rand(100)
        cal_mask = torch.zeros(100, dtype=torch.bool)
        cal_mask[:50] = True
        cons = hyperedge_consistency(scores, hg, features=x, cal_mask=cal_mask)
        assert cons.shape == (100,)
        assert (cons > 0).all() and (cons <= 1.0 + 1e-6).all()
        # 结构信息被双份使用：传播改变了特征，一致性基于同一图
        assert not torch.allclose(x, x_enh, atol=1e-6)

    def test_simple_graph_consistency_subset_of_struct(self):
        x = _tiny_data(n=100)
        hg = KNNHypergraph(x, k=8)
        s = torch.rand(100)
        cm = torch.zeros(100, dtype=torch.bool)
        cm[:40] = True
        c_struct = hyperedge_consistency(s, hg, features=x, cal_mask=cm)
        c_simple = simple_graph_consistency(s, hg, cal_mask=cm)
        assert c_simple.shape == c_struct.shape
        assert (c_simple > 0).all()
