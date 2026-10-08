"""创新点一: 超图驱动的免训练测试时自适应 (Training-free Hypergraph TTA)。

对标 TUNE (AAAI 2026) 的可学习 MLP 图对齐器 X' = X + MLP(X)，
StructCP 用零参数的超图拉普拉斯传播替代:

    X^{(t+1)} = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(t)}

零参数、零反传，天然适配资源受限部署。

同一个超图结构同时服务于:
  1) 方法层: 特征增强 (本文件)
  2) 理论层: 超边一致性权重 (adapters/conformal.py)
—— 即"同一结构、双重功能"的内生耦合。
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class KNNHypergraph:
    """k-NN 超图。每个节点及其 k 近邻构成一条超边。

    内部以稀疏关联矩阵 H (n_v x n_e) 表示，避免依赖 dhg 的稠密实现，
    以适配 8GB 显存约束 (R7)。

    approx=True 时使用 hnswlib 近似最近邻，缓解大规模图 (T-Social 等)
    的可扩展性审稿质疑；默认精确 k-NN (sklearn) 以保持主实验可复现。
    """

    def __init__(self, features: torch.Tensor, k: int = 10, metric: str = "cosine",
                 approx: bool = False, hnsw_m: int = 16, hnsw_ef: int = 128):
        self.k = k
        x = features.detach().cpu().numpy().astype(np.float32)
        n = x.shape[0]
        self.num_v = n
        self.num_e = n

        if approx:
            import hnswlib
            space = "l2" if metric == "euclidean" else "cosine"
            index = hnswlib.Index(space=space, dim=x.shape[1])
            index.init_index(max_elements=max(n, 1), ef_construction=hnsw_ef, M=hnsw_m)
            index.add_items(x)
            index.set_ef(hnsw_ef)
            knn_k = min(k + 1, n)
            labels, _ = index.knn_query(x, k=knn_k)
            indices = labels.astype(np.int64)
        else:
            nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric, n_jobs=-1)
            nbrs.fit(x)
            _, indices = nbrs.kneighbors(x)
        self.knn_idx = torch.from_numpy(indices.astype(np.int64))  # (n, k+1) 含自身

        # 超边 e_i = {i} ∪ N_k(i)
        rows = torch.from_numpy(indices.reshape(-1).astype(np.int64))
        cols = torch.arange(n).repeat_interleave(indices.shape[1])
        self._inc_idx = torch.stack([rows, cols])  # H 的非零索引
        self._nnz = self._inc_idx.shape[1]
        self._device = torch.device("cpu")
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        n_v, n_e = self.num_v, self.num_e
        vals = torch.ones(self._nnz)
        H = torch.sparse_coo_tensor(self._inc_idx, vals, (n_v, n_e)).coalesce()
        self.H = H
        # 节点度 D_v = sum_e H[v,e]; 超边度 D_e = sum_v H[v,e]
        dv = torch.sparse.sum(H, dim=1).to_dense().clamp(min=1)
        de = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1)
        self.Dv_inv_sqrt = dv.pow(-0.5)
        self.De_inv = de.pow(-1.0)

    def to(self, device):
        self.H = self.H.to(device)
        self.Dv_inv_sqrt = self.Dv_inv_sqrt.to(device)
        self.De_inv = self.De_inv.to(device)
        self.knn_idx = self.knn_idx.to(device)
        self._device = device
        return self

    # ------------------------------------------------------------------
    def smooth(self, x: torch.Tensor, deg_gate: bool = False) -> torch.Tensor:
        """一次 HGNN 传播: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} x。

        deg_gate=True 时 W 不再是单位阵, 而是超边度自适应的无参数门控
        (B 方案, 轻量 RFHND 思想): 高度超边 (密集社区) 降权、稀疏超边保权,
        缓解社区型异常被同配传播过度同质化。门控 = sigmoid(log(de) - log(de_med)),
        高度超边 -> <1 (降权), 稀疏超边 -> >1 (保权); 乘到 De_inv 上, 不影响
        稀疏关联矩阵 H 与一致性权重 (双重功能保持)。
        """
        de = self.De_inv
        if deg_gate:
            with torch.no_grad():
                de_med = self.De_inv.median()
                gate = torch.sigmoid(torch.log(self.De_inv + 1e-8) - torch.log(de_med + 1e-8))
            de = self.De_inv * gate
        h = self.Dv_inv_sqrt.unsqueeze(1) * x            # D_v^{-1/2} X
        h = torch.sparse.mm(self.H.t(), h)               # H^T ...
        h = de.unsqueeze(1) * h                           # D_e^{-1} (·gate) ...
        h = torch.sparse.mm(self.H, h)                   # H ...
        return self.Dv_inv_sqrt.unsqueeze(1) * h         # D_v^{-1/2} ...


def hypergraph_propagation(
    features: torch.Tensor,
    hypergraph: KNNHypergraph,
    num_layers: int = 2,
    alpha_res: float = 0.5,
    beta_highpass: float = 0.0,
    deg_gate: bool = False,
) -> torch.Tensor:
    """免训练超图传播。

    Args:
        alpha_res: 低通残差系数。X <- (1-a-b) * X + a * lowpass(X) + b * highpass(X)。
                   防止过平滑抹掉异常信号 (纯传播会让异常与正常同质化)。
        beta_highpass: 高通系数 (A 方案, 轻量 HyperNATE 思路)。highpass = X - lowpass(X)
                       保留异常与正常的局部差异, 直接对症异配/离群场景。
                       beta=0 时退化为原纯低通残差行为 (向后兼容)。
        deg_gate: 超边度自适应门控 (B 方案), 见 smooth()。

    注意 (2026-09-02 清理): 此处曾有一个 "ESS 触发高通门" 的实现, 已被移除。原因是
    它在架构上不可能生效且语义错误:
      (1) 循环依赖 —— 伪正常池的 ESS 由 *传播之后* 的打分/一致性/加权保形给出,
          传播函数内部无法观测到自己输出的下游量;
      (2) 空转 —— `if pseudo_normal_count > 0:` 守卫下, 该参数无调用方传入 (默认 0),
          且 if/else 两支均赋 `effective_beta = beta_highpass`, 触发与否无差别;
      (3) 语义错 —— 判据用的是样本数比 `pseudo_normal_count / num_v`, 而非
          ESS = (Σw)²/Σw², 与论文所述 "ESS" 不是同一个量。
    ESS 门控现已上移到 `adapters/pipeline.py` 的 run_comparison, 采用两遍式
    (先以 beta=0 传播并测量伪正常池 ESS, 再决定是否以 beta>0 重传播)。
    """
    effective_beta = beta_highpass

    enhanced = features
    for _ in range(num_layers):
        lp = hypergraph.smooth(enhanced, deg_gate=deg_gate)
        hp = enhanced - lp                      # 高通分量 = 原始 - 低通
        enhanced = (1.0 - alpha_res - effective_beta) * enhanced \
                   + alpha_res * lp \
                   + effective_beta * hp
    return enhanced


@torch.no_grad()
def hyperedge_consistency(
    scores: torch.Tensor,
    hypergraph: KNNHypergraph,
    features: torch.Tensor | None = None,
    consistency_features: torch.Tensor | None = None,
    cal_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """超边一致性 c_i ∈ (0, 1]。数值越大 = 越像"群体性正常模式"。

    关键区分 (新正常类别 vs 真异常)：
      * 新正常类别 (novel normality): 检测器给高分, 但它们在超图上**成簇聚集**,
        超边内分数彼此接近、特征紧致 → 高一致性。
      * 真异常 (anomaly): 分数高且**局部离群**, 与邻域分数差异大、特征分散
        → 低一致性。

    因此 c 由两部分组成:
      c_score  = exp(-|s_i - median(s_Ni)| / scale_s)   分数与邻域中位数的偏离
      c_feat   = exp(-d_i / scale_f)                    到超边质心的特征距离
      c = sqrt(c_score * c_feat)

    cal_mask (可选): 与 `scores` 同长的 bool 张量, 标记校准集节点。
    尺度项 scale_s=mean(dev), scale_f=mean(dist) 只在这些节点上取均值, 从而
    **不依赖测试节点统计量** (审稿质疑#2: 在线推理无法预知整个测试批)。若为 None
    则退回全输入均值 (保持向后兼容, 但主流程必须传 cal_mask)。
    """
    s = scores.detach().flatten()
    idx = hypergraph.knn_idx                       # (n, k+1)
    nb = s[idx]

    # --- 分数偏离度: 节点自身分数 vs 超边内中位数 ---
    med = nb.median(dim=1).values
    dev = (s - med).abs()
    scale_s = dev[cal_mask].mean() if cal_mask is not None else dev.mean()
    c_score = torch.exp(-dev / (scale_s + eps))

    # --- 特征紧致度: 到超边质心的距离 ---
    # consistency_features 默认与传播特征 features 解耦: 用 raw 特征算一致性
    # (论文 §4.3 消融证明 raw 特征比 propagated 更能保留局部不连续性, 见 tab:consistency_feat)。
    cf = consistency_features if consistency_features is not None else features
    if cf is not None:
        f = cf.detach()
        f = f / (f.norm(dim=1, keepdim=True) + eps)
        centroid = f[idx].mean(dim=1)
        centroid = centroid / (centroid.norm(dim=1, keepdim=True) + eps)
        dist = 1.0 - (f * centroid).sum(dim=1)     # cosine 距离
        scale_f = dist[cal_mask].mean() if cal_mask is not None else dist.mean()
        c_feat = torch.exp(-dist / (scale_f + eps))
    else:
        c_feat = torch.ones_like(c_score)

    c = torch.sqrt(c_score * c_feat)
    return c.clamp(min=eps, max=1.0)


@torch.no_grad()
def contrastive_consistency(
    features: torch.Tensor,
    hypergraph: KNNHypergraph,
    cal_normal_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """对比一致性 c_i^struct ∈ (0, 1)（替换原公式 2 的"绝对一致性"）。

    原公式 2 只看"节点-邻居紧致度"（绝对一致性），导致高度同质的**欺诈环**
    （Elliptic 等社区型异常）获得与常态密集聚类同级的权重，被误纳入伪正常池、
    推高阈值，最终 TPR 从 ~0.94 暴跌至 ~0.41（社区异常致命缺陷）。

    对比一致性引入第二个参照系——校准集常态原型 μC：
        ci_local  = cos(x_i, x̄_N(i))    局部凝聚度：与邻居均值特征的余弦相似度
        ci_global = cos(x_i, μC)        全局背景度：与常态原型的余弦相似度
        c_i^struct = σ(ci_local) · σ(ci_global)   （乘积门控）

    物理含义：
      * 常态密集聚类：ci_local 高（与邻居像）**且** ci_global 高（与常态原型像）
        → 两因子都大 → 高权重（予以提权）。
      * 欺诈环：ci_local 高（环内互相像）**但** ci_global 低（偏离常态原型）
        → 乘积门控中 σ(ci_global) 被压低 → ci_struct 自动压低，
        **解除对社区异常的误提权**，从根上修复社区异常 TPR 崩溃。

    μC 仅由校准集正常节点特征计算（cal_normal_mask 标记），不依赖测试批统计量，
    保持流式/归纳部署能力（审稿质疑#2 同口径）。

    Args:
        features: 用于一致性判断的特征（论文 §4.3 建议用 raw 特征以保留局部不连续性）。
        hypergraph: KNNHypergraph（提供 k-NN 邻居索引）。
        cal_normal_mask: 与 features 同长的 bool 张量，标记**校准集正常**节点，
            用于估计常态原型 μC = 归一化后的校准正常特征均值。
        eps: 数值稳定性常数。
    """
    f = features.detach()
    f = f / (f.norm(dim=1, keepdim=True) + eps)          # (n, d) L2 归一化

    # --- 局部凝聚度: 与邻居均值特征的余弦相似度 ---
    idx = hypergraph.knn_idx                             # (n, k+1) 含自身
    nb = f[idx].mean(dim=1)                              # 邻居均值 x̄_N(i)
    nb = nb / (nb.norm(dim=1, keepdim=True) + eps)
    ci_local = (f * nb).sum(dim=1)                       # cos(x_i, x̄_N(i)) ∈ [-1, 1]

    # --- 全局背景度: 与校准常态原型 μC 的余弦相似度 ---
    mu_c = f[cal_normal_mask].mean(dim=0)                # 常态原型 μC
    mu_c = mu_c / (mu_c.norm() + eps)
    ci_global = f @ mu_c                                 # cos(x_i, μC) ∈ [-1, 1]

    # --- 乘积门控: 全局疏离 (欺诈环) 时 σ(ci_global) 压低整体 ---
    c_struct = torch.sigmoid(ci_local) * torch.sigmoid(ci_global)
    return c_struct.clamp(min=eps, max=1.0)


@torch.no_grad()
def simple_graph_consistency(
    scores: torch.Tensor,
    hypergraph: KNNHypergraph,
    cal_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Simple-Graph-CP 基线用的极简"图结构一致性"权重。

    仅保留 StructCP 一致性权重的 *分数邻域一致性* 单分量，即
        c_i = exp(- |s_i - median(s_{N(i)})| / scale_s)
    它完全依赖 k-NN 图结构，但 **不做** 以下 StructCP 的复杂工程：
      * 特征紧致度 c_feat (到超边质心的 cosine 距离 log-linear 项)
      * Eq.8 的 test-batch median/std 标准化、clip[0.05,20]、归一化
    因此它是 StructCP 的 "同架构、更简单" 对照：共用加权共形 + 伪正常扩增
    机制，但用最朴素的图结构信号作为权重来源。用以隔离 "结构权重本身"
    与 "一致性 log-linear 复杂变换" 各自的贡献 (对应论文 Tab.weight_ablation
    的 Simple-Graph-CP 行)。

    cal_mask (可选): 尺度项 scale_s=mean(dev) 仅在校准集节点上取均值,
    去除对测试批统计量的依赖 (审稿质疑#2)。None 则退回全输入均值。
    """
    s = scores.detach().flatten()
    idx = hypergraph.knn_idx                       # (n, k+1)
    nb = s[idx]
    med = nb.median(dim=1).values
    dev = (s - med).abs()
    scale_s = dev[cal_mask].mean() if cal_mask is not None else dev.mean()
    c_score = torch.exp(-dev / (scale_s + eps))
    return c_score.clamp(min=eps, max=1.0)
