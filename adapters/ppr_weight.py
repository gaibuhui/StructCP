"""GRAPHLCP 式 PPR 结构权重 —— 思路2落地（针对社区型异常 TPR 坍缩）。

【问题】社区型异常（Elliptic 欺诈环 / TFinance 洗钱环）**内部高同配**，
`hyperedge_consistency` 只刻画局部 k-NN 超边内的同质性，因此给它们很高的
一致性 c：局部与周围一致 → 高 c → 被误当"伪正常"注入校准池推高阈值 →
TPR 坍缩（见附录 B 的 homophily 分析：Elliptic 异常侧同配性 0.814，最高）。

【思路2】不再依赖可能失效的简单同质性假设，改用可捕捉**长程/全局结构角色**
的个性化 PageRank（PPR）来计算权重：c_ppr(v) = 从"校准正常骨架"出发的
PPR 稳态概率。社区型异常即使局部同配，其所属结构环与正常骨架的结构连接弱，
c_ppr 显著低于正常 → 不再被误当伪正常 → TPR 修复。

【三步（对齐 GRAPHLCP）】
  1. 特征感知图密化：PCA 降维 → 各向异性高斯核（带宽由 PCA 特征值导出）
     → 高置信度 top-k 近邻边 → 密化图（增强稀疏图的长程可达性）；
  2. PPR 核：在密化图上幂迭代计算"从校准正常节点均匀 seed 出发"的 PPR；
  3. 集成到 CP：把 c_ppr 作为单节点结构权重，与一致性权重做几何调制
     c = sqrt(c_consistency * c_ppr)，再注入现有 `structcp_threshold`
     （伪正常筛选 c>=τ + 加权分位），其余流程不变。

【为什么用"调制"而非"替换"】novel-normality（测试期新正常类别）在一致性上
成簇聚集（高 c，StructCP 靠它扩增校准池修正 FPR），但不在"校准正常 seed"里、
PPR 得分偏低。纯替换会让 novel 正常失去扩增能力 → 伤 FPR；几何调制保留一致性
的 FPR 修正能力，同时用 PPR 压低社区型异常 —— 两者互补。

【可扩展性】PPR 幂迭代在 CPU（scipy.sparse）完成，不占 8GB 显存；PCA 与
top-k 近邻密化用 numpy/sklearn，TFinance(39k 节点) 规模秒级完成。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def pca_project(x, pca_dim: int = 16):
    """PCA 降维（numpy SVD）。返回 (z, eigvals)。

    - z: (n, pca_dim) 主成分得分（去中心后投影）
    - eigvals: 各主成分方差，用于导出各向异性高斯核带宽
    """
    X = _to_numpy(x).astype(np.float64)
    n, d = X.shape
    pca_dim = max(1, min(int(pca_dim), d, n))
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    z = U[:, :pca_dim] * S[:pca_dim]                      # (n, pca_dim)
    eig = S[:pca_dim] ** 2 / max(n - 1, 1)                # 主成分方差
    return z, np.clip(eig, 1e-12, None)


def _unique_undirected_pairs(src, dst, n_nodes):
    """无向去重：构造 (min, max) 唯一键，双向边只保留一次。返回去重后的 (src, dst)。"""
    mn = np.minimum(src, dst)
    mx = np.maximum(src, dst)
    key = mn.astype(np.int64) * (n_nodes + 1) + mx.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq = np.ones(len(ks), dtype=bool)
    uniq[1:] = ks[1:] != ks[:-1]
    # 返回 (min, max) 排序的无向边，保证与 lookup 键 (min,max) 一致
    return mn[order][uniq], mx[order][uniq]


def anisotropic_density_knn(x, pca_dim: int = 16, k: int = 15, h: float = 1.0,
                            max_edges=None):
    """GRAPHLCP 步骤1（密化）：PCA + 各向异性高斯核的 top-k 近邻加边。

    在 PCA 空间把每维除以 h*sqrt(λ_c)（带宽由 PCA 特征值导出），使欧氏距离
    等价于各向异性高斯核的对数形式；相似度 sim = exp(-0.5*d²)。返回去重后的
    无向边 (src, dst) 及其相似度。
    """
    z, eig = pca_project(x, pca_dim)
    scale = np.sqrt(eig)
    scale = np.clip(scale, 1e-6, None)
    zs = z / (h * scale)                                  # 各向异性加权坐标
    n = zs.shape[0]
    n_nbr = min(int(k) + 1, n)
    nn = NearestNeighbors(n_neighbors=n_nbr, metric="euclidean", n_jobs=-1)
    nn.fit(zs)
    dist, idx = nn.kneighbors(zs)
    dist, idx = dist[:, 1:], idx[:, 1:]                   # 去掉自身
    if idx.shape[1] == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))
    sim = np.exp(-0.5 * dist ** 2)                        # 各向异性高斯核
    src = np.repeat(np.arange(n), idx.shape[1])
    dst = idx.ravel()
    m = src != dst
    src, dst, s = src[m], dst[m], sim.ravel()[m]
    # 双向化（先存旧值，避免赋值顺序污染）
    _src, _dst, _s = src, dst, s
    src = np.concatenate([_src, _dst])
    dst = np.concatenate([_dst, _src])
    s = np.concatenate([_s, _s])
    # 无向去重（保留首次 sim，对称）
    usrc, udst = _unique_undirected_pairs(src, dst, n)
    # 去重后重建 sim（取该无向边的最大相似度）
    key = np.minimum(usrc, udst).astype(np.int64) * (n + 1) \
        + np.maximum(usrc, udst).astype(np.int64)
    lookup = {}
    for a, b, sv in zip(src.tolist(), dst.tolist(), s.tolist()):
        kk = (min(a, b), max(a, b))
        if kk not in lookup or sv > lookup[kk]:
            lookup[kk] = sv
    s_out = np.array([lookup[(int(a), int(b))] for a, b in zip(usrc.tolist(), udst.tolist())],
                     dtype=np.float64)
    if max_edges is not None and len(usrc) > max_edges:
        keep = np.argsort(-s_out)[:max_edges]
        usrc, udst, s_out = usrc[keep], udst[keep], s_out[keep]
    return usrc, udst, s_out


def densify_graph(x, edge_index, pca_dim: int = 16, density_k: int = 15,
                  h: float = 1.0, max_edges=None):
    """返回密化 edge_index [2, E']（原图 + 高置信度近邻边，无向去重）。"""
    ei = _to_numpy(edge_index)
    if ei.shape[0] != 2:
        ei = ei.T
    n = x.shape[0] if not isinstance(x, torch.Tensor) else x.shape[0]
    usrc, udst, _ = anisotropic_density_knn(x, pca_dim, density_k, h, max_edges)
    all_src = np.concatenate([ei[0].astype(np.int64), usrc])
    all_dst = np.concatenate([ei[1].astype(np.int64), udst])
    src_u, dst_u = _unique_undirected_pairs(all_src, all_dst, n)
    out = np.vstack([src_u, dst_u])
    return torch.from_numpy(out).long()


# ---------------------------------------------------------------------------
# PPR 核（步骤2）
# ---------------------------------------------------------------------------
def ppr_from_seed(edge_index, n, seed, restart: float = 0.15, K: int = 60,
                  tol: float = 1e-6):
    """幂迭代 PPR：π ← (1-α)·Pᵀ·π + α·seed（P 为行归一化转移矩阵）。

    seed: (n,) 非负向量，会归一化为概率分布。
    返回稠密 π (n,)。图按无向处理（入边=出边）。
    """
    ei = _to_numpy(edge_index)
    if ei.shape[0] != 2:
        ei = ei.T
    src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)
    m = src != dst
    src, dst = src[m], dst[m]
    data = np.ones(src.shape[0], dtype=np.float64)
    A = sp.coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    out_deg = np.clip(out_deg, 1, None)
    P = sp.diags(1.0 / out_deg) @ A
    PT = P.T.tocsr()
    seed = np.asarray(seed, dtype=np.float64).ravel()
    seed = seed / seed.sum()
    pi = seed.copy()
    alpha = float(restart)
    for _ in range(int(K)):
        pi_new = (1.0 - alpha) * (PT @ pi) + alpha * seed
        delta = float(np.abs(pi_new - pi).sum())
        pi = pi_new
        if delta < tol:
            break
    return pi


def rank01(v):
    """秩归一化到 (0,1]：秩 1 → 1/n，秩 n → 1.0，单调保序。

    PPR 分数是幂律分布，min-max 会把大部分值压到 0 附近；秩归一化对尺度稳健，
    且与 `hyperedge_consistency` 的 c（越大越正常，值域 (0,1]）语义一致。
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    order = np.argsort(np.argsort(v))                     # 秩 (0..n-1)，含并列
    return (order + 1) / max(len(v), 1)


# ---------------------------------------------------------------------------
# 主入口（步骤1+2+3）
# ---------------------------------------------------------------------------
@torch.no_grad()
def ppr_structure_score(x, edge_index, seed_mask, pca_dim: int = 16,
                        density_k: int = 15, h: float = 1.0,
                        restart: float = 0.15, K: int = 60,
                        max_edges=None, seed_mode: str = "seed_mask",
                        normalize: str = "rank"):
    """计算单节点 PPR 结构得分 c_ppr ∈ (0,1]，越大越接近"正常骨架"。

    流程：密化图 → seed 的全局 PPR → 秩归一化。

    Args:
        x: (n, d) 特征（用 TTA 后的 x_sp，与 hyperedge_consistency 同输入）
        edge_index: [2, E] 原始图边（会与密化边合并）
        seed_mask: (n,) bool，PPR 重启 seed 的节点集合（语义上应传"校准正常节点"，
            即 PPR 衡量"从正常骨架出发的随机游走稳态落点概率"）
        seed_mode: "seed_mask"=seed_mask 内节点均匀 seed；"uniform"=全图均匀（对照）
        normalize: "rank"=秩归一化；"minmax"=min-max
    """
    x_np = _to_numpy(x)
    ei_np = _to_numpy(edge_index)
    n = x_np.shape[0]
    sm_np = _to_numpy(seed_mask).astype(bool)

    # 1) 图密化
    ei_d = densify_graph(x_np, ei_np, pca_dim, density_k, h, max_edges)
    ei_d_np = _to_numpy(ei_d)

    # 2) seed：seed_mask 内节点均匀分布
    if seed_mode == "seed_mask":
        seed = sm_np.astype(np.float64)
    else:  # "uniform" 全图均匀（对照：相当于普通 PageRank）
        seed = np.ones(n, dtype=np.float64)
    seed = seed / seed.sum()

    # 3) PPR 核
    pi = ppr_from_seed(ei_d_np, n, seed, restart, K)

    # 4) 归一化到 (0,1]
    if normalize == "rank":
        c = rank01(pi)
    else:
        lo, hi = float(pi.min()), float(pi.max())
        c = (pi - lo) / (hi - lo + 1e-12)
    c = np.clip(c, 1e-8, 1.0)
    return torch.from_numpy(c.astype(np.float32)).to(x.device)
