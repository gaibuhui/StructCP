"""Step 5: 无参数拓扑结构异常得分 (弥补纯特征子空间的缺陷)。

对每个节点计算三类无参数拓扑信号并归一化融合:
  1) 邻域特征相似度均值  (异常节点常与邻居特征不一致)
  2) 节点度偏差          (过高/过低度偏离正常分布)
  3) 局部结构熵          (邻居度分布熵, 异常局部结构紊乱)

输出 s_topo in [0,1], 分数越高越异常。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler


def topology_anomaly_score(
    x: np.ndarray,
    edge_index: np.ndarray,
    degree_hub_thresh: int = 200,
) -> np.ndarray:
    """返回每个节点的拓扑异常分 s_topo (N,)。

    注意: 拓扑信号应基于**有判别力的嵌入**计算 (推荐传入 BWGNN 嵌入 Z,
    而非原始特征 x), 否则异常节点的邻域相似度区分度弱。

    Args:
        x: (N, d) 节点特征/嵌入 (标准化后)
        edge_index: (2, E) 无向边
        degree_hub_thresh: hub 裁剪阈值 (与 Step1 一致, 仅用于度偏差归一化封顶)
    """
    N = x.shape[0]
    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    # 无向邻接 (去重), 稀疏表示, 避免 (N,N) 稠密矩阵在大数据集爆内存
    edges = np.vstack([np.stack([src, dst], 1), np.stack([dst, src], 1)])
    edges = np.unique(edges, axis=0)
    adj = csr_matrix(
        (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(N, N),
    )
    # 对称化 (去重后已近似无向, 保险起见)
    adj = adj.maximum(adj.T)
    deg = np.asarray(adj.sum(1)).ravel().astype(np.float32)

    # 1) 邻域特征相似度均值 (cosine) —— 向量化: 用稀疏邻接做局部加权池化
    x_n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    # 归一化嵌入的邻接加权均值: adj @ x_n, 再逐节点与自身余弦相似度
    neigh_mean = adj.dot(x_n)                       # (N, d) 邻居嵌入均值
    row_deg = deg.copy()
    row_deg[row_deg == 0] = 1.0
    neigh_mean = neigh_mean / row_deg[:, None]      # 除以度, 得到均值
    sim_self = np.sum(x_n * neigh_mean, axis=1)     # (N,) 与邻居均值余弦相似度
    s_sim = (1.0 - sim_self).astype(np.float32)     # 异常分 = 1 - 平均相似度

    # 2) 节点度偏差 (偏离正常度分布的程度, 用 |log(deg+1) - 中位数| 归一)
    logdeg = np.log1p(deg)
    med = np.median(logdeg) + 1e-8
    s_deg = np.abs(logdeg - med) / (np.std(logdeg) + 1e-8)
    s_deg = np.clip(s_deg, 0, 5) / 5.0

    # 3) 局部结构熵 (邻居度分布熵, 异常节点局部结构紊乱 -> 熵高) —— 向量化
    logdeg_rep = adj.dot(logdeg) / row_deg         # 邻居 logdeg 均值
    # 用邻居度均值代理分布熵 (向量化, 等价逻辑: 邻居度分布越分散越紊乱)
    s_ent = np.abs(logdeg - logdeg_rep)            # 与邻居偏离越大越异常
    s_ent = s_ent / (np.std(logdeg) + 1e-8)
    s_ent = np.clip(s_ent, 0, 5) / 5.0

    # 归一化融合 (等权, 可在主脚本调 alpha 前做 0-1 归一)
    mm = MinMaxScaler()
    feat = np.vstack([s_sim, s_deg, s_ent]).T
    feat_n = mm.fit_transform(feat)
    s_topo = feat_n.mean(1).astype(np.float32)
    return s_topo
