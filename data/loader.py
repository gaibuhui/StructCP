"""GAD 数据加载 + 正常性偏移(normality shift)构造。

数据源: datasets/GAD/{Amazon,YelpChi}.mat  (CARE-GNN / BWGNN 官方格式)

正常性偏移设定 (对标 TUNE, AAAI 2026):
  训练阶段只见到"部分正常模式"，测试阶段出现"新正常类别"。
  由于 Amazon/YelpChi 只有二值标签(正常/异常)，
  我们对正常节点做 KMeans 聚类得到伪正常类别 (normality prototypes)，
  将其中一部分簇标记为 "novel normality"，训练集中完全剔除，
  仅在测试期出现 —— 这会让冻结检测器把它们误判为异常(误报激增)，
  正是 StructCP 要修正的场景。
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "GAD")

# T-Finance / T-Social (Rethinking GNN-AD, ICML'22) 原始 DGL 文件位置
# 文件为 DGL save_graphs 格式 (torch 2.6 新序列化), 由外部脚本转换为通用 .npz 存放于此目录。
TFS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "datasets", "TFS"
)


class GADData:
    """轻量数据容器。"""

    def __init__(self, x, edge_index, y, train_mask, cal_mask, test_mask, novel_mask, name):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_mask = train_mask
        self.cal_mask = cal_mask
        self.test_mask = test_mask
        self.novel_mask = novel_mask  # 新正常类别节点 (测试期才出现)
        self.name = name

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_nodes(self):
        return self.x.shape[0]

    def to(self, device):
        for k in ["x", "edge_index", "y", "train_mask", "cal_mask", "test_mask", "novel_mask"]:
            setattr(self, k, getattr(self, k).to(device))
        return self

    def __repr__(self):
        return (
            f"GADData({self.name}: n={self.num_nodes}, d={self.num_features}, "
            f"E={self.edge_index.shape[1]}, anomaly={self.y.float().mean():.4f}, "
            f"train={int(self.train_mask.sum())}, cal={int(self.cal_mask.sum())}, "
            f"test={int(self.test_mask.sum())}, novel={int(self.novel_mask.sum())})"
        )


def _to_edge_index(adj: sp.spmatrix) -> torch.Tensor:
    adj = sp.coo_matrix(adj)
    ei = np.vstack([adj.row, adj.col])
    return torch.from_numpy(ei).long()


def load_gad(
    name: str = "Amazon",
    root: str = DEFAULT_ROOT,
    relation: str = "homo",
    train_ratio: float = 0.4,
    cal_ratio: float = 0.2,
    n_normality_clusters: int = 6,
    n_novel_clusters: int = 2,
    shift_strength: float = None,
    seed: int = 42,
    subsample: int = None,
    knn_backbone: int = None,
    knn_only: bool = False,
) -> GADData:
    # 偏移强度消融: 用 novel 簇占比 shift_strength (1/6~3/6) 控制偏移幅度,
    # 覆盖默认 n_novel_clusters, 不影响其余默认行为。
    if shift_strength is not None:
        n_novel_clusters = max(1, int(round(shift_strength * n_normality_clusters)))
    if name == "Elliptic":
        return load_elliptic(
            train_ratio=train_ratio, cal_ratio=cal_ratio,
            n_normality_clusters=n_normality_clusters,
            n_novel_clusters=n_novel_clusters, seed=seed,
        )
    if name in ("TFinance", "TSocial"):
        return load_tfs(
            name=name, train_ratio=train_ratio, cal_ratio=cal_ratio,
            n_normality_clusters=n_normality_clusters,
            n_novel_clusters=n_novel_clusters, seed=seed,
            subsample=subsample, knn_backbone=knn_backbone, knn_only=knn_only,
        )
    # TUNE (AAAI'26) 复现涉及的额外数据集 (DGL 图格式, 来自 TUNE_official/datasets)。
    # 它们在 StructCP 主表框架内采用统一的 KMeans novel-normality 协议 (与 GADBench/TFS 一致),
    # 而非 TUNE 自己的 few-shot / unseen-normal 协议, 以保证跨数据集方法学可比。
    TUNE_NAMES = {"photo", "computer", "weibo", "tolokers", "questions", "reddit"}
    if name in TUNE_NAMES:
        return load_tune_gad(
            name=name, train_ratio=train_ratio, cal_ratio=cal_ratio,
            n_normality_clusters=n_normality_clusters,
            n_novel_clusters=n_novel_clusters, seed=seed,
            subsample=subsample, knn_backbone=knn_backbone, knn_only=knn_only,
        )
    path = os.path.join(root, f"{name}.mat")
    mat = loadmat(path)

    feats = mat["features"]
    x = np.asarray(feats.todense() if sp.issparse(feats) else feats, dtype=np.float32)
    y = np.asarray(mat["label"]).flatten().astype(np.int64)

    # 多关系融合: 仅对 YelpChi 有意义 (homo/net_rur/net_rtr/net_rsr)。
    # relation="multi" 时把全部 relation 相加成单一无向图, 保留更丰富的拓扑信号。
    if relation == "multi":
        rels = [k for k in mat.keys() if k.startswith("net_") or k == "homo"]
        adj = None
        for k in rels:
            a = mat[k]
            a = a + a.T  # 保证对称
            adj = a if adj is None else (adj + a)
    else:
        adj = mat[relation]
        adj = adj + adj.T  # 保证对称

    x = StandardScaler().fit_transform(x).astype(np.float32)
    edge_index = _to_edge_index(adj)

    n = x.shape[0]
    rng = np.random.RandomState(seed)

    # ---- 构造正常性偏移: 对正常节点聚类, 挑出 novel normality 簇 ----
    normal_idx = np.where(y == 0)[0]
    km = KMeans(n_clusters=n_normality_clusters, random_state=seed, n_init=10)
    cl = km.fit_predict(x[normal_idx])
    # 选最小的若干簇作为 novel normality (更贴近"罕见新正常模式")
    sizes = np.bincount(cl, minlength=n_normality_clusters)
    novel_clusters = np.argsort(sizes)[:n_novel_clusters]
    novel_flag = np.isin(cl, novel_clusters)
    novel_nodes = normal_idx[novel_flag]

    novel_mask = np.zeros(n, dtype=bool)
    novel_mask[novel_nodes] = True

    # ---- 划分 train / cal / test ----
    # train 与 cal 均不包含 novel normality (模拟部署前未见过)
    seen = np.where(~novel_mask)[0]
    rng.shuffle(seen)
    n_tr = int(len(seen) * train_ratio)
    n_cal = int(len(seen) * cal_ratio)
    tr_idx = seen[:n_tr]
    cal_idx = seen[n_tr : n_tr + n_cal]
    te_seen = seen[n_tr + n_cal :]
    # 测试集 = 剩余已见节点 + 全部 novel normality 节点 → 分布偏移
    te_idx = np.concatenate([te_seen, novel_nodes])

    def _mask(idx):
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return torch.from_numpy(m)

    return GADData(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(y),
        train_mask=_mask(tr_idx),
        cal_mask=_mask(cal_idx),
        test_mask=_mask(te_idx),
        novel_mask=torch.from_numpy(novel_mask),
        name=name,
    )


def _build_gad_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    train_ratio: float = 0.4,
    cal_ratio: float = 0.2,
    n_normality_clusters: int = 8,
    n_novel_clusters: int = 2,
    seed: int = 42,
    name: str = "",
) -> GADData:
    """由 (feature, label, edge_index[2,E]) 数组构造 GADData, 含 novel-normality 偏移。

    复用逻辑: 对正常节点 KMeans 聚类 -> 取最小簇作为 novel normality (测试专属),
    其余节点划分 train/cal/test。label 约定: 0=正常, 1=异常。
    """
    n = x.shape[0]
    x = StandardScaler().fit_transform(x).astype(np.float32)
    rng = np.random.RandomState(seed)

    normal_idx = np.where(y == 0)[0]
    km = KMeans(n_clusters=n_normality_clusters, random_state=seed, n_init=10)
    cl = km.fit_predict(x[normal_idx])
    sizes = np.bincount(cl, minlength=n_normality_clusters)
    novel_clusters = np.argsort(sizes)[:n_novel_clusters]
    novel_flag = np.isin(cl, novel_clusters)
    novel_nodes = normal_idx[novel_flag]

    novel_mask = np.zeros(n, dtype=bool)
    novel_mask[novel_nodes] = True

    seen = np.where(~novel_mask)[0]
    rng.shuffle(seen)
    n_tr = int(len(seen) * train_ratio)
    n_cal = int(len(seen) * cal_ratio)
    tr_idx = seen[:n_tr]
    cal_idx = seen[n_tr : n_tr + n_cal]
    te_seen = seen[n_tr + n_cal :]
    te_idx = np.concatenate([te_seen, novel_nodes])

    def _mask(idx):
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return torch.from_numpy(m)

    return GADData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.from_numpy(y),
        train_mask=_mask(tr_idx),
        cal_mask=_mask(cal_idx),
        test_mask=_mask(te_idx),
        novel_mask=torch.from_numpy(novel_mask),
        name=name,
    )


def _knn_edge_index(x: np.ndarray, k: int = 10, mode: str = "connect"):
    """在特征空间构造 k-NN 无向边 (双向, 去重)。"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=8)
    nn.fit(x)
    _, idx = nn.kneighbors(x)
    rows, cols = [], []
    for i in range(x.shape[0]):
        for j in idx[i, 1:]:
            rows.append(i)
            cols.append(int(j))
    ei = np.vstack([rows, cols]).astype(np.int64)
    # 去重 (无向)
    ei_sorted = np.sort(ei, axis=0)
    _, uniq = np.unique(ei_sorted, axis=1, return_index=True)
    ei = ei[:, np.sort(uniq)]
    return ei


def load_tfs(
    name: str = "TFinance",
    train_ratio: float = 0.4,
    cal_ratio: float = 0.2,
    n_normality_clusters: int = 8,
    n_novel_clusters: int = 2,
    seed: int = 42,
    subsample: int = None,
    knn_backbone: int = None,
    knn_only: bool = False,
) -> GADData:
    """T-Finance / T-Social (Rethinking GNN-AD, ICML'22) 数据集。

    - T-Finance: 39,357 节点 / 42.4M 边 / 10 维连续特征 / 异常率 4.58% (金融交易图)。
    - T-Social: 5,781,065 节点 / 146.2M 边 / 10 维特征 / 异常率 3.01% (大规模社交图)。

    二者均无官方 train/test 划分, 沿用与其他 GAD 数据集一致的 KMeans novel-normality
    构造, 以统一方法学做跨数据集泛化验证。

    T-Social 规模 (5.78M 节点 / 1.46 亿边) 超过 8GB 显存, 默认对节点做子采样
    (subsample, 默认 100k) 再做训练/校准/测试划分; 子图边由全局 edge_index 裁剪得到。
    由于 100k 随机子采样下诱导原图边极稀疏 (平均度 <1), 仅用原图边时 BWGNN 退化为
    MLP; 故 T-Social 默认额外补充特征 k-NN 骨干边 (knn_backbone=10), 使骨干保留结构信号。
    """
    npz_path = os.path.join(TFS_ROOT, "processed", f"{name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"missing {npz_path}; run: python scripts/convert_tfs.py --name {name}"
        )
    d = np.load(npz_path)
    feat = d["feature"].astype(np.float32)
    y = d["label"].astype(np.int64).ravel()
    edge_index = d["edge_index"].astype(np.int64)

    n_total = feat.shape[0]
    if subsample is not None and n_total > subsample:
        rng = np.random.RandomState(seed)
        # 分层子采样, 保持异常比例
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = min(len(pos), max(1, int(subsample * (len(pos) / n_total))))
        n_neg = subsample - n_pos
        sel_pos = rng.choice(pos, size=n_pos, replace=False)
        sel_neg = rng.choice(neg, size=n_neg, replace=False)
        sel = np.concatenate([sel_pos, sel_neg])
        sel = np.sort(sel)
        remap = -np.ones(n_total, dtype=np.int64)
        remap[sel] = np.arange(sel.shape[0])
        # 裁剪原图边: 仅保留两端都在子集内的边
        keep = np.isin(edge_index[0], sel) & np.isin(edge_index[1], sel)
        ei_sub = edge_index[:, keep]
        edge_index = np.vstack([remap[ei_sub[0]], remap[ei_sub[1]]]).astype(np.int64)
        feat = feat[sel]
        y = y[sel]
        print(f"[load_tfs] {name} subsampled -> {feat.shape[0]} nodes, "
              f"{edge_index.shape[1]} induced-edges")
        if knn_backbone is not None:
            knn_ei = _knn_edge_index(feat, k=knn_backbone)
            edge_index = np.hstack([edge_index, knn_ei]).astype(np.int64)
            print(f"[load_tfs] + knn backbone (k={knn_backbone}) -> "
                  f"{edge_index.shape[1]} total edges")

    if knn_only:
        # 8GB 显存/有限主机内存硬约束: TFinance(42.4M 边)/TSocial(146M 边) 原图边
        # 密集化会 OOM。knn_only 丢弃原始稠密边, 仅用特征空间 k-NN 骨干边
        # (knn_backbone 默认 10)。节点/特征/标签均为真实数据, 仅图结构换成
        # 特征 k-NN —— 与 T-Social 子采样策略一致, 属真实实验(非模拟)。
        if knn_backbone is None:
            knn_backbone = 10
        edge_index = _knn_edge_index(feat, k=knn_backbone)
        print(f"[load_tfs] knn_only=True -> using feature k-NN edges (k={knn_backbone}), "
              f"{edge_index.shape[1]} edges (original dense edges dropped)")

    return _build_gad_from_arrays(
        x=feat, y=y, edge_index=edge_index,
        train_ratio=train_ratio, cal_ratio=cal_ratio,
        n_normality_clusters=n_normality_clusters,
        n_novel_clusters=n_novel_clusters, seed=seed, name=name,
    )


def load_elliptic(
    root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "EllipticBitcoin"),
    train_ratio: float = 0.4,
    cal_ratio: float = 0.2,
    n_normality_clusters: int = 8,
    n_novel_clusters: int = 2,
    seed: int = 42,
) -> GADData:
    """Elliptic Bitcoin 数据集 (真实金融交易图, 203k 节点 / 165 维 / 带时间步)。

    用途: 作为**第三个独立领域** (金融 vs 电商 Amazon / 评论 YelpChi) 验证跨数据集泛化。

    标签映射 (PyG): 0=licit(正常), 1=illicit(异常), 2=unknown(无标签)。
    仅 licit/illicit 节点参与训练/校准/测试划分; unknown 节点丢弃。

    NOTE: Elliptic 的已知标签高度集中在 t=0 快照, 后期时间步几乎无标签,
    故**无法做纯时间切分的跨时间偏移**评测。本函数沿用与其他数据集一致的
    KMeans novel-normality 构造 (在真实 165 维金融特征上), 以统一方法学对比。
    """
    from torch_geometric.datasets import EllipticBitcoinDataset

    g = EllipticBitcoinDataset(root=root)[0]
    x_raw = g.x.numpy().astype(np.float32)
    y_all = g.y.numpy().astype(np.int64)
    edge_index = g.edge_index.numpy().astype(np.int64)

    # 仅保留有标签节点 (licit/illicit), 丢弃 unknown(2)
    known = y_all != 2
    x = x_raw[known]
    y = y_all[known]
    # 重新映射边: 仅保留两端都在 known 集合内的边, 并压缩索引
    known_idx = np.where(known)[0]
    remap = -np.ones(y_all.shape[0], dtype=np.int64)
    remap[known_idx] = np.arange(known_idx.shape[0])
    src, dst = edge_index[0], edge_index[1]
    keep = known[src] & known[dst]
    src, dst = remap[src[keep]], remap[dst[keep]]
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()

    # 复用统一构造块: 传 remap 后的原始特征 (helper 内做 StandardScaler 归一化),
    # 边为 remap 后的 coo。时间步列一并纳入归一化口径, 与其他数据集保持一致。
    x_remapped = x_raw[known_idx]
    return _build_gad_from_arrays(
        x=x_remapped, y=y, edge_index=edge_index.numpy().astype(np.int64),
        train_ratio=train_ratio, cal_ratio=cal_ratio,
        n_normality_clusters=n_normality_clusters,
        n_novel_clusters=n_novel_clusters, seed=seed, name="Elliptic",
    )


# ---------------------------------------------------------------------------
# TUNE (AAAI'26) 复现涉及的额外数据集
# ---------------------------------------------------------------------------
# 这些数据集以 DGL 图格式存放于 baselines/TUNE_official/datasets/{name}。
# 每个图含 ndata: feature / label / train_mask / val_mask / test_mask
# (photo/computer/reddit 额外含 newnormal_masks / train_masks / val_masks / test_masks)。
#
# 在 StructCP 主表框架内, 我们**不**沿用 TUNE 的 few-shot/unseen-normal 协议, 而是
# 复用与 GADBench/TFS 完全一致的 KMeans novel-normality 构造 (见 _build_gad_from_arrays):
#   - 对正常节点 KMeans 聚类, 取最小簇为 novel normality (仅测试期出现)
#   - train/cal 不含 novel, test = 剩余已见节点 + novel 节点 → 分布偏移
# 这样主表所有数据集共用同一方法学, 跨数据集可比。
TUNE_GRAPH_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets", "TUNE",
)


def load_tune_gad(
    name: str = "photo",
    train_ratio: float = 0.4,
    cal_ratio: float = 0.2,
    n_normality_clusters: int = 8,
    n_novel_clusters: int = 2,
    seed: int = 42,
    subsample: int = None,
    knn_backbone: int = None,
    knn_only: bool = False,
) -> GADData:
    """加载 TUNE 复现涉及的额外数据集 (DGL 图) 并构造与 StructCP 主表一致的
    novel-normality 偏移划分。

    Args:
        name: 数据集名, 支持 photo/computer/weibo/tolokers/questions/reddit。
        subsample / knn_backbone / knn_only: 与 TFS 一致的显存/内存约束补充边。
    """
    import dgl  # 延迟导入, 避免无关数据集也依赖 dgl
    from dgl.data.utils import load_graphs

    graph_path = os.path.join(TUNE_GRAPH_ROOT, name)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"missing TUNE graph: {graph_path}")
    g = load_graphs(graph_path)[0][0]

    feat = g.ndata["feature"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64).ravel()

    # DGL 图 -> PyG 无向 edge_index (双向, 去重)
    src, dst = g.edges()
    ei = np.vstack([src.numpy(), dst.numpy()]).astype(np.int64)
    ei = np.hstack([ei, np.flipud(ei)])  # 无向: 补反向边
    ei_sorted = np.sort(ei, axis=0)
    _, uniq = np.unique(ei_sorted, axis=1, return_index=True)
    edge_index = ei[:, np.sort(uniq)]

    n_total = feat.shape[0]
    if subsample is not None and n_total > subsample:
        rng = np.random.RandomState(seed)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = min(len(pos), max(1, int(subsample * (len(pos) / n_total))))
        n_neg = subsample - n_pos
        sel_pos = rng.choice(pos, size=n_pos, replace=False)
        sel_neg = rng.choice(neg, size=n_neg, replace=False)
        sel = np.sort(np.concatenate([sel_pos, sel_neg]))
        remap = -np.ones(n_total, dtype=np.int64)
        remap[sel] = np.arange(sel.shape[0])
        keep = np.isin(edge_index[0], sel) & np.isin(edge_index[1], sel)
        ei_sub = edge_index[:, keep]
        edge_index = np.vstack([remap[ei_sub[0]], remap[ei_sub[1]]]).astype(np.int64)
        feat = feat[sel]
        y = y[sel]
        print(f"[load_tune_gad] {name} subsampled -> {feat.shape[0]} nodes, "
              f"{edge_index.shape[1]} induced-edges")
        if knn_backbone is not None:
            knn_ei = _knn_edge_index(feat, k=knn_backbone)
            edge_index = np.hstack([edge_index, knn_ei]).astype(np.int64)

    if knn_only:
        if knn_backbone is None:
            knn_backbone = 10
        edge_index = _knn_edge_index(feat, k=knn_backbone)
        print(f"[load_tune_gad] knn_only=True -> feature k-NN edges (k={knn_backbone}), "
              f"{edge_index.shape[1]} edges")

    return _build_gad_from_arrays(
        x=feat, y=y, edge_index=edge_index,
        train_ratio=train_ratio, cal_ratio=cal_ratio,
        n_normality_clusters=n_normality_clusters,
        n_novel_clusters=n_novel_clusters, seed=seed, name=name,
    )
