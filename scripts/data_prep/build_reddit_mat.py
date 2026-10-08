"""
build_reddit_mat.py
===================
将 GADBench 保存的 reddit DGL 图文件（dgl.save_graphs 格式）转换为 TA-GGAD 训练所需的
`reddit.mat`，字段与官方其它 14 个 .mat 保持一致：

    Network   : scipy.sparse.coo_matrix (N, N)  无自环邻接（TA-GGAD 的 Dataset 类会自己加 sp.eye 做对称归一化）
    Attributes: scipy.sparse.coo_matrix (N, F)  节点特征（dense 转 coo）
    Label     : numpy.ndarray (N, 1) uint8       异常标签（1=异常，0=正常）

数据来源（按 R4 不复制，仅读取本地已有文件）：
    TTHyper/baselines/GADBench/datasets/reddit   (GADBench 自带示例数据集，7.24MB)

说明：
    - GADBench 的 reddit 图默认含自环（num_edges = 真实边 + N）。TA-GGAD 的 Dataset 类
      对 reddit 走 `normalize_adj(adj + sp.eye(N))`，因此这里存"无自环"原始边，避免双重自环。
    - 仅取 trial 0 的 train/val/test mask 用于统计（不影响 .mat 结构，TA-GGAD 用 shot 抽样而非官方 mask）。

运行（CBP 环境，DGL 1.1.3 已验证可读该文件）：
    cd /media/lixin/新加卷/数据集/test/StructCP && \\
    source <CONDA_PREFIX>/bin/activate CBP && \\
    export PYTHONPATH="/media/lixin/新加卷/数据集/test/StructCP" && \\
    python scripts/build_reddit_mat.py
"""

# --- StructCP 项目根定位（深度无关）：任意脚本深度下均可定位根目录 ---
import os as _os, sys as _sys
def _structcp_root():
    _p = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(10):
        if _os.path.isfile(_os.path.join(_p, 'configs', 'default.yaml')):
            return _p
        _p = _os.path.dirname(_p)
    return _p
if _structcp_root() not in _sys.path:
    _sys.path.insert(0, _structcp_root())


import os
import argparse

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from dgl.data.utils import load_graphs

# ----------------------------------------------------------------------------
# 路径默认值（遵循项目目录约定：GADBench 数据在 TTHyper，TA-GGAD 在 StructCP）
# ----------------------------------------------------------------------------
DEFAULT_SRC = os.path.join(
    _structcp_root(),
    "TTHyper", "baselines", "GADBench", "datasets", "reddit",
)
DEFAULT_DST = os.path.join(
    _structcp_root(),
    "baselines", "ta_ggad_src", "dataset", "reddit.mat",
)


def main():
    parser = argparse.ArgumentParser(description="Convert GADBench reddit DGL -> TA-GGAD .mat")
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help="GADBench reddit DGL 图文件路径（dgl.save_graphs 格式）")
    parser.add_argument("--dst", default=DEFAULT_DST,
                        help="输出 TA-GGAD reddit.mat 路径")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"找不到 GADBench reddit 文件: {args.src}\n"
                                f"请确认 TTHyper/baselines/GADBench/datasets/reddit 是否存在。")

    print(f"[load] 读取 DGL 图: {args.src}")
    graph = load_graphs(args.src)[0][0]

    num_nodes = graph.num_nodes()
    src, dst = graph.edges()

    # 去掉自环，保留真实边（TA-GGAD 自行加自环归一化）
    mask = (src != dst)
    src_n = src[mask].numpy()
    dst_n = dst[mask].numpy()
    num_edges = src_n.shape[0]
    print(f"[info] 节点数={num_nodes}, 真实边数(去自环)={num_edges}, "
          f"原边数(含自环)={graph.num_edges()}")

    # 无向图：TA-GGAD 的 Network 为对称无向邻接（与 cora/pubmed 等一致）
    rows = np.concatenate([src_n, dst_n])
    cols = np.concatenate([dst_n, src_n])
    vals = np.ones(rows.shape[0], dtype=np.float64)
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    adj.sum_duplicates()          # 合并重复边（对称后可能出现的平行边）
    adj = adj.tocoo().astype(np.float64)

    # 节点特征：(N, F) dense -> coo
    feat = graph.ndata["feature"].cpu().numpy().astype(np.float64)
    feat_coo = sp.coo_matrix(feat).astype(np.float64)
    print(f"[info] 特征维度={feat.shape}, 异常标签数={int(graph.ndata['label'].sum())}")

    # 标签：(N,) -> (N, 1) uint8
    label = graph.ndata["label"].cpu().numpy().astype(np.uint8).reshape(-1, 1)

    # 保存（与 TA-GGAD 其它 .mat 同构：仅 Network / Attributes / Label 三字段）
    out_dir = os.path.dirname(args.dst)
    os.makedirs(out_dir, exist_ok=True)
    sio.savemat(args.dst, {
        "Network": adj,
        "Attributes": feat_coo,
        "Label": label,
    })
    print(f"[save] 已写出: {args.dst}")

    # 回读校验
    chk = sio.loadmat(args.dst)
    assert sp.isspmatrix_coo(chk["Network"]), "Network 非 coo_matrix"
    assert sp.isspmatrix_coo(chk["Attributes"]), "Attributes 非 coo_matrix"
    assert chk["Label"].shape == (num_nodes, 1), "Label 形状错误"
    assert int(chk["Label"].sum()) == int(label.sum()), "标签数不一致"
    print(f"[check] 校验通过: Network={chk['Network'].shape}, "
          f"Attributes={chk['Attributes'].shape}, Label={chk['Label'].shape}, "
          f"异常数={int(chk['Label'].sum())}")


if __name__ == "__main__":
    main()
