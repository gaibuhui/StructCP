"""
build_taggad_mats.py
=====================
将本地已有的原始数据转换为 TA-GGAD 所需的 .mat 格式
(字段: Network=coo_matrix, Attributes=coo_matrix, Label=ndarray)。

TA-GGAD 的 utils.Dataset 读取约定 (见 baselines/ta_ggad_src/utils.py):
    data = sio.loadmat(f"{prefix}{name}.mat")
    adj  = data['Network']
    feat = data['Attributes']
    label = data['Label']

支持的数据集:
  1. tfinance : 来自 datasets/TFS/processed/TFinance.npz
                (keys: feature[N,D], label[N], edge_index[2,E], num_nodes)
  2. elliptic : 来自 datasets/EllipticBitcoin/raw/ 的三个 CSV
                (elliptic_txs_classes.csv / _features.csv / _edgelist.csv)

其余 4 个缺失数据集 (Facebook, Flickr, questions, photo) 需从各自的
原始来源下载, 本脚本不覆盖, 需另行获取后转换。
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
import sys
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = "/media/lixin/新加卷/数据集/test"
OUT_DIR = os.path.join(ROOT, "StructCP/baselines/ta_ggad_src/dataset")


def save_mat(name, adj_coo, feat, label):
    """adj_coo: scipy sparse; feat: dense [N,D] or sparse; label: [N] or [N,1]."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if not sp.issparse(feat):
        feat = sp.coo_matrix(np.asarray(feat, dtype=np.float32))
    else:
        feat = feat.tocsr().tocoo().astype(np.float32)
    label = np.asarray(label).reshape(-1, 1).astype(np.int64)
    out = {
        "Network": adj_coo.tocsr().tocoo().astype(np.float32),
        "Attributes": feat,
        "Label": label,
    }
    path = os.path.join(OUT_DIR, f"{name}.mat")
    sio.savemat(path, out)
    print(f"[OK] saved {path}  nodes={adj_coo.shape[0]}  feat_dim={feat.shape[1]}")


def build_tfinance():
    npz = os.path.join(ROOT, "datasets/TFS/processed/TFinance.npz")
    if not os.path.exists(npz):
        print("[SKIP] tfinance: missing", npz)
        return
    f = np.load(npz, allow_pickle=True)
    x = np.asarray(f["feature"], dtype=np.float32)        # [N, D]
    y = np.asarray(f["label"]).reshape(-1)                # [N]
    ei = np.asarray(f["edge_index"], dtype=np.int64)      # [2, E]
    N = x.shape[0]
    adj = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32),
                         (ei[0], ei[1])), shape=(N, N))
    # 确保对称 (无向)
    adj = adj.maximum(adj.T)
    save_mat("tfinance", adj, x, y)


def build_elliptic():
    base = os.path.join(ROOT, "datasets/EllipticBitcoin/raw")
    cls = os.path.join(base, "elliptic_txs_classes.csv")
    feat = os.path.join(base, "elliptic_txs_features.csv")
    edge = os.path.join(base, "elliptic_txs_edgelist.csv")
    if not (os.path.exists(cls) and os.path.exists(feat) and os.path.exists(edge)):
        print("[SKIP] elliptic: missing raw csv in", base)
        return
    import pandas as pd
    labels = pd.read_csv(cls).to_numpy()
    node_features = pd.read_csv(feat, header=None).to_numpy()
    edges = pd.read_csv(edge).to_numpy()

    node_dict = {i: labels[i][0] for i in range(labels.shape[0])}
    inv = {v: k for k, v in node_dict.items()}
    N = labels.shape[0]
    new_labels = np.zeros(N, dtype=np.int64)
    marks = labels[:, 1] != "unknown"
    features = node_features[:, 1:].astype(np.float32)
    new_labels[labels[:, 1] == "1"] = 1

    # 时间特征作为划分参考 (GADBench 设定下仅生成图, 不需要 mask)
    row = np.array([inv[e[0]] for e in edges], dtype=np.int64)
    col = np.array([inv[e[1]] for e in edges], dtype=np.int64)
    adj = sp.coo_matrix((np.ones(row.shape[0], dtype=np.float32),
                         (row, col)), shape=(N, N))
    adj = adj.maximum(adj.T)
    save_mat("elliptic", adj, features, new_labels)


if __name__ == "__main__":
    want = sys.argv[1:] or ["tfinance", "elliptic"]
    for ds in want:
        if ds == "tfinance":
            build_tfinance()
        elif ds == "elliptic":
            build_elliptic()
        else:
            print("[WARN] unknown dataset for local build:", ds)
