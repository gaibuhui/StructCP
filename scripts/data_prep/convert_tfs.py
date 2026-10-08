"""Convert DGL-saved T-Finance / T-Social graphs to a unified .npz.

The original files (datasets/TFS/dataset/tfinance|tsocial) are produced by
Rethinking-GNN-AD (ICML'22) and stored via DGL ``save_graphs`` (torch 2.6
serialization). This helper reads them with DGL and writes a plain ``.npz``
that ``data/loader.py`` can load without DGL installed.

Usage:
    python scripts/convert_tfs.py --name TFinance
    python scripts/convert_tfs.py --name TSocial
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

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "..") if False else os.path.dirname(os.path.dirname(HERE))
DATASET_DIR = os.path.join(ROOT, "datasets", "TFS")
OUT_DIR = os.path.join(DATASET_DIR, "processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["TFinance", "TSocial"], required=True)
    ap.add_argument("--src", default=None, help="override DGL source file")
    args = ap.parse_args()

    from dgl.data.utils import load_graphs

    src = args.src or os.path.join(DATASET_DIR, "dataset", args.name.lower(), args.name.lower())
    print(f"[convert] loading {src}")
    g, _ = load_graphs(src)
    gg = g[0]

    feat = gg.ndata["feature"].cpu().numpy().astype(np.float32)
    lab = gg.ndata["label"].cpu().numpy()
    if lab.ndim > 1:
        lab = lab.argmax(1)
    lab = lab.astype(np.int64).ravel()

    # edge_index as [2, E] numpy long (coo)
    src_nodes, dst_nodes = gg.edges()
    edge_index = np.vstack([
        src_nodes.cpu().numpy().astype(np.int64),
        dst_nodes.cpu().numpy().astype(np.int64),
    ])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.name}.npz")
    np.savez(
        out_path,
        feature=feat,
        label=lab,
        edge_index=edge_index,
        num_nodes=np.int64(gg.num_nodes()),
    )
    print(f"[convert] wrote {out_path}")
    print(f"  num_nodes={gg.num_nodes()} num_edges={gg.num_edges()} "
          f"feat={feat.shape} anomaly_ratio={float((lab == 1).mean()):.4f}")


if __name__ == "__main__":
    main()
