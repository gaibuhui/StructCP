"""Compute the estimable homophily-deviation proxy epsilon_hom (Claim 1 / Theorem 1).

Theoretical background
----------------------
Claim 1 links the weighted-exchangeability gap to a *homophily deviation*
parameter epsilon_hom.  In the paper epsilon_hom is *not* a free parameter: it
has a concrete, data-driven estimator that we report in Tab.~\ref{tab:eps_hom}.

Estimator
---------
We report two complementary, fully computable quantities:

1. delta (spectral gap of the k-NN incidence)
   epsilon_hom is tied (Claim 1) to the spectral gap delta of the normalized
   hypergraph Laplacian L = I - Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}.
   We estimate delta as the second-smallest eigenvalue of L (the Fiedler-like
   value), computed with sparse Lanczos for large graphs.

2. h_dev = 1 - edge_homophily
   The *empirical* homophily deviation: build a k-NN graph on raw features,
   then edge_homophily = fraction of edges whose endpoints share the (true)
   majority label within the k-NN neighborhood.  h_dev is large when anomalous
   clusters are internally consistent (community-type: Elliptic/T-Finance) and
   small when anomalies are outliers (outlier-type: Amazon/YelpChi).

This provides the *checkable* bridge between Theorem 1 and the data: when
h_dev is large AND the anomaly type is community-type, Assumption H is violated
and the bound does not apply -- exactly the regime where TPR drops.

Usage
-----
    cd StructCP && source <conda>/CBP && export PYTHONPATH=".../StructCP" \
        && python scripts/compute_eps_hom.py --datasets Amazon YelpChi Elliptic TFinance
"""
from __future__ import annotations

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
import json
import os
import sys

import numpy as np
import torch
from scipy.sparse.linalg import eigsh

sys.path.insert(0, _structcp_root())

from adapters.hypergraph_tta import KNNHypergraph  # noqa: E402
from data.loader import load_gad  # noqa: E402

DATA_DIR = "./datasets/GAD"


@torch.no_grad()
def estimate_epsilon_hom(name: str, k: int = 10) -> dict:
    print(f"[eps_hom] loading {name} ...")
    data = load_gad(name, root=DATA_DIR)
    x = torch.as_tensor(np.asarray(data.x, dtype=np.float32))
    y = torch.as_tensor(np.asarray(data.y, dtype=np.int64)).flatten()
    test_mask = torch.as_tensor(np.asarray(data.test_mask, dtype=bool)).flatten()

    # --- normalized hypergraph Laplacian L = I - Dv^{-1/2} H De^{-1} H^T Dv^{-1/2} (sparse)
    hg = KNNHypergraph(x, k=k, metric="cosine")
    Hc = hg.H.coalesce()
    n = hg.num_v
    dv = torch.sparse.sum(Hc, dim=1).to_dense().clamp(min=1).numpy().astype(np.float64)
    de = torch.sparse.sum(Hc, dim=0).to_dense().clamp(min=1).numpy().astype(np.float64)
    import scipy.sparse as sp
    Hsp = sp.coo_matrix((
        Hc.values().numpy().astype(np.float64),
        (Hc.indices()[0].numpy(), Hc.indices()[1].numpy()),
    ), shape=(n, n)).tocsr()
    Dv_inv_sqrt = sp.diags(dv ** -0.5)
    De_inv = sp.diags(de ** -1.0)
    L = sp.eye(n, format="csr") - Dv_inv_sqrt @ Hsp @ De_inv @ Hsp.T @ Dv_inv_sqrt
    L = L.tocsc()
    # second smallest eigenvalue (spectral gap delta) via sparse Lanczos
    try:
        evals = eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-3)
        delta = float(sorted(evals)[1])
    except Exception as e:  # pragma: no cover
        print(f"  [warn] eigsh failed: {e}; setting delta=NaN")
        delta = float("nan")

    # --- empirical homophily deviation h_dev on the k-NN graph (test nodes only)
    knn = hg.knn_idx.numpy()  # (n, k+1) includes self
    ynp = y.numpy()
    idx_test = np.where(test_mask.numpy())[0]
    same = 0
    total = 0
    # Assumption-H diagnostic: split edge-homophily by node class.
    # If ANOMALOUS nodes are themselves highly homophilous (h_dev_anom ~ 0),
    # then high consistency c_i no longer implies "true normal" -> Assumption H
    # is violated (community-type regime). We report this split explicitly.
    same_n, tot_n = 0, 0
    same_a, tot_a = 0, 0
    for i in idx_test:
        nb = knn[i, 1:]  # exclude self
        votes = ynp[nb]
        majority = np.bincount(votes, minlength=2).argmax()
        same += int(ynp[i] == majority)
        total += 1
        if ynp[i] == 0:
            same_n += int(majority == 0)
            tot_n += 1
        else:
            same_a += int(majority == 1)
            tot_a += 1
    edge_hom = same / max(total, 1)
    h_dev = 1.0 - edge_hom
    edge_hom_normal = same_n / max(tot_n, 1)
    edge_hom_anom = same_a / max(tot_a, 1)
    # Assumption-H violation score: how homophilous anomalous nodes are
    # (high => Assumption H fails => Theorem 1 does not apply).
    h_violation = 1.0 - edge_hom_anom

    return {
        "dataset": name,
        "n_test": int(total),
        "spectral_gap_delta": delta,
        "edge_homophily": float(edge_hom),
        "epsilon_hom_proxy": float(h_dev),
        "edge_homophily_normal": float(edge_hom_normal),
        "edge_homophily_anomaly": float(edge_hom_anom),
        "assumptionH_violation": float(h_violation),
        "anomaly_type": "community" if name in ("Elliptic", "TFinance") else "outlier",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="./outputs/eps_hom_table.json")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        rows.append(estimate_epsilon_hom(ds, k=args.k))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    # LaTeX table rows
    print("\n=== epsilon_hom proxy + Assumption-H diagnostic (Tab. eps_hom) ===")
    print(f"{'Dataset':<10} | type      | delta  | eps_hom | hom_anom | H_viol |")
    for r in rows:
        print(f"{r['dataset']:<10} | {r['anomaly_type']:<9} | "
              f"{r['spectral_gap_delta']:.4f} | {r['epsilon_hom_proxy']:.4f}  | "
              f"{r['edge_homophily_anomaly']:.4f}  | {r['assumptionH_violation']:.4f} |")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
