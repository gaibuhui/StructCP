
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
import numpy as np
from tqdm import tqdm
import torch
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Amazon", "YelpChi", "Elliptic", "TFinance"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    # Hyperparameter search space
    k_list = [5, 10, 15, 20]
    L_list = [1, 2, 3]
    tau_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    gamma_list = [0, 0.5, 1, 1.5, 2]

    results = {}
    for ds in args.datasets:
        results[ds] = []
        for k in tqdm(k_list, desc=f"Processing {ds} k"):
            for L in L_list:
                for tau in tau_list:
                    for gamma in gamma_list:
                        for seed in args.seeds:
                            # Run StructCP with current hyperparameters
                            cmd = f"python scripts/run_structcp.py --dataset {ds} --k {k} --L {L} --tau {tau} --gamma {gamma} --alpha {args.alpha} --seed {seed} --use_calib_stat"
                            result = subprocess.run(cmd, shell=True, cwd=os.getcwd(), capture_output=True, text=True)
                            # Parse result
                            if result.returncode == 0:
                                # Parse output for FPR, TPR, F1
                                lines = result.stdout.split('\n')
                                for line in lines:
                                    if "StructCP" in line and "(target alpha" in line:
                                        # Find the next lines with results
                                        for i, l in enumerate(lines):
                                            if "StructCP" in l:
                                                result_line = lines[i+2]
                                                if len(result_line.split()) >= 6:
                                                    fpr = float(result_line.split()[4])
                                                    tpr = float(result_line.split()[5])
                                                    f1 = float(result_line.split()[6])
                                                    break
                                results[ds].append({
                                    "k": k,
                                    "L": L,
                                    "tau": tau,
                                    "gamma": gamma,
                                    "seed": seed,
                                    "fpr": fpr,
                                    "tpr": tpr,
                                    "f1": f1,
                                    "valid": fpr <= args.alpha
                                })
    # Save results
    with open("outputs/grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Grid search results saved to outputs/grid_search_results.json")

if __name__ == "__main__":
    main()