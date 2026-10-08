#!/usr/bin/env python3
"""运行单个基线方法的评测脚本"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# 添加项目根目录到path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from data.loader import load_gad
from adapters.registry import build

def run_baseline(method: str, dataset: str, device: str = "cuda"):
    # 加载数据
    data = load_gad(name=dataset, seed=42)
    data = data.to(device)
    
    # 构建模型
    adapter = build(method, device=device)
    
    # 训练/加载模型
    print(f"[info] Running {method} on {dataset}...")
    adapter.fit(data)
    
    # 计算分数
    scores = adapter.score(data)
    
    # 计算指标
    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    y_test = y[test_mask]
    scores_test = scores[test_mask]
    
    auc = roc_auc_score(y_test, scores_test)
    ap = average_precision_score(y_test, scores_test)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs/baseline_subset/fill")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}__{method}.json")
    
    with open(output_path, "w") as f:
        json.dump({"AUROC": auc, "AUPRC": ap, "auc": auc, "ap": ap}, f, indent=2)
    
    print(f"[info] Saved results to {output_path}")
    print(f"[info] AUROC: {auc:.4f}, AUPRC: {ap:.4f}")
    
    return auc, ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="基线方法名称，如ARC、GADT3等")
    parser.add_argument("--dataset", required=True, help="数据集名称，如Amazon、YelpChi等")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    args = parser.parse_args()
    
    run_baseline(args.method, args.dataset, args.device)