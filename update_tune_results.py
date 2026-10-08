#!/usr/bin/env python3
"""更新comparison_table.json，填入我们刚刚训练的GCN/GAT结果"""
import os
import sys
import json
import numpy as np

# 添加项目根目录到path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from data.loader import load_gad

def load_results(dataset, model):
    """加载我们训练的模型结果"""
    result_path = os.path.join(root_dir, "outputs", "baseline_checkpoints", f"{dataset}__{model}.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return json.load(f)
    return None

def main():
    # 读取原始的comparison_table.json
    with open("/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table.json", "r") as f:
        data = json.load(f)
    
    # 要处理的TUNE数据集
    tune_datasets = ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]
    models = ["gcn", "gat"]
    
    # 遍历所有数据集和模型
    for dataset in tune_datasets:
        for model in models:
            model_key = model.upper()
            # 加载结果
            result = load_results(dataset, model)
            if result:
                # 更新table中的条目
                if model_key in data["table"] and dataset in data["table"][model_key]:
                    data["table"][model_key][dataset]["auc"] = result["auc"]
                    data["table"][model_key][dataset]["status"] = "ok"
                    data["table"][model_key][dataset]["reason"] = ""
                    print(f"Updated {model_key} {dataset}: {result['auc']:.4f}")
    
    # 保存更新后的文件
    output_path = "/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table_final.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nUpdated comparison table saved to {output_path}")
    print("\nSummary of updates:")
    for dataset in tune_datasets:
        for model in models:
            result = load_results(dataset, model)
            if result:
                print(f"  {model.upper()} {dataset}: {result['auc']:.4f}")

if __name__ == "__main__":
    main()