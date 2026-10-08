#!/usr/bin/env python3
"""检查GCN/GAT模型的分数是否需要翻转"""
import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from data.loader import load_gad
from models.gcn_encoder import GCNEncoder
from models.gat_encoder import GATEncoder

def check_score_direction(dataset_name, model_type, device="cpu"):
    # 加载数据
    data = load_gad(name=dataset_name, seed=42)
    
    # 加载模型
    ckpt_path = f"checkpoint/{model_type.upper()}_{dataset_name}_homo.pth"
    if not os.path.exists(ckpt_path):
        print(f"Missing checkpoint for {model_type} {dataset_name}")
        return None
    
    # 构建模型
    in_channels = data.num_features
    if model_type == "gcn":
        model = GCNEncoder(in_channels=in_channels, hidden_channels=128, out_channels=2)
    elif model_type == "gat":
        model = GATEncoder(in_channels=in_channels, hidden_channels=128, out_channels=2)
    
    # 加载权重
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    
    # 计算分数
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        # 原始分数：异常类概率
        scores_original = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        # 翻转分数：正常类概率
        scores_flipped = F.softmax(out, dim=1)[:, 0].cpu().numpy()
    
    # 测试集标签
    test_mask = data.test_mask.cpu().numpy()
    y_true = data.y.cpu().numpy()
    
    y_test = y_true[test_mask]
    scores_orig = scores_original[test_mask]
    scores_flip = scores_flipped[test_mask]
    
    # 计算AUC
    auc_orig = roc_auc_score(y_test, scores_orig)
    auc_flip = roc_auc_score(y_test, scores_flip)
    
    # 判断是否需要翻转
    need_flip = auc_flip > auc_orig + 0.05
    print(f"=== {dataset_name} {model_type.upper()} ===")
    print(f"Original AUC: {auc_orig:.4f}")
    print(f"Flipped AUC: {auc_flip:.4f}")
    print(f"Need flip: {need_flip}")
    print(f"Final recommended AUC: {auc_flip if need_flip else auc_orig}")
    print()
    
    return {
        "dataset": dataset_name,
        "model": model_type,
        "auc_original": auc_orig,
        "auc_flipped": auc_flip,
        "need_flip": need_flip,
        "final_auc": auc_flip if need_flip else auc_orig
    }

def main():
    tune_datasets = ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]
    models = ["gcn", "gat"]
    
    results = []
    for dataset in tune_datasets:
        for model in models:
            res = check_score_direction(dataset, model)
            if res:
                results.append(res)
    
    # 统计
    print("\n=== Summary ===")
    for res in results:
        status = "FLIPPED" if res["need_flip"] else "NOT FLIPPED"
        print(f"{res['model'].upper()} {res['dataset']}: {res['final_auc']:.4f} ({status})")
    
    # 保存结果
    import json
    with open("/media/lixin/新加卷/数据集/test/StructCP/outputs/score_direction_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to outputs/score_direction_check.json")

if __name__ == "__main__":
    main()