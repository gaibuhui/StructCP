#!/usr/bin/env python3
"""训练GCN/GAT模型，对齐TUNE novel-normality协议，仅使用正常样本训练"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# 添加项目根目录到path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from data.loader import load_gad
from models.gcn_encoder import GCNEncoder
from models.gat_encoder import GATEncoder


def train_normal_only(model, data, device, epochs=100, lr=1e-3):
    """仅使用正常样本训练模型"""
    model = model.to(device)
    
    # 只使用正常样本的训练mask
    train_mask = data.train_mask & (data.y == 0)
    normal_train_idx = torch.where(train_mask)[0]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"[train] 正常样本训练集大小: {len(normal_train_idx)}/{data.num_nodes}")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out[normal_train_idx], data.y[normal_train_idx].to(device))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model


def evaluate(model, data, device):
    """评估模型在test set上的AUROC"""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        scores = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        
        test_mask = data.test_mask.cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        y_test = y_true[test_mask]
        scores_test = scores[test_mask]
        
        auc = roc_auc_score(y_test, scores_test)
        ap = average_precision_score(y_test, scores_test)
        
        print(f"[eval] Test AUROC: {auc:.4f}, AUPRC: {ap:.4f}")
        return auc, ap


def save_checkpoint(model, dataset_name, model_type, save_dir="checkpoint"):
    """保存模型checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{model_type}_{dataset_name}_homo.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "dataset": dataset_name
    }, ckpt_path)
    print(f"[save] Checkpoint saved to {ckpt_path}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["photo", "computer", "weibo", "tolokers", "questions", "reddit"])
    parser.add_argument("--model", required=True, choices=["gcn", "gat"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()
    
    print(f"=== Training {args.model} on {args.dataset} ===")
    
    # 加载数据
    data = load_gad(name=args.dataset, seed=42)
    print(f"[load] {data}")
    
    # 构建模型
    in_channels = data.num_features
    if args.model == "gcn":
        model = GCNEncoder(in_channels=in_channels, hidden_channels=args.hidden, out_channels=2)
    elif args.model == "gat":
        model = GATEncoder(in_channels=in_channels, hidden_channels=args.hidden, out_channels=2)
    
    # 训练模型
    model = train_normal_only(model, data, args.device, args.epochs, args.lr)
    
    # 评估模型
    auc, ap = evaluate(model, data, args.device)
    
    # 保存checkpoint
    ckpt_path = save_checkpoint(model, args.dataset, args.model.upper())
    
    # 保存结果
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "auc": float(auc),
        "ap": float(ap),
        "checkpoint": ckpt_path
    }
    
    output_dir = os.path.join(root_dir, "outputs", "baseline_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{args.dataset}__{args.model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"=== Done! AUROC: {auc:.4f}")
    return auc, ap


if __name__ == "__main__":
    main()