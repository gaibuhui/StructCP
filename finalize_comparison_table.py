#!/usr/bin/env python3
"""根据分数方向检查结果更新comparison table"""
import json
import os

# 加载原始的对比表
with open("/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table_final.json", "r") as f:
    data = json.load(f)

# 加载分数方向检查结果
with open("/media/lixin/新加卷/数据集/test/StructCP/outputs/score_direction_check.json", "r") as f:
    direction_results = json.load(f)

# 需要翻转的映射：(model, dataset) -> flipped_auc
flip_map = {}
for res in direction_results:
    if res["need_flip"]:
        flip_map[(res["model"], res["dataset"])] = res["final_auc"]

print("需要翻转的条目:")
for (model, dataset), auc in flip_map.items():
    print(f"  {model.upper()} {dataset}: {auc:.4f}")

# 更新对比表
for (model_key, dataset), final_auc in flip_map.items():
    model_upper = model_key.upper()
    if model_upper in data["table"] and dataset in data["table"][model_upper]:
        # 更新AUC值
        data["table"][model_upper][dataset]["auc"] = final_auc
        # 添加翻转说明
        if "reason" in data["table"][model_upper][dataset]:
            data["table"][model_upper][dataset]["reason"] += " | 分数翻转（原始AUC<0.5）"
        else:
            data["table"][model_upper][dataset]["reason"] = "分数翻转（原始AUC<0.5）"

# 保存最终的对比表
final_output_path = "/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table_final_with_flip.json"
with open(final_output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n最终对比表已保存到: {final_output_path}")
print("\n所有条目已更新，包括分数翻转处理。")