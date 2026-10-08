import json
import os

# 读取当前的对比表文件
file_path = "/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# 需要标记为OOM的方法和数据集
oom_updates = {
    "DOMINANT": ["TFinance"],
    "ARC": ["TFinance"],
    "GADT3": ["TFinance"],
    "TA-GGAD": ["TFinance"]
}

# 更新每个条目的状态
for method, datasets in oom_updates.items():
    for dataset in datasets:
        if method in data["table"] and dataset in data["table"][method]:
            data["table"][method][dataset]["status"] = "oom"
            data["table"][method][dataset]["reason"] = "算力限制 (全图 OOM)"

# 更新note字段中的描述
old_note_part = "TFinance 上 DOMINANT/ARC/GADT3/TA-GGAD 全图 OOM -> N/A"
new_note_part = "TFinance 上 DOMINANT/ARC/GADT3/TA-GGAD 全图 OOM -> OOM"
data["note"] = data["note"].replace(old_note_part, new_note_part)

# 更新na_breakdown统计
# 原来的算力限制计数是3，现在增加1个DOMINANT，变为4
data["na_breakdown"]["算力限制 (全图 OOM)"] = 4
# 原来的"其他: 未跑通"是1，现在DOMINANT已经移到算力限制，所以变为0
data["na_breakdown"]["其他: 未跑通"] = 0

# 确保na_total正确（总N/A数量不变，只是分类调整）
data["na_total"] = sum(data["na_breakdown"].values())

# 生成新的文件名
output_file = "/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table_updated_oom.json"
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"更新完成，结果已保存到: {output_file}")
print(f"已将TFinance数据集上的DOMINANT/ARC/GADT3/TA-GGAD标记为OOM状态")