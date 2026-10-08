import json
import os

# 读取当前的comparison_table.json
with open("/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table.json", "r") as f:
    data = json.load(f)

# 更新TFinance部分的各个方法的status为oom
methods_to_mark_oom = ["ARC", "GADT3", "TA-GGAD", "DOMINANT"]
for method in methods_to_mark_oom:
    if method in data["table"] and "TFinance" in data["table"][method]:
        data["table"][method]["TFinance"]["status"] = "oom"
        data["table"][method]["TFinance"]["reason"] = "算力限制 (TFinance 全图 OOM)"

# 更新GCN和GAT在TUNE数据集上的status
for method in ["GCN", "GAT"]:
    for dataset in ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]:
        if method in data["table"] and dataset in data["table"][method]:
            data["table"][method][dataset]["status"] = "missing"
            data["table"][method][dataset]["reason"] = "骨干不支持 (无 TUNE checkpoint 权重)"

# 更新GAD-NR在TUNE数据集上的status
for dataset in ["photo", "computer", "weibo", "tolokers", "questions", "reddit"]:
    if "GAD-NR" in data["table"] and dataset in data["table"]["GAD-NR"]:
        data["table"]["GAD-NR"][dataset]["status"] = "failed"
        data["table"]["GAD-NR"][dataset]["reason"] = "骨干不支持 (TUNE 上 loss=nan 发散)"

# 保存更新后的文件
with open("/media/lixin/新加卷/数据集/test/StructCP/outputs/compare_all/comparison_table_updated.json", "w") as f:
    json.dump(data, f, indent=2)

print("Updated comparison_table.json saved to outputs/compare_all/comparison_table_updated.json")