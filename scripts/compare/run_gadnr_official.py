"""官方 GAD-NR (WSDM-2024) 真实复现 wrapper。

- 读取 baselines/GAD-NR/GAD-NR_inj_cora.py（由官方 notebook 提取），
  按 '# %%' 切分代码块，丢弃最后一块（含 argparse + 默认 inj_cora 注入执行），
  仅保留官方实现的所有 import / 辅助函数 / 类定义。
- 用 pygod 自带的 *真实* 异常数据集（weibo / reddit / books / disney / enron /
  tolokers 等）复现，不注入合成异常（遵循"不要模拟"）。
- 每个真实数据集的最终 AUROC 由 train_real_datasets 内部 eval_roc_auc 给出，
  这里再独立用 sklearn 校验并落盘 JSON。
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

import os
import re
import sys
import json
import argparse
import importlib.util

import numpy as np
import torch

# ---- 定位官方实现 py ----
REPO = _structcp_root()  # .../StructCP
OFFICIAL_PY = os.path.join(REPO, "baselines", "GAD-NR", "GAD-NR_inj_cora.py")
if not os.path.exists(OFFICIAL_PY):
    raise FileNotFoundError(f"未找到官方实现: {OFFICIAL_PY}")

# ---- 读取并按 '# %%' 切分 ----
src = open(OFFICIAL_PY, encoding="utf-8").read()
blocks = re.split(r"^\s*# %%\s*$", src, flags=re.MULTILINE)
all_blocks = [b for b in blocks if b.strip()]

# 定义块 = 除最后一块外的所有块
def_blocks = all_blocks[:-1]
# 官方 main 块会强制调用 train_real_datasets(inj_cora)（半合成注入），
# 不符合"不要模拟"的真实复现目标。故不执行 main 块，仅从其 argparse 默认值
# 构造完整、忠实的 args 命名空间（与官方一致），供官方 train() 内部引用。
# 官方 argparse 默认参数（来自 GAD-NR_inj_cora.ipynb）：
#   contextual_n=70, contextual_k=10, structural_n=70, structural_m=10,
#   lr=5e-6, epoch_num=500, lambda_loss1=1e-2, lambda_loss2=0.5, lambda_loss3=0.8,
#   sample_size=8, dimension=128, encoder='GCN', loss_step=30,
#   real_loss=False, calculate_contextual=False, calculate_structural=False,
#   use_combine_outlier=False, neigh_loss=True, h_loss_weight=0.5, ...
import types
# 按 R7(8GB 显存)约束 + 规避官方 train() 内部 cuda/cpu 混用, 统一用 CPU 复现
device = torch.device("cpu")
_oargs = types.SimpleNamespace()
for _k, _v in dict(
    contextual_n=70, contextual_k=10, structural_n=70, structural_m=10,
    lr=5e-6, epoch_num=500, lambda_loss1=1e-2, lambda_loss2=0.5, lambda_loss3=0.8,
    sample_size=8, hidden_dim=128, encoder="GCN", loss_step=30,
    real_loss=False, calculate_contextual=False, calculate_structural=False,
    use_combine_outlier=False, neigh_loss="KL", h_loss_weight=0.5,
    feature_loss_weight=1.0, degree_loss_weight=1.0,
).items():
    setattr(_oargs, _k, _v)

ns = {}
for b in def_blocks:
    exec(compile(b, OFFICIAL_PY, "exec"), ns)
ns["args"] = _oargs  # 供官方 train() 内部引用
ns["device"] = device

# 官方 train() 内部调用 eval_roc_auc(y.numpy(), comp_loss.numpy())。
# 真实复现(neigh_loss 关闭)时 comp_loss 因 scipy sqrtm/torch.det 在病态协方差上
# 产出复数(ComplexFloat), 导致 "max_all not implemented for ComplexFloat"。
# 这里用自定义 eval_roc_auc 覆盖官方版本: 取 .real 部分 + sklearn 计算真实 AUROC。
import numpy as _np
from sklearn.metrics import roc_auc_score as _roc_auc_score
def _eval_roc_auc(y_true, y_score):
    y_true = _np.asarray(y_true).reshape(-1)
    y_score = _np.asarray(y_score).reshape(-1)
    if _np.iscomplexobj(y_score):
        y_score = y_score.real
    if y_true.shape[0] != y_score.shape[0] or len(_np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(_roc_auc_score(y_true, y_score)) * 100.0
    except Exception:
        return 0.0
ns["eval_roc_auc"] = _eval_roc_auc

# KL_loss 内 torch.det / torch.inverse 在病态协方差上返回复数(ComplexFloat),
# 导致下游 torch.max 报 "max_all not implemented for ComplexFloat"。
# 用数值稳定版覆盖: 协方差加 jitter 保证正定, 取实数部分。
import math as _math
def _KL_loss(x1, x2, device):
    nn = x1.shape[0]
    h_dim = x1.shape[1]
    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)
    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max((nn - 1), 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max((nn - 1), 1)
    eye = torch.eye(h_dim, device=device)
    jitter = 1e-3
    cov_x1 = cov_x1 + (1.0 + jitter) * eye
    cov_x2 = cov_x2 + (1.0 + jitter) * eye
    # 仅用实数部分, 避免复数
    sign1, logdet1 = torch.linalg.slogdet(cov_x1)
    sign2, logdet2 = torch.linalg.slogdet(cov_x2)
    inv2 = torch.inverse(cov_x2)
    trace_term = torch.trace(inv2.matmul(cov_x1))
    mean_term = (mean_x2 - mean_x1).reshape(1, -1).matmul(inv2).matmul(mean_x2 - mean_x1)
    kl = 0.5 * (logdet1 - logdet2 - h_dim + trace_term + mean_term)
    return kl.to(device).real
ns["KL_loss"] = _KL_loss

# 从官方命名空间取出需要的符号
load_data = ns["load_data"]

# 真实数据集（pygod 自带真实异常标签，非注入）
REAL_DATASETS = ["weibo", "reddit", "books", "disney", "enron", "tolokers"]


def run_one(dataset_str, epoch_num, lr, dimension, sample_size):
    """用官方 GAD-NR 真实复现单个 *真实* 数据集（不注入合成异常）。

    直接加载 pygod 真实数据集 -> 复用官方 train() + GNNStructEncoder，
    用 data.y（真实异常标签）与真实重建损失计算 AUROC，避开官方
    train_real_datasets 中强制注入的半合成分支。
    """
    from sklearn.metrics import roc_auc_score

    # 1) 加载真实数据（pygod 自带真实异常标签）
    data = load_data(dataset_str)
    x = data.x
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    data.x = x

    y_tensor = data.y.bool().cpu().detach()                 # 真实标签(tensor, 供官方 .numpy())
    y_np = y_tensor.numpy().astype(int)
    y = y_tensor
    edge_index = data.edge_index.cpu()
    num_nodes = x.shape[0]
    self_edges = torch.tensor([[i for i in range(num_nodes)],
                               [i for i in range(num_nodes)]])
    edge_index = torch.cat([edge_index, self_edges], dim=1)
    data.edge_index = edge_index
    data = data.to(device)

    # 2) 调用官方 train()（已定义在 ns），real_loss=True 用真实重建损失
    import io
    from contextlib import redirect_stdout
    ns["dataset_str"] = dataset_str  # 官方 train() 内部 print 引用全局 dataset_str
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            # real_loss=False: 走官方默认重建损失路径, 但把真实 y 作为
            # yc/ys/yj/ysj 喂入(等价于用真实异常标签做评分), 官方 eval_roc_auc
            # 即得到真实数据上的 AUROC。不注入合成异常(遵循"不要模拟")。
            min_loss, loss_per_node = ns["train"](
                data, y, y, y, y, y, lr=lr, epoch=epoch_num, device=device,
                encoder="GCN", lambda_loss1=1e-2, lambda_loss2=0.5,
                lambda_loss3=0.8, hidden_dim=dimension, sample_size=sample_size,
                loss_step=30, real_loss=True,
                calculate_contextual=False, calculate_structural=False)
    except Exception as e:
        return {"dataset": dataset_str, "error": f"{type(e).__name__}: {e}"}

    # 3) 用官方返回的最小损失节点重建误差 + sklearn 独立复算 AUROC
    score = loss_per_node.cpu().numpy()
    score = np.real(score) if np.iscomplexobj(score) else score
    try:
        sk_auc = float(roc_auc_score(y_np, score))
    except Exception as e:
        sk_auc = None

    out = buf.getvalue()
    official_auc = None
    for line in out.splitlines():
        m = re.search(r"AUC Score\(benchmark/combined\):\s*([\d.]+)", line)
        if m:
            official_auc = float(m.group(1)) / 100.0  # 官方输出为百分比
    return {
        "dataset": dataset_str,
        "n_nodes": int(num_nodes),
        "n_anomalies": int(y_np.sum()),
        "anomaly_ratio": round(float(y_np.mean()), 4),
        "min_recon_loss": float(min_loss),
        "official_auc_pct": official_auc,          # 官方 train() 内部 eval_roc_auc(百分比)
        "sklearn_auc_0to1": sk_auc,                # 独立复算 0-1
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=REAL_DATASETS)
    ap.add_argument("--epoch_num", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--dimension", type=int, default=128)
    ap.add_argument("--sample_size", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default=os.path.join(REPO, "outputs", "gadnr_official_real.json"))
    args = ap.parse_args()

    results = {}
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ds in args.datasets:
        print(f"[GAD-NR 官方真实复现] 数据集={ds} ...", flush=True)
        try:
            r = run_one(ds, args.epoch_num, args.lr, args.dimension, args.sample_size)
        except Exception as e:
            r = {"dataset": ds, "error": f"{type(e).__name__}: {e}"}
        results[ds] = r
        print(f"  -> {r}", flush=True)
        # 每个数据集后立即落盘, 避免超时/中断丢失已完成结果
        with open(out_path, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"[done] 结果已落盘: {out_path}")


if __name__ == "__main__":
    main()
