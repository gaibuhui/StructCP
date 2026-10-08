"""指标计算：分类指标 + AUC 一站式封装。

FPR/TPR/Precision/F1 语义与 `adapters/conformal.evaluate_coverage` 完全一致
（此处仅为多返回合并提供便利，不重实现阈值逻辑）。
"""
from __future__ import annotations

import numpy as np
import torch

from adapters.conformal import evaluate_coverage


def auc_scores(y_true, y_score) -> dict[str, float]:
    """AUROC / AUPRC（y_score 越大越异常）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true).ravel()
    s = np.asarray(y_score.cpu().numpy() if torch.is_tensor(y_score) else y_score).ravel()
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPRC": float(average_precision_score(y, s)),
    }


def classification_metrics(y_true, y_score, threshold) -> dict[str, float]:
    """AUROC + AUPRC + evaluate_coverage 的全部字段合并。"""
    out = auc_scores(y_true, y_score)
    out.update(evaluate_coverage(
        torch.as_tensor(y_score, dtype=torch.float32).flatten(),
        torch.as_tensor(y_true, dtype=torch.long).flatten(),
        torch.as_tensor(threshold, dtype=torch.float32),
    ))
    return out
