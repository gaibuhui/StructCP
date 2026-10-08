"""Step 1: 严格子集采样封装 (解决显存爆炸核心关键, 论文可直接写)。

在 data.loader.load_gad 基础上叠加论文固定协议:
  - 分层随机采样, 保持原图异常比例不变 (GAD 公平性)
  - 剔除 degree > max_degree 的超级 hub (破坏子空间结构)
  - 固定规模: 源域正常 = n_train_norm, 测试集 = n_test (含正常/异常)

输出: GADData 子集 (特征标准化已在 loader 内完成, F 维不变)。
"""
from __future__ import annotations

import numpy as np

from data.loader import load_gad


def stratified_subset(
    data,
    n_train_norm: int = 3000,
    n_test: int = 10000,
    max_degree: int | None = None,
    degree_quantile: float = 0.98,
    seed: int = 42,
    cal_ratio: float = 0.2,
):
    """对 load_gad 返回的 GADData 做论文固定子集裁剪。

    Hub 裁剪策略 (论文可写, 公平且适配异质度分布):
      默认按度分位动态阈值 max_degree = percentile(deg, degree_quantile),
      对稀疏图 (Elliptic) 几乎不裁剪, 对稠密二分图 (Amazon/TFinance) 只
      剔除最极端 top (100-q)% 的超级 hub, 避免固定阈值砍光主体节点。
      仍可用 --max_degree 强制固定阈值 (论文方案原值 200)。

    Args:
        data:             原始 GADData (全量或 TFS 子采样后)
        n_train_norm:     源域正常训练节点数 (默认 3000)
        n_test:           测试集节点数 (默认 10000, 含正常/异常)
        max_degree:       hub 固定阈值 (None 时用语义分位)
        degree_quantile:  动态阈值分位 (默认 0.98)
        seed:             固定随机种子
        cal_ratio:        从训练正常节点中再分 cal_ratio 作校准
    """
    rng = np.random.RandomState(seed)
    # 采样在 CPU 上进行, 调用方可能传入 cuda 张量
    x = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    ei = data.edge_index.cpu().numpy().astype(np.int64)
    n = x.shape[0]

    # --- hub 裁剪: 计算度, 按分位或固定阈值剔除超级 hub ---
    deg = np.bincount(ei.ravel(), minlength=n)
    if max_degree is None:
        thr = float(np.quantile(deg, degree_quantile))
        thr = max(int(thr), 1)
    else:
        thr = int(max_degree)
    keep = deg <= thr
    # 若裁剪后正常节点不足以支撑 n_train_norm, 放宽阈值至 99.5 分位保底
    n_norm = int(keep[y == 0].sum())
    if n_norm < n_train_norm and max_degree is None:
        thr2 = max(int(np.quantile(deg, 0.995)), thr)
        keep = deg <= thr2
        thr = thr2
    if keep.sum() < n:
        print(f"[subset] hub 裁剪: {n} -> {int(keep.sum())} "
              f"(剔除 degree>{thr} 的 {int((~keep).sum())} 个 hub)")
    idx = np.where(keep)[0]
    remap = -np.ones(n, dtype=np.int64)
    remap[idx] = np.arange(idx.shape[0])
    sub_ei = ei[:, keep[ei[0]] & keep[ei[1]]]
    sub_ei = np.vstack([remap[sub_ei[0]], remap[sub_ei[1]]]).astype(np.int64)
    x = x[idx]
    y = y[idx]
    n = x.shape[0]

    normal = np.where(y == 0)[0]
    abnormal = np.where(y == 1)[0]

    # --- 训练正常: 从正常节点抽 n_train_norm, 但始终为测试集预留正常节点 ---
    # 测试集正常预留量: 至少 50 个, 保证 AUC 可计算 (少样本对照公平性)
    test_normal_reserve = max(50, int(n_test * 0.5))
    n_norm_avail = len(normal)
    n_train_actual = min(n_train_norm, max(0, n_norm_avail - test_normal_reserve))
    if n_train_actual > 0:
        tr_norm = rng.choice(normal, size=n_train_actual, replace=False)
    else:
        tr_norm = normal[:0]
    tr_norm_set = set(tr_norm.tolist())

    # cal 从 tr_norm 中再分 (同分布, 不影响 novel 协议)
    n_cal = int(len(tr_norm) * cal_ratio)
    rng.shuffle(tr_norm)
    cal_idx = tr_norm[:n_cal]
    train_idx = tr_norm[n_cal:]

    # --- 测试集: 始终同时含正常与异常两类 (保证评测指标有效, 避免 AUC=nan) ---
    rest_normal = np.array([i for i in range(n) if i not in tr_norm_set], dtype=np.int64)
    abn_ratio = len(abnormal) / max(1, n)
    # 异常数: 维持原比例但至少 50 个 (防止极不均衡), 且不超过可用异常
    n_ab = int(n_test * abn_ratio)
    n_ab = min(len(abnormal), max(50, n_ab))
    # 正常数: 补足到 n_test, 不超过可用剩余正常
    n_no = min(len(rest_normal), max(50, n_test - n_ab))
    # 若正常不足, 反向缩减异常数以满足 n_test
    if n_no < max(50, n_test - n_ab):
        n_ab = min(len(abnormal), max(50, n_test - n_no))
    if len(abnormal) > n_ab:
        te_ab = rng.choice(abnormal, size=n_ab, replace=False)
    else:
        te_ab = abnormal
    if len(rest_normal) > n_no:
        te_no = rng.choice(rest_normal, size=n_no, replace=False)
    else:
        te_no = rest_normal
    te_idx = np.concatenate([te_ab, te_no])

    def _mask(arr):
        m = np.zeros(n, dtype=bool)
        m[arr] = True
        return m

    from data.loader import GADData
    return GADData(
        x=data.x.new_tensor(x),
        edge_index=data.edge_index.new_tensor(sub_ei),
        y=data.y.new_tensor(y),
        train_mask=data.train_mask.new_tensor(_mask(train_idx)),
        cal_mask=data.train_mask.new_tensor(_mask(cal_idx)),
        test_mask=data.train_mask.new_tensor(_mask(te_idx)),
        novel_mask=data.novel_mask.new_tensor(
            data.novel_mask.cpu().numpy().astype(bool)[idx]
        ) if data.novel_mask is not None else data.train_mask.new_tensor(np.zeros(n, bool)),
        name=data.name + f"_sub{n_train_norm}-{n_test}",
    )
