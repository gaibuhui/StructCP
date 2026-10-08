"""对比方法注册表 (全自动串跑框架)。

每个条目: method_name -> 返回 adapter 实例的工厂函数 (或类)。
已接入: BWGNN / GADT3 / TA-GGAD / ARC (复用既有 adapter 逻辑, 包装为 BaselineAdapter)。
待接入: 列表里其余 20+ 方法, 写在各自 baselines/<NAME>/adapter.py 后在此注册。

注意: 本项目统一评测协议为 StructCP 的 novel-normality 偏移
(data.loader.load_gad), 所有方法都在同一 GADData 上算分, 保证可比。
"""
from __future__ import annotations

import torch

from .baseline_base import BaselineAdapter

# ---------------------------------------------------------------------------
# 已接入方法 (复用既有 adapter 实现, 做统一接口包装)
# ---------------------------------------------------------------------------


class _FrozenGNNEncoderAdapter(BaselineAdapter):
    """通用冻结 GNN 编码器适配器 (BWGNN/GCN/GAT/GAD-NR 共用)。

    直接加载 StructCP 已训练好的 checkpoint/<NAME>_<dataset>.pth,
    用 anomaly_score (softmax 异常类) 产分, **越大越异常**。
    作为 GAD 监督/无监督基线对比, 不重新训练 (对齐论文 frozen-backbone 设定)。
    """
    #: 子类/工厂设置
    model_key: str = "BWGNN"      # checkpoint 前缀, 如 BWGNN / GCN / GAT / GADNR
    module_path: str = "models.bwgnn.BWGNN"
    hidden: int = 128

    def __init__(self, device: str = "cpu", **kw):
        super().__init__(device=device)

    def _build_model(self, in_channels):
        import importlib, inspect
        mod_name, cls_name = self.module_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        sig = inspect.signature(cls.__init__)
        kwargs = dict(in_channels=in_channels, hidden_channels=self.hidden,
                      out_channels=2, d=2, heads=8, att_out=16)
        # 只保留类 __init__ 接受的参数, 避免 BWGNN/GCN 拒绝 heads/att_out
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return cls(**filtered)

    def fit(self, data):
        import os, torch as _t
        key = self.model_key.lower()  # 实际文件名前缀为小写, 如 bwgnn/gcn/gat/gadnr
        name = data.name
        candidates = [
            f"checkpoint/{key}_{name}.pth",
            f"checkpoint/{key}_{name}_h{self.hidden}.pth",
            f"checkpoint/{key}_{name}_homo.pth",
            f"checkpoint/{self.model_key}_{name}.pth",
        ]
        ckpt = next((c for c in candidates if os.path.exists(c)), None)
        if ckpt is None:
            raise FileNotFoundError(
                f"找不到 {self.model_key} 权重, 尝试过: {candidates}"
            )
        sd = _t.load(ckpt, map_location=self.device, weights_only=False)
        in_ch = int(sd.get("in_channels", data.x.shape[1]))
        # 从 state_dict 推断真实 hidden (部分权重如 BWGNN 实为 h64, 文件名 h128 是误导)
        hidden = self.hidden
        for k, v in sd["state_dict"].items():
            if k.endswith("lin_out1.bias"):
                hidden = int(v.shape[0]); break
        self.hidden = hidden
        self.model = self._build_model(in_ch).to(self.device)
        self.model.load_state_dict(sd["state_dict"])
        self.model.freeze()
        return self

    @torch.no_grad()
    def score(self, data):
        return self.model.anomaly_score(
            data.x.to(self.device), data.edge_index.to(self.device)
        ).cpu()


class _BWGNNAdapter(_FrozenGNNEncoderAdapter):
    name, model_key, module_path = "BWGNN", "BWGNN", "models.bwgnn.BWGNN"


class _GCNAdapter(_FrozenGNNEncoderAdapter):
    name, model_key, module_path = "GCN", "GCN", "models.gcn_encoder.GCNEncoder"


class _GATAdapter(_FrozenGNNEncoderAdapter):
    name, model_key, module_path = "GAT", "GAT", "models.gat_encoder.GATEncoder"


def _make_bwgnn(**kw):
    return _BWGNNAdapter(**kw)


def _make_gcn(**kw):
    return _GCNAdapter(**kw)


def _make_gat(**kw):
    return _GATAdapter(**kw)


# 复用既有 adapter 的薄包装器 -------------------------------------------------

def _import_arc_adapter():
    """取 ARCAdapter（ARC 与 TA-GGAD 共用）。

    2026-09-20 状态说明：`baselines/ta_ggad_pyg/`（PyG 重写版）在本机已不存在，
    仅遗留 `baselines/ta_ggad_extracted.txt` 与官方原仓 `baselines/TA-GGAD/`。
    因此 ARC / TA-GGAD 两条基线当前无法实例化；`outputs/` 中的历史结果来自
    该模块尚存时的运行。补回方式（二选一）：
      (a) 从原开发机恢复 `baselines/ta_ggad_pyg/{__init__,adapter,model,tta}.py`；
      (b) 参照 `baselines/TA-GGAD/model.py` 重写一个 PyG 版 adapter，
          暴露 `ARCAdapter(dim, device)` 与 `.fit(data)/.score(data)` 接口。
    """
    try:
        from baselines.ta_ggad_pyg.adapter import ARCAdapter
    except ImportError as e:
        raise ImportError(
            "ARC/TA-GGAD 适配器不可用：缺少 baselines/ta_ggad_pyg/adapter.py。"
            "见 adapters/registry.py::_import_arc_adapter 的说明。"
            f"（原始错误：{e}）"
        ) from e
    return ARCAdapter


def _make_arc(dim=None, **kw):
    ARCAdapter = _import_arc_adapter()
    return ARCAdapter(dim or 100, device=kw.get("device", "cpu"))


def _make_gadt3(ndim_s=None, **kw):
    from baselines.gadt3_pyg.adapter import GADT3Adapter
    return GADT3Adapter(ndim_s or 100, device=kw.get("device", "cpu"))


def _make_taggad(**kw):
    # TA-GGAD 官方检测器核心即 ARC 模型, 与 ARC 共用同一实现。
    return _make_arc(**kw)


class _LazyDimAdapter(BaselineAdapter):
    """延迟构建适配器: fit(data) 时用 data.x.shape[1] 作为真实输入维构造底层 adapter。

    解决 ARC/GADT3/TA-GGAD 原 make 函数中 dim/ndim_s 写死 100 导致的维度不匹配;
    同时把 GADT3 的 fit_source+fit_target 统一映射到 fit+fit_target 接口。
    """

    def __init__(self, factory, needs_source_fit: bool = False, device: str = "cpu", **kw):
        super().__init__(device=device)
        self._factory = factory
        self._needs_source_fit = needs_source_fit
        self._kw = kw
        self._inner = None

    def fit(self, data):
        in_dim = int(data.x.shape[1])
        self._inner = self._factory(in_dim, device=self.device, **self._kw)
        # 先训练 (GADT3 用 fit_source, 其余用 fit)
        if self._needs_source_fit:
            self._inner.fit_source(data)
        else:
            self._inner.fit(data)
        # 测试时自适应 (统一接口)
        if hasattr(self._inner, "fit_target"):
            self._inner.fit_target(data)
        return self

    @torch.no_grad()
    def score(self, data):
        return self._inner.score(data)


def _make_dominant(**kw):
    from .dominant import DOMINANTAdapter
    kw.setdefault("device", "cpu")
    return DOMINANTAdapter(**kw)


def _make_unprompt(**kw):
    from .unprompt import _make_unprompt as _f
    return _f(**kw)


def _make_cola(**kw):
    from .cola import CoLAAdapter
    kw.setdefault("device", "cpu")
    return CoLAAdapter(**kw)


def _make_gadnr(**kw):
    from .gadnr import GADNRAdapter
    kw.setdefault("device", "cpu")
    return GADNRAdapter(**kw)


def _make_arc_lazy(**kw):
    ARCAdapter = _import_arc_adapter()
    return _LazyDimAdapter(lambda d, **a: ARCAdapter(d, **a), device=kw.get("device", "cpu"))


def _make_gadt3_lazy(**kw):
    from baselines.gadt3_pyg.adapter import GADT3Adapter
    return _LazyDimAdapter(lambda d, **a: GADT3Adapter(d, **a),
                           needs_source_fit=True, device=kw.get("device", "cpu"))


def _make_taggad_lazy(**kw):
    # TA-GGAD 官方检测器核心即 ARC 模型 (见 baselines/ta_ggad_pyg/model.py 标题
    # "TA-GGAD 的 ARC 模型"; 官方基准 run_gadt3_vs_taggad.py 亦用 ARCAdapter 产出
    # TA-GGAD 结果)。因此 TA-GGAD 与 ARC 共用同一实现, 结果数值一致, 不再有
    # "TAGGADAdapter" 独立类。对比表将二者并列并注明等价关系。
    # 注: ta_ggad_pyg 源码缺失时, 二者会同时不可用, 见 _import_arc_adapter()。
    return _make_arc_lazy(**kw)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

REGISTRY = {
    "BWGNN": _make_bwgnn,
    "GCN": _make_gcn,
    "GAT": _make_gat,
    "ARC": _make_arc_lazy,
    "GADT3": _make_gadt3_lazy,
    "TA-GGAD": _make_taggad_lazy,
    "DOMINANT": _make_dominant,
    "UNPrompt": _make_unprompt,
    "CoLA": _make_cola,
    "GAD-NR": _make_gadnr,
}

# 论文对比表中全部方法 (含尚未接入的, 标记 TODO 由后续 adapter 补齐)
ALL_METHODS = [
    # GGAD
    "TA-GGAD", "ARC", "UNPrompt", "AnomalyGFM",
    # 监督
    "GCN", "GAT", "BGNN", "BWGNN", "GHRN", "CAGAD",
    # 半监督
    "S-GAD",
    # 无监督
    "DOMINANT", "CoLA", "HCM-A", "TAM", "SmoothGNN", "SpaceGNN", "GCTAM",
    # 其他
    "CONSISGAD", "GAD-NR", "GADT3", "TUNE",
]


def register(name: str, factory):
    REGISTRY[name] = factory


def build(name: str, **kw) -> BaselineAdapter:
    if name not in REGISTRY:
        raise KeyError(f"方法 {name} 尚未接入 (REGISTRY 无条目)。请先写 adapter 并 register。")
    return REGISTRY[name](**kw)
