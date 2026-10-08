"""utils —— StructCP 共享工具库。

提供深度无关的项目根定位、统一随机种子、配置加载、checkpoint 加载、
结果读写/聚合/manifest、指标计算等基础设施，供 `scripts/**` 复用。

使用约定（重构后）：任何脚本启动时先
    from utils.paths import project_root
    ROOT = project_root(__file__)
（或使用每个脚本内注入的 `_structcp_root()` bootstrap，二者等价。）
"""
from .paths import project_root
from .seed import set_seed
from .config import load_config_yaml
from .checkpoint import load_frozen_detector
from .results import save_json, load_json, aggregate_metric, ResultManifest
from .metrics import classification_metrics

__all__ = [
    "project_root",
    "set_seed",
    "load_config_yaml",
    "load_frozen_detector",
    "save_json",
    "load_json",
    "aggregate_metric",
    "ResultManifest",
    "classification_metrics",
]
