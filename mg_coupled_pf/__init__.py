"""项目对外入口（中文注释版）。

导出最常用的三个对象：
- `SimulationConfig`：总配置数据结构
- `load_config`：配置读取函数
- `CoupledSimulator`：主求解器
"""

from .config import SimulationConfig, load_config
from .simulator import CoupledSimulator
from .validation import ConfigAuditReport, audit_config, save_audit_report, summarize_audit_report

__all__ = [
    "SimulationConfig",
    "load_config",
    "CoupledSimulator",
    "ConfigAuditReport",
    "audit_config",
    "save_audit_report",
    "summarize_audit_report",
]
