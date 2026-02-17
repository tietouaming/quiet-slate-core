"""跨尺度机器学习子系统。

目标：
1. 由微结构相场（phi/c/eta/epspeq/应力）预测宏观腐蚀与力学指标；
2. 提供可复现实验的数据集构建、训练、推理接口；
3. 提供反向设计优化器（基于 surrogate 的候选搜索）。
"""

from .dataset import (
    MacroTargetSpec,
    MultiScaleDatasetConfig,
    build_multiscale_dataset_from_cases,
    load_multiscale_dataset_npz,
    save_multiscale_dataset_npz,
)
from .features import extract_micro_descriptors, extract_micro_tensor
from .inverse_design import (
    DesignVariableSpec,
    InverseDesignConfig,
    InverseDesignResult,
    run_inverse_design,
)
from .models import MacroCorrosionPredictor, MultiScaleModelConfig, load_multiscale_model, save_multiscale_model
from .train import TrainingConfig, train_multiscale_model

__all__ = [
    "MacroTargetSpec",
    "MultiScaleDatasetConfig",
    "build_multiscale_dataset_from_cases",
    "load_multiscale_dataset_npz",
    "save_multiscale_dataset_npz",
    "extract_micro_descriptors",
    "extract_micro_tensor",
    "DesignVariableSpec",
    "InverseDesignConfig",
    "InverseDesignResult",
    "run_inverse_design",
    "MacroCorrosionPredictor",
    "MultiScaleModelConfig",
    "load_multiscale_model",
    "save_multiscale_model",
    "TrainingConfig",
    "train_multiscale_model",
]

