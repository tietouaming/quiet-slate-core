"""跨尺度机器学习子系统。

目标：
1. 由微结构相场（phi/c/eta/epspeq/应力）预测宏观腐蚀与力学指标；
2. 提供可复现实验的数据集构建、训练、推理接口；
3. 提供反向设计优化器（基于 surrogate 的候选搜索）。
"""

from .dataset import (
    MacroTargetSpec,
    MultiFidelityDatasetConfig,
    MultiScaleDatasetConfig,
    build_multifidelity_dataset_from_cases,
    build_multiscale_dataset_from_cases,
    load_multiscale_dataset_npz,
    save_multiscale_dataset_npz,
)
from .advanced_models import (
    AdvancedModelConfig,
    HybridUAFNOMacroPredictor,
    MultiFidelityHybridPredictor,
    build_advanced_model,
    load_advanced_model,
    save_advanced_model,
)
from .advanced_train import AdvancedTrainingConfig, train_advanced_multiscale_model
from .active_learning import ActiveLearningConfig, run_active_learning_loop
from .closed_loop_design import ClosedLoopDesignConfig, run_closed_loop_design
from .features import extract_micro_descriptors, extract_micro_tensor
from .inverse_design import (
    DesignVariableSpec,
    InverseDesignConfig,
    InverseDesignResult,
    run_inverse_design,
)
from .models import MacroCorrosionPredictor, MultiScaleModelConfig, load_multiscale_model, save_multiscale_model
from .physics_metrics import PhysicsMetricConfig, evaluate_physics_metrics
from .train import TrainingConfig, train_multiscale_model
from .visualization import (
    plot_active_learning_progress,
    plot_parity_by_target,
    plot_training_history,
    plot_uncertainty_calibration,
)

__all__ = [
    "MacroTargetSpec",
    "MultiFidelityDatasetConfig",
    "MultiScaleDatasetConfig",
    "build_multifidelity_dataset_from_cases",
    "build_multiscale_dataset_from_cases",
    "load_multiscale_dataset_npz",
    "save_multiscale_dataset_npz",
    "AdvancedModelConfig",
    "HybridUAFNOMacroPredictor",
    "MultiFidelityHybridPredictor",
    "build_advanced_model",
    "save_advanced_model",
    "load_advanced_model",
    "AdvancedTrainingConfig",
    "train_advanced_multiscale_model",
    "ActiveLearningConfig",
    "run_active_learning_loop",
    "ClosedLoopDesignConfig",
    "run_closed_loop_design",
    "PhysicsMetricConfig",
    "evaluate_physics_metrics",
    "plot_training_history",
    "plot_parity_by_target",
    "plot_uncertainty_calibration",
    "plot_active_learning_progress",
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
