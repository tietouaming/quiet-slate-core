"""配置模块（中文注释版）。

本文件统一定义了项目中所有可配置参数的数据结构，包含：
1. 几何与网格参数（DomainConfig）
2. 时间推进与数值稳定参数（NumericsConfig）
3. 腐蚀相场参数（CorrosionConfig）
4. 孪晶相场参数（TwinningConfig）
5. 晶体塑性参数（CrystalPlasticityConfig）
6. 力学参数（MechanicsConfig）
7. 机器学习代理模型参数（MLConfig）
8. 扩展耦合接口参数（ExtensionsConfig）
9. 运行时输出参数（RuntimeConfig）

使用方式：
- 通过 `load_config(path)` 从 YAML 读取并覆盖默认值。
- 若未提供滑移系配置文件，则回退到内置二维 HCP 投影滑移系。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DomainConfig:
    """二维计算域与初始几何配置。"""
    nx: int = 256
    ny: int = 160
    lx_um: float = 400.0
    ly_um: float = 250.0
    solid_phase_threshold: float = 0.5
    mesh_refine_near_notch: bool = False
    mesh_refine_strength_x: float = 8.0
    mesh_refine_strength_y: float = 6.0
    mesh_refine_sigma_x_um: float = 30.0
    mesh_refine_sigma_y_um: float = 20.0
    mesh_refine_samples: int = 20000
    initial_geometry: str = "notch"
    notch_tip_x_um: float = 80.0
    notch_center_y_um: float = 125.0
    notch_depth_um: float = 55.0
    notch_half_opening_um: float = 30.0
    interface_width_um: float = 2.0
    initial_pit_radius_um: float = 1.5
    add_initial_pit_seed: bool = True
    half_space_direction: str = "x"
    half_space_interface_x_um: float = -1.0
    half_space_interface_y_um: float = -1.0
    half_space_solid_on_lower_side: bool = True
    add_half_space_triangular_notch: bool = False
    half_space_notch_tip_x_um: float = 140.0
    half_space_notch_center_y_um: float = 125.0
    half_space_notch_depth_um: float = 100.0
    half_space_notch_half_opening_um: float = 25.0
    half_space_notch_open_to_positive_x: bool = True
    half_space_notch_sharpness_um: float = 0.25
    tip_zoom_width_um: float = 12.0
    tip_zoom_height_um: float = 9.0


@dataclass
class NumericsConfig:
    """数值求解与时间推进配置。"""
    dt_s: float = 1e-4
    n_steps: int = 3000
    save_every: int = 100
    mechanics_substeps: int = 12
    mechanics_adaptive_substeps: bool = False
    mechanics_min_substeps: int = 2
    mechanics_residual_tol: float = 0.0
    mechanics_residual_rel_tol: float = 0.0
    mechanics_relaxation: float = 0.22
    seed: int = 42
    use_torch_compile: bool = True
    mixed_precision: bool = False
    inference_mode: bool = True
    dtype: str = "float32"
    diffusion_auto_substeps: bool = True
    diffusion_dt_safety: float = 0.9
    diffusion_substeps_cap: int = 256
    diffusion_integrator: str = "subcycled_euler"
    diffusion_sts_nu: float = 0.05
    diffusion_sts_max_stages: int = 128
    mechanics_update_every: int = 1
    phi_integrator: str = "imex_helmholtz"
    eta_integrator: str = "imex_variable"
    imex_solver_iters: int = 80
    imex_solver_tol_abs: float = 1e-8
    imex_solver_tol_rel: float = 1e-5
    imex_relaxation: float = 1.0
    # 标量隐式线性求解器：
    # - pcg: 仅用预条件共轭梯度
    # - bicgstab: 仅用 BiCGStab
    # - auto: 先 PCG，若不收敛自动回退 BiCGStab
    scalar_linear_solver: str = "auto"
    mechanics_solver: str = "hybrid_cg_relax"
    mechanics_cg_max_iters: int = 120
    mechanics_cg_tol_abs: float = 5.0
    mechanics_cg_tol_rel: float = 5e-3
    mechanics_cg_diag_shift: float = 1e-6
    mechanics_krylov_method: str = "gmres"
    mechanics_gmres_restart: int = 24
    mechanics_gmres_max_restarts: int = 6
    mechanics_krylov_fail_streak_to_pause: int = 3
    mechanics_krylov_pause_steps: int = 30
    mechanics_use_ml_initial_guess: bool = False
    # 标量场离散边界条件：
    # - neumann: 零法向梯度/零通量（默认）
    # - periodic: 周期边界
    # - dirichlet0: 零值 Dirichlet
    scalar_bc: str = "neumann"
    # 力学自适应触发阈值（当 mechanics_update_every>1 时可提前触发力学更新）
    mechanics_trigger_phi_max_delta: float = 0.0
    mechanics_trigger_eta_max_delta: float = 0.0


@dataclass
class CorrosionConfig:
    """腐蚀相场与浓度扩散参数。"""
    # From Kovacevic et al. (2023), with normalized concentration form.
    D_l_m2_s: float = 1.0e-10
    D_s_m2_s: float = 1.0e-13
    c_l_eq_mol_L: float = 0.57
    c_s_eq_mol_L: float = 71.44
    # Normalized concentration (cMg) used in governing equations.
    c_l_eq_norm: float = 0.0
    c_s_eq_norm: float = 1.0
    cMg_init_liquid: float = 0.0
    cMg_init_solid: float = 1.0
    cMg_min: float = 0.0
    cMg_max: float = 1.0
    gamma_J_m2: float = 0.5
    interface_thickness_um: float = 4.0
    A_J_m3: float = 6.0e7
    temperature_K: float = 310.15
    molar_volume_m3_mol: float = 13.998e-6
    gas_constant_J_mol_K: float = 8.314462618
    L0: float = 5.0e-8
    pitting_beta: float = 0.75
    pitting_alpha: float = 3.0
    pitting_N: int = 2
    pitting_min_factor: float = 0.2
    pitting_max_factor: float = 5.0
    yield_strain_for_mech: float = 0.002
    include_mech_term_in_phi_variation: bool = False
    include_twin_grad_term_in_phi_variation: bool = False


@dataclass
class TwinningConfig:
    """孪晶相场参数（当前实现为单序参数等效描述）。"""
    L_eta: float = 0.1295
    W_barrier_MPa: float = 8.86
    kappa_eta: float = 6.0e-3
    h_power: int = 3
    gamma_twin: float = 0.129
    twin_crss_MPa: float = 30.0
    twin_shear_dir_angle_deg: float = 35.0
    twin_plane_normal_angle_deg: float = 125.0
    scale_twin_gradient_by_hphi: bool = True
    langevin_nucleation_noise: float = 2e-4
    nucleation_center_radius_um: float = 3.0


@dataclass
class CrystalPlasticityConfig:
    """晶体塑性本构参数与滑移系配置。"""
    gamma0_s_inv: float = 1.0
    m_rate_sensitivity: float = 20.0
    h_MPa: float = 120.0
    q_latent: float = 0.5
    use_overstress: bool = True
    gamma_dot_max_s_inv: float = 50.0
    hardening_rate_cap_MPa_s: float = 20000.0
    elastic_shortcut: bool = False
    basal_crss_MPa: float = 13.0
    prismatic_crss_MPa: float = 57.0
    pyramidal_crss_MPa: float = 100.0
    slip_systems_file: str = "configs/slip_systems_hcp_2d.json"


@dataclass
class MechanicsConfig:
    """力学平衡求解参数（小变形、准静态）。"""
    C11_GPa: float = 63.5
    C12_GPa: float = 25.9
    C13_GPa: float = 21.7
    C33_GPa: float = 66.5
    C44_GPa: float = 18.4
    lambda_GPa: float = 38.0
    mu_GPa: float = 16.3
    # loading_mode:
    # - eigenstrain: 通过均匀外加应变 external_strain_x 进入应变分解
    # - dirichlet_x: 通过位移边界施加（左端 ux=0，右端 ux=u_right）
    loading_mode: str = "eigenstrain"
    external_strain_x: float = 0.001
    # 若 < 0，则在 dirichlet_x 下自动取 external_strain_x * lx_um。
    dirichlet_right_displacement_um: float = -1.0
    plane_strain: bool = True
    residual_damping_MPa_um: float = 80.0
    max_abs_displacement_um: float = 0.0
    max_abs_strain: float = 0.1
    strict_solid_stress_only: bool = True


@dataclass
class MLConfig:
    """机器学习代理模型训练与推理策略配置。"""
    enabled: bool = True
    mode: str = "predictor_corrector"
    model_path: str = "artifacts/ml/surrogate_latest.pt"
    model_arch: str = "tiny_unet"
    model_hidden: int = 32
    dw_hidden: int = 24
    dw_depth: int = 2
    fno_width: int = 32
    fno_modes_x: int = 24
    fno_modes_y: int = 16
    fno_depth: int = 4
    afno_width: int = 24
    afno_modes_x: int = 20
    afno_modes_y: int = 12
    afno_depth: int = 4
    afno_expansion: float = 2.0
    train_epochs: int = 30
    train_batch_size: int = 8
    train_lr: float = 5e-4
    train_split: float = 0.9
    rollout_every: int = 3
    residual_gate: float = 0.5
    # 经典启发式门控（均值/幅值阈值）仍保留，用于快速筛除异常预测。
    max_field_delta: float = 0.06
    max_mean_phi_drop: float = 0.002
    max_mean_eta_rise: float = 0.01
    max_mean_phi_abs_delta: float = 8e-4
    max_mean_eta_abs_delta: float = 1e-3
    max_mean_c_abs_delta: float = 5e-3
    max_mean_epspeq_abs_delta: float = 5e-4
    # 物理残差门控：对 surrogate 候选状态计算离散 PDE 残差并做阈值判定。
    enable_pde_residual_gate: bool = True
    pde_residual_phi_abs_max: float = 5e-2
    pde_residual_eta_abs_max: float = 5e-2
    pde_residual_c_abs_max: float = 5e-2
    pde_residual_mech_abs_max: float = 5e1
    # 能量门控：限制“无外功项下的离散总能量”异常上升。
    enable_energy_gate: bool = True
    energy_abs_increase_max: float = 5e-2
    energy_rel_increase_max: float = 2e-3
    # 不确定性门控：通过输入微扰多次推理估计 surrogate 局部敏感度。
    enable_uncertainty_gate: bool = False
    uncertainty_samples: int = 3
    uncertainty_jitter_std: float = 5e-4
    uncertainty_gate: float = 2.5e-2
    # 自适应 rollout：根据接受/拒绝历史动态调整 surrogate 触发频率。
    adaptive_rollout: bool = True
    rollout_min_every: int = 1
    rollout_max_every: int = 8
    rollout_success_streak_to_increase: int = 6
    rollout_reject_streak_to_decrease: int = 2
    enable_surrogate_mechanics_correction: bool = True
    anchor_physics_every: int = 0
    max_consecutive_reject_before_pause: int = 8
    pause_steps_after_reject_burst: int = 120
    disable_surrogate_after_reject_burst: bool = True
    surrogate_delta_scale: float = 1.0
    surrogate_update_mechanics_fields: bool = True


@dataclass
class ExtensionsConfig:
    """可扩展耦合接口（电场、氢脆）占位配置。"""
    enable_electric_field: bool = False
    enable_hydrogen_embrittlement: bool = False
    electric_field_strength_V_m: float = 0.0
    hydrogen_reference_ppm: float = 0.0


@dataclass
class RuntimeConfig:
    """运行时配置（设备、输出路径、可视化开关）。"""
    device: str = "auto"
    output_dir: str = "artifacts/sim_notch"
    case_name: str = "mg_notch_2d"
    clean_output: bool = True
    render_intermediate_fields: bool = True
    render_final_clouds: bool = True
    render_grid_figure: bool = True


@dataclass
class SimulationConfig:
    """项目总配置容器。"""
    domain: DomainConfig = field(default_factory=DomainConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    corrosion: CorrosionConfig = field(default_factory=CorrosionConfig)
    twinning: TwinningConfig = field(default_factory=TwinningConfig)
    crystal_plasticity: CrystalPlasticityConfig = field(default_factory=CrystalPlasticityConfig)
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    slip_systems: List[Dict[str, Any]] = field(default_factory=list)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，用于 YAML 覆盖默认配置。"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def default_slip_systems_2d() -> List[Dict[str, Any]]:
    """返回二维投影下的默认 HCP 滑移系集合。"""
    systems: List[Dict[str, Any]] = []
    # 3 basal <a>
    for i, angle in enumerate([0.0, 60.0, 120.0], start=1):
        systems.append(
            {
                "name": f"basal_{i}",
                "family": "basal",
                "direction_angle_deg": angle,
                "normal_angle_deg": angle + 90.0,
            }
        )
    # 3 prismatic <a>
    for i, angle in enumerate([30.0, 90.0, 150.0], start=1):
        systems.append(
            {
                "name": f"prismatic_{i}",
                "family": "prismatic",
                "direction_angle_deg": angle,
                "normal_angle_deg": angle + 90.0,
            }
        )
    # 6 pyramidal-II <a+c> projected to 2D
    for i, angle in enumerate([15.0, 45.0, 75.0, 105.0, 135.0, 165.0], start=1):
        systems.append(
            {
                "name": f"pyramidal_{i}",
                "family": "pyramidal",
                "direction_angle_deg": angle,
                "normal_angle_deg": angle + 90.0,
            }
        )
    return systems


def _load_slip_systems(path: Path) -> List[Dict[str, Any]]:
    """从 JSON 读取滑移系；若文件不存在则使用内置默认值。"""
    if path.exists():
        # 用户显式提供时，完全以外部文件为准。
        return json.loads(path.read_text(encoding="utf-8"))
    # 回退到内置默认滑移系，保证最小可运行。
    return default_slip_systems_2d()


def load_config(config_path: str | Path | None = None) -> SimulationConfig:
    """读取并构建 SimulationConfig。

    参数:
    - config_path: YAML 配置路径；为空时返回纯默认配置。

    返回:
    - 完整的 SimulationConfig，同时已完成滑移系加载。
    """
    cfg = SimulationConfig()
    if config_path is not None:
        p = Path(config_path)
        # 使用 safe_load 避免执行任意对象构造。
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        merged = {
            "domain": cfg.domain.__dict__.copy(),
            "numerics": cfg.numerics.__dict__.copy(),
            "corrosion": cfg.corrosion.__dict__.copy(),
            "twinning": cfg.twinning.__dict__.copy(),
            "crystal_plasticity": cfg.crystal_plasticity.__dict__.copy(),
            "mechanics": cfg.mechanics.__dict__.copy(),
            "ml": cfg.ml.__dict__.copy(),
            "extensions": cfg.extensions.__dict__.copy(),
            "runtime": cfg.runtime.__dict__.copy(),
        }
        _deep_update(merged, data)
        # 逐子配置回填到 dataclass，保留类型约束。
        cfg.domain = DomainConfig(**merged["domain"])
        cfg.numerics = NumericsConfig(**merged["numerics"])
        cfg.corrosion = CorrosionConfig(**merged["corrosion"])
        cfg.twinning = TwinningConfig(**merged["twinning"])
        cfg.crystal_plasticity = CrystalPlasticityConfig(**merged["crystal_plasticity"])
        cfg.mechanics = MechanicsConfig(**merged["mechanics"])
        cfg.ml = MLConfig(**merged["ml"])
        cfg.extensions = ExtensionsConfig(**merged["extensions"])
        cfg.runtime = RuntimeConfig(**merged["runtime"])
    slip_path = Path(cfg.crystal_plasticity.slip_systems_file)
    # 最终统一加载滑移系（JSON 或默认）。
    cfg.slip_systems = _load_slip_systems(slip_path)
    return cfg
