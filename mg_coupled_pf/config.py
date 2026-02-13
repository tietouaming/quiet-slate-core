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

from dataclasses import dataclass, field, fields
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
    # 统一界面厚度参数（几何初值默认使用该值，腐蚀自由能可选择继承该值）。
    interface_width_um: float = 2.0
    # 初始 pit 种子高斯核标准差（um）。若 <=0，则回退兼容旧参数 initial_pit_radius_um。
    initial_pit_sigma_um: float = -1.0
    # 旧参数：历史版本名称保留用于兼容；推荐使用 initial_pit_sigma_um。
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
    # phi-IMEX 中 L0 的选取策略：
    # - max: 使用全场最大值（最稳健）
    # - p95: 使用 95% 分位（兼顾稳定与耗散）
    # - mean: 使用全场均值（仅建议做对照）
    phi_imex_l0_strategy: str = "max"
    phi_imex_l0_quantile: float = 0.95
    phi_imex_l0_safety: float = 1.05
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
    # 物理步 cMg 质量投影（在裁剪后把均值拉回参考值）。
    concentration_mass_projection: bool = False
    concentration_mass_projection_iters: int = 2


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
    # <=0 时自动继承 DomainConfig.interface_width_um，实现单一界面厚度来源。
    interface_thickness_um: float = -1.0
    A_J_m3: float = 6.0e7
    temperature_K: float = 310.15
    molar_volume_m3_mol: float = 13.998e-6
    gas_constant_J_mol_K: float = 8.314462618
    # 腐蚀相场迁移率系数。
    # 若 `L0_unit=1_over_MPa_s`，与本代码 MPa 能量尺度直接匹配；
    # 若 `L0_unit=1_over_Pa_s`，内部会自动乘 1e6 进行单位换算。
    L0: float = 5.0e-8
    L0_unit: str = "1_over_MPa_s"
    pitting_beta: float = 0.75
    pitting_alpha: float = 3.0
    pitting_N: int = 2
    pitting_min_factor: float = 0.2
    pitting_max_factor: float = 5.0
    yield_strain_for_mech: float = 0.002
    # 速率型机械耦合：优先使用 epspeq_dot 而非累计 epspeq，避免“历史塑性永久放大”。
    use_epspeq_rate_for_mobility: bool = True
    epspeq_dot_ref_s_inv: float = 1.0e-3
    k_epsdot: float = 1.0
    include_mech_term_in_phi_variation: bool = False
    include_twin_grad_term_in_phi_variation: bool = False
    # 若启用 include_mech_term_in_phi_variation，则在自由能中加入一致的 -h(phi)*e_mech 近似项，
    # 使 energy gate 与 PDE 演化目标保持一致。
    # 默认关闭，保持 mobility-only 路线与门控逻辑自洽。
    include_mech_term_in_free_energy: bool = False
    # c 方程是否使用“化学势一致”写法：
    # mu = A_mpa * (c - h(phi)*(cs-cl) - cl)，c_t = div(M*grad(mu))，其中 M = D / A_mpa。
    # 该写法与 chem 能量项保持同一尺度。
    concentration_use_mu_form: bool = True


@dataclass
class TwinningConfig:
    """孪晶相场参数（当前实现为单序参数等效描述）。"""
    # 序参量动力学系数（与 MPa 能量尺度配套，单位近似 1/(MPa*s)）。
    # 注意：其物理含义与几何孪晶剪切 gamma_twin 完全不同，不能混淆。
    L_eta: float = 0.1295
    W_barrier_MPa: float = 8.86
    kappa_eta: float = 6.0e-3
    # 几何孪晶剪切幅值（Mg {10-12} 扩展孪晶常见量级约 0.129）。
    gamma_twin: float = 0.129
    twin_crss_MPa: float = 30.0
    twin_shear_dir_angle_deg: float = 35.0
    twin_plane_normal_angle_deg: float = 125.0
    scale_twin_gradient_by_hphi: bool = True
    # 形核源项幅值（默认确定性源，不引入随机噪声）。
    nucleation_source_amp: float = 2e-4
    # 兼容旧配置键名，后续将逐步移除。
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
    # 孪晶-母相强度学耦合（最小可用版本）：
    # 通过 eta->f_twin 插值对 CRSS 与硬化速率做相依赖缩放。
    use_phase_dependent_strength: bool = True
    matrix_crss_scale: float = 1.0
    twin_crss_scale: float = 0.85
    matrix_hardening_scale: float = 1.0
    twin_hardening_scale: float = 1.2
    # 晶体取向（Bunge ZXZ 欧拉角，单位 degree）。
    crystal_orientation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # 孪晶重取向：绕给定轴旋转一定角度（单位 degree）。
    twin_reorientation_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    twin_reorientation_angle_deg: float = 86.3
    # 默认优先使用 3D 晶体学滑移系定义文件。
    slip_systems_file: str = "configs/slip_systems_hcp_3d.json"


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
    strict_solid_stress_only: bool = False
    # 各向异性 HCP 弹性开关：开启后使用 Cij + 取向旋转，而非各向同性 lambda/mu。
    use_anisotropic_hcp: bool = False
    # 与 CP 保持一致的取向参数（Bunge ZXZ）。
    crystal_orientation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    twin_reorientation_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    twin_reorientation_angle_deg: float = 86.3
    # 是否根据 h(eta) 在母相/孪晶刚度之间插值。
    anisotropic_blend_with_twin: bool = True
    # 力学离散算子的边界处理（与标量场边界分离）。
    mechanics_bc: str = "neumann"
    # 可选按轴边界条件：若非空则覆盖 mechanics_bc。
    mechanics_bc_x: str = ""
    mechanics_bc_y: str = ""


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
    # 物理残差门控：对 surrogate 候选状态计算“无量纲”离散 PDE 残差并判定阈值。
    enable_pde_residual_gate: bool = True
    pde_residual_phi_abs_max: float = 5e-2
    pde_residual_eta_abs_max: float = 5e-2
    pde_residual_c_abs_max: float = 5e-2
    pde_residual_mech_abs_max: float = 5e-2
    # 能量门控：限制“无外功项下的离散总能量”异常上升。
    enable_energy_gate: bool = True
    # `true` 时，energy gate 使用严格与当前演化方程匹配的 E_gate（变分核心项）；
    # `false` 时使用诊断总能量 E_diag（含弹性能等诊断项）。
    energy_gate_use_variational_core: bool = True
    energy_abs_increase_max: float = 5e-2
    energy_rel_increase_max: float = 2e-3
    # 不确定性门控：通过输入微扰多次推理估计 surrogate 局部敏感度。
    enable_uncertainty_gate: bool = False
    uncertainty_samples: int = 3
    uncertainty_jitter_std: float = 5e-4
    uncertainty_gate: float = 2.5e-2
    # tiny/dw 架构启用坐标通道（x/L, y/L），提升边界条件与位置感知能力。
    surrogate_add_coord_features: bool = True
    # surrogate 输出后强制投影位移边界约束（ux 左端固定、可选右端 Dirichlet、uy 锚点）。
    surrogate_enforce_displacement_projection: bool = True
    # 力学专用 ML 初值器：仅用于位移初值，不替代力学平衡求解。
    mechanics_warmstart_enabled: bool = False
    mechanics_warmstart_model_path: str = "artifacts/ml/mech_warmstart_latest.pt"
    mechanics_warmstart_hidden: int = 24
    mechanics_warmstart_add_coord_features: bool = True
    # surrogate 质量守恒门控：约束 cMg 全域平均的单步变化。
    enable_mass_gate: bool = True
    mass_abs_delta_max: float = 1e-4
    mass_rel_delta_max: float = 1e-4
    # surrogate 接受后是否做 cMg 质量投影（保持与前一步均值一致）。
    enforce_mass_projection_on_accept: bool = False
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
    # 将“位移更新”和“塑性场更新”解耦，默认禁止 surrogate 直接改塑性历史量。
    surrogate_update_plastic_fields: bool = False
    # 当 surrogate 允许更新塑性场时，是否在 accept 后同步推进 CP 内变量（g/gamma_accum）。
    surrogate_sync_cp_state_on_accept: bool = True
    # surrogate / warmstart 通道量纲归一化（位移、应变量）开关。
    enable_field_scaling: bool = True


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


def _filter_dataclass_kwargs(dc_type, payload: Dict[str, Any]) -> Dict[str, Any]:
    """过滤 dataclass 未声明字段，保证旧配置键名不致使加载失败。"""
    names = {f.name for f in fields(dc_type)}
    return {k: v for k, v in payload.items() if k in names}


def default_slip_systems_hcp_3d() -> List[Dict[str, Any]]:
    """返回 3D 晶体学定义的默认 HCP 滑移系集合（投影到 2D 使用）。"""
    systems: List[Dict[str, Any]] = []
    # basal <a>，法向沿 c 轴。
    basal_dirs = [
        [1.0, 0.0, 0.0],
        [-0.5, 0.8660254, 0.0],
        [-0.5, -0.8660254, 0.0],
    ]
    for i, s in enumerate(basal_dirs, start=1):
        systems.append({"name": f"basal_{i}", "family": "basal", "s_crystal": s, "n_crystal": [0.0, 0.0, 1.0]})
    # prismatic <a>，棱柱面法向位于 basal 面内。
    prismatic_defs = [
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([-0.8660254, -0.5, 0.0], [-0.5, 0.8660254, 0.0]),
        ([0.8660254, -0.5, 0.0], [-0.5, -0.8660254, 0.0]),
    ]
    for i, (s, n) in enumerate(prismatic_defs, start=1):
        systems.append({"name": f"prismatic_{i}", "family": "prismatic", "s_crystal": s, "n_crystal": n})
    # pyramidal <c+a>（简化 3D 定义，保持 s·n=0 且含 c 分量）。
    pyramidal_defs = [
        ([0.7, 0.0, 0.714], [0.0, 0.917, -0.399]),
        ([0.35, 0.606, 0.714], [-0.794, 0.458, -0.399]),
        ([-0.35, 0.606, 0.714], [-0.794, -0.458, -0.399]),
        ([-0.7, 0.0, 0.714], [0.0, -0.917, -0.399]),
        ([-0.35, -0.606, 0.714], [0.794, -0.458, -0.399]),
        ([0.35, -0.606, 0.714], [0.794, 0.458, -0.399]),
    ]
    for i, (s, n) in enumerate(pyramidal_defs, start=1):
        systems.append({"name": f"pyramidal_{i}", "family": "pyramidal", "s_crystal": s, "n_crystal": n})
    return systems


def _load_slip_systems(path: Path) -> List[Dict[str, Any]]:
    """从 JSON 读取滑移系；若文件不存在则使用内置默认值。"""
    if path.exists():
        # 用户显式提供时，完全以外部文件为准。
        return json.loads(path.read_text(encoding="utf-8"))
    # 回退到内置默认滑移系，保证最小可运行。
    return default_slip_systems_hcp_3d()


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
        cfg.domain = DomainConfig(**_filter_dataclass_kwargs(DomainConfig, merged["domain"]))
        cfg.numerics = NumericsConfig(**_filter_dataclass_kwargs(NumericsConfig, merged["numerics"]))
        cfg.corrosion = CorrosionConfig(**_filter_dataclass_kwargs(CorrosionConfig, merged["corrosion"]))
        cfg.twinning = TwinningConfig(**_filter_dataclass_kwargs(TwinningConfig, merged["twinning"]))
        cfg.crystal_plasticity = CrystalPlasticityConfig(
            **_filter_dataclass_kwargs(CrystalPlasticityConfig, merged["crystal_plasticity"])
        )
        cfg.mechanics = MechanicsConfig(**_filter_dataclass_kwargs(MechanicsConfig, merged["mechanics"]))
        cfg.ml = MLConfig(**_filter_dataclass_kwargs(MLConfig, merged["ml"]))
        cfg.extensions = ExtensionsConfig(**_filter_dataclass_kwargs(ExtensionsConfig, merged["extensions"]))
        cfg.runtime = RuntimeConfig(**_filter_dataclass_kwargs(RuntimeConfig, merged["runtime"]))
    slip_path = Path(cfg.crystal_plasticity.slip_systems_file)
    # 最终统一加载滑移系（JSON 或默认）。
    cfg.slip_systems = _load_slip_systems(slip_path)
    return cfg
