"""配置与物理一致性审计模块（中文注释版）。

目标：
1. 在仿真启动前识别“会导致结果失真或数值不稳”的配置问题；
2. 给出可执行的修复建议，而非仅报错；
3. 提供可机器读取的 JSON 报告，便于批量回归与 CI 集成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

from .config import SimulationConfig


@dataclass
class AuditIssue:
    """单条审计问题。"""

    level: str
    code: str
    message: str
    recommendation: str = ""
    path: str = ""
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "recommendation": self.recommendation,
            "path": self.path,
            "value": self.value,
        }


@dataclass
class ConfigAuditReport:
    """配置审计报告。"""

    config_path: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    issues: List[AuditIssue] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, *, recommendation: str = "", path: str = "", value: Any = None) -> None:
        """添加审计项。"""
        self.issues.append(
            AuditIssue(
                level=str(level).lower().strip(),
                code=str(code),
                message=str(message),
                recommendation=str(recommendation),
                path=str(path),
                value=value,
            )
        )

    def add_error(self, code: str, message: str, *, recommendation: str = "", path: str = "", value: Any = None) -> None:
        """添加 error 级问题。"""
        self.add("error", code, message, recommendation=recommendation, path=path, value=value)

    def add_warning(self, code: str, message: str, *, recommendation: str = "", path: str = "", value: Any = None) -> None:
        """添加 warning 级问题。"""
        self.add("warning", code, message, recommendation=recommendation, path=path, value=value)

    def add_info(self, code: str, message: str, *, recommendation: str = "", path: str = "", value: Any = None) -> None:
        """添加 info 级问题。"""
        self.add("info", code, message, recommendation=recommendation, path=path, value=value)

    @property
    def error_count(self) -> int:
        """error 数量。"""
        return sum(1 for x in self.issues if x.level == "error")

    @property
    def warning_count(self) -> int:
        """warning 数量。"""
        return sum(1 for x in self.issues if x.level == "warning")

    @property
    def info_count(self) -> int:
        """info 数量。"""
        return sum(1 for x in self.issues if x.level == "info")

    @property
    def passed(self) -> bool:
        """是否通过（无 error）。"""
        return self.error_count == 0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "config_path": self.config_path,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "counts": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "infos": self.info_count,
            },
            "issues": [x.to_dict() for x in self.issues],
        }


def _norm3(v: List[float]) -> float:
    """三维向量二范数。"""
    if len(v) != 3:
        return 0.0
    return float((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5)


def _dot3(a: List[float], b: List[float]) -> float:
    """三维向量点积。"""
    if len(a) != 3 or len(b) != 3:
        return 0.0
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _approx_dx_dy_um(cfg: SimulationConfig) -> tuple[float, float]:
    """估计均匀网格下的 dx,dy（用于审计上界估计）。"""
    nx = max(int(cfg.domain.nx), 2)
    ny = max(int(cfg.domain.ny), 2)
    dx = float(cfg.domain.lx_um) / float(nx - 1)
    dy = float(cfg.domain.ly_um) / float(ny - 1)
    return max(dx, 1e-12), max(dy, 1e-12)


def _estimate_diffusion_dt_limit(cfg: SimulationConfig) -> float:
    """估算显式扩散稳定上限（秒）。"""
    dx_um, dy_um = _approx_dx_dy_um(cfg)
    dx = dx_um * 1e-6
    dy = dy_um * 1e-6
    dmax = max(float(cfg.corrosion.D_l_m2_s), float(cfg.corrosion.D_s_m2_s), 1e-20)
    return 1.0 / (2.0 * dmax * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))))


def _audit_domain(cfg: SimulationConfig, rep: ConfigAuditReport) -> None:
    """几何与网格配置审计。"""
    if int(cfg.domain.nx) < 16 or int(cfg.domain.ny) < 16:
        rep.add_error(
            "DOMAIN_GRID_TOO_SMALL",
            "网格过粗，难以解析界面与应力集中。",
            recommendation="将 nx, ny 至少提升到 64 以上。",
            path="domain.nx/domain.ny",
            value={"nx": cfg.domain.nx, "ny": cfg.domain.ny},
        )
    elif int(cfg.domain.nx) < 64 or int(cfg.domain.ny) < 64:
        rep.add_warning(
            "DOMAIN_GRID_COARSE",
            "网格较粗，结果可能网格依赖明显。",
            recommendation="建议至少使用 64x64，并对缺口附近做加密。",
            path="domain.nx/domain.ny",
            value={"nx": cfg.domain.nx, "ny": cfg.domain.ny},
        )

    if float(cfg.domain.lx_um) <= 0.0 or float(cfg.domain.ly_um) <= 0.0:
        rep.add_error(
            "DOMAIN_SIZE_INVALID",
            "计算域尺寸必须为正数。",
            recommendation="检查 domain.lx_um 与 domain.ly_um。",
            path="domain.lx_um/domain.ly_um",
            value={"lx_um": cfg.domain.lx_um, "ly_um": cfg.domain.ly_um},
        )


def _audit_corrosion(cfg: SimulationConfig, rep: ConfigAuditReport) -> None:
    """腐蚀与相场参数审计。"""
    c = cfg.corrosion
    mode = str(getattr(c, "interface_thickness_definition", "tanh_half_width")).strip().lower()
    if mode not in {"tanh_half_width", "legacy", "legacy_energy_equiv", "legacy_energy_equivalent"}:
        rep.add_error(
            "CORR_IFACE_MODE_INVALID",
            "interface_thickness_definition 取值非法。",
            recommendation="使用 tanh_half_width 或 legacy_energy_equiv。",
            path="corrosion.interface_thickness_definition",
            value=mode,
        )

    ell_um = float(c.interface_thickness_um)
    if ell_um <= 0.0:
        ell_um = float(cfg.domain.interface_width_um)
    if ell_um <= 0.0:
        rep.add_error(
            "CORR_IFACE_THICKNESS_INVALID",
            "界面厚度参数必须为正数。",
            recommendation="设置 corrosion.interface_thickness_um > 0 或 domain.interface_width_um > 0。",
            path="corrosion.interface_thickness_um",
            value=c.interface_thickness_um,
        )
    else:
        dx_um, dy_um = _approx_dx_dy_um(cfg)
        hmin = min(dx_um, dy_um)
        cells = ell_um / max(hmin, 1e-12)
        if cells < 2.0:
            rep.add_error(
                "CORR_IFACE_UNDERRESOLVED",
                "界面厚度分辨率不足（小于 2 个网格）。",
                recommendation="增大界面厚度或提高网格分辨率。",
                path="corrosion.interface_thickness_um",
                value={"ell_um": ell_um, "cells": cells},
            )
        elif cells < 4.0:
            rep.add_warning(
                "CORR_IFACE_MARGINAL_RESOLUTION",
                "界面厚度分辨率偏低（建议至少 4 个网格）。",
                recommendation="提高网格或适当增大界面厚度参数。",
                path="corrosion.interface_thickness_um",
                value={"ell_um": ell_um, "cells": cells},
            )

    if float(c.D_l_m2_s) <= 0.0 or float(c.D_s_m2_s) <= 0.0:
        rep.add_error(
            "CORR_DIFFUSION_NONPOSITIVE",
            "扩散系数必须为正数。",
            recommendation="检查 corrosion.D_l_m2_s 与 corrosion.D_s_m2_s。",
            path="corrosion.D_l_m2_s/corrosion.D_s_m2_s",
            value={"D_l": c.D_l_m2_s, "D_s": c.D_s_m2_s},
        )

    if float(c.c_l_eq_norm) >= float(c.c_s_eq_norm):
        rep.add_warning(
            "CORR_EQ_CONCENTRATION_ORDER",
            "归一化平衡浓度通常应满足液相 < 固相。",
            recommendation="确认 c_l_eq_norm 与 c_s_eq_norm 的归一化口径。",
            path="corrosion.c_l_eq_norm/corrosion.c_s_eq_norm",
            value={"c_l_eq_norm": c.c_l_eq_norm, "c_s_eq_norm": c.c_s_eq_norm},
        )

    if bool(getattr(c, "use_epspeq_rate_for_mobility", True)):
        if float(getattr(c, "epspeq_dot_ref_s_inv", 0.0)) <= 0.0:
            rep.add_error(
                "CORR_EPSDOT_REF_INVALID",
                "epspeq_dot_ref_s_inv 必须为正数。",
                recommendation="设置 corrosion.epspeq_dot_ref_s_inv > 0。",
                path="corrosion.epspeq_dot_ref_s_inv",
                value=getattr(c, "epspeq_dot_ref_s_inv", 0.0),
            )
    if float(getattr(c, "epspeq_twin_weight", 0.0)) < 0.0:
        rep.add_error(
            "CORR_TWIN_WEIGHT_NEGATIVE",
            "epspeq_twin_weight 不应为负数。",
            recommendation="设置 corrosion.epspeq_twin_weight >= 0。",
            path="corrosion.epspeq_twin_weight",
            value=getattr(c, "epspeq_twin_weight", 0.0),
        )

    dt_lim = _estimate_diffusion_dt_limit(cfg)
    dt = float(cfg.numerics.dt_s)
    mode_d = str(getattr(cfg.numerics, "diffusion_integrator", "subcycled_euler")).strip().lower()
    if mode_d in {"explicit", "euler", "subcycled_euler"} and not bool(getattr(cfg.numerics, "diffusion_auto_substeps", True)):
        if dt > 1.05 * dt_lim:
            rep.add_error(
                "NUM_DT_EXCEEDS_DIFFUSION_LIMIT",
                "时间步长超过显式扩散稳定上限。",
                recommendation="减小 dt_s 或开启 diffusion_auto_substeps / STS。",
                path="numerics.dt_s",
                value={"dt_s": dt, "diffusion_dt_limit_s": dt_lim},
            )
        elif dt > 0.8 * dt_lim:
            rep.add_warning(
                "NUM_DT_NEAR_DIFFUSION_LIMIT",
                "时间步长接近显式扩散稳定上限。",
                recommendation="建议留出安全裕量，或改用 STS/子步。",
                path="numerics.dt_s",
                value={"dt_s": dt, "diffusion_dt_limit_s": dt_lim},
            )


def _audit_twinning_and_cp(cfg: SimulationConfig, rep: ConfigAuditReport) -> None:
    """孪晶与晶体塑性配置审计。"""
    tw = cfg.twinning
    cp = cfg.crystal_plasticity
    if float(tw.twin_crss_MPa) <= 0.0:
        rep.add_error(
            "TW_CRSS_NONPOSITIVE",
            "孪晶 CRSS 必须为正值。",
            recommendation="设置 twinning.twin_crss_MPa > 0。",
            path="twinning.twin_crss_MPa",
            value=tw.twin_crss_MPa,
        )
    if float(tw.gamma_twin) <= 0.0:
        rep.add_error(
            "TW_GAMMA_NONPOSITIVE",
            "孪晶剪切幅值 gamma_twin 必须为正值。",
            recommendation="设置 twinning.gamma_twin > 0。",
            path="twinning.gamma_twin",
            value=tw.gamma_twin,
        )
    n_var = int(getattr(tw, "n_variants", 1))
    if n_var < 1:
        rep.add_error(
            "TW_VARIANT_COUNT_INVALID",
            "孪晶变体数量必须 >= 1。",
            recommendation="设置 twinning.n_variants >= 1。",
            path="twinning.n_variants",
            value=n_var,
        )
    idx_cfg = [int(v) for v in list(getattr(tw, "twin_variant_indices", []))]
    if idx_cfg and len(idx_cfg) < max(n_var, 1):
        rep.add_warning(
            "TW_VARIANT_INDEX_LIST_SHORT",
            "显式孪晶变体索引数量少于 n_variants，剩余变体将回退到默认顺序。",
            recommendation="补全 twinning.twin_variant_indices 或降低 twinning.n_variants。",
            path="twinning.twin_variant_indices",
            value={"n_variants": n_var, "len(indices)": len(idx_cfg)},
        )
    if float(cp.gamma0_s_inv) < 0.0:
        rep.add_error(
            "CP_GAMMA0_NEGATIVE",
            "晶体塑性参考剪切速率 gamma0_s_inv 不应为负数。",
            recommendation="设置 crystal_plasticity.gamma0_s_inv >= 0。",
            path="crystal_plasticity.gamma0_s_inv",
            value=cp.gamma0_s_inv,
        )
    if float(cp.m_rate_sensitivity) <= 0.0:
        rep.add_error(
            "CP_RATE_SENSITIVITY_INVALID",
            "速率敏感指数必须为正值。",
            recommendation="设置 crystal_plasticity.m_rate_sensitivity > 0。",
            path="crystal_plasticity.m_rate_sensitivity",
            value=cp.m_rate_sensitivity,
        )

    e_cp = list(getattr(cp, "crystal_orientation_euler_deg", [0.0, 0.0, 0.0]))
    e_mech = list(getattr(cfg.mechanics, "crystal_orientation_euler_deg", [0.0, 0.0, 0.0]))
    if [float(x) for x in e_cp] != [float(x) for x in e_mech]:
        rep.add_warning(
            "ORI_CP_MECH_MISMATCH",
            "CP 与力学模块的晶体取向不一致，可能导致耦合偏差。",
            recommendation="建议统一 crystal_orientation_euler_deg。",
            path="crystal_plasticity.crystal_orientation_euler_deg/mechanics.crystal_orientation_euler_deg",
            value={"cp": e_cp, "mech": e_mech},
        )

    if not cfg.slip_systems:
        rep.add_error(
            "CP_SLIP_SYSTEM_EMPTY",
            "滑移系为空。",
            recommendation="检查 slip_systems_file 或默认系统构造。",
            path="slip_systems",
            value=0,
        )
    if not cfg.twin_systems:
        rep.add_error(
            "TW_SYSTEM_EMPTY",
            "孪晶系统为空。",
            recommendation="检查 twin_systems_file 或默认系统构造。",
            path="twin_systems",
            value=0,
        )
    else:
        n_avail = len(cfg.twin_systems)
        if n_var > n_avail:
            rep.add_warning(
                "TW_VARIANT_COUNT_EXCEEDS_LIBRARY",
                "请求的孪晶变体数量超过可用系统数，多余部分将被截断/复用。",
                recommendation="减小 twinning.n_variants 或提供更多 twin_systems。",
                path="twinning.n_variants",
                value={"n_variants": n_var, "n_available": n_avail},
            )
        for j, idx in enumerate(idx_cfg):
            if idx < 0 or idx >= n_avail:
                rep.add_error(
                    "TW_VARIANT_INDEX_OUT_OF_RANGE",
                    "孪晶变体索引越界。",
                    recommendation="确保每个 twinning.twin_variant_indices 都在 [0, len(twin_systems)-1]。",
                    path=f"twinning.twin_variant_indices[{j}]",
                    value={"index": idx, "n_available": n_avail},
                )

    for i, s in enumerate(cfg.slip_systems):
        sv = [float(v) for v in s.get("s_crystal", [])]
        nv = [float(v) for v in s.get("n_crystal", [])]
        ns = _norm3(sv)
        nn = _norm3(nv)
        dot = abs(_dot3(sv, nv))
        if abs(ns - 1.0) > 1e-4 or abs(nn - 1.0) > 1e-4 or dot > 1e-3:
            rep.add_error(
                "CP_SLIP_SYSTEM_NOT_ORTHONORMAL",
                "滑移系 s/n 未正确归一正交。",
                recommendation="检查系统定义并执行正交化。",
                path=f"slip_systems[{i}]",
                value={"|s|": ns, "|n|": nn, "|s·n|": dot},
            )

    for i, s in enumerate(cfg.twin_systems):
        sv = [float(v) for v in s.get("s_crystal", [])]
        nv = [float(v) for v in s.get("n_crystal", [])]
        ns = _norm3(sv)
        nn = _norm3(nv)
        dot = abs(_dot3(sv, nv))
        if abs(ns - 1.0) > 1e-4 or abs(nn - 1.0) > 1e-4 or dot > 1e-3:
            rep.add_error(
                "TW_SYSTEM_NOT_ORTHONORMAL",
                "孪晶系 s/n 未正确归一正交。",
                recommendation="检查系统定义并执行正交化。",
                path=f"twin_systems[{i}]",
                value={"|s|": ns, "|n|": nn, "|s·n|": dot},
            )


def _audit_mechanics(cfg: SimulationConfig, rep: ConfigAuditReport) -> None:
    """力学模型审计。"""
    m = cfg.mechanics
    if bool(m.use_anisotropic_hcp):
        for k in ("C11_GPa", "C12_GPa", "C13_GPa", "C33_GPa", "C44_GPa"):
            if float(getattr(m, k, 0.0)) <= 0.0:
                rep.add_error(
                    "MECH_HCP_STIFFNESS_NONPOSITIVE",
                    "HCP 各向异性刚度参数必须为正数。",
                    recommendation=f"检查 mechanics.{k}。",
                    path=f"mechanics.{k}",
                    value=getattr(m, k),
                )
    else:
        if float(m.mu_GPa) <= 0.0:
            rep.add_error(
                "MECH_MU_NONPOSITIVE",
                "剪切模量 mu_GPa 必须为正数。",
                recommendation="设置 mechanics.mu_GPa > 0。",
                path="mechanics.mu_GPa",
                value=m.mu_GPa,
            )

    mode = str(getattr(m, "loading_mode", "eigenstrain")).strip().lower()
    if mode in {"dirichlet_x", "dirichlet", "ux_dirichlet"}:
        # 允许自动计算，但自动计算值为 0 时给 warning。
        right_u = float(getattr(m, "dirichlet_right_displacement_um", -1.0))
        if right_u < 0.0 and abs(float(getattr(m, "external_strain_x", 0.0))) <= 1e-12:
            rep.add_warning(
                "MECH_DIRICHLET_RIGHT_ZERO",
                "Dirichlet 载荷模式下右边界位移将自动推导为 0。",
                recommendation="显式设置 dirichlet_right_displacement_um 或 external_strain_x。",
                path="mechanics.dirichlet_right_displacement_um",
                value=right_u,
            )


def _audit_ml(cfg: SimulationConfig, rep: ConfigAuditReport) -> None:
    """ML 代理配置审计。"""
    ml = cfg.ml
    if not bool(ml.enabled):
        return

    model_path = Path(str(ml.model_path))
    if not model_path.exists():
        rep.add_warning(
            "ML_MODEL_MISSING",
            "ML 已启用但模型文件不存在，将回退纯物理步或降级逻辑。",
            recommendation="训练后写入 model_path 或关闭 ml.enabled。",
            path="ml.model_path",
            value=str(model_path),
        )

    if bool(getattr(ml, "surrogate_update_plastic_fields", False)) and not bool(
        getattr(ml, "surrogate_sync_cp_state_on_accept", True)
    ):
        rep.add_warning(
            "ML_PLASTIC_WITHOUT_CP_SYNC",
            "surrogate 允许更新塑性场但关闭了 CP 同步，可能导致内变量漂移。",
            recommendation="建议开启 ml.surrogate_sync_cp_state_on_accept。",
            path="ml.surrogate_sync_cp_state_on_accept",
            value=False,
        )

    if bool(getattr(ml, "enable_energy_gate", False)):
        if bool(cfg.corrosion.include_mech_term_in_phi_variation) and not bool(
            getattr(cfg.corrosion, "include_mech_term_in_free_energy", False)
        ):
            rep.add_warning(
                "ML_ENERGY_GATE_VARIATIONAL_MISMATCH",
                "启用了能量门控，但机械-phi 变分与自由能配置不一致。",
                recommendation="建议统一 include_mech_term_in_phi_variation 与 include_mech_term_in_free_energy。",
                path="ml.enable_energy_gate",
                value=True,
            )
    out_mode = str(getattr(ml, "surrogate_output_mode", "delta")).strip().lower()
    if out_mode not in {"delta", "absolute"}:
        rep.add_error(
            "ML_SURROGATE_OUTPUT_MODE_INVALID",
            "surrogate_output_mode 仅支持 delta 或 absolute。",
            recommendation="设置 ml.surrogate_output_mode 为 delta（推荐）或 absolute。",
            path="ml.surrogate_output_mode",
            value=out_mode,
        )
    if float(getattr(ml, "local_pde_exceed_frac_max", 0.0)) < 0.0:
        rep.add_error(
            "ML_LOCAL_PDE_FRAC_NEGATIVE",
            "local_pde_exceed_frac_max 不能为负数。",
            recommendation="设置 ml.local_pde_exceed_frac_max >= 0。",
            path="ml.local_pde_exceed_frac_max",
            value=getattr(ml, "local_pde_exceed_frac_max", 0.0),
        )
    for k in (
        "local_pde_phi_abs_max",
        "local_pde_eta_abs_max",
        "local_pde_c_abs_max",
        "local_pde_mech_abs_max",
    ):
        if float(getattr(ml, k, 0.0)) < 0.0:
            rep.add_error(
                "ML_LOCAL_PDE_THRESHOLD_NEGATIVE",
                "局部 PDE 阈值不能为负数。",
                recommendation=f"设置 ml.{k} >= 0。",
                path=f"ml.{k}",
                value=getattr(ml, k, 0.0),
            )


def audit_config(cfg: SimulationConfig, *, config_path: str | Path | None = None) -> ConfigAuditReport:
    """执行完整配置审计。"""
    rep = ConfigAuditReport(config_path=str(config_path or ""))
    _audit_domain(cfg, rep)
    _audit_corrosion(cfg, rep)
    _audit_twinning_and_cp(cfg, rep)
    _audit_mechanics(cfg, rep)
    _audit_ml(cfg, rep)
    if rep.passed:
        rep.add_info(
            "CONFIG_AUDIT_PASSED",
            "配置审计通过（无 error）。",
            recommendation="可进入仿真/训练流程。",
        )
    return rep


def save_audit_report(report: ConfigAuditReport, path: str | Path) -> Path:
    """保存审计报告 JSON。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def summarize_audit_report(report: ConfigAuditReport) -> str:
    """生成人类可读的一行摘要。"""
    st = "PASS" if report.passed else "FAIL"
    return (
        f"[{st}] errors={report.error_count} "
        f"warnings={report.warning_count} infos={report.info_count}"
    )
