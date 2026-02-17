"""数值稳定性与一致性测试（中文注释版）。"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from mg_coupled_pf import CoupledSimulator, load_config
from mg_coupled_pf.crystal_plasticity import CrystalPlasticityModel
from mg_coupled_pf.mechanics import MechanicsModel
from mg_coupled_pf.operators import double_well_prime, grad_xy, smooth_heaviside, smooth_heaviside_prime


def test_short_run_stability() -> None:
    """短程纯物理运行稳定性测试。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.domain.nx = 96
    cfg.domain.ny = 64
    cfg.numerics.n_steps = 24
    cfg.numerics.mechanics_substeps = 8
    cfg.numerics.save_every = 1000
    cfg.ml.enabled = False

    sim = CoupledSimulator(cfg)
    max_sigma = 0.0
    max_epspeq = 0.0
    for i in range(1, cfg.numerics.n_steps + 1):
        diag = sim.step(i)
        max_sigma = max(max_sigma, diag.max_sigma_h)
        max_epspeq = max(max_epspeq, diag.avg_epspeq)

    assert math.isfinite(max_sigma)
    assert math.isfinite(max_epspeq)
    assert max_sigma < 5000.0
    assert max_epspeq < 5.0
    assert torch.isfinite(sim.state["phi"]).all()


def test_diffusion_sts_close_to_subcycled_euler() -> None:
    """验证 STS 与子循环欧拉在短程上结果接近。"""
    def run(mode: str):
        cfg = load_config("configs/notch_case.yaml")
        cfg.runtime.device = "cpu"
        cfg.domain.nx = 64
        cfg.domain.ny = 48
        cfg.numerics.dt_s = 4e-4
        cfg.numerics.n_steps = 10
        cfg.numerics.save_every = 1000
        cfg.numerics.mechanics_substeps = 4
        cfg.numerics.diffusion_integrator = mode
        cfg.numerics.diffusion_sts_nu = 0.05
        cfg.numerics.diffusion_sts_max_stages = 64
        cfg.ml.enabled = False

        sim = CoupledSimulator(cfg)
        for i in range(1, cfg.numerics.n_steps + 1):
            sim.step(i)
        return sim.state

    ref = run("subcycled_euler")
    sts = run("sts")

    phi_mae = float(torch.mean(torch.abs(ref["phi"] - sts["phi"])).item())
    c_mae = float(torch.mean(torch.abs(ref["c"] - sts["c"])).item())
    eta_mae = float(torch.mean(torch.abs(ref["eta"] - sts["eta"])).item())

    assert phi_mae < 8e-3
    assert c_mae < 8e-3
    assert eta_mae < 8e-3


def test_phi_eta_imex_close_to_explicit_small_dt() -> None:
    """验证 IMEX 与显式格式在小步长下结果一致。"""
    def run(phi_mode: str, eta_mode: str):
        cfg = load_config("configs/notch_case.yaml")
        cfg.runtime.device = "cpu"
        cfg.domain.nx = 64
        cfg.domain.ny = 48
        cfg.numerics.dt_s = 1e-4
        cfg.numerics.n_steps = 8
        cfg.numerics.save_every = 1000
        cfg.numerics.mechanics_substeps = 4
        cfg.numerics.phi_integrator = phi_mode
        cfg.numerics.eta_integrator = eta_mode
        cfg.ml.enabled = False
        sim = CoupledSimulator(cfg)
        for i in range(1, cfg.numerics.n_steps + 1):
            sim.step(i)
        return sim.state

    ref = run("explicit_euler", "explicit_euler")
    imx = run("imex_helmholtz", "imex_variable")
    phi_mae = float(torch.mean(torch.abs(ref["phi"] - imx["phi"])).item())
    eta_mae = float(torch.mean(torch.abs(ref["eta"] - imx["eta"])).item())
    c_mae = float(torch.mean(torch.abs(ref["c"] - imx["c"])).item())
    assert phi_mae < 1.5e-2
    assert eta_mae < 1.5e-2
    assert c_mae < 1.5e-2


def test_phi_eta_imex_stability_large_dt() -> None:
    """验证 IMEX 在较大步长下保持有界与非 NaN。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 64
    cfg.domain.ny = 48
    cfg.numerics.dt_s = 8e-4
    cfg.numerics.n_steps = 6
    cfg.numerics.save_every = 1000
    cfg.numerics.phi_integrator = "imex_helmholtz"
    cfg.numerics.eta_integrator = "imex_variable"
    cfg.ml.enabled = False
    sim = CoupledSimulator(cfg)
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    assert torch.isfinite(sim.state["phi"]).all()
    assert torch.isfinite(sim.state["eta"]).all()
    assert float(sim.state["phi"].min()) >= 0.0
    assert float(sim.state["phi"].max()) <= 1.0
    assert float(sim.state["eta"].min()) >= 0.0
    assert float(sim.state["eta"].max()) <= 1.0


def test_cp_overstress_elastic_shortcut() -> None:
    """验证晶体塑性弹性捷径逻辑。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.crystal_plasticity.elastic_shortcut = True
    dev = torch.device("cpu")
    cp = CrystalPlasticityModel(cfg, device=dev, dtype=torch.float32)
    st = cp.init_state(ny=6, nx=8)

    sigma_xx = torch.full((1, 1, 6, 8), 2.0, dtype=torch.float32, device=dev)
    sigma_yy = torch.full((1, 1, 6, 8), 1.5, dtype=torch.float32, device=dev)
    sigma_xy = torch.full((1, 1, 6, 8), 0.5, dtype=torch.float32, device=dev)
    eta = torch.zeros((1, 1, 6, 8), dtype=torch.float32, device=dev)

    out = cp.update(
        cp_state=st,
        sigma_xx=sigma_xx,
        sigma_yy=sigma_yy,
        sigma_xy=sigma_xy,
        eta=eta,
        dt_s=1e-3,
    )

    assert out["plastic_active"] is False
    assert torch.max(torch.abs(out["epsp_dot_xx"])).item() == 0.0
    assert torch.max(torch.abs(out["epsp_dot_yy"])).item() == 0.0
    assert torch.max(torch.abs(out["epsp_dot_xy"])).item() == 0.0
    assert torch.max(torch.abs(out["epspeq_dot"])).item() == 0.0
    assert torch.allclose(out["g"], st["g"])
    assert torch.allclose(out["gamma_accum"], st["gamma_accum"])


def test_mechanics_hybrid_fallback_remains_stable() -> None:
    """验证 hybrid 力学求解在 Krylov 失败后可稳定回退。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 72
    cfg.domain.ny = 48
    cfg.numerics.n_steps = 10
    cfg.numerics.save_every = 1000
    cfg.numerics.mechanics_solver = "hybrid_cg_relax"
    cfg.ml.enabled = False
    sim = CoupledSimulator(cfg)
    max_sigma = 0.0
    for i in range(1, cfg.numerics.n_steps + 1):
        d = sim.step(i)
        max_sigma = max(max_sigma, d.max_sigma_h)
    assert max_sigma < 2000.0
    # 至少应执行过一次预测力学求解。
    assert sim.stats["mech_predict_solve"] == cfg.numerics.n_steps


def test_cp_schmid_tensor_mixture_endpoints() -> None:
    """验证 Schmid 张量混合在 eta 端点退化为各相本征值。"""
    cfg = load_config("configs/notch_case.yaml")
    cp = CrystalPlasticityModel(cfg, device=torch.device("cpu"), dtype=torch.float32)
    eta0 = torch.zeros((1, 1, 4, 5), dtype=torch.float32)
    eta1 = torch.ones((1, 1, 4, 5), dtype=torch.float32)
    t0 = cp._effective_schmid_tensors(eta0)
    t1 = cp._effective_schmid_tensors(eta1)
    pxx_m = cp.pxx_m.view(1, cp.n_sys, 1, 1).expand_as(t0["p_xx"])
    pxx_t = cp.pxx_t.view(1, cp.n_sys, 1, 1).expand_as(t1["p_xx"])
    assert torch.allclose(t0["p_xx"], pxx_m, atol=1e-6, rtol=1e-6)
    assert torch.allclose(t1["p_xx"], pxx_t, atol=1e-6, rtol=1e-6)


def test_mechanics_gmres_hybrid_converges_short_run() -> None:
    """验证 GMRES-hybrid 在短程中能稳定收敛。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 64
    cfg.domain.ny = 48
    cfg.numerics.n_steps = 6
    cfg.numerics.save_every = 1000
    cfg.numerics.mechanics_solver = "hybrid_cg_relax"
    cfg.numerics.mechanics_krylov_method = "gmres"
    cfg.numerics.mechanics_cg_max_iters = 160
    cfg.numerics.mechanics_cg_tol_abs = 8.0
    cfg.numerics.mechanics_cg_tol_rel = 8e-3
    cfg.numerics.mechanics_gmres_restart = 24
    cfg.numerics.mechanics_gmres_max_restarts = 8
    cfg.ml.enabled = False
    sim = CoupledSimulator(cfg)
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    assert sim.stats["mech_cg_converged"] > 0
    assert sim.stats["mech_cg_failed"] == 0


def test_surrogate_gate_rejects_by_pde_residual() -> None:
    """验证 surrogate 门控可由 PDE 残差触发拒绝。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 48
    cfg.domain.ny = 32
    cfg.ml.enabled = False
    cfg.ml.max_field_delta = 10.0
    cfg.ml.max_mean_phi_drop = 10.0
    cfg.ml.max_mean_eta_rise = 10.0
    cfg.ml.max_mean_phi_abs_delta = 10.0
    cfg.ml.max_mean_eta_abs_delta = 10.0
    cfg.ml.max_mean_c_abs_delta = 10.0
    cfg.ml.max_mean_epspeq_abs_delta = 10.0
    cfg.ml.residual_gate = 10.0
    cfg.ml.enable_pde_residual_gate = True
    cfg.ml.pde_residual_phi_abs_max = 1e-6
    cfg.ml.pde_residual_c_abs_max = 1e6
    cfg.ml.pde_residual_eta_abs_max = 1e6
    cfg.ml.pde_residual_mech_abs_max = 1e6
    cfg.ml.enable_energy_gate = False
    sim = CoupledSimulator(cfg)
    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in prev.items()}
    nxt["phi"] = torch.clamp(prev["phi"] * 0.0 + 0.5, 0.0, 1.0)
    ok = sim._surrogate_update_is_valid(prev, nxt)
    assert ok is False
    assert sim.stats["ml_reject_pde_phi"] > 0


def test_surrogate_gate_rejects_by_energy_increase() -> None:
    """验证 surrogate 门控可由自由能上升触发拒绝。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 48
    cfg.domain.ny = 32
    cfg.ml.enabled = False
    cfg.ml.max_field_delta = 10.0
    cfg.ml.max_mean_phi_drop = 10.0
    cfg.ml.max_mean_eta_rise = 10.0
    cfg.ml.max_mean_phi_abs_delta = 10.0
    cfg.ml.max_mean_eta_abs_delta = 10.0
    cfg.ml.max_mean_c_abs_delta = 10.0
    cfg.ml.max_mean_epspeq_abs_delta = 10.0
    cfg.ml.residual_gate = 10.0
    cfg.ml.enable_pde_residual_gate = False
    cfg.ml.enable_energy_gate = True
    cfg.ml.energy_abs_increase_max = 0.0
    cfg.ml.energy_rel_increase_max = 0.0
    sim = CoupledSimulator(cfg)
    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in prev.items()}
    nxt["phi"] = torch.clamp(prev["phi"] + 0.1, 0.0, 1.0)
    ok = sim._surrogate_update_is_valid(prev, nxt)
    assert ok is False
    assert sim.stats["ml_reject_energy"] > 0


def test_surrogate_gate_rejects_by_mass_delta() -> None:
    """验证 surrogate 门控可由 cMg 全域质量变化触发拒绝。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 40
    cfg.domain.ny = 28
    cfg.ml.enabled = False
    cfg.ml.max_field_delta = 10.0
    cfg.ml.max_mean_phi_drop = 10.0
    cfg.ml.max_mean_eta_rise = 10.0
    cfg.ml.max_mean_phi_abs_delta = 10.0
    cfg.ml.max_mean_eta_abs_delta = 10.0
    cfg.ml.max_mean_c_abs_delta = 10.0
    cfg.ml.max_mean_epspeq_abs_delta = 10.0
    cfg.ml.residual_gate = 10.0
    cfg.ml.enable_pde_residual_gate = False
    cfg.ml.enable_energy_gate = False
    cfg.ml.enable_mass_gate = True
    cfg.ml.mass_abs_delta_max = 1e-8
    cfg.ml.mass_rel_delta_max = 0.0
    sim = CoupledSimulator(cfg)
    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in prev.items()}
    nxt["c"] = torch.clamp(prev["c"] + 5e-3, 0.0, 1.0)
    ok = sim._surrogate_update_is_valid(prev, nxt)
    assert ok is False
    assert sim.stats["ml_reject_mass"] > 0


def test_mechanics_dirichlet_x_boundary_enforced() -> None:
    """验证位移边界加载模式下右边界位移被严格约束。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 48
    cfg.domain.ny = 32
    cfg.numerics.n_steps = 4
    cfg.numerics.save_every = 1000
    cfg.ml.enabled = False
    cfg.mechanics.loading_mode = "dirichlet_x"
    cfg.mechanics.dirichlet_right_displacement_um = 0.05
    sim = CoupledSimulator(cfg)
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    ux = sim.state["ux"]
    left = ux[:, :, :, 0]
    right = ux[:, :, :, -1]
    assert float(torch.max(torch.abs(left)).item()) < 1e-8
    assert float(torch.max(torch.abs(right - 0.05)).item()) < 1e-6


def test_plane_strain_sigma_h_includes_sigma_zz() -> None:
    """验证平面应变下 sigma_h 计算包含 sigma_zz 分量。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.mechanics.plane_strain = True
    mech = MechanicsModel(cfg)
    phi = torch.ones((1, 1, 4, 5), dtype=torch.float32)
    ex = torch.full_like(phi, 0.01)
    ey = torch.full_like(phi, -0.002)
    exy = torch.full_like(phi, 0.0)
    sig = mech.constitutive_stress(phi, ex, ey, exy)
    lam = cfg.mechanics.lambda_GPa * 1e3
    mu = cfg.mechanics.mu_GPa * 1e3
    tr = ex + ey
    sxx = lam * tr + 2.0 * mu * ex
    syy = lam * tr + 2.0 * mu * ey
    szz = lam * tr
    sh_ref = (sxx + syy + szz) / 3.0
    err = torch.max(torch.abs(sig["sigma_h"] - sh_ref)).item()
    assert err < 1e-6


def test_diffusion_rhs_conservative_for_zero_flux_case() -> None:
    """零通量边界、无耦合漂移时扩散项应近似守恒。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 48
    cfg.domain.ny = 32
    cfg.numerics.scalar_bc = "neumann"
    sim = CoupledSimulator(cfg)
    torch.manual_seed(1)
    c = torch.rand_like(sim.state["c"])
    phi = sim.state["phi"]
    hphi = torch.clamp(phi, 0.0, 1.0)
    D = cfg.corrosion.D_s_m2_s * hphi + (1.0 - hphi) * cfg.corrosion.D_l_m2_s
    gx_phi = torch.zeros_like(phi)
    gy_phi = torch.zeros_like(phi)
    corr = torch.zeros_like(phi)
    rhs = sim._diffusion_rhs(c, D, corr, gx_phi, gy_phi, phi)
    # 面积加权平均变化率应接近 0。
    mass_rate = float(sim._mean_field(rhs).item())
    assert abs(mass_rate) < 5e-9


def test_mechanics_adaptive_trigger_updates_early() -> None:
    """当微结构突变超过阈值时，力学应提前更新。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 40
    cfg.domain.ny = 28
    cfg.numerics.mechanics_update_every = 100
    cfg.numerics.mechanics_trigger_phi_max_delta = 1e-4
    cfg.numerics.mechanics_trigger_eta_max_delta = 0.0
    sim = CoupledSimulator(cfg)
    # 第一步会执行力学并建立参考态。
    sim._physics_step(cfg.numerics.dt_s, 1)
    n1 = sim.stats["mech_predict_solve"]
    # 非周期步手工施加较大相场扰动，触发自适应力学更新。
    sim.state["phi"] = torch.clamp(sim.state["phi"], 0.0, 1.0)
    sim.state["phi"][:, :, 6:9, 6:9] = torch.clamp(sim.state["phi"][:, :, 6:9, 6:9] - 0.02, 0.0, 1.0)
    sim._physics_step(cfg.numerics.dt_s, 2)
    n2 = sim.stats["mech_predict_solve"]
    assert n2 == n1 + 1


def test_scalar_linear_solver_auto_fallback_to_bicgstab() -> None:
    """auto 模式下 PCG 未收敛时应回退到 BiCGStab。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 24
    cfg.domain.ny = 16
    cfg.numerics.scalar_linear_solver = "auto"
    cfg.numerics.imex_solver_iters = 1
    cfg.numerics.imex_solver_tol_abs = 1e-18
    cfg.numerics.imex_solver_tol_rel = 1e-18
    sim = CoupledSimulator(cfg)

    torch.manual_seed(2)
    rhs = torch.rand((1, 1, 16, 24), dtype=torch.float32)

    def apply_a(u: torch.Tensor) -> torch.Tensor:
        # 构造带平流型偏置的非对称算子，降低 PCG 单步收敛概率。
        return u + 0.3 * (torch.roll(u, shifts=-1, dims=3) - u)

    x = sim._solve_scalar_linear(apply_a, rhs, x0=torch.zeros_like(rhs), diag_inv=None)
    assert torch.isfinite(x).all()
    assert sim.stats["scalar_pcg_calls"] >= 1
    assert sim.stats["scalar_solver_fallback"] >= 1
    assert sim.stats["scalar_bicgstab_calls"] >= 1


def test_concentration_mass_projection_applied_in_physics_step() -> None:
    """开启质量投影后，物理步应记录投影应用计数。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 36
    cfg.domain.ny = 24
    cfg.numerics.n_steps = 1
    cfg.numerics.concentration_mass_projection = True
    cfg.numerics.concentration_mass_projection_iters = 2
    sim = CoupledSimulator(cfg)
    sim._physics_step(cfg.numerics.dt_s, 1)
    assert sim.stats["mass_projection_applied"] >= 1


def test_phi_residual_uses_allen_cahn_l_times_laplacian_form() -> None:
    """验证 phi 残差使用 L*(nonlin-kappa*lap) 而非 div(L*kappa*grad)。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 40
    cfg.domain.ny = 28
    cfg.corrosion.include_mech_term_in_phi_variation = False
    cfg.corrosion.include_twin_grad_term_in_phi_variation = False
    sim = CoupledSimulator(cfg)

    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in sim.state.items()}
    # 构造线性 phi 场，便于显式识别是否引入额外 grad(L)·grad(phi) 假项。
    xx = torch.linspace(0.2, 0.8, cfg.domain.nx, dtype=sim.state["phi"].dtype).view(1, 1, 1, -1)
    nxt["phi"] = xx.expand_as(prev["phi"])
    nxt["c"] = torch.full_like(prev["c"], 0.5)
    nxt["eta"] = torch.zeros_like(prev["eta"])
    dt = float(cfg.numerics.dt_s)

    mech = sim._estimate_state_mechanics(nxt)
    got = sim._compute_pde_residuals(prev, nxt, mech, dt_s=dt)

    phi = torch.nan_to_num(nxt["phi"], nan=0.5, posinf=1.0, neginf=0.0)
    cbar = torch.nan_to_num(nxt["c"], nan=0.5, posinf=cfg.corrosion.cMg_max, neginf=cfg.corrosion.cMg_min)
    hphi = smooth_heaviside(phi, clamp_input=False)
    hphi_d = smooth_heaviside_prime(phi, clamp_input=False)
    delta_eq = cfg.corrosion.c_s_eq_norm - cfg.corrosion.c_l_eq_norm
    chem_drive = -cfg.corrosion.A_J_m3 / 1e6 * (cbar - hphi * delta_eq - cfg.corrosion.c_l_eq_norm) * delta_eq * hphi_d
    omega, kappa_phi = sim._omega_kappa_phi()
    phi_nonlin = chem_drive + omega * double_well_prime(phi, clamp_input=False)
    L_phi = torch.clamp(sim._corrosion_mobility(state=nxt, mech=mech), min=0.0)
    phi_lap_kappa = sim._div_diffusive_fv(
        phi,
        torch.full_like(phi, float(kappa_phi)),
        dx=sim.dx_m,
        dy=sim.dy_m,
        x_coords=sim.x_coords_m,
        y_coords=sim.y_coords_m,
        bc=sim._scalar_bc(),
    )
    r_expected = (phi - prev["phi"]) / max(dt, 1e-12) + L_phi * (phi_nonlin - phi_lap_kappa)
    pde_expected = sim._masked_mean_abs(r_expected) * dt
    assert abs(float(got["pde_phi"]) - float(pde_expected)) < 1e-10


def test_phi_twin_gradient_variation_sign_consistent() -> None:
    """验证孪晶梯度能对 phi 变分项符号与自由能定义一致（正号）。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 40
    cfg.domain.ny = 28
    cfg.corrosion.include_mech_term_in_phi_variation = False
    cfg.corrosion.include_twin_grad_term_in_phi_variation = True
    cfg.twinning.scale_twin_gradient_by_hphi = True
    sim = CoupledSimulator(cfg)

    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in sim.state.items()}
    xx = torch.linspace(0.1, 0.9, cfg.domain.nx, dtype=sim.state["phi"].dtype).view(1, 1, 1, -1)
    yy = torch.linspace(0.0, 1.0, cfg.domain.ny, dtype=sim.state["eta"].dtype).view(1, 1, -1, 1)
    nxt["phi"] = xx.expand_as(prev["phi"])
    nxt["eta"] = yy.expand_as(prev["eta"])
    nxt["c"] = torch.full_like(prev["c"], 0.45)
    dt = float(cfg.numerics.dt_s)

    mech = sim._estimate_state_mechanics(nxt)
    got = sim._compute_pde_residuals(prev, nxt, mech, dt_s=dt)

    phi = torch.nan_to_num(nxt["phi"], nan=0.5, posinf=1.0, neginf=0.0)
    cbar = torch.nan_to_num(nxt["c"], nan=0.5, posinf=cfg.corrosion.cMg_max, neginf=cfg.corrosion.cMg_min)
    eta = torch.nan_to_num(nxt["eta"], nan=0.0, posinf=1.0, neginf=0.0)
    hphi = smooth_heaviside(phi, clamp_input=False)
    hphi_d = smooth_heaviside_prime(phi, clamp_input=False)
    delta_eq = cfg.corrosion.c_s_eq_norm - cfg.corrosion.c_l_eq_norm
    chem_drive = -cfg.corrosion.A_J_m3 / 1e6 * (cbar - hphi * delta_eq - cfg.corrosion.c_l_eq_norm) * delta_eq * hphi_d
    omega, kappa_phi = sim._omega_kappa_phi()
    phi_nonlin = chem_drive + omega * double_well_prime(phi, clamp_input=False)
    gx_eta_um, gy_eta_um = grad_xy(
        eta,
        sim.dx_um,
        sim.dy_um,
        x_coords=sim.x_coords_um,
        y_coords=sim.y_coords_um,
        bc=sim._scalar_bc(),
    )
    twin_grad_e = 0.5 * cfg.twinning.kappa_eta * (gx_eta_um * gx_eta_um + gy_eta_um * gy_eta_um)
    phi_nonlin = phi_nonlin + hphi_d * twin_grad_e
    L_phi = torch.clamp(sim._corrosion_mobility(state=nxt, mech=mech), min=0.0)
    phi_lap_kappa = sim._div_diffusive_fv(
        phi,
        torch.full_like(phi, float(kappa_phi)),
        dx=sim.dx_m,
        dy=sim.dy_m,
        x_coords=sim.x_coords_m,
        y_coords=sim.y_coords_m,
        bc=sim._scalar_bc(),
    )
    r_expected = (phi - prev["phi"]) / max(dt, 1e-12) + L_phi * (phi_nonlin - phi_lap_kappa)
    pde_expected = sim._masked_mean_abs(r_expected) * dt
    assert abs(float(got["pde_phi"]) - float(pde_expected)) < 1e-9


def test_free_energy_includes_mech_phi_coupling_when_enabled() -> None:
    """验证 include_mech_term_in_free_energy 会写入与 phi 方程一致的机械耦合能项。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 36
    cfg.domain.ny = 24
    cfg.corrosion.include_mech_term_in_phi_variation = True
    cfg.corrosion.include_mech_term_in_free_energy = True
    sim = CoupledSimulator(cfg)
    state = {k: v.clone() for k, v in sim.state.items()}
    mech = sim._estimate_state_mechanics(state)

    e_with = sim._compute_free_energy(state, mech)
    cfg.corrosion.include_mech_term_in_free_energy = False
    sim_ref = CoupledSimulator(cfg)
    sim_ref.state = {k: v.clone() for k, v in state.items()}
    e_without = sim_ref._compute_free_energy(state, mech)

    hphi = smooth_heaviside(torch.clamp(state["phi"], 0.0, 1.0))
    e_mech = sim._mech_phi_coupling_energy_density(mech)
    expected_delta = -float(sim._mean_field(hphi * e_mech).item())
    got_delta = float(e_with - e_without)
    assert abs(got_delta - expected_delta) < 1e-6


def test_energy_gate_auto_skips_when_nonvariational_mech_term_enabled() -> None:
    """当机械驱动未写入自由能时，energy gate 应自动跳过。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 40
    cfg.domain.ny = 28
    cfg.ml.enabled = False
    cfg.ml.max_field_delta = 10.0
    cfg.ml.max_mean_phi_drop = 10.0
    cfg.ml.max_mean_eta_rise = 10.0
    cfg.ml.max_mean_phi_abs_delta = 10.0
    cfg.ml.max_mean_eta_abs_delta = 10.0
    cfg.ml.max_mean_c_abs_delta = 10.0
    cfg.ml.max_mean_epspeq_abs_delta = 10.0
    cfg.ml.residual_gate = 10.0
    cfg.ml.enable_pde_residual_gate = False
    cfg.ml.enable_energy_gate = True
    cfg.ml.energy_gate_use_variational_core = False
    cfg.corrosion.include_mech_term_in_phi_variation = True
    cfg.corrosion.include_mech_term_in_free_energy = False
    sim = CoupledSimulator(cfg)
    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in prev.items()}
    nxt["phi"] = torch.clamp(prev["phi"] + 0.02, 0.0, 1.0)
    _ = sim._surrogate_update_is_valid(prev, nxt)
    assert sim.stats["ml_energy_gate_skipped_nonvariational"] > 0


def test_phi_imex_l0_strategy_max_not_lower_than_mean() -> None:
    """L0=max 策略在同一场上不应小于 mean 策略。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 12
    cfg.domain.ny = 8
    sim = CoupledSimulator(cfg)
    l = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 2.0]]]], dtype=torch.float32)
    cfg.numerics.phi_imex_l0_safety = 1.0
    cfg.numerics.phi_imex_l0_strategy = "mean"
    l_mean = sim._select_phi_imex_l0(l)
    cfg.numerics.phi_imex_l0_strategy = "max"
    l_max = sim._select_phi_imex_l0(l)
    assert l_max >= l_mean
    assert abs(l_max - 2.0) < 1e-6


def test_mechanics_dirichlet_nonzero_right_value_enters_ux_gradient() -> None:
    """验证非零右边界位移会进入 ux 导数离散，而非被当作 0。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.mechanics.loading_mode = "dirichlet_x"
    cfg.mechanics.dirichlet_right_displacement_um = 1.6
    cfg.mechanics.mechanics_bc = "neumann"
    mech = MechanicsModel(cfg)

    h, w = 10, 24
    ux_line = torch.linspace(0.0, cfg.mechanics.dirichlet_right_displacement_um, w, dtype=torch.float32).view(1, 1, 1, -1)
    ux = ux_line.expand(1, 1, h, w).clone()
    uy = torch.zeros_like(ux)
    eps = mech._disp_strain_only(ux, uy, dx_um=1.0, dy_um=1.0, x_coords_um=None, y_coords_um=None)
    ex = eps["eps_xx"]
    # 对线性位移场，内部导数应为常数；最右端一侧差分也应为正，不应塌缩为 0。
    interior_mean = float(torch.mean(ex[:, :, :, 1:-1]).item())
    right_mean = float(torch.mean(ex[:, :, :, -1]).item())
    assert interior_mean > 1e-4
    assert right_mean > 1e-4


def test_diffusion_rhs_mu_form_matches_explicit_mu_divergence() -> None:
    """化学势一致写法应等价于 `div(M*grad(mu))` 的显式实现。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.ml.enabled = False
    cfg.domain.nx = 32
    cfg.domain.ny = 24
    sim = CoupledSimulator(cfg)
    phi = torch.clamp(sim.state["phi"] * 0.85 + 0.1, 0.0, 1.0)
    c = torch.clamp(sim.state["c"] * 0.9 + 0.02, cfg.corrosion.cMg_min, cfg.corrosion.cMg_max)
    hphi = smooth_heaviside(phi)
    hphi_d = smooth_heaviside_prime(phi)
    D = cfg.corrosion.D_s_m2_s * hphi + (1.0 - hphi) * cfg.corrosion.D_l_m2_s
    gx_phi, gy_phi = grad_xy(
        phi,
        sim.dx_m,
        sim.dy_m,
        x_coords=sim.x_coords_m,
        y_coords=sim.y_coords_m,
        bc=sim._scalar_bc(),
    )
    corr = hphi_d * (cfg.corrosion.c_l_eq_norm - cfg.corrosion.c_s_eq_norm)

    cfg.corrosion.concentration_use_mu_form = True
    rhs_mu = sim._diffusion_rhs(c, D, corr, gx_phi, gy_phi, phi)
    A_mpa = cfg.corrosion.A_J_m3 / 1e6
    mu = A_mpa * (c - hphi * (cfg.corrosion.c_s_eq_norm - cfg.corrosion.c_l_eq_norm) - cfg.corrosion.c_l_eq_norm)
    M = D / max(A_mpa, 1e-12)
    rhs_ref = sim._div_diffusive_fv(
        mu,
        M,
        dx=sim.dx_m,
        dy=sim.dy_m,
        x_coords=sim.x_coords_m,
        y_coords=sim.y_coords_m,
        bc=sim._scalar_bc(),
    )
    err = float(torch.max(torch.abs(rhs_mu - rhs_ref)).item())
    assert err < 1e-8


def test_mechanics_stress_divergence_uses_traction_free_on_neumann_boundaries() -> None:
    """Neumann 轴上应按 traction-free 处理边界面通量。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.mechanics.mechanics_bc = "neumann"
    mech = MechanicsModel(cfg)
    h, w = 12, 18
    sxx = torch.ones((1, 1, h, w), dtype=torch.float32)
    syy = torch.zeros_like(sxx)
    sxy = torch.zeros_like(sxx)
    div_x, div_y = mech.stress_divergence(sxx, syy, sxy, dx_um=1.0, dy_um=1.0)
    # 常值 sigma_xx 在 traction-free x 边界下，内部散度为 0，边界两列出现符号相反的通量跳变。
    interior = float(torch.max(torch.abs(div_x[:, :, :, 1:-1])).item())
    left = float(torch.mean(div_x[:, :, :, 0]).item())
    right = float(torch.mean(div_x[:, :, :, -1]).item())
    assert interior < 1e-6
    assert left > 0.5
    assert right < -0.5
    assert float(torch.max(torch.abs(div_y)).item()) < 1e-6


def test_cp_phase_dependent_strength_affects_plastic_rate() -> None:
    """孪晶相强度缩放开启时，eta 改变应影响塑性应变率。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.crystal_plasticity.use_phase_dependent_strength = True
    cfg.crystal_plasticity.matrix_crss_scale = 1.0
    cfg.crystal_plasticity.twin_crss_scale = 0.7
    cfg.crystal_plasticity.matrix_hardening_scale = 1.0
    cfg.crystal_plasticity.twin_hardening_scale = 1.0
    cp = CrystalPlasticityModel(cfg, device=torch.device("cpu"), dtype=torch.float32)
    cp_state = cp.init_state(ny=8, nx=10)
    sigma_xx = torch.full((1, 1, 8, 10), 16.0)
    sigma_yy = torch.zeros_like(sigma_xx)
    sigma_xy = torch.full_like(sigma_xx, 18.0)
    eta_m = torch.zeros_like(sigma_xx)
    eta_t = torch.ones_like(sigma_xx)
    out_m = cp.update(cp_state, sigma_xx, sigma_yy, sigma_xy, eta_m, dt_s=1e-4)
    out_t = cp.update(cp_state, sigma_xx, sigma_yy, sigma_xy, eta_t, dt_s=1e-4)
    m_rate = float(torch.mean(out_m["epspeq_dot"]).item())
    t_rate = float(torch.mean(out_t["epspeq_dot"]).item())
    assert t_rate > m_rate


def test_twin_nucleation_activation_positive_shear_only() -> None:
    """孪晶形核激活仅应对有利剪切方向开启。"""
    cfg = load_config("configs/notch_case.yaml")
    sim = CoupledSimulator(cfg)
    tau = torch.tensor([[[[-50.0, 0.0, 20.0, 30.0, 45.0]]]], dtype=torch.float32)
    act = sim._twin_nucleation_activation(tau)
    # twin_crss=30：<=30 时为 0，>30 时线性开启。
    assert float(act[0, 0, 0, 0].item()) == 0.0
    assert float(act[0, 0, 0, 1].item()) == 0.0
    assert float(act[0, 0, 0, 2].item()) == 0.0
    assert float(act[0, 0, 0, 3].item()) == 0.0
    assert float(act[0, 0, 0, 4].item()) > 0.0


def test_surrogate_limit_freezes_plastic_fields_by_default() -> None:
    """默认配置下 surrogate 不应直接更新塑性历史场。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.ml.enabled = False
    cfg.ml.surrogate_update_mechanics_fields = True
    cfg.ml.surrogate_update_plastic_fields = False
    sim = CoupledSimulator(cfg)
    prev = {k: v.clone() for k, v in sim.state.items()}
    nxt = {k: v.clone() for k, v in sim.state.items()}
    nxt["ux"] = prev["ux"] + 0.02
    nxt["uy"] = prev["uy"] - 0.02
    nxt["epspeq"] = prev["epspeq"] + 0.2
    nxt["epsp_xx"] = prev["epsp_xx"] + 0.2
    nxt["epsp_yy"] = prev["epsp_yy"] - 0.2
    nxt["epsp_xy"] = prev["epsp_xy"] + 0.2
    out = sim._limit_surrogate_update(prev, nxt)
    # 位移可更新（受限幅），塑性场必须回退到 prev。
    assert float(torch.mean(torch.abs(out["ux"] - prev["ux"])).item()) > 0.0
    assert float(torch.mean(torch.abs(out["uy"] - prev["uy"])).item()) > 0.0
    assert torch.allclose(out["epspeq"], prev["epspeq"])
    assert torch.allclose(out["epsp_xx"], prev["epsp_xx"])
    assert torch.allclose(out["epsp_yy"], prev["epsp_yy"])
    assert torch.allclose(out["epsp_xy"], prev["epsp_xy"])


def test_surrogate_accept_syncs_cp_state_when_plastic_enabled() -> None:
    """当允许 surrogate 更新塑性场时，accept 后应同步推进 CP 内变量。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.ml.enabled = False
    cfg.ml.surrogate_update_plastic_fields = True
    cfg.ml.surrogate_sync_cp_state_on_accept = True
    sim = CoupledSimulator(cfg)
    # 构造高应力，确保 CP 发生非零滑移累积。
    shape = sim.state["phi"].shape
    dev = sim.state["phi"].device
    dtp = sim.state["phi"].dtype
    mech_for_cp = {
        "sigma_xx": torch.full(shape, 40.0, device=dev, dtype=dtp),
        "sigma_yy": torch.full(shape, 5.0, device=dev, dtype=dtp),
        "sigma_xy": torch.full(shape, 60.0, device=dev, dtype=dtp),
    }
    g0 = sim.cp_state["g"].clone()
    ga0 = sim.cp_state["gamma_accum"].clone()
    sim._sync_cp_state_on_surrogate_accept(dt_s=cfg.numerics.dt_s, mech_for_cp=mech_for_cp)
    # 至少应使累计滑移增加；g 一般也会增大（硬化）。
    assert float(torch.mean(sim.cp_state["gamma_accum"] - ga0).item()) > 0.0
    assert float(torch.mean(sim.cp_state["g"] - g0).item()) >= 0.0


def test_load_config_ignores_deprecated_unknown_fields(tmp_path: Path) -> None:
    """旧配置多余键（如 h_power）不应导致 load_config 失败。"""
    cfg_text = """
domain:
  nx: 16
  ny: 12
twinning:
  h_power: 5
  twin_crss_MPa: 31.0
"""
    p = tmp_path / "legacy_cfg.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.domain.nx == 16
    assert cfg.domain.ny == 12
    assert abs(cfg.twinning.twin_crss_MPa - 31.0) < 1e-12


def test_mechanics_anisotropic_hcp_stress_is_finite() -> None:
    """开启 HCP 各向异性后，应力计算应保持有限值并返回完整分量。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.mechanics.use_anisotropic_hcp = True
    cfg.mechanics.plane_strain = True
    mech = MechanicsModel(cfg)
    phi = torch.ones((1, 1, 5, 6), dtype=torch.float32)
    ex = torch.full_like(phi, 0.006)
    ey = torch.full_like(phi, -0.0015)
    exy = torch.full_like(phi, 0.001)
    eta = torch.zeros_like(phi)
    sig = mech.constitutive_stress(phi, ex, ey, exy, eta=eta)
    for k in ("sigma_xx", "sigma_yy", "sigma_xy", "sigma_zz", "sigma_h"):
        assert torch.isfinite(sig[k]).all()


def test_mechanics_anisotropic_twin_blend_changes_stress() -> None:
    """各向异性+孪晶刚度混合开启时，eta 从 0->1 应改变应力响应。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.mechanics.use_anisotropic_hcp = True
    cfg.mechanics.anisotropic_blend_with_twin = True
    cfg.mechanics.crystal_orientation_euler_deg = [20.0, 25.0, 10.0]
    # 避免绕 c 轴旋转导致 TI 刚度不变，测试时使用 x 轴重取向。
    cfg.mechanics.twin_reorientation_axis = [1.0, 0.0, 0.0]
    cfg.mechanics.twin_reorientation_angle_deg = 32.0
    mech = MechanicsModel(cfg)
    phi = torch.ones((1, 1, 5, 6), dtype=torch.float32)
    ex = torch.full_like(phi, 0.004)
    ey = torch.full_like(phi, -0.001)
    exy = torch.full_like(phi, 0.002)
    eta0 = torch.zeros_like(phi)
    eta1 = torch.ones_like(phi)
    s0 = mech.constitutive_stress(phi, ex, ey, exy, eta=eta0)
    s1 = mech.constitutive_stress(phi, ex, ey, exy, eta=eta1)
    delta = float(torch.mean(torch.abs(s1["sigma_xx"] - s0["sigma_xx"])).item())
    assert delta > 1e-4


def test_omega_kappa_phi_respects_tanh_half_width_convention() -> None:
    """interface_thickness_definition=tanh_half_width 时应使用解析定标关系。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.corrosion.gamma_J_m2 = 0.5
    cfg.corrosion.interface_thickness_um = 2.5
    cfg.corrosion.interface_thickness_definition = "tanh_half_width"
    sim = CoupledSimulator(cfg)
    omega, kappa = sim._omega_kappa_phi()
    ell_m = cfg.corrosion.interface_thickness_um * 1e-6
    omega_ref = 3.0 * cfg.corrosion.gamma_J_m2 / (8.0 * ell_m) / 1e6
    kappa_ref = 3.0 * cfg.corrosion.gamma_J_m2 * ell_m / 1e6
    assert abs(omega - omega_ref) / max(abs(omega_ref), 1e-12) < 1e-10
    assert abs(kappa - kappa_ref) / max(abs(kappa_ref), 1e-12) < 1e-10


def test_corrosion_mobility_uses_solid_sigma_when_enabled() -> None:
    """开启 use_solid_sigma_for_corrosion 后，应优先使用未掩膜固相应力。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.ml.enabled = False
    sim = CoupledSimulator(cfg)
    st = {k: v.clone() for k, v in sim.state.items()}
    mech = {
        "sigma_h": torch.zeros_like(st["phi"]),
        "sigma_h_solid": torch.full_like(st["phi"], 120.0),
    }
    cfg.corrosion.use_solid_sigma_for_corrosion = False
    l_masked = sim._corrosion_mobility(state=st, mech=mech)
    cfg.corrosion.use_solid_sigma_for_corrosion = True
    l_solid = sim._corrosion_mobility(state=st, mech=mech)
    assert float(torch.mean(l_solid - l_masked).item()) > 0.0


def test_total_epspeq_dot_includes_twin_rate_component() -> None:
    """总等效塑性速率应包含孪晶转变应变速率贡献。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 48
    cfg.domain.ny = 32
    cfg.numerics.n_steps = 1
    cfg.numerics.dt_s = 2e-4
    cfg.ml.enabled = False
    # 关闭滑移塑性，仅保留孪晶项，便于验证分解逻辑。
    cfg.crystal_plasticity.gamma0_s_inv = 0.0
    cfg.twinning.twin_crss_MPa = -1e6
    cfg.twinning.nucleation_source_amp = 2e-3
    cfg.corrosion.epspeq_twin_weight = 1.0

    sim = CoupledSimulator(cfg)
    sim.step(1)
    twin_dot = sim.state["epspeq_dot_twin"]
    slip_dot = sim.state["epspeq_dot_slip"]
    total_dot = sim.state["epspeq_dot"]
    assert float(torch.mean(twin_dot).item()) > 0.0
    assert float(torch.max(torch.abs(slip_dot)).item()) < 1e-10
    assert float(torch.mean(torch.abs(total_dot - twin_dot)).item()) < 1e-8


def test_mechanics_anisotropic_plane_stress_enforces_sigma_zz_zero() -> None:
    """各向异性平面应力分支应满足 sigma_zz=0（通过 Schur 消元）。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.mechanics.use_anisotropic_hcp = True
    cfg.mechanics.plane_strain = False
    mech = MechanicsModel(cfg)
    phi = torch.ones((1, 1, 4, 5), dtype=torch.float32)
    eta = torch.zeros_like(phi)
    ex = torch.full_like(phi, 0.004)
    ey = torch.full_like(phi, -0.001)
    exy = torch.full_like(phi, 0.0015)
    sig = mech.constitutive_stress(phi, ex, ey, exy, eta=eta)
    assert float(torch.max(torch.abs(sig["sigma_zz_solid"])).item()) < 1e-8


def test_multivariant_eta_state_consistency_and_sum_constraint() -> None:
    """多孪晶变体开启后，应保持 eta=Σeta_v 且可选约束 Σeta_v<=1。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 40
    cfg.domain.ny = 28
    cfg.numerics.n_steps = 2
    cfg.numerics.dt_s = 2e-4
    cfg.ml.enabled = False
    cfg.twinning.n_variants = 2
    cfg.twinning.twin_variant_indices = [0, 1]
    cfg.twinning.enforce_variant_sum_le_one = True
    # 提高孪晶激活概率，便于触发两变体更新。
    cfg.twinning.twin_crss_MPa = -1e6
    cfg.twinning.nucleation_source_amp = 1e-3
    sim = CoupledSimulator(cfg)
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    assert "eta_v1" in sim.state and "eta_v2" in sim.state
    eta_sum = sim.state["eta_v1"] + sim.state["eta_v2"]
    err = float(torch.max(torch.abs(sim.state["eta"] - eta_sum)).item())
    assert err < 1e-6
    assert float(torch.max(eta_sum).item()) <= 1.0001


def test_multivariant_short_run_remains_finite() -> None:
    """多孪晶变体短程运行应保持有限值并完成力学-孪晶耦合。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 48
    cfg.domain.ny = 32
    cfg.numerics.n_steps = 3
    cfg.numerics.dt_s = 1e-4
    cfg.ml.enabled = False
    cfg.twinning.n_variants = 2
    cfg.twinning.twin_variant_indices = [0, 1]
    sim = CoupledSimulator(cfg)
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    for k in ("phi", "c", "eta", "eta_v1", "eta_v2", "epspeq"):
        assert torch.isfinite(sim.state[k]).all()
