"""数值稳定性与一致性测试（中文注释版）。"""

from __future__ import annotations

import math

import torch

from mg_coupled_pf import CoupledSimulator, load_config
from mg_coupled_pf.crystal_plasticity import CrystalPlasticityModel
from mg_coupled_pf.mechanics import MechanicsModel


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
