"""配置审计模块测试。"""

from __future__ import annotations

from pathlib import Path

from mg_coupled_pf import audit_config, load_config, save_audit_report


def test_default_notch_config_has_no_errors() -> None:
    """默认算例配置应通过审计（允许 warning）。"""
    cfg = load_config("configs/notch_case.yaml")
    rep = audit_config(cfg, config_path="configs/notch_case.yaml")
    assert rep.error_count == 0
    assert rep.passed is True


def test_invalid_interface_thickness_mode_is_error() -> None:
    """非法界面厚度口径应触发 error。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.corrosion.interface_thickness_definition = "bad_mode"
    rep = audit_config(cfg)
    assert rep.error_count >= 1
    assert any(x.code == "CORR_IFACE_MODE_INVALID" for x in rep.issues)


def test_underresolved_interface_is_error() -> None:
    """界面厚度分辨率不足应触发 error。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.domain.nx = 128
    cfg.domain.ny = 96
    cfg.domain.lx_um = 400.0
    cfg.domain.ly_um = 250.0
    cfg.corrosion.interface_thickness_um = 0.02
    rep = audit_config(cfg)
    assert any(x.code == "CORR_IFACE_UNDERRESOLVED" for x in rep.issues)


def test_orientation_mismatch_warns() -> None:
    """CP 与力学取向不一致时应触发 warning。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.crystal_plasticity.crystal_orientation_euler_deg = [0.0, 0.0, 0.0]
    cfg.mechanics.crystal_orientation_euler_deg = [30.0, 15.0, 5.0]
    rep = audit_config(cfg)
    assert any(x.code == "ORI_CP_MECH_MISMATCH" for x in rep.issues)


def test_non_orthonormal_slip_system_is_error() -> None:
    """若 slip 系统被破坏为非正交，审计应拦截。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.slip_systems[0]["s_crystal"] = [1.0, 0.0, 0.0]
    cfg.slip_systems[0]["n_crystal"] = [1.0, 0.0, 0.0]
    rep = audit_config(cfg)
    assert any(x.code == "CP_SLIP_SYSTEM_NOT_ORTHONORMAL" for x in rep.issues)


def test_save_audit_report_roundtrip(tmp_path: Path) -> None:
    """审计报告应可写出为 JSON。"""
    cfg = load_config("configs/notch_case.yaml")
    rep = audit_config(cfg, config_path="configs/notch_case.yaml")
    out = save_audit_report(rep, tmp_path / "cfg_audit.json")
    assert out.exists()
    txt = out.read_text(encoding="utf-8")
    assert "config_path" in txt
    assert "counts" in txt

