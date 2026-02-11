"""扩展耦合器模块（中文注释版）。

用于给主求解器提供“可插拔”的多物理扩展入口，目前包含：
- 电场耦合占位实现；
- 氢脆耦合占位实现。

说明：
- 当前实现为工程化接口，便于后续替换为更严格的物理模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import SimulationConfig


class CouplerBase:
    """耦合器抽象基类。"""
    name: str = "base"

    def corrosion_mobility_multiplier(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回腐蚀迁移率倍率场（默认全 1）。"""
        return torch.ones_like(state["phi"])

    def concentration_drift_flux(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """返回浓度漂移通量 (Jx, Jy)（默认全 0）。"""
        z = torch.zeros_like(state["phi"])
        return z, z

    def update_material_overrides(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Example keys: yield_scale, elastic_scale.
        # 返回值会被主求解器合并并传给晶体塑性模块。
        return {}


@dataclass
class ElectricFieldCoupler(CouplerBase):
    """电场耦合器：提供迁移率倍率与漂移通量占位项。"""
    name: str = "electric_field"
    field_strength_V_m: float = 0.0
    mobility_coeff: float = 1e-3

    def corrosion_mobility_multiplier(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Placeholder extension: electric field can accelerate anodic dissolution.
        # 当前为线性增益示例，便于后续替换为 Butler-Volmer 等更严格关系。
        E = abs(self.field_strength_V_m)
        gain = 1.0 + self.mobility_coeff * E
        return torch.full_like(state["phi"], gain)

    def concentration_drift_flux(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Nernst-Planck-like drift term J_drift = mu*z*c*E.
        # Here E is prescribed along +x as an extension hook.
        c = state["c"]
        jx = self.mobility_coeff * c * self.field_strength_V_m
        jy = torch.zeros_like(jx)
        return jx, jy


@dataclass
class HydrogenEmbrittlementCoupler(CouplerBase):
    """氢脆耦合器：提供强度退化和腐蚀加速占位项。"""
    name: str = "hydrogen_embrittlement"
    reference_ppm: float = 0.0
    strength_reduction_coeff: float = 0.15
    corrosion_accel_coeff: float = 0.2

    def _hydrogen_proxy(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Placeholder hydrogen proxy: normalized Mg ion concentration in liquid near interface.
        # 后续可替换为显式氢扩散-陷阱模型输出。
        return torch.clamp(state["c"], 0.0, 1.0)

    def corrosion_mobility_multiplier(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self._hydrogen_proxy(state)
        return 1.0 + self.corrosion_accel_coeff * h

    def update_material_overrides(self, state: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self._hydrogen_proxy(state)
        yield_scale = 1.0 - self.strength_reduction_coeff * h
        return {"yield_scale": torch.clamp(yield_scale, min=0.2)}


def build_couplers(cfg: SimulationConfig) -> List[CouplerBase]:
    """按配置构建启用的耦合器列表。"""
    couplers: List[CouplerBase] = []
    if cfg.extensions.enable_electric_field:
        couplers.append(
            ElectricFieldCoupler(field_strength_V_m=cfg.extensions.electric_field_strength_V_m)
        )
    if cfg.extensions.enable_hydrogen_embrittlement:
        couplers.append(
            HydrogenEmbrittlementCoupler(reference_ppm=cfg.extensions.hydrogen_reference_ppm)
        )
    return couplers
