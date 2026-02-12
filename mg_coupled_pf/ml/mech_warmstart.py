"""力学初值器（Mechanics Warmstart）模块。

用途：
- 学习从微结构场到位移场初值的映射；
- 仅作为 Krylov 线性求解初值，不替代力学平衡方程。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .surrogate import TinyUNet, _maybe_compile


MECH_INPUT_ORDER = ["phi", "c", "eta", "epspeq", "epsp_xx", "epsp_yy", "epsp_xy"]
MECH_OUTPUT_ORDER = ["ux", "uy"]


def mech_state_to_tensor(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """将状态字典打包为力学初值器输入张量。"""
    if not state:
        raise ValueError("mech_state_to_tensor received empty state.")
    ref = next(iter(state.values()))
    chans: List[torch.Tensor] = []
    for k in MECH_INPUT_ORDER:
        if k in state:
            chans.append(state[k])
        else:
            chans.append(torch.zeros_like(ref))
    return torch.cat(chans, dim=1)


def tensor_to_mech_fields(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """将网络输出拆分为 ux/uy 两个通道。"""
    return {"ux": x[:, 0:1], "uy": x[:, 1:2]}


@dataclass
class MechanicsWarmstartPredictor:
    """力学初值器推理包装器。"""

    model: nn.Module
    device: torch.device
    channels_last: bool = False
    add_coord_features: bool = True
    hidden: int = 24
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _use_dirichlet_x(loading_mode: str) -> bool:
        m = str(loading_mode).strip().lower()
        return m in {"dirichlet_x", "dirichlet", "ux_dirichlet"}

    def predict(
        self,
        state: Dict[str, torch.Tensor],
        *,
        loading_mode: str = "eigenstrain",
        right_disp_um: float = 0.0,
        enforce_anchor: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测位移初值并施加硬边界约束。"""
        self.model.eval()
        amp_ctx = nullcontext()
        with torch.inference_mode(), amp_ctx:
            x = mech_state_to_tensor(state)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = self.model(x)
        out = tensor_to_mech_fields(y)
        ux = torch.nan_to_num(out["ux"], nan=0.0, posinf=0.0, neginf=0.0)
        uy = torch.nan_to_num(out["uy"], nan=0.0, posinf=0.0, neginf=0.0)
        # 与力学求解器一致的硬约束，避免初值破坏边界条件。
        ux[:, :, :, 0] = 0.0
        if self._use_dirichlet_x(loading_mode):
            ux[:, :, :, -1] = float(right_disp_um)
        if enforce_anchor:
            uy[:, :, 0, 0] = 0.0
        return ux, uy


def build_mechanics_warmstart(
    *,
    device: torch.device,
    hidden: int = 24,
    add_coord_features: bool = True,
    use_torch_compile: bool = False,
    channels_last: bool = False,
) -> MechanicsWarmstartPredictor:
    """构建力学初值器。"""
    model = TinyUNet(
        in_ch=len(MECH_INPUT_ORDER),
        hidden=int(hidden),
        out_ch=len(MECH_OUTPUT_ORDER),
        add_coords=bool(add_coord_features),
    ).to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile(model, use_torch_compile)
    return MechanicsWarmstartPredictor(
        model=model,
        device=device,
        channels_last=channels_last and device.type == "cuda",
        add_coord_features=bool(add_coord_features),
        hidden=int(hidden),
    )


def save_mechanics_warmstart(predictor: MechanicsWarmstartPredictor, path: str | Path) -> None:
    """保存力学初值器模型与元信息。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.model.state_dict(),
            "meta": {
                "hidden": int(predictor.hidden),
                "add_coord_features": bool(predictor.add_coord_features),
                "input_order": list(MECH_INPUT_ORDER),
                "output_order": list(MECH_OUTPUT_ORDER),
            },
        },
        p,
    )


def load_mechanics_warmstart(
    path: str | Path,
    *,
    device: torch.device,
    fallback_hidden: int = 24,
    fallback_add_coord_features: bool = True,
    use_torch_compile: bool = False,
    channels_last: bool = False,
) -> MechanicsWarmstartPredictor:
    """加载力学初值器。"""
    payload = torch.load(Path(path), map_location=device)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    hidden = int(meta.get("hidden", fallback_hidden))
    add_coord = bool(meta.get("add_coord_features", fallback_add_coord_features))
    predictor = build_mechanics_warmstart(
        device=device,
        hidden=hidden,
        add_coord_features=add_coord,
        use_torch_compile=False,
        channels_last=channels_last,
    )
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model_sd = predictor.model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            filtered[k] = v
    predictor.model.load_state_dict(filtered, strict=False)
    predictor.model.eval()
    predictor.model = _maybe_compile(predictor.model, use_torch_compile)
    predictor.hidden = hidden
    predictor.add_coord_features = add_coord
    return predictor

