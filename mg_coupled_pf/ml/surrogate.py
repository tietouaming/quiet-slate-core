"""ML surrogate 模型模块（中文注释版）。

作用：
- 将多物理状态场映射到统一张量表示；
- 提供多种代理网络（TinyUNet / DWUNet / FNO2D / AFNO2D）；
- 统一封装模型构建、推理、保存与加载逻辑。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import platform
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operators import smooth_heaviside
from .scaling import sanitize_field_scales


FIELD_ORDER = ["phi", "c", "eta", "ux", "uy", "epspeq", "epsp_xx", "epsp_yy", "epsp_xy"]


def state_to_tensor(state: Dict[str, torch.Tensor], field_scales: Dict[str, float] | None = None) -> torch.Tensor:
    """按固定通道顺序将状态字典拼接为网络输入。"""
    if not state:
        raise ValueError("state_to_tensor received empty state.")
    scales = sanitize_field_scales(FIELD_ORDER, field_scales)
    ref = next(iter(state.values()))
    chans: List[torch.Tensor] = []
    for k in FIELD_ORDER:
        if k in state:
            chans.append(state[k] / scales[k])
        else:
            # 兼容旧状态字典：缺失通道以零场填充。
            chans.append(torch.zeros_like(ref))
    return torch.cat(chans, dim=1)


def tensor_to_state(x: torch.Tensor, field_scales: Dict[str, float] | None = None) -> Dict[str, torch.Tensor]:
    """将网络输出张量拆回状态字典。"""
    scales = sanitize_field_scales(FIELD_ORDER, field_scales)
    out: Dict[str, torch.Tensor] = {}
    for i, k in enumerate(FIELD_ORDER):
        out[k] = x[:, i : i + 1] * scales[k]
    return out


def _normalize_arch(name: str) -> str:
    """将模型架构别名归一化为标准名称。"""
    n = str(name).strip().lower()
    if n in {"tiny_unet", "unet", "tinyunet"}:
        return "tiny_unet"
    if n in {"dw_unet", "mobile_unet", "depthwise_unet"}:
        return "dw_unet"
    if n in {"fno2d", "fno", "fno_lite", "fno-lite"}:
        return "fno2d"
    if n in {"afno2d", "afno", "afno_lite", "afno-lite"}:
        return "afno2d"
    raise ValueError(f"Unsupported surrogate architecture: {name}")


def arch_requires_full_precision(model_arch: str) -> bool:
    """判断该架构是否应在推理时禁用混合精度。"""
    return _normalize_arch(model_arch) in {"fno2d", "afno2d"}


def _coord_channels_like(x: torch.Tensor) -> torch.Tensor:
    """构造与输入同批量/分辨率的归一化坐标通道 `[x/L, y/L]`。"""
    b, _, h, w = x.shape
    yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([xx, yy], dim=1)


def _soft_project_01(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """可微 [0,1] 软投影，区间内近似恒等、区间外平滑回推。"""
    y = x - F.softplus(x - 1.0, beta=beta) / beta + F.softplus(-x, beta=beta) / beta
    return y


class ConvBlock(nn.Module):
    """基础卷积块。"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """轻量 U-Net。"""
    def __init__(self, in_ch: int, hidden: int = 32, out_ch: int | None = None, add_coords: bool = True):
        super().__init__()
        self.add_coords = bool(add_coords)
        in_proj_ch = int(in_ch + (2 if self.add_coords else 0))
        o = int(in_ch if out_ch is None else out_ch)
        self.enc1 = ConvBlock(in_proj_ch, hidden)
        self.down = nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(hidden * 2, hidden * 2)
        self.up = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.dec = ConvBlock(hidden * 2, hidden)
        self.out = nn.Conv2d(hidden, o, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_coords:
            x = torch.cat([x, _coord_channels_like(x)], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        u = self.up(e2)
        cat = torch.cat([u, e1], dim=1)
        d = self.dec(cat)
        return self.out(d)


class DWConvBlock(nn.Module):
    """深度可分离卷积块。"""
    def __init__(self, c_in: int, c_out: int, expansion: int = 2):
        super().__init__()
        mid = max(c_in, c_out) * max(1, int(expansion))
        self.pw1 = nn.Conv2d(c_in, mid, 1)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.pw2 = nn.Conv2d(mid, c_out, 1)
        groups = 4 if c_out % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.pw1(x))
        x = F.gelu(self.dw(x))
        x = self.pw2(x)
        x = self.norm(x)
        return F.gelu(x)


class DWUNet(nn.Module):
    """深度可分离卷积版 U-Net。"""
    def __init__(
        self,
        in_ch: int,
        hidden: int = 24,
        depth: int = 2,
        out_ch: int | None = None,
        add_coords: bool = True,
    ):
        super().__init__()
        self.add_coords = bool(add_coords)
        in_proj_ch = int(in_ch + (2 if self.add_coords else 0))
        o = int(in_ch if out_ch is None else out_ch)
        h = max(8, int(hidden))
        d = max(1, int(depth))
        self.stem = nn.Conv2d(in_proj_ch, h, 3, padding=1)
        self.enc = nn.ModuleList([DWConvBlock(h, h) for _ in range(d)])
        self.down = nn.Conv2d(h, h * 2, 3, stride=2, padding=1)
        self.mid = nn.ModuleList([DWConvBlock(h * 2, h * 2) for _ in range(d + 1)])
        self.up = nn.ConvTranspose2d(h * 2, h, 4, stride=2, padding=1)
        self.dec_in = DWConvBlock(h * 2, h)
        self.dec = nn.ModuleList([DWConvBlock(h, h) for _ in range(max(0, d - 1))])
        self.out = nn.Conv2d(h, o, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_coords:
            x = torch.cat([x, _coord_channels_like(x)], dim=1)
        x0 = F.gelu(self.stem(x))
        x1 = x0
        for blk in self.enc:
            x1 = x1 + blk(x1)
        x2 = self.down(x1)
        for blk in self.mid:
            x2 = x2 + blk(x2)
        x3 = self.up(x2)
        x4 = torch.cat([x3, x1], dim=1)
        x4 = self.dec_in(x4)
        for blk in self.dec:
            x4 = x4 + blk(x4)
        return self.out(x4)


class SpectralConv2d(nn.Module):
    """频域卷积层（FNO 核心算子）。"""
    def __init__(self, in_ch: int, out_ch: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.modes_x = max(1, int(modes_x))
        self.modes_y = max(1, int(modes_y))
        scale = 1.0 / max(1, in_ch * out_ch)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(self.in_ch, self.out_ch, self.modes_y, self.modes_x, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(self.in_ch, self.out_ch, self.modes_y, self.modes_x, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, self.out_ch, h, w // 2 + 1, device=x.device, dtype=torch.cfloat)

        my = min(self.modes_y, h)
        mx = min(self.modes_x, w // 2 + 1)
        if my > 0 and mx > 0:
            out_ft[:, :, :my, :mx] = self._compl_mul2d(x_ft[:, :, :my, :mx], self.weight_pos[:, :, :my, :mx])
            out_ft[:, :, -my:, :mx] = self._compl_mul2d(x_ft[:, :, -my:, :mx], self.weight_neg[:, :, :my, :mx])

        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    """FNO 残差块。"""
    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes_x=modes_x, modes_y=modes_y)
        self.w = nn.Conv2d(width, width, 1)
        groups = 4 if width % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.w(x)
        y = self.norm(y)
        return F.gelu(y)


class FNO2D(nn.Module):
    """二维 Fourier Neural Operator。"""
    def __init__(self, in_ch: int, width: int = 32, modes_x: int = 24, modes_y: int = 16, depth: int = 4, out_ch: int | None = None):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(self.in_ch if out_ch is None else out_ch)
        self.width = max(8, int(width))
        self.modes_x = max(2, int(modes_x))
        self.modes_y = max(2, int(modes_y))
        self.depth = max(1, int(depth))
        self.in_proj = nn.Conv2d(self.in_ch + 2, self.width, 1)
        self.blocks = nn.ModuleList([FNOBlock(self.width, self.modes_x, self.modes_y) for _ in range(self.depth)])
        self.mid_proj = nn.Conv2d(self.width, self.width, 1)
        self.out_proj = nn.Conv2d(self.width, self.out_ch, 1)

    def _coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, self._coord_grid(x)], dim=1)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = F.gelu(self.mid_proj(x))
        return self.out_proj(x)


class AFNOLiteBlock(nn.Module):
    """轻量 AFNO 频域块。"""
    def __init__(self, width: int, modes_x: int, modes_y: int, expansion: float = 2.0):
        super().__init__()
        self.width = int(width)
        self.modes_x = max(1, int(modes_x))
        self.modes_y = max(1, int(modes_y))
        hid = max(8, int(round(self.width * float(expansion))))
        self.fc1 = nn.Linear(2 * self.width, 2 * hid)
        self.fc2 = nn.Linear(2 * hid, 2 * self.width)
        self.local = nn.Conv2d(self.width, self.width, 1)
        groups = 4 if self.width % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, self.width)

    def _mix(self, x_ft: torch.Tensor, y_slice: slice, x_slice: slice) -> torch.Tensor:
        z = x_ft[:, :, y_slice, x_slice]
        if z.numel() == 0:
            return z
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, Y, X, C]
        zr = z.real
        zi = z.imag
        feat = torch.cat([zr, zi], dim=-1)
        feat = F.gelu(self.fc1(feat))
        feat = self.fc2(feat)
        fr, fi = torch.chunk(feat, 2, dim=-1)
        out = torch.complex(fr, fi).permute(0, 3, 1, 2).contiguous()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        out_ft = x_ft.clone()
        my = min(self.modes_y, h)
        mx = min(self.modes_x, w // 2 + 1)
        if my > 0 and mx > 0:
            out_ft[:, :, :my, :mx] = self._mix(x_ft, slice(0, my), slice(0, mx))
            out_ft[:, :, -my:, :mx] = self._mix(x_ft, slice(h - my, h), slice(0, mx))
        y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        y = y + self.local(x.float())
        y = self.norm(y)
        return F.gelu(y).to(dtype=x.dtype)


class AFNO2D(nn.Module):
    """二维 AFNO 网络。"""
    def __init__(
        self,
        in_ch: int,
        width: int = 24,
        modes_x: int = 20,
        modes_y: int = 12,
        depth: int = 4,
        expansion: float = 2.0,
        out_ch: int | None = None,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(self.in_ch if out_ch is None else out_ch)
        self.width = max(8, int(width))
        self.modes_x = max(2, int(modes_x))
        self.modes_y = max(2, int(modes_y))
        self.depth = max(1, int(depth))
        self.expansion = float(expansion)
        self.in_proj = nn.Conv2d(self.in_ch + 2, self.width, 1)
        self.blocks = nn.ModuleList(
            [AFNOLiteBlock(self.width, self.modes_x, self.modes_y, expansion=self.expansion) for _ in range(self.depth)]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, self.out_ch, 1),
        )

    def _coord_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, self._coord_grid(x)], dim=1)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.out_proj(x)


@dataclass
class SurrogatePredictor:
    """surrogate 推理包装器。"""
    model: nn.Module
    device: torch.device
    residual_gate: float = 2.5
    channels_last: bool = False
    model_arch: str = "tiny_unet"
    add_coord_features: bool = True
    enforce_displacement_constraints: bool = False
    loading_mode: str = "eigenstrain"
    dirichlet_right_ux: float = 0.0
    enforce_uy_anchor: bool = True
    allow_plastic_outputs: bool = True
    field_scales: Dict[str, float] = field(default_factory=dict)
    arch_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.field_scales = sanitize_field_scales(FIELD_ORDER, self.field_scales)

    @staticmethod
    def _use_dirichlet_x(loading_mode: str) -> bool:
        """判断当前加载模式是否需要右端位移 Dirichlet 约束。"""
        m = str(loading_mode).strip().lower()
        return m in {"dirichlet_x", "dirichlet", "ux_dirichlet"}

    def _project_mechanics_constraints(self, nxt: Dict[str, torch.Tensor]) -> None:
        """将 surrogate 输出硬投影到位移约束空间。"""
        if not self.enforce_displacement_constraints:
            return
        if "ux" in nxt:
            nxt["ux"][:, :, :, 0] = 0.0
            if self._use_dirichlet_x(self.loading_mode):
                nxt["ux"][:, :, :, -1] = float(self.dirichlet_right_ux)
        if self.enforce_uy_anchor and "uy" in nxt:
            nxt["uy"][:, :, 0, 0] = 0.0

    def predict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.model.eval()
        if self.device.type == "cuda" and self.model_arch in {"fno2d", "afno2d"}:
            amp_ctx = torch.autocast(device_type="cuda", enabled=False)
        else:
            amp_ctx = nullcontext()
        with torch.inference_mode(), amp_ctx:
            x_base = state_to_tensor(state, field_scales=self.field_scales)
            x = x_base
            if self.model_arch in {"fno2d", "afno2d"}:
                x = x.float()
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            dx = self.model(x)
            x1 = x_base + dx.to(dtype=x_base.dtype)
        nxt = tensor_to_state(x1, field_scales=self.field_scales)
        nxt["phi"] = _soft_project_01(torch.nan_to_num(nxt["phi"], nan=0.5, posinf=1.0, neginf=0.0))
        nxt["c"] = _soft_project_01(torch.nan_to_num(nxt["c"], nan=0.5, posinf=1.0, neginf=0.0))
        nxt["eta"] = _soft_project_01(torch.nan_to_num(nxt["eta"], nan=0.0, posinf=1.0, neginf=0.0))
        nxt["ux"] = torch.nan_to_num(nxt["ux"], nan=0.0, posinf=0.0, neginf=0.0)
        nxt["uy"] = torch.nan_to_num(nxt["uy"], nan=0.0, posinf=0.0, neginf=0.0)
        self._project_mechanics_constraints(nxt)
        nxt["epspeq"] = torch.clamp(nxt["epspeq"], 0.0, 1e6)
        nxt["epsp_xx"] = torch.clamp(torch.nan_to_num(nxt["epsp_xx"], nan=0.0, posinf=0.0, neginf=0.0), min=-1.0, max=1.0)
        nxt["epsp_yy"] = torch.clamp(torch.nan_to_num(nxt["epsp_yy"], nan=0.0, posinf=0.0, neginf=0.0), min=-1.0, max=1.0)
        nxt["epsp_xy"] = torch.clamp(torch.nan_to_num(nxt["epsp_xy"], nan=0.0, posinf=0.0, neginf=0.0), min=-1.0, max=1.0)
        if not self.allow_plastic_outputs:
            # 仅预测微结构与位移，塑性历史量由物理 CP 路径维护。
            nxt["epspeq"] = state["epspeq"]
            nxt["epsp_xx"] = state["epsp_xx"]
            nxt["epsp_yy"] = state["epsp_yy"]
            nxt["epsp_xy"] = state["epsp_xy"]
        # 连续相场门控，避免硬阈值在界面处引入数值跳变。
        solid = smooth_heaviside(nxt["phi"], clamp_input=False)
        nxt["eta"] = nxt["eta"] * solid
        nxt["epspeq"] = nxt["epspeq"] * solid
        nxt["epsp_xx"] = nxt["epsp_xx"] * solid
        nxt["epsp_yy"] = nxt["epsp_yy"] * solid
        nxt["epsp_xy"] = nxt["epsp_xy"] * solid
        return nxt


def _maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    """按平台能力尝试启用 torch.compile。"""
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        p0 = next(model.parameters())
        device_type = p0.device.type
    except StopIteration:
        device_type = "cpu"
    if device_type == "cuda":
        # Windows CUDA wheels commonly lack a working Triton backend.
        if platform.system().lower().startswith("win"):
            return model
        if importlib.util.find_spec("triton") is None:
            return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model


def _build_model(
    model_arch: str,
    in_ch: int,
    out_ch: int,
    *,
    add_coord_features: bool = True,
    hidden: int = 32,
    dw_hidden: int = 24,
    dw_depth: int = 2,
    fno_width: int = 32,
    fno_modes_x: int = 24,
    fno_modes_y: int = 16,
    fno_depth: int = 4,
    afno_width: int = 24,
    afno_modes_x: int = 20,
    afno_modes_y: int = 12,
    afno_depth: int = 4,
    afno_expansion: float = 2.0,
) -> tuple[nn.Module, str, Dict[str, Any]]:
    """按配置构建模型并返回(模型, 架构名, 架构参数)。"""
    arch = _normalize_arch(model_arch)
    if arch == "tiny_unet":
        kwargs = {"hidden": int(hidden), "add_coord_features": bool(add_coord_features)}
        return (
            TinyUNet(
                in_ch=in_ch,
                hidden=kwargs["hidden"],
                out_ch=out_ch,
                add_coords=kwargs["add_coord_features"],
            ),
            arch,
            kwargs,
        )
    if arch == "dw_unet":
        kwargs = {
            "dw_hidden": int(dw_hidden),
            "dw_depth": int(dw_depth),
            "add_coord_features": bool(add_coord_features),
        }
        return (
            DWUNet(
                in_ch=in_ch,
                hidden=kwargs["dw_hidden"],
                depth=kwargs["dw_depth"],
                out_ch=out_ch,
                add_coords=kwargs["add_coord_features"],
            ),
            arch,
            kwargs,
        )
    if arch == "afno2d":
        kwargs = {
            "afno_width": int(afno_width),
            "afno_modes_x": int(afno_modes_x),
            "afno_modes_y": int(afno_modes_y),
            "afno_depth": int(afno_depth),
            "afno_expansion": float(afno_expansion),
        }
        return (
            AFNO2D(
                in_ch=in_ch,
                width=kwargs["afno_width"],
                modes_x=kwargs["afno_modes_x"],
                modes_y=kwargs["afno_modes_y"],
                depth=kwargs["afno_depth"],
                expansion=kwargs["afno_expansion"],
                out_ch=out_ch,
            ),
            arch,
            kwargs,
        )
    kwargs = {
        "fno_width": int(fno_width),
        "fno_modes_x": int(fno_modes_x),
        "fno_modes_y": int(fno_modes_y),
        "fno_depth": int(fno_depth),
    }
    return (
        FNO2D(
            in_ch=in_ch,
            width=kwargs["fno_width"],
            modes_x=kwargs["fno_modes_x"],
            modes_y=kwargs["fno_modes_y"],
            depth=kwargs["fno_depth"],
            out_ch=out_ch,
        ),
        arch,
        kwargs,
    )


def build_surrogate(
    device: torch.device,
    use_torch_compile: bool = False,
    channels_last: bool = False,
    model_arch: str = "tiny_unet",
    add_coord_features: bool = True,
    hidden: int = 32,
    dw_hidden: int = 24,
    dw_depth: int = 2,
    fno_width: int = 32,
    fno_modes_x: int = 24,
    fno_modes_y: int = 16,
    fno_depth: int = 4,
    afno_width: int = 24,
    afno_modes_x: int = 20,
    afno_modes_y: int = 12,
    afno_depth: int = 4,
    afno_expansion: float = 2.0,
    field_scales: Dict[str, float] | None = None,
) -> SurrogatePredictor:
    """构建可直接用于推理的 SurrogatePredictor。"""
    model, arch, arch_kwargs = _build_model(
        model_arch=model_arch,
        in_ch=len(FIELD_ORDER),
        out_ch=len(FIELD_ORDER),
        add_coord_features=add_coord_features,
        hidden=hidden,
        dw_hidden=dw_hidden,
        dw_depth=dw_depth,
        fno_width=fno_width,
        fno_modes_x=fno_modes_x,
        fno_modes_y=fno_modes_y,
        fno_depth=fno_depth,
        afno_width=afno_width,
        afno_modes_x=afno_modes_x,
        afno_modes_y=afno_modes_y,
        afno_depth=afno_depth,
        afno_expansion=afno_expansion,
    )
    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile(model, use_torch_compile)
    return SurrogatePredictor(
        model=model,
        device=device,
        channels_last=channels_last and device.type == "cuda",
        model_arch=arch,
        add_coord_features=bool(add_coord_features),
        field_scales=sanitize_field_scales(FIELD_ORDER, field_scales),
        arch_kwargs=arch_kwargs,
    )


def save_surrogate(predictor: SurrogatePredictor, path: str | Path) -> None:
    """保存 surrogate 权重与元信息。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": predictor.model.state_dict(),
            "meta": {
                "model_arch": predictor.model_arch,
                "arch_kwargs": predictor.arch_kwargs,
                "add_coord_features": bool(predictor.add_coord_features),
                "field_scales": dict(predictor.field_scales),
                "field_order": FIELD_ORDER,
            },
        },
        p,
    )


def load_surrogate(
    path: str | Path,
    device: torch.device,
    use_torch_compile: bool = False,
    *,
    fallback_model_arch: str = "tiny_unet",
    fallback_arch_kwargs: Dict[str, Any] | None = None,
    fallback_field_scales: Dict[str, float] | None = None,
) -> SurrogatePredictor:
    """加载 surrogate 并恢复到可推理状态。"""
    payload = torch.load(Path(path), map_location=device)
    meta: Dict[str, Any] = {}
    if isinstance(payload, dict):
        meta_raw = payload.get("meta")
        if isinstance(meta_raw, dict):
            meta = meta_raw

    arch_kwargs: Dict[str, Any] = {}
    if isinstance(fallback_arch_kwargs, dict):
        arch_kwargs.update(fallback_arch_kwargs)
    meta_kwargs = meta.get("arch_kwargs")
    if isinstance(meta_kwargs, dict):
        arch_kwargs.update(meta_kwargs)
    field_scales: Dict[str, float] = {}
    if isinstance(fallback_field_scales, dict):
        field_scales.update(fallback_field_scales)
    meta_scales = meta.get("field_scales")
    if isinstance(meta_scales, dict):
        field_scales.update({str(k): float(v) for k, v in meta_scales.items()})

    arch = _normalize_arch(str(meta.get("model_arch", fallback_model_arch)))
    add_coord_features = bool(meta.get("add_coord_features", arch_kwargs.get("add_coord_features", True)))
    predictor = build_surrogate(
        device=device,
        use_torch_compile=False,
        channels_last=device.type == "cuda",
        model_arch=arch,
        add_coord_features=add_coord_features,
        hidden=int(arch_kwargs.get("hidden", 32)),
        dw_hidden=int(arch_kwargs.get("dw_hidden", 24)),
        dw_depth=int(arch_kwargs.get("dw_depth", 2)),
        fno_width=int(arch_kwargs.get("fno_width", 32)),
        fno_modes_x=int(arch_kwargs.get("fno_modes_x", 24)),
        fno_modes_y=int(arch_kwargs.get("fno_modes_y", 16)),
        fno_depth=int(arch_kwargs.get("fno_depth", 4)),
        afno_width=int(arch_kwargs.get("afno_width", 24)),
        afno_modes_x=int(arch_kwargs.get("afno_modes_x", 20)),
        afno_modes_y=int(arch_kwargs.get("afno_modes_y", 12)),
        afno_depth=int(arch_kwargs.get("afno_depth", 4)),
        afno_expansion=float(arch_kwargs.get("afno_expansion", 2.0)),
        field_scales=field_scales,
    )
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    # 兼容旧 checkpoint（输入通道数变更时会出现 shape mismatch）。
    model_sd = predictor.model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            filtered[k] = v
    predictor.model.load_state_dict(filtered, strict=False)
    predictor.model.eval()
    predictor.model = _maybe_compile(predictor.model, use_torch_compile)
    predictor.model_arch = arch
    predictor.add_coord_features = bool(add_coord_features)
    predictor.field_scales = sanitize_field_scales(FIELD_ORDER, field_scales)
    predictor.arch_kwargs = dict(arch_kwargs)
    return predictor
