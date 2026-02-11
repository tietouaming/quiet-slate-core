"""训练数据加载兼容性测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train_surrogate import _load_snapshot_tensor


def test_snapshot_loader_fills_missing_epsp_channels(tmp_path: Path) -> None:
    """旧快照缺少 epsp_xx/yy/xy 时应自动零填充。"""
    p = tmp_path / "snapshot_000001.npz"
    arr = np.zeros((1, 1, 8, 10), dtype=np.float32)
    np.savez_compressed(
        p,
        phi=arr + 1.0,
        c=arr + 0.2,
        eta=arr + 0.3,
        ux=arr,
        uy=arr,
        epspeq=arr + 0.05,
    )
    x = _load_snapshot_tensor(p)
    # FIELD_ORDER 长度为 9，末三通道应由零填充得到。
    assert tuple(x.shape) == (1, 9, 8, 10)
    assert float(x[:, 6:9].abs().max().item()) == 0.0
