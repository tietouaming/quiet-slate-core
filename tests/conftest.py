"""Pytest session bootstrap.

确保仓库根目录位于 `sys.path`，便于测试按模块路径导入 `scripts.*`。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

