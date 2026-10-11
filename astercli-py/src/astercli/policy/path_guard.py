"""路径围栏（对齐 Java policy.PathGuard 语义）。

定位：HITL 之前的 LLM 输入合法性检查，不是沙箱。
解决三类越界：绝对路径逃出项目根、.. 穿越、符号链接逃逸。
不存在的路径也能校验（write_file 新建文件场景）：
向上找最近的存在祖先解析真实路径，再把剩余段接回。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from .exceptions import PolicyException


class PathGuard:
    def __init__(self, root: Union[str, Path]) -> None:
        if root is None or not str(root).strip():
            raise ValueError("项目根路径不能为空")
        candidate = Path(os.path.abspath(str(root)))
        real = candidate
        # macOS 上 /var → /private/var 这类符号链接必须先把根本身展开，
        # 否则目标解析成 /private/... 而根还是 /var/...，包含判断永远失败
        if candidate.exists():
            try:
                real = Path(os.path.realpath(candidate))
            except OSError:
                pass
        self._root = real

    @property
    def root_path(self) -> Path:
        return self._root

    def resolve_safe(self, raw: str) -> Path:
        """校验路径在项目根之内，返回规范化的绝对路径；越界抛 PolicyException。"""
        if raw is None or not str(raw).strip():
            raise PolicyException("路径不能为空")
        p = Path(raw)
        resolved = Path(os.path.abspath(p)) if p.is_absolute() else (self._root / p)
        resolved = Path(os.path.normpath(resolved))
        real_resolved = self._resolve_real_path(resolved)
        if not self._is_within(real_resolved, self._root):
            raise PolicyException(f"路径越界: {raw} 不在项目根 {self._root} 之内")
        return real_resolved

    @staticmethod
    def _is_within(target: Path, root: Path) -> bool:
        return target == root or root in target.parents

    @staticmethod
    def _resolve_real_path(target: Path) -> Path:
        """向上找最近的存在祖先解析符号链接，再把剩余段接回。"""
        existing = target
        while not existing.exists():
            parent = existing.parent
            if parent == existing:
                return Path(os.path.abspath(target))
            existing = parent
        try:
            real_existing = Path(os.path.realpath(existing))
            remainder = os.path.relpath(target, existing)
            return Path(os.path.normpath(real_existing / remainder))
        except OSError:
            return Path(os.path.abspath(target))
