"""固定高度临时活动区（对齐 Java `InlineActivityDisplay`）。

思考 / 阻塞操作期间在正文区下方维护一块可重绘区域：spinner + 标题 + 已用时间 +
最近几行 reasoning 预览。只清理自己刚打印过的行，不碰 transcript 与输入区，
因此不会出现「按屏幕绝对行号清屏」导致的错位。

设计取舍：Java 版用 250ms 定时器驱动 spinner；Python 侧在每次 reasoning delta
到达时重绘（reasoning 流本身就是高频事件），避免额外线程抢 stdout。
"""

from __future__ import annotations

import shutil
import time
from typing import Optional

CLEAR_LINE = "\x1b[2K"
CLEAR_TO_EOL = "\x1b[K"
FAINT = "\x1b[2m"
RESET = "\x1b[0m"

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
MAX_REASONING_CHARS = 4096
MAX_REASONING_ROWS = 4


class ActivityArea:
    """临时活动面板；非 TTY（管道 / 测试）下自动降级为 no-op。"""

    def __init__(self, out, enabled: bool) -> None:
        self._out = out
        self._enabled = enabled
        self._active = False
        self._label = "Thinking"
        self._show_cancel_hint = True
        self._reasoning = ""
        self._started = 0.0
        self._frame = 0
        self._rendered_rows = 0

    @property
    def active(self) -> bool:
        return self._active

    def begin(self, label: str = "Thinking", *, show_cancel_hint: bool = True) -> None:
        if not self._enabled:
            return
        self._clear()
        self._label = (label or "Thinking").strip()
        self._show_cancel_hint = show_cancel_hint
        self._reasoning = ""
        self._started = time.monotonic()
        self._frame = 0
        self._active = True
        self._render()

    def append(self, delta: str) -> None:
        if not delta:
            return
        if not self._active and self._enabled:
            self.begin()
        self._reasoning += delta
        if len(self._reasoning) > MAX_REASONING_CHARS:
            self._reasoning = self._reasoning[-MAX_REASONING_CHARS:]
        if self._active:
            self._render()

    def end(self) -> bool:
        """结束并清空区域；返回此前是否真的绘制过内容。"""
        if not self._active:
            return False
        self._active = False
        drawn = self._rendered_rows > 0
        self._clear()
        self._reasoning = ""
        return drawn

    # ---------- 内部 ----------

    def _render(self) -> None:
        if not self._active or not self._enabled:
            return
        self._clear()
        lines = self._build_lines()
        for index, line in enumerate(lines):
            self._out.write(line + CLEAR_TO_EOL)
            if index < len(lines) - 1:
                self._out.write("\n")
        self._rendered_rows = len(lines)
        self._frame += 1
        self._out.flush()

    def _clear(self) -> None:
        rows = self._rendered_rows
        if rows <= 0:
            return
        if rows > 1:
            self._out.write(f"\x1b[{rows - 1}A")
        self._out.write("\r")
        for index in range(rows):
            self._out.write(CLEAR_LINE)
            if index < rows - 1:
                self._out.write("\n")
        if rows > 1:
            self._out.write(f"\x1b[{rows - 1}A")
        self._out.write("\r")
        self._rendered_rows = 0
        self._out.flush()

    def _build_lines(self) -> list[str]:
        cols = _columns()
        elapsed = max(0, int(time.monotonic() - self._started))
        if not self._show_cancel_hint:
            return [
                FAINT + "  ✢ " + self._label + "..." + RESET,
                FAINT + "    " + _progress_bar(cols) + " " + str(_progress_percent(elapsed)) + "%" + RESET,
            ]
        suffix = f" (esc to cancel, {elapsed}s)" if self._show_cancel_hint else f" ({elapsed}s)"
        spinner = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        lines = [FAINT + f"  {spinner} {self._label}..." + suffix + RESET]
        for line in _reasoning_lines(self._reasoning)[-MAX_REASONING_ROWS:]:
            lines.append(FAINT + "  │ " + _clip(line, cols - 4) + RESET)
        return lines


def _columns() -> int:
    return max(20, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)


def _clip(text: str, limit: int) -> str:
    if limit <= 1:
        return text[:limit]
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reasoning_lines(text: str) -> list[str]:
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        normalized = " ".join(raw.split())
        if normalized:
            lines.append(normalized)
    return lines


def _progress_bar(cols: int) -> str:
    width = max(10, min(40, cols - 12))
    percent = _progress_percent(0)
    filled = max(1, min(width - 1, round(width * percent / 100)))
    return "▰" * filled + "▱" * (width - filled)


def _progress_percent(elapsed_seconds: int) -> int:
    # 与 Java 一致的渐近曲线：15s 达到约 63%，上限 95%
    import math

    curve = 1.0 - math.exp(-max(0, elapsed_seconds) * 1000 / 15_000.0)
    return max(1, min(95, round(curve * 95)))


def make_activity(console) -> ActivityArea:
    """按 rich Console 能力创建活动区；非终端自动禁用。"""
    file = getattr(console, "file", None)
    enabled = bool(getattr(console, "is_terminal", False)) and file is not None
    return ActivityArea(file, enabled)
