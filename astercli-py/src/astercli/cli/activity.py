"""思考期临时活动区（rich `Live` 实现）。

思考 / 阻塞操作期间在正文区下方维护一块临时区域：spinner + 标题 + 已用时间 +
最近几行 reasoning 预览。`Live(transient=True)` 结束时整块清除，完整 reasoning
由 `ConsoleRenderer` 落到正文引用块，因此不会污染 transcript。

非 TTY（管道 / 测试）下自动降级为 no-op。
"""

from __future__ import annotations

import math
import time

from rich.live import Live
from rich.text import Text

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
MAX_REASONING_CHARS = 4096
MAX_REASONING_ROWS = 4


class ActivityArea:
    """临时活动面板；非 TTY（管道 / 测试）下自动降级为 no-op。"""

    def __init__(self, console, enabled: bool) -> None:
        self._console = console
        self._enabled = enabled
        self._live: Live | None = None
        self._label = "Thinking"
        self._show_cancel_hint = True
        self._reasoning = ""
        self._started = 0.0
        self._frame = 0

    @property
    def active(self) -> bool:
        return self._live is not None

    def begin(self, label: str = "Thinking", *, show_cancel_hint: bool = True) -> None:
        if not self._enabled:
            return
        self.end()
        self._label = (label or "Thinking").strip()
        self._show_cancel_hint = show_cancel_hint
        self._reasoning = ""
        self._started = time.monotonic()
        self._frame = 0
        # get_renderable：每次自动刷新都重算耗时与 spinner，无需额外定时线程
        self._live = Live(
            get_renderable=self._build,
            console=self._console,
            refresh_per_second=8,
            transient=True,
            vertical_overflow="visible",
        )
        self._live.start(refresh=True)

    def append(self, delta: str) -> None:
        if not delta:
            return
        if self._live is None and self._enabled:
            self.begin()
        self._reasoning += delta
        if len(self._reasoning) > MAX_REASONING_CHARS:
            self._reasoning = self._reasoning[-MAX_REASONING_CHARS:]
        if self._live is not None:
            self._live.refresh()

    def end(self) -> bool:
        """结束并清除区域；返回此前是否真的存在活动区。"""
        live, self._live = self._live, None
        if live is None:
            return False
        live.stop()
        self._reasoning = ""
        return True

    # ---------- 内部 ----------

    def _build(self) -> Text:
        cols = _columns(self._console)
        elapsed = max(0, int(time.monotonic() - self._started))
        text = Text(style="#8b949e")
        if not self._show_cancel_hint:
            text.append(f"  ✢ {self._label}...", style="bold #a78bfa")
            text.append("\n")
            percent = _progress_percent(elapsed)
            text.append(f"    {_progress_bar(cols, percent)} {percent}%", style="#7ee787")
            return text
        spinner = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        self._frame += 1
        text.append(
            f"  {spinner} {self._label}... (ctrl+c to cancel, {elapsed}s)", style="bold #a78bfa"
        )
        for line in _reasoning_lines(self._reasoning)[-MAX_REASONING_ROWS:]:
            text.append("\n")
            text.append("  │ " + _clip(line, cols - 4))
        return text


def _columns(console) -> int:
    width = getattr(console, "width", 120) or 120
    return max(20, width - 1)


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


def _progress_bar(cols: int, percent: int) -> str:
    width = max(10, min(40, cols - 12))
    filled = max(1, min(width - 1, round(width * percent / 100)))
    return "▰" * filled + "▱" * (width - filled)


def _progress_percent(elapsed_seconds: int) -> int:
    # 与 Java 一致的渐近曲线：15s 达到约 63%，上限 95%
    curve = 1.0 - math.exp(-max(0, elapsed_seconds) * 1000 / 15_000.0)
    return max(1, min(95, round(curve * 95)))


def make_activity(console) -> ActivityArea:
    """按 rich Console 能力创建活动区；非终端自动禁用。"""
    enabled = bool(getattr(console, "is_terminal", False))
    return ActivityArea(console, enabled)
