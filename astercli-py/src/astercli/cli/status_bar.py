"""底部状态区（对齐 Java `BottomStatusBar` 的两行 dock）。

Java 版由 JLine `Status` 托管底部保留区，输入期和任务执行期都可见；Python 侧
改为在每次提示符前打印两行（rich 没有常驻底栏）：第一行 = 模式 + MCP/Skill 摘要，
第二行 = 模型 / phase / 上下文 / token / 耗时 / cwd。

`format_status_lines` / `status_text` 都是纯函数，方便单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from .render_blocks import display_width, fit_to_columns

CONTEXT_BAR_WIDTH = 8

BASE = "fg:#8a8a8a"
MODE = "fg:#5fd700 bold"
MODEL = "fg:#00afff bold"
PHASE_IDLE = "fg:#5fd700"
PHASE_ACTIVE = "fg:#ffd700 bold"
CTX = "fg:#00afff bold"
TOKEN = "fg:#ffd700"
CACHE = "fg:#ff5fff"
ELAPSED = "fg:#ffd700"
CWD = "fg:#8a8a8a"


@dataclass
class StatusState:
    model: str = ""
    phase: str = "idle"
    mode: str = "HITL 审批"
    mcp_summary: str = ""
    skill_summary: str = ""
    context_tokens: int = 0
    context_window: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    elapsed_ms: int = 0
    cwd: str = ""

    def update(self, **fields) -> None:
        for key, value in fields.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def context_percent(self) -> int:
        if self.context_window <= 0:
            return 0
        return max(0, min(100, round(self.context_tokens * 100 / self.context_window)))

    def environment(self) -> str:
        parts = []
        if self.mcp_summary:
            parts.append(self.mcp_summary)
        if self.skill_summary:
            parts.append(self.skill_summary)
        return " · ".join(parts)


def format_status_lines(state: StatusState, cols: int) -> list[str]:
    cols = max(20, cols)
    left = f" {state.mode}"
    right = state.environment()
    if right:
        gap = max(1, cols - display_width(left) - display_width(right) - 2)
        first = fit_to_columns(left + " " * gap + right + " ", cols)
    else:
        first = fit_to_columns(left, cols)
    return [first, fit_to_columns(_footer(state), cols)]


def _footer(state: StatusState) -> str:
    return "  ".join(field for field in _footer_fields(state) if field)


def _context_segment(state: StatusState) -> str:
    if state.context_window <= 0:
        return ""
    percent = state.context_percent()
    filled = max(0, min(CONTEXT_BAR_WIDTH, round(CONTEXT_BAR_WIDTH * percent / 100)))
    bar = "▰" * filled + "▱" * (CONTEXT_BAR_WIDTH - filled)
    return f"ctx {bar} {percent}%"


def _tokens(value: int) -> str:
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}m"


def _elapsed(millis: int) -> str:
    seconds = millis / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def _compact_cwd(cwd: str) -> str:
    if not cwd:
        return ""
    try:
        home = str(Path.home())
        return "~" + cwd[len(home):] if cwd.startswith(home) else cwd
    except (OSError, ValueError):
        return cwd


def status_text(state: StatusState, cols: int) -> Text:
    """两行状态区的富文本形态（第一行整段 base 色，第二行字段分色）。"""
    cols = max(20, cols)
    first, second = format_status_lines(state, cols)
    text = Text(first, style=BASE)
    text.append("\n")
    width = 0
    for index, field in enumerate(_footer_fields(state)):
        if not field:
            continue
        if index:
            text.append("  ", style=BASE)
            width += 2
        text.append(field, style=_field_style(field))
        width += display_width(field)
    if width > cols:
        # 超宽时退回整段裁剪，避免折行把状态区撑成三行
        return Text(first + "\n" + second, style=BASE)
    if width < cols:
        text.append(" " * (cols - width), style=BASE)
    return text


def print_status(console, state: StatusState) -> None:
    """在提示符前打印两行状态区。"""
    console.print(status_text(state, max(20, console.width)))


def _footer_fields(state: StatusState) -> list[str]:
    model = state.model.strip() or "Auto Model"
    phase = state.phase.strip() or "idle"
    fields = [f" Auto Model · {model}", phase, _context_segment(state)]
    if state.input_tokens or state.output_tokens or state.cached_tokens:
        token_text = f"in {_tokens(state.input_tokens)} out {_tokens(state.output_tokens)}"
        if state.cached_tokens:
            token_text += f" cache {_tokens(state.cached_tokens)}"
        fields.append(token_text)
    if state.elapsed_ms > 0:
        fields.append(_elapsed(state.elapsed_ms))
    cwd = _compact_cwd(state.cwd)
    if cwd:
        fields.append(cwd)
    return fields


def _field_style(field: str) -> str:
    if field.startswith(" Auto Model"):
        return MODEL
    if field.startswith("ctx "):
        return CTX
    if field.startswith(("in ", "out ")):
        return TOKEN
    if field.startswith("~") or field.startswith("/"):
        return CWD
    if field.endswith("s") or field.endswith("m00s"):
        return ELAPSED
    if field == "idle":
        return PHASE_IDLE
    return PHASE_ACTIVE
