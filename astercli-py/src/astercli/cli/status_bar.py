"""底部状态区（两次输入之间的两行摘要）。

第一行 = 上下文资产统计：AGENTS.md 文件 / MCP server / Skill / 工具数；
第二行 = 模型 + 上下文占用条 + 工作目录，有 token 统计时在末尾补一段暗色字段。

rich 没有常驻底栏，状态区在每次提示符前打印。
`format_status_lines` / `status_text` 都是纯函数，方便单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from .panels import CYAN, DIM, GREY, GREEN, WHITE, YELLOW
from .render_blocks import display_width, fit_to_columns

CONTEXT_BAR_WIDTH = 10
BAR_FILLED = "█"
BAR_EMPTY = "░"
BAR_EMPTY_STYLE = "#30363d"
COUNT_STYLE = f"bold {CYAN}"


@dataclass
class StatusState:
    model: str = ""
    phase: str = "idle"
    agents_files: int = 0
    mcp_servers: int = 0
    skills: int = 0
    tools: int = 0
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


def format_status_lines(state: StatusState, cols: int) -> list[str]:
    cols = max(20, cols)
    first = (
        f"{state.agents_files} AGENTS.md files · {state.mcp_servers} MCP servers · "
        f"{state.skills} skills · Tools {state.tools}"
    )
    return [fit_to_columns(first, cols), fit_to_columns(_second_line(state, cols), cols)]


def _second_line(state: StatusState, cols: int) -> str:
    parts = [state.model.strip() or "Auto Model", _context_segment(state)]
    cwd = _compact_cwd(state.cwd)
    if cwd:
        parts.append(cwd)
    text = "   ".join(part for part in parts if part)
    tokens = _token_segment(state)
    # token 统计是低优先级信息：放不下就整段省略，避免被裁成 "in 2…"
    if tokens and display_width(text) + 3 + display_width(tokens) <= cols:
        text += "   " + tokens
    return text


def _context_segment(state: StatusState) -> str:
    percent = state.context_percent()
    filled = max(0, min(CONTEXT_BAR_WIDTH, round(CONTEXT_BAR_WIDTH * percent / 100)))
    return BAR_FILLED * filled + BAR_EMPTY * (CONTEXT_BAR_WIDTH - filled) + f" {percent}%"


def _token_segment(state: StatusState) -> str:
    if not (state.input_tokens or state.output_tokens or state.cached_tokens):
        return ""
    text = f"in {_tokens(state.input_tokens)} out {_tokens(state.output_tokens)}"
    if state.cached_tokens:
        text += f" cache {_tokens(state.cached_tokens)}"
    if state.elapsed_ms > 0:
        text += f" {_elapsed(state.elapsed_ms)}"
    return text


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
    """两行状态区的富文本形态：数字高亮、标签暗色、占用条分色。"""
    cols = max(20, cols)
    text = Text()
    _append_counts(text, state)
    text.append("\n")
    _append_model_line(text, state, cols)
    if max(display_width(line) for line in text.plain.split("\n")) > cols:
        # 超宽时退回整段裁剪，避免折行把状态区撑成三行
        first, second = format_status_lines(state, cols)
        return Text(first + "\n" + second, style=GREY)
    return text


def print_status(console, state: StatusState) -> None:
    """在提示符前打印两行状态区。"""
    console.print(status_text(state, max(20, console.width)))


def _append_counts(text: Text, state: StatusState) -> None:
    # "Tools" 的计数在标签之后，其余统计是"数字 + 标签"
    fields = (
        (state.agents_files, "AGENTS.md files", True),
        (state.mcp_servers, "MCP servers", True),
        (state.skills, "skills", True),
        (state.tools, "Tools", False),
    )
    for index, (value, label, value_first) in enumerate(fields):
        if index:
            text.append(" · ", style=DIM)
        if value_first:
            text.append(str(value), style=COUNT_STYLE)
            text.append(f" {label}", style=GREY)
        else:
            text.append(label, style=GREY)
            text.append(f" {value}", style=COUNT_STYLE)


def _append_model_line(text: Text, state: StatusState, cols: int) -> None:
    text.append(state.model.strip() or "Auto Model", style=f"bold {WHITE}")
    text.append("   ")
    percent = state.context_percent()
    filled = max(0, min(CONTEXT_BAR_WIDTH, round(CONTEXT_BAR_WIDTH * percent / 100)))
    color = GREEN if state.phase == "idle" else YELLOW
    text.append(BAR_FILLED * filled, style=color)
    text.append(BAR_EMPTY * (CONTEXT_BAR_WIDTH - filled), style=BAR_EMPTY_STYLE)
    text.append(f" {percent}%", style=color)
    cwd = _compact_cwd(state.cwd)
    if cwd:
        text.append("   ")
        text.append(cwd, style=GREY)
    tokens = _token_segment(state)
    if tokens:
        used = display_width(text.plain.split("\n")[-1])
        if used + 3 + display_width(tokens) <= cols:
            text.append("   ")
            text.append(tokens, style=DIM)
