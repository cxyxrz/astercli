"""圆角面板与首屏 banner（统一视觉语言）。

首屏、思考区、工具调用、工具结果共用同一套样式：圆角边框 + 居中标题 + 区块品牌色。
所有面板走 rich `Panel`，宽度跟随 console，非 TTY 下同样可用（测试直接断言文本）。
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

GREY = "#8b949e"
DIM = "#6e7681"
WHITE = "#e6edf3"
GREEN = "#7ee787"
PURPLE = "#a78bfa"
YELLOW = "#e3b341"
RED = "#ff7b72"
CYAN = "#56d4dd"

# 首屏左上角 logo：粗块 "A"，与右侧信息区等高
LOGO_ROWS = (
    "   ████  ",
    "  ██████ ",
    " ██    ██",
    " ████████",
    " ██    ██",
    " ██    ██",
)

TOOL_RESULT_MAX_LINES = 12
TOOL_RESULT_MAX_CHARS = 1600


# ---------- 首屏 ----------


def logo_text() -> Text:
    return Text("\n".join(LOGO_ROWS), style=f"bold {GREEN}")


def startup_banner(
    *,
    title: str,
    version: str,
    subtitle: str,
    bullets: Iterable[str],
    hint: str,
    width: int,
) -> Group:
    """首屏：logo + 版本信息 + What's new 面板 + 分隔线 + 右侧提示。"""
    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column(width=len(LOGO_ROWS[0]), no_wrap=True)
    table.add_column(width=22, no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(logo_text(), _identity(title, version, subtitle), _whats_new(version, bullets))
    return Group(
        table,
        Text(""),
        Text("─" * max(10, width - 1), style=DIM),
        Text(hint, justify="right", style=DIM),
        Text(""),
    )


def _identity(title: str, version: str, subtitle: str) -> Text:
    text = Text()
    text.append(f"{title} v{version}", style=f"bold {WHITE}")
    text.append("\n\n")
    text.append(subtitle, style=f"bold {WHITE}")
    return text


def _whats_new(version: str, bullets: Iterable[str]) -> Panel:
    body = Text()
    for index, item in enumerate(bullets):
        if index:
            body.append("\n")
        body.append("- ", style=GREY)
        body.append(item, style=WHITE)
    body.append("\n")
    body.append("/help for more", style=f"italic {CYAN}")
    return _panel(body, Text(f"What's new (v{version})", style=f"bold {GREEN}"), GREY)


# ---------- 正文区块 ----------


def thinking_panel(reasoning: str) -> Panel:
    body = Text(reasoning.strip() or "（无思考内容）", style=WHITE)
    return _panel(body, Text("Thinking", style=f"bold {PURPLE}"), PURPLE)


def tool_use_panel(name: str, args: Optional[dict]) -> Panel:
    payload = json.dumps(args or {}, ensure_ascii=False, indent=2)
    body = Text()
    body.append("name  ", style=GREY)
    body.append(name, style=f"bold {WHITE}")
    body.append("\n")
    body.append("input ", style=GREY)
    body.append(payload, style=WHITE)
    return _panel(body, Text("Tool Use", style=f"bold {YELLOW}"), YELLOW)


def tool_result_panel(name: str, content: str, ok: bool = True) -> Panel:
    text, truncated = _clip_result(content)
    color = GREEN if ok else RED
    status = "ok" if ok else "error"
    body = Text(text or "（无输出）", style=WHITE if ok else RED)
    if truncated:
        body.append("\n… 输出过长已截断", style=DIM)
    return _panel(body, Text(f"Tool Result · {name} · {status}", style=f"bold {color}"), color)


def _panel(body: Text, title: Text, color: str) -> Panel:
    return Panel(
        body,
        title=title,
        title_align="center",
        border_style=color,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _clip_result(content: str) -> tuple[str, bool]:
    text = (content or "").strip()
    if not text:
        return "", False
    lines = text.splitlines()
    truncated = len(lines) > TOOL_RESULT_MAX_LINES or len(text) > TOOL_RESULT_MAX_CHARS
    text = "\n".join(lines[:TOOL_RESULT_MAX_LINES])
    if len(text) > TOOL_RESULT_MAX_CHARS:
        text = text[:TOOL_RESULT_MAX_CHARS]
    return text, truncated
