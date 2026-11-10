"""行内渲染基础块（纯逻辑，不依赖 rich / prompt_toolkit）。

参照 Java 版 `render/inline` 包，在 Python 侧提供三样能力：

1. CJK 安全宽度计算与截断（对齐 `InlineRenderer.displayWidth` / `fitToColumns`）；
2. 可折叠块（对齐 `FoldableBlock`：折叠态一行，展开态多行 + 收起提示）；
3. 工具调用分组文案与行内 unified diff（对齐 `ToolCallRenderer` / `InlineDiffRenderer`）。

折叠块本身只维护状态，重绘由 `ConsoleRenderer.toggle_last_block()` 负责。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
COLLAPSE_FOOTER = "⏷ 收起 (ctrl+o)"
EXPAND_HINT = "(ctrl+o to expand)"
KEY_PARAM_MAX_CHARS = 80


# ---------- 终端宽度 ----------


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def display_width(text: str) -> int:
    """按终端列宽计算字符串宽度（CJK / 全角 / emoji 记 2 列）。"""
    width = 0
    for ch in strip_ansi(text):
        if unicodedata.combining(ch):
            continue
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            width += 1
    return width


def fit_to_columns(text: str, cols: int, ellipsis: str = "…") -> str:
    """按显示宽度截断，超宽时以省略号收尾。"""
    cols = max(1, cols)
    if display_width(text) <= cols:
        return text
    limit = cols - display_width(ellipsis)
    if limit <= 0:
        return ellipsis
    out, width = [], 0
    for ch in text:
        step = display_width(ch)
        if width + step > limit:
            break
        out.append(ch)
        width += step
    return "".join(out) + ellipsis


def wrapped_rows(text: str, cols: int) -> int:
    """估算一段文本在给定列宽下占用的物理行数。"""
    cols = max(1, cols)
    rows, col = 0, 0
    for ch in strip_ansi(text):
        if ch == "\n":
            rows += 1
            col = 0
            continue
        col += display_width(ch)
        if col >= cols:
            rows += 1
            col = 0
    return rows + (1 if col else 0)


# ---------- 可折叠块 ----------


@dataclass
class FoldableBlock:
    """折叠态只占一行，展开态输出完整内容 + 收起提示（对齐 Java FoldableBlock）。"""

    header: str
    expanded_lines: list[str] = field(default_factory=list)
    footer: str = COLLAPSE_FOOTER
    expanded: bool = False

    def toggle(self) -> None:
        self.expanded = not self.expanded

    def lines(self) -> list[str]:
        if not self.expanded:
            return [self.header]
        if not self.footer:
            return list(self.expanded_lines)
        return [*self.expanded_lines, self.footer]

    def rows(self, cols: int) -> int:
        return sum(max(1, wrapped_rows(line, cols)) for line in self.lines())


# ---------- 工具调用文案 ----------

TOOL_LABELS = {
    "read_file": "📖 读取 {n} 个文件",
    "write_file": "✏️ 写入 {n} 个文件",
    "list_dir": "📂 列出 {n} 个目录",
    "glob_files": "🔎 匹配 {n} 次文件",
    "grep_code": "🔍 搜索代码 {n} 次",
    "execute_command": "⚡ 执行 {n} 条命令",
    "search_code": "🧠 语义检索 {n} 次",
    "web_search": "🌐 联网搜索 {n} 次",
    "web_fetch": "📰 抓取 {n} 个网页",
    "load_skill": "📚 加载 {n} 个 Skill",
}

KEY_PARAMS = {
    "read_file": "path",
    "write_file": "path",
    "list_dir": "path",
    "glob_files": "pattern",
    "grep_code": "pattern",
    "execute_command": "command",
    "search_code": "query",
    "web_search": "query",
    "web_fetch": "url",
    "load_skill": "name",
}

# 这些工具单次调用时直接用具名形态展示，信息密度更高（对齐 Java toolCollapsedLabel）
CALL_STYLE_LABELS = {
    "read_file": "📖 ReadFile({detail})",
    "list_dir": "📂 ListDir({detail})",
    "grep_code": "🔍 GrepCode({detail})",
    "execute_command": "⚡ Shell({detail})",
    "search_code": "🧠 SearchCode(\"{detail}\")",
    "web_search": "🌐 WebSearch(\"{detail}\")",
    "web_fetch": "📰 WebFetch({detail})",
    "load_skill": "📚 LoadSkill({detail})",
}


def tool_label(name: str, count: int) -> str:
    template = TOOL_LABELS.get(name)
    if template:
        return template.format(n=count)
    if name.startswith("mcp__"):
        return _mcp_label(name, count)
    return f"🔧 {name} × {count}"


def _mcp_label(name: str, count: int) -> str:
    parts = name.split("__", 2)
    display = f"{parts[1]}.{parts[2]}" if len(parts) == 3 else name
    return f"🔌 调用 MCP 工具 {display}" + ("" if count == 1 else f" × {count}")


def key_param(name: str, args: dict) -> str:
    """提取工具的关键参数，未知工具回退到截断后的 JSON。"""
    key = KEY_PARAMS.get(name)
    if key:
        value = args.get(key)
        text = "" if value is None else str(value).replace("\n", " ")
    else:
        try:
            text = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(args)
    if name == "web_fetch" and text:
        text = re.sub(r"^https?://", "", text).rstrip("/")
    if len(text) > KEY_PARAM_MAX_CHARS:
        text = text[: KEY_PARAM_MAX_CHARS - 3] + "..."
    return text


def call_label(name: str, call: dict) -> str:
    """单次调用的紧凑标签；无法提取关键参数时退回分组标签。"""
    detail = key_param(name, call.get("args") or {})
    template = CALL_STYLE_LABELS.get(name)
    if not template or not detail:
        return tool_label(name, 1) + (f" · {detail}" if detail else "")
    return template.format(detail=detail)


def group_calls(calls: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for call in calls:
        grouped.setdefault(str(call.get("name") or "?"), []).append(call)
    return grouped


def collapsed_header(grouped: dict[str, list[dict]]) -> str:
    if len(grouped) == 1:
        name, calls = next(iter(grouped.items()))
        label = call_label(name, calls[0]) if len(calls) == 1 else tool_label(name, len(calls))
        return f"⏵ {_strip_leading_icon(label)} {EXPAND_HINT}"
    total = sum(len(calls) for calls in grouped.values())
    return f"⏵ {len(grouped)} 组工具调用 / {total} 次 {EXPAND_HINT}"


def expanded_lines(grouped: dict[str, list[dict]]) -> list[str]:
    lines: list[str] = []
    for name, calls in grouped.items():
        lines.append(f"  {tool_label(name, len(calls))}")
        for call in calls:
            detail = key_param(name, call.get("args") or {})
            if detail:
                lines.append(f"    └ {detail}")
    return lines


def _strip_leading_icon(label: str) -> str:
    """折叠态去掉 emoji 前缀，视觉更紧凑。"""
    first, _, rest = label.partition(" ")
    if rest and display_width(first) <= 2:
        return rest
    return label


def build_tool_block(calls: list[dict]) -> Optional[FoldableBlock]:
    grouped = group_calls(calls)
    if not grouped:
        return None
    return FoldableBlock(collapsed_header(grouped), expanded_lines(grouped))


def result_is_error(text: str) -> bool:
    return text.startswith(("工具执行失败", "参数校验失败", "工具调用参数解析失败"))


# ---------- 行内 diff ----------

CONTEXT_LINES = 2


@dataclass(frozen=True)
class DiffLine:
    kind: str  # header / hunk / add / delete / context / meta
    text: str


def render_diff(path: str, before: Optional[str], after: Optional[str]) -> list[DiffLine]:
    """把一次写入渲染成 unified diff（对齐 Java InlineDiffRenderer）。"""
    lines = [DiffLine("header", f"📝 {path or '(unnamed)'}")]
    if before is None and after is None:
        return lines + [DiffLine("meta", "  (空 diff)")]
    if before is None:
        body = [line for line in (after or "").splitlines() if line]
        lines.append(DiffLine("hunk", f"@@ -0,0 +1,{len(body)} @@"))
        lines.extend(DiffLine("add", f"+{line}") for line in body)
        return lines
    if after is None:
        body = [line for line in before.splitlines() if line]
        lines.append(DiffLine("hunk", f"@@ -1,{len(body)} +0,0 @@"))
        lines.extend(DiffLine("delete", f"-{line}") for line in body)
        return lines
    if before == after:
        return lines + [DiffLine("meta", "  (内容未变)")]

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    ops = compute_diff(before_lines, after_lines)
    for hunk in group_hunks(ops):
        lines.append(DiffLine("hunk", hunk.header()))
        for op in hunk.ops:
            if op.kind == "equal":
                lines.append(DiffLine("context", f" {op.text}"))
            elif op.kind == "add":
                lines.append(DiffLine("add", f"+{op.text}"))
            else:
                lines.append(DiffLine("delete", f"-{op.text}"))
    return lines


@dataclass(frozen=True)
class _Op:
    kind: str  # equal / add / delete
    text: str
    before_index: int
    after_index: int


def compute_diff(before: list[str], after: list[str]) -> list[_Op]:
    """朴素 LCS diff：单文件规模足够，避免引入第三方依赖。"""
    n, m = len(before), len(after)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = (
                dp[i + 1][j + 1] + 1
                if before[i] == after[j]
                else max(dp[i + 1][j], dp[i][j + 1])
            )
    ops: list[_Op] = []
    i = j = 0
    while i < n and j < m:
        if before[i] == after[j]:
            ops.append(_Op("equal", before[i], i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            ops.append(_Op("delete", before[i], i, -1))
            i += 1
        else:
            ops.append(_Op("add", after[j], -1, j))
            j += 1
    ops.extend(_Op("delete", before[k], k, -1) for k in range(i, n))
    ops.extend(_Op("add", after[k], -1, k) for k in range(j, m))
    return ops


@dataclass(frozen=True)
class _Hunk:
    before_start: int
    before_count: int
    after_start: int
    after_count: int
    ops: list[_Op]

    def header(self) -> str:
        return (
            f"@@ -{self.before_start + 1},{self.before_count}"
            f" +{self.after_start + 1},{self.after_count} @@"
        )


def group_hunks(ops: list[_Op], context: int = CONTEXT_LINES) -> list[_Hunk]:
    """把连续相同段切成 hunk，每个 hunk 保留 context 行上下文。"""
    hunks: list[_Hunk] = []
    index = 0
    while index < len(ops):
        while index < len(ops) and ops[index].kind == "equal":
            index += 1
        if index >= len(ops):
            break
        start = max(0, index - context)
        end, equal_run = index, 0
        while end < len(ops):
            if ops[end].kind == "equal":
                equal_run += 1
                if equal_run >= 2 * context:
                    break
            else:
                equal_run = 0
            end += 1
        close = min(len(ops), end + context - equal_run)
        chunk = ops[start:close]
        hunks.append(
            _Hunk(
                _first(chunk, "before_index"),
                sum(1 for op in chunk if op.kind != "add"),
                _first(chunk, "after_index"),
                sum(1 for op in chunk if op.kind != "delete"),
                chunk,
            )
        )
        index = close
    return hunks


def _first(ops: list[_Op], attr: str) -> int:
    for op in ops:
        value = getattr(op, attr)
        if value >= 0:
            return value
    return 0
