"""行内渲染基础块（纯逻辑，不依赖 rich）。

参照 Java 版 `render/inline` 包，在 Python 侧提供：

1. CJK 安全宽度计算与截断（对齐 `InlineRenderer.displayWidth` / `fitToColumns`）；
2. 工具结果错误判定；
3. 行内 unified diff（对齐 `InlineDiffRenderer`）。

实际绘制由 `ConsoleRenderer` 用 rich 组件完成，这里不产生任何 ANSI 输出。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


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


# ---------- 工具结果 ----------


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
