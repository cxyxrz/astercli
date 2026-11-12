"""渲染器。

NullRenderer：无 TUI 基类（测试 / 脚本路径），只累积缓冲不输出；
ConsoleRenderer：rich + ANSI 实现，参照 Java 版 `render/inline` 包补齐四类能力：

1. 思考期固定高度活动区（spinner + 最近 reasoning 预览），结束后落完整引用块；
2. 工具调用折叠块（`⏵ 读取 3 个文件 (ctrl+o to expand)`），Ctrl+O 由 REPL 触发展开；
3. 代码围栏折叠（`⏵ code: python (42 行, ctrl+o to expand)`）；
4. 写文件行内 unified diff。

reasoning 语义：openai_compat 基类保证每个 chunk 的
additional_kwargs["reasoning_content"] 是增量片段；take_reasoning() 为 drain 语义
（返回自上次调用以来的增量并清空），保证多轮循环中每条 assistant 消息只带本轮
reasoning（对齐 Java：Message.assistant 只携带当轮 reasoningContent）。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessageChunk

from .activity import make_activity
from .render_blocks import (
    FoldableBlock,
    build_tool_block,
    render_diff,
    result_is_error,
    wrapped_rows,
)

REASONING_KEYS = ("reasoning_content", "reasoning")
TOOL_PREVIEW_CHARS = 100

DIFF_STYLES = {
    "header": "bold cyan",
    "hunk": "cyan",
    "add": "green",
    "delete": "red",
    "context": "",
    "meta": "grey50",
}


class NullRenderer:
    def __init__(self) -> None:
        self._reasoning_drain: list = []
        self.content_text: list = []
        self.reasoning_text: list = []
        self.tool_call_log: list = []
        self.tool_result_log: list = []
        self.halt_log: list = []
        self.diff_log: list = []
        self._thinking = False

    # ---------- 渲染接口 ----------

    def begin_turn(self) -> None:
        """NullRenderer 保留累积日志（测试会跨调用读取），仅 TUI 形态重置界面状态。"""

    def begin_thinking(self) -> None:
        self._thinking = True

    def end_thinking(self) -> None:
        self._thinking = False

    def stream_chunk(self, chunk: AIMessageChunk) -> None:
        reasoning = self._reasoning_delta(chunk)
        if reasoning:
            self._reasoning_drain.append(reasoning)
            self.reasoning_text.append(reasoning)
        for text in _content_text(chunk.content):
            self.content_text.append(text)

    def take_reasoning(self) -> str:
        value = "".join(self._reasoning_drain)
        self._reasoning_drain.clear()
        return value

    def tool_calls(self, calls: list) -> None:
        self.tool_call_log.extend(calls)

    def tool_results(self, calls: list, results: list) -> None:
        self.tool_result_log.extend(results)

    def diff(self, path: str, before: Optional[str], after: Optional[str]) -> None:
        self.diff_log.append((path, before, after))

    def halt(self, message: str) -> None:
        self.halt_log.append(message)

    def echo_input(self, text: str) -> None:
        pass

    def toggle_last_block(self) -> bool:
        return False

    def usage(self, budget) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    # ---------- 内部 ----------

    @staticmethod
    def _reasoning_delta(chunk: AIMessageChunk) -> Optional[str]:
        kwargs = chunk.additional_kwargs or {}
        for key in REASONING_KEYS:
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
        return None


class ConsoleRenderer(NullRenderer):
    def __init__(self, console, status=None) -> None:
        super().__init__()
        self._console = console
        self._status = status
        self._activity = make_activity(console)
        self._reasoning_shown = False
        self._turn_reasoning: list[str] = []
        self._printed_any = False
        self._content_buffer = ""
        self._in_fence = False
        self._fence_lang = ""
        self._fence_lines: list[str] = []
        self._blocks: list[FoldableBlock] = []
        # 最近一次折叠块之后的所有输出，Ctrl+O 展开时整体重绘
        self._tail: list[dict] = []
        self._tail_rows = 0

    # ---------- 状态栏 ----------

    @property
    def status(self):
        return self._status

    def _set_phase(self, phase: str) -> None:
        if self._status is not None:
            self._status.update(phase=phase)

    # ---------- 一轮生命周期 ----------

    def begin_turn(self) -> None:
        super().begin_turn()
        self._reasoning_shown = False
        self._turn_reasoning.clear()
        self._printed_any = False
        self._content_buffer = ""
        self._in_fence = False
        self._fence_lang = ""
        self._fence_lines.clear()
        self._blocks.clear()
        self._tail.clear()
        self._tail_rows = 0
        self._set_phase("thinking")

    # ---------- 思考区 ----------

    def begin_thinking(self) -> None:
        super().begin_thinking()
        self._reasoning_shown = False
        self._turn_reasoning.clear()
        self._activity.begin("Thinking")
        self._set_phase("thinking")

    def stream_chunk(self, chunk: AIMessageChunk) -> None:
        reasoning = self._reasoning_delta(chunk)
        if reasoning:
            self._reasoning_drain.append(reasoning)
            self.reasoning_text.append(reasoning)
            self._turn_reasoning.append(reasoning)
            self._activity.append(reasoning)
        content = _content_text(chunk.content)
        if not content:
            return
        self._close_activity()
        for text in content:
            self.content_text.append(text)
            self._feed_content(text)
        self._set_phase("responding")

    def end_thinking(self) -> None:
        super().end_thinking()
        self._close_activity()
        self._flush_content()
        self._set_phase("tools" if self._blocks else "idle")

    def _close_activity(self) -> None:
        """结束 live 区；若期间有 reasoning，则把完整内容落到正文区。"""
        if self._activity.active:
            self._activity.end()
        if self._reasoning_shown or not self._turn_reasoning:
            return
        self._reasoning_shown = True
        self._print_reasoning()

    def _print_reasoning(self) -> None:
        text = "".join(self._turn_reasoning)
        if not text.strip():
            return
        if self._printed_any:
            self._emit_text_line()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            self._emit_styled(f"│ {line}" if line.strip() else "│", "grey50")
        self._emit_text_line()

    # ---------- 正文 ----------

    def _feed_content(self, text: str) -> None:
        self._content_buffer += text
        while "\n" in self._content_buffer:
            line, self._content_buffer = self._content_buffer.split("\n", 1)
            self._process_content_line(line)

    def _process_content_line(self, line: str) -> None:
        stripped = line.strip()
        if not self._in_fence and stripped.startswith("```"):
            self._in_fence = True
            self._fence_lang = stripped[3:].strip()
            self._fence_lines = [line]
            return
        if self._in_fence:
            self._fence_lines.append(line)
            if stripped == "```":
                self._close_fence()
            return
        self._emit_text(line + "\n")

    def _close_fence(self) -> None:
        language = self._fence_lang or "code"
        body_count = max(0, len(self._fence_lines) - 2)
        block = FoldableBlock(
            header=f"⏵ code: {language} ({body_count} 行, ctrl+o to expand)",
            expanded_lines=list(self._fence_lines),
        )
        self._in_fence = False
        self._fence_lang = ""
        self._fence_lines = []
        self._emit_block(block)

    def _flush_content(self) -> None:
        if self._in_fence:
            # 流式中途结束（异常 / 被截断）：围栏未闭合时按原文输出，避免丢内容
            remainder, self._content_buffer = self._content_buffer, ""
            if remainder:
                self._fence_lines.append(remainder)
            for line in self._fence_lines:
                self._emit_text(line + "\n")
            self._in_fence = False
            self._fence_lang = ""
            self._fence_lines = []
            return
        if not self._content_buffer:
            return
        remainder, self._content_buffer = self._content_buffer, ""
        # 补换行：流式最后一行通常没有 \n，若直接 end="" 输出，
        # 下一次 prompt_toolkit 渲染提示符/底栏时会把行坐标算错，造成重叠
        self._emit_text_line(remainder)

    # ---------- 工具调用 / 结果 / diff ----------

    def tool_calls(self, calls: list) -> None:
        super().tool_calls(calls)
        self._close_activity()
        self._flush_content()
        block = build_tool_block(calls)
        if block is not None:
            self._emit_block(block)
        self._set_phase("tools")

    def tool_results(self, calls: list, results: list) -> None:
        super().tool_results(calls, results)
        for message in results:
            content = str(getattr(message, "content", "") or "")
            name = str(getattr(message, "name", "") or "?")
            failed = result_is_error(content)
            preview = " ".join(content.split())[:TOOL_PREVIEW_CHARS]
            mark = "✗" if failed else "✓"
            self._emit_styled_line(f"  {mark} {name}: {preview}", "red" if failed else "grey50")
        self._set_phase("thinking")

    def diff(self, path: str, before: Optional[str], after: Optional[str]) -> None:
        super().diff(path, before, after)
        self._emit_text_line()
        for item in render_diff(path, before, after):
            self._emit_styled_line(item.text, DIFF_STYLES.get(item.kind, ""))

    # ---------- 折叠块展开（Ctrl+O） ----------

    def toggle_last_block(self) -> bool:
        if not self._blocks:
            return False
        rows = _terminal_rows()
        block = self._blocks[-1]
        before_rows = self._tail_rows
        block.toggle()
        self._tail_rows = self._measure_tail()
        if before_rows >= rows - 2 or self._tail_rows >= rows - 2:
            block.toggle()
            self._tail_rows = before_rows
            self._emit_styled_line("  （该块超出可视区域，无法就地展开）", "yellow")
            return False
        self._redraw_tail()
        return True

    def _emit_block(self, block: FoldableBlock) -> None:
        self._blocks.append(block)
        self._tail = []
        self._tail_rows = 0
        self._tail_print_block(block)
        self._tail_rows += block.rows(self._columns())

    def _redraw_tail(self) -> None:
        out = self._console.file
        rows = self._tail_rows
        if rows <= 0 or out is None:
            return
        out.write(f"\x1b[{rows}A\r\x1b[J")
        out.flush()
        for entry in self._tail:
            if entry["kind"] == "block":
                for line in entry["block"].lines():
                    self._console.print(line, style="grey50", soft_wrap=True)
            elif entry["kind"] == "styled":
                self._console.print(entry["text"], style=entry["style"], soft_wrap=True)
            else:
                self._console.print(entry["text"], end="", soft_wrap=True, markup=False, highlight=False)
        out.flush()

    def _measure_tail(self) -> int:
        cols = self._columns()
        rows = 0
        for entry in self._tail:
            if entry["kind"] == "block":
                rows += entry["block"].rows(cols)
            else:
                rows += max(1, wrapped_rows(entry["text"], cols))
        return rows

    # ---------- 基础输出 ----------

    def _emit_text(self, text: str) -> None:
        self._printed_any = True
        self._console.print(text, end="", soft_wrap=True, markup=False, highlight=False)
        self._record_tail({"kind": "text", "text": text})

    def _emit_text_line(self, text: str = "") -> None:
        self._printed_any = True
        self._console.print(text, soft_wrap=True, markup=False, highlight=False)
        self._record_tail({"kind": "text", "text": text + "\n"})

    def _emit_styled_line(self, text: str, style: str = "") -> None:
        self._printed_any = True
        self._console.print(text, style=style or None, soft_wrap=True)
        self._record_tail({"kind": "styled", "text": text, "style": style})

    def _emit_styled(self, text: str, style: str) -> None:
        self._printed_any = True
        self._console.print(text, style=style, soft_wrap=False)
        self._record_tail({"kind": "styled", "text": text, "style": style})

    def _tail_print_block(self, block: FoldableBlock) -> None:
        self._printed_any = True
        self._tail.append({"kind": "block", "block": block})
        for line in block.lines():
            self._console.print(line, style="grey50", soft_wrap=True)

    def _record_tail(self, entry: dict) -> None:
        self._tail.append(entry)

    def _columns(self) -> int:
        return max(20, getattr(self._console, "width", 120))

    # ---------- 其他输出 ----------

    def halt(self, message: str) -> None:
        super().halt(message)
        self._emit_styled_line(message, "red")
        self._set_phase("idle")

    def usage(self, budget) -> None:
        if self._status is not None:
            self._status.update(
                input_tokens=budget.total_input_tokens,
                output_tokens=budget.total_output_tokens,
                cached_tokens=budget.total_cached_input_tokens,
            )
            return
        self._console.print(
            f"⚡ tokens in={budget.total_input_tokens} out={budget.total_output_tokens} "
            f"cache={budget.total_cached_input_tokens} · iters={budget.iteration}",
            style="grey50",
        )

    def info(self, message: str) -> None:
        self._close_activity()
        self._console.print(message, style="cyan", markup=False, highlight=False)

    def echo_input(self, text: str) -> None:
        """提交回显：暗色整行块，对齐 JLine 侧 `>` 前缀写回 transcript。"""
        for line in text.split("\n"):
            self._console.print(f"> {line}", style="grey50", soft_wrap=True, markup=False, highlight=False)

    def warn(self, message: str) -> None:
        self._close_activity()
        self._console.print(message, style="yellow", markup=False, highlight=False)


def _content_text(content) -> list[str]:
    """把 chunk.content 统一成文本片段列表（str / block 列表两种形态）。"""
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return parts
    return []


def _terminal_rows() -> int:
    import shutil

    return max(10, shutil.get_terminal_size(fallback=(120, 24)).lines)
