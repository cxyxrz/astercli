"""渲染器（rich 原生实现）。

NullRenderer：无 TUI 基类（测试 / 脚本路径），只累积缓冲不输出；
ConsoleRenderer：全部输出走 rich Console / Text / Syntax / Live，参照 Java 版
`render/inline` 包补齐三类能力：

1. 思考期活动区（rich `Live`，spinner + 最近 reasoning 预览），结束后落完整引用块；
2. 工具调用一行摘要（不再提供展开）；
3. 代码围栏用 `rich.syntax.Syntax` 渲染，写文件行内 unified diff。

reasoning 语义：openai_compat 基类保证每个 chunk 的
additional_kwargs["reasoning_content"] 是增量片段；take_reasoning() 为 drain 语义
（返回自上次调用以来的增量并清空），保证多轮循环中每条 assistant 消息只带本轮
reasoning（对齐 Java：Message.assistant 只携带当轮 reasoningContent）。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessageChunk
from rich.syntax import Syntax
from rich.text import Text

from .activity import make_activity
from .panels import thinking_panel, tool_result_panel, tool_use_panel
from .render_blocks import render_diff, result_is_error

REASONING_KEYS = ("reasoning_content", "reasoning")

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
        self._set_phase("idle")

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
        self._printed_any = True
        self._console.print(thinking_panel(text))

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
        body = self._fence_lines[1:-1]
        language = self._fence_lang or "text"
        self._in_fence = False
        self._fence_lang = ""
        self._fence_lines = []
        if not body:
            return
        if self._printed_any:
            self._emit_text_line()
        code = "\n".join(body)
        try:
            self._console.print(
                Syntax(code, language, line_numbers=False, word_wrap=True, background_color="default")
            )
        except Exception:  # noqa: BLE001 —— 未知 lexer / 渲染异常时退回纯文本
            self._emit_text(code + "\n")

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
        # 补换行：流式最后一行通常没有 \n，保留裸行会让下一次提示符紧跟其后
        self._emit_text_line(remainder)

    # ---------- 工具调用 / 结果 / diff ----------

    def tool_calls(self, calls: list) -> None:
        super().tool_calls(calls)
        self._close_activity()
        self._flush_content()
        for call in calls:
            name = str(call.get("name") or "?")
            if self._printed_any:
                self._emit_text_line()
            self._printed_any = True
            self._console.print(tool_use_panel(name, call.get("args") or {}))
        self._set_phase("tools")

    def tool_results(self, calls: list, results: list) -> None:
        super().tool_results(calls, results)
        for message in results:
            content = str(getattr(message, "content", "") or "")
            name = str(getattr(message, "name", "") or "?")
            if self._printed_any:
                self._emit_text_line()
            self._printed_any = True
            self._console.print(tool_result_panel(name, content, ok=not result_is_error(content)))
        self._set_phase("thinking")

    def diff(self, path: str, before: Optional[str], after: Optional[str]) -> None:
        super().diff(path, before, after)
        self._close_activity()
        self._emit_text_line()
        for item in render_diff(path, before, after):
            self._emit_styled_line(item.text, DIFF_STYLES.get(item.kind, ""))

    # ---------- 基础输出 ----------

    def _emit_text(self, text: str) -> None:
        self._printed_any = True
        self._console.print(text, end="", soft_wrap=True, markup=False, highlight=False)

    def _emit_text_line(self, text: str = "") -> None:
        self._printed_any = True
        self._console.print(text, soft_wrap=True, markup=False, highlight=False)

    def _emit_styled_line(self, text: str, style: str = "") -> None:
        self._printed_any = True
        self._console.print(Text(text, style=style or ""), soft_wrap=True)

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
        self._console.print(Text(message, style="cyan"), markup=False, highlight=False)

    def echo_input(self, text: str) -> None:
        """提交回显：暗色整行块，对齐 JLine 侧 `>` 前缀写回 transcript。"""
        for line in text.split("\n"):
            self._console.print(f"> {line}", style="grey50", soft_wrap=True, markup=False, highlight=False)

    def warn(self, message: str) -> None:
        self._close_activity()
        self._console.print(Text(message, style="yellow"), markup=False, highlight=False)


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
