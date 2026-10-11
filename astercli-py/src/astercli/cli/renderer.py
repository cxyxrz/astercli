"""渲染器。

NullRenderer：无 TUI 基类（测试 / 脚本路径），只累积缓冲不输出；
ConsoleRenderer：rich 流式实现——reasoning 灰字、正文白字、工具调用 dim 行。

reasoning 语义：openai_compat 基类保证每个 chunk 的
additional_kwargs["reasoning_content"] 是增量片段；take_reasoning() 为
drain 语义（返回自上次调用以来的增量并清空），保证多轮循环中
每条 assistant 消息只带本轮的 reasoning（对齐 Java：Message.assistant
只携带当轮 reasoningContent）。
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import AIMessageChunk

REASONING_KEYS = ("reasoning_content", "reasoning")


class NullRenderer:
    def __init__(self) -> None:
        self._reasoning_drain: list = []
        self.content_text: list = []
        self.reasoning_text: list = []
        self.tool_call_log: list = []
        self.tool_result_log: list = []
        self.halt_log: list = []
        self._thinking = False

    # ---------- 渲染接口 ----------

    def begin_thinking(self) -> None:
        self._thinking = True

    def end_thinking(self) -> None:
        self._thinking = False

    def stream_chunk(self, chunk: AIMessageChunk) -> None:
        reasoning = self._reasoning_delta(chunk)
        if reasoning:
            self._reasoning_drain.append(reasoning)
            self.reasoning_text.append(reasoning)
        content = chunk.content
        if isinstance(content, str) and content:
            self.content_text.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    self.content_text.append(block["text"])

    def take_reasoning(self) -> str:
        value = "".join(self._reasoning_drain)
        self._reasoning_drain.clear()
        return value

    def tool_calls(self, calls: list) -> None:
        self.tool_call_log.extend(calls)

    def tool_results(self, calls: list, results: list) -> None:
        self.tool_result_log.extend(results)

    def halt(self, message: str) -> None:
        self.halt_log.append(message)

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
    def __init__(self, console) -> None:
        super().__init__()
        self._console = console
        self._thought_started = False

    def begin_thinking(self) -> None:
        super().begin_thinking()
        self._thought_started = False

    def end_thinking(self) -> None:
        super().end_thinking()
        if self._thought_started:
            self._console.print()  # reasoning 后换行，与正文分隔
            self._thought_started = False
        else:
            # 没有 reasoning 的模型（如关闭 thinking）也要和上轮输出留出空行
            self._console.print()

    def stream_chunk(self, chunk: AIMessageChunk) -> None:
        reasoning = self._reasoning_delta(chunk)
        if reasoning:
            if not self._thought_started:
                self._console.print("✱ ", style="dim grey50", end="")
                self._thought_started = True
            self._console.print(_as_text(reasoning), style="grey50", end="")
            self._reasoning_drain.append(reasoning)
            self.reasoning_text.append(reasoning)
        content = chunk.content
        if isinstance(content, str) and content:
            self._console.print(_as_text(content), end="")
            self.content_text.append(content)

    def tool_calls(self, calls: list) -> None:
        super().tool_calls(calls)
        for call in calls:
            args_text = json.dumps(call.get("args", {}), ensure_ascii=False)
            if len(args_text) > 160:
                args_text = args_text[:160] + "…"
            self._console.print(f"→ {call.get('name')}({args_text})", style="cyan")

    def tool_results(self, calls: list, results: list) -> None:
        super().tool_results(calls, results)
        for msg in results:
            preview = (msg.content or "").replace("\n", " ")[:100]
            self._console.print(f"  ✓ {msg.name}: {preview}", style="grey50")

    def halt(self, message: str) -> None:
        super().halt(message)
        self._console.print(message, style="red")

    def usage(self, budget) -> None:
        self._console.print(
            f"⚡ tokens in={budget.total_input_tokens} out={budget.total_output_tokens} "
            f"cache={budget.total_cached_input_tokens} · iters={budget.iteration}",
            style="grey50",
        )

    def info(self, message: str) -> None:
        self._console.print(message, style="cyan")

    def warn(self, message: str) -> None:
        self._console.print(message, style="yellow")


def _as_text(text: str):
    from rich.text import Text

    return Text(text)
