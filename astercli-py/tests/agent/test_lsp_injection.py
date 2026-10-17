"""ReAct 图 LSP 诊断注入测试（对齐 Java AgentLspDiagnosticsTest）：

write_file 写坏代码 → 工具出口钩子收集 → 下一轮 LLM 调用前作为
user message 注入 `[LSP 诊断注入]`，Agent 据此修复。零网络：假模型按脚本回放。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from astercli.agent import AgentBudget
from astercli.agent.graph import build_react_graph
from astercli.config import Settings
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.lsp import LspManager
from astercli.policy import PathGuard
from astercli.tools.builtin.files import register_file_tools
from astercli.tools.registry import ToolRegistry


class RecordingScriptedModel(BaseChatModel):
    """按脚本顺序回放消息，并记录每次 astream 收到的完整 messages。"""

    script: list = []
    call_count: int = 0
    seen_messages: list = []
    last_tools: object = None

    @property
    def _llm_type(self) -> str:
        return "scripted-recording"

    def bind_tools(self, tools, **kwargs):
        self.last_tools = tools
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("测试只走 astream 路径")

    async def _astream(self, messages, stop=None, **kwargs):
        self.seen_messages.append(list(messages))
        idx = self.call_count
        self.call_count += 1
        msg = self.script[idx] if idx < len(self.script) else AIMessage(content="(end)")
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=msg.content,
                tool_calls=msg.tool_calls,
                usage_metadata=msg.usage_metadata,
            )
        )


class RecordingRenderer:
    def __init__(self):
        self.halts = []
        self.infos = []
        self.recorded_tool_calls = []
        self.recorded_tool_results = []
        self.reasoning_buffer = []

    def begin_thinking(self):
        pass

    def end_thinking(self):
        pass

    def stream_chunk(self, chunk):
        pass

    def take_reasoning(self):
        return ""

    def tool_calls(self, calls):
        self.recorded_tool_calls.append(calls)

    def tool_results(self, calls, results):
        self.recorded_tool_results.append(results)

    def halt(self, message):
        self.halts.append(message)

    def info(self, message):
        self.infos.append(message)


def _tool_call(name, args, id_):
    return [{"name": name, "args": args, "id": id_, "type": "tool_call"}]


async def _run(tmp_path: Path, script: list, *, with_lsp: bool):
    model = RecordingScriptedModel(script=script)
    settings = Settings(load_files=False)
    guard = PathGuard(tmp_path)
    registry = ToolRegistry(settings=settings)
    register_file_tools(registry, guard)
    if with_lsp:
        registry.set_lsp_manager(LspManager(tmp_path, settings))
    renderer = RecordingRenderer()
    runtime = ProviderRuntime(
        client=model, provider="glm", model="scripted", spec=PROVIDER_SPECS["glm"]
    )
    graph = build_react_graph(runtime, registry, renderer, tool_mode="slim")
    budget = AgentBudget(stagnation_window=3, hard_max_iterations=50)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage("写一个有语法问题的 Python 文件")],
            "budget": budget,
            "halt_reason": None,
        }
    )
    return model, renderer, result


_BROKEN = "def broken(:\n    pass\n"


async def test_写入坏文件后下一轮注入诊断(tmp_path):
    script = [
        AIMessage(
            content="",
            tool_calls=_tool_call(
                "write_file", {"path": "broken.py", "content": _BROKEN}, "call_1"
            ),
        ),
        AIMessage(content="已看到诊断，现在修复。"),
    ]

    model, renderer, result = await _run(tmp_path, script, with_lsp=True)

    assert model.call_count == 2
    second_messages = model.seen_messages[1]
    injected = [
        m
        for m in second_messages
        if isinstance(m, HumanMessage) and "[LSP 诊断注入]" in m.content
    ]
    assert injected, "第二轮 LLM 请求前应注入 LSP 诊断"
    assert "broken.py" in injected[0].content
    assert "[error]" in injected[0].content
    # 注入消息跨轮保留在最终 state 历史（对齐 Java conversationHistory.add(user)）
    assert any(
        isinstance(m, HumanMessage) and "[LSP 诊断注入]" in m.content
        for m in result["messages"]
    )
    # renderer 侧显示诊断块
    assert any("🔎 LSP 诊断" in text for text in renderer.infos)


async def test_未挂载_manager_时不注入(tmp_path):
    script = [
        AIMessage(
            content="",
            tool_calls=_tool_call(
                "write_file", {"path": "broken.py", "content": _BROKEN}, "call_1"
            ),
        ),
        AIMessage(content="完成。"),
    ]

    model, renderer, _ = await _run(tmp_path, script, with_lsp=False)

    assert model.call_count == 2
    second_messages = model.seen_messages[1]
    assert not any(
        isinstance(m, HumanMessage) and "[LSP 诊断注入]" in m.content
        for m in second_messages
    )
    assert not any("🔎 LSP 诊断" in text for text in renderer.infos)


async def test_修复后不再注入诊断(tmp_path):
    fixed = "def broken():\n    pass\n"
    script = [
        AIMessage(
            content="",
            tool_calls=_tool_call(
                "write_file", {"path": "broken.py", "content": _BROKEN}, "call_1"
            ),
        ),
        AIMessage(
            content="",
            tool_calls=_tool_call(
                "write_file", {"path": "broken.py", "content": fixed}, "call_2"
            ),
        ),
        AIMessage(content="已修复。"),
    ]

    model, renderer, _ = await _run(tmp_path, script, with_lsp=True)

    assert model.call_count == 3
    # 第二轮：坏文件诊断已注入；第三轮：文件已修复 → flush 无新诊断
    second_injected = [
        m
        for m in model.seen_messages[1]
        if isinstance(m, HumanMessage) and "[LSP 诊断注入]" in m.content
    ]
    assert second_injected
    third_injected = [
        m
        for m in model.seen_messages[2]
        if isinstance(m, HumanMessage) and "[LSP 诊断注入]" in m.content
    ]
    # 跨轮保留：上一轮注入消息仍出现在第三轮历史中，且没有新增注入
    # （第三轮的注入消息与第二轮是同一批对象，identity 集合相等）
    assert {id(m) for m in third_injected} == {id(m) for m in second_injected}
    # renderer 只在坏文件那一轮显示过一次诊断块 → 修复后无新增注入
    assert sum("🔎 LSP 诊断" in text for text in renderer.infos) == 1
