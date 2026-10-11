"""LangGraph ReAct 循环测试：ScriptedModel 驱动，不发真实请求。

覆盖：两轮 ReAct（tool_calls → ToolMessage → content）、硬轮数兜底、
停滞检测、token 预算、usage 记账、xfyun 纯对话不 bind_tools。

langchain-core 的 astream 契约：_astream 须 yield ChatGenerationChunk，
astream 对外展开为其中的 message（AIMessage / AIMessageChunk 均可）。
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk

from astercli.agent.budget import AgentBudget, ExitReason
from astercli.agent.graph import build_react_graph, extract_tool_calls
from astercli.config import Settings
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.memory import ConversationHistoryCompactor
from astercli.policy import PathGuard
from astercli.tools.builtin.files import register_file_tools
from astercli.tools.registry import ToolRegistry


class ScriptedModel(BaseChatModel):
    """按脚本顺序返回消息的假模型；bind_tools 记录 schemas 并返回自身。

    langchain-core astream 契约：_astream 须 yield ChatGenerationChunk，
    且其中的 message 必须是 BaseMessageChunk 子类 → 脚本里的 AIMessage
    会被包装为 AIMessageChunk 再对外吐出。
    """

    script: list = []
    call_count: int = 0
    last_tools: object = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.last_tools = tools
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("测试只走 astream 路径")

    @staticmethod
    def _to_chunk(msg: AIMessage) -> AIMessageChunk:
        return AIMessageChunk(
            content=msg.content,
            tool_calls=msg.tool_calls,
            usage_metadata=msg.usage_metadata,
            additional_kwargs=msg.additional_kwargs,
        )

    async def _astream(self, messages, stop=None, **kwargs):
        idx = self.call_count
        self.call_count += 1
        msg = self.script[idx] if idx < len(self.script) else AIMessage(content="(end)")
        yield ChatGenerationChunk(message=self._to_chunk(msg))


class EmptyModel(ScriptedModel):
    """只吐非 AIMessage 的 chunk，用于验证空响应兜底（final is None）。"""

    async def _astream(self, messages, stop=None, **kwargs):
        self.call_count += 1
        yield ChatGenerationChunk(message=HumanMessageChunk(content="ignored"))


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
        reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
        if reasoning:
            self.reasoning_buffer.append(reasoning)

    def take_reasoning(self):
        out = "".join(self.reasoning_buffer)
        self.reasoning_buffer.clear()
        return out

    def tool_calls(self, calls):
        self.recorded_tool_calls.append(calls)

    def tool_results(self, calls, results):
        self.recorded_tool_results.append(results)

    def halt(self, message):
        self.halts.append(message)

    def info(self, message):
        self.infos.append(message)


def tool_call(name, args, id_):
    return [{"name": name, "args": args, "id": id_, "type": "tool_call"}]


def make_runtime(model, spec_key="glm"):
    return ProviderRuntime(
        client=model, provider=spec_key, model="scripted", spec=PROVIDER_SPECS[spec_key]
    )


def make_setup(project, model, *, budget=None, spec_key="glm"):
    guard = PathGuard(project)
    registry = ToolRegistry(settings=Settings(load_files=False))
    register_file_tools(registry, guard)
    renderer = RecordingRenderer()
    budget = budget or AgentBudget(stagnation_window=3, hard_max_iterations=50)
    runtime = make_runtime(model, spec_key=spec_key)
    graph = build_react_graph(runtime, registry, renderer, tool_mode="slim")
    return graph, budget, renderer, runtime


async def run(project, script, *, budget=None, spec_key="glm", model=None):
    assert model is None or script is None
    m = model if model is not None else ScriptedModel(script=script)
    graph, b, renderer, runtime = make_setup(project, m, budget=budget, spec_key=spec_key)
    result = await graph.ainvoke({"messages": [HumanMessage(content="读 README")], "budget": b})
    return result, renderer, b, runtime


# ---------- 两轮 ReAct ----------


async def test_two_round_react_flow(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("read_file", {"path": "README.md"}, "call_1")),
        AIMessage(content="README 的内容是 hello world"),
    ]
    result, renderer, budget, _ = await run(project, script)

    assert result.get("halt_reason") is None
    messages = result["messages"]
    # [user, AIMessage(tool_call), ToolMessage, AIMessage(final)]
    assert len(messages) == 4
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"
    assert messages[2].name == "read_file"
    assert "hello world" in messages[2].content
    assert messages[3].content == "README 的内容是 hello world"
    assert budget.iteration == 2
    assert renderer.halts == []
    assert len(renderer.recorded_tool_calls) == 1
    assert len(renderer.recorded_tool_results) == 1


async def test_slim_tool_schemas_bound_to_llm(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("list_dir", {"path": "."}, "c1")),
        AIMessage(content="done"),
    ]
    _, _, _, runtime = await run(project, script)
    model = runtime.client
    assert model.call_count == 2
    assert model.last_tools is not None
    # slim 模式：parameters 只留 type=object + required（read_file 为首个注册工具）
    first = model.last_tools[0]["function"]
    assert first["name"] == "read_file"
    assert first["parameters"] == {"type": "object", "required": ["path"]}
    assert "参数:" in first["description"]


async def test_xfyun_never_binds_tools(project):
    _, _, _, runtime = await run(
        project, [AIMessage(content="纯对话回答")], spec_key="xfyun"
    )
    model = runtime.client
    assert model.call_count == 1
    assert model.last_tools is None


async def test_usage_metadata_recorded(project):
    script = [
        AIMessage(
            content="答案",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
    ]
    _, _, budget, _ = await run(project, script)
    assert budget.total_input_tokens == 10
    assert budget.total_output_tokens == 5
    assert budget.total_cached_input_tokens == 0


async def test_cached_tokens_recorded(project):
    script = [
        AIMessage(
            content="答案",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"cached_tokens": 4},
            },
        )
    ]
    _, _, budget, _ = await run(project, script)
    assert budget.total_cached_input_tokens == 4


# ---------- 兜底 ----------


async def test_hard_max_iterations_halts(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("list_dir", {"path": "."}, "c"))
    ] * 5
    budget = AgentBudget(hard_max_iterations=1)
    result, renderer, b, _ = await run(project, script, budget=budget)
    assert result.get("halt_reason")
    assert "硬轮数上限（1）" in result["halt_reason"]
    assert renderer.halts
    assert b.iteration == 1


async def test_stagnation_detected_halts(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("list_dir", {"path": "."}, "c"))
    ] * 5
    budget = AgentBudget(stagnation_window=2, hard_max_iterations=50)
    result, renderer, b, _ = await run(project, script, budget=budget)
    assert result.get("halt_reason")
    assert "疑似死循环" in result["halt_reason"]
    assert b.iteration == 2  # 第 2 轮记录重复签名，第 3 轮入口被拦


async def test_token_budget_halts(project):
    # 第 1 轮带工具调用 + usage(100+100) → 第 2 轮入口 check 命中 token 预算
    script = [
        AIMessage(
            content="",
            tool_calls=tool_call("list_dir", {"path": "."}, "c"),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 100,
                "total_tokens": 200,
            },
        ),
        AIMessage(content="second"),
    ]
    budget = AgentBudget(token_budget=150)
    result, _, b, _ = await run(project, script, budget=budget)
    assert result.get("halt_reason")
    assert "Token 预算已用尽" in result["halt_reason"]
    assert b.total_input_tokens + b.total_output_tokens == 200


async def test_empty_llm_response_halts(project):
    result, renderer, _, _ = await run(project, None, model=EmptyModel(script=[]))
    assert result.get("halt_reason") == "LLM 未返回内容"
    assert renderer.halts == ["LLM 未返回内容"]


# ---------- extract_tool_calls ----------


def test_extract_tool_calls_prefers_tool_calls():
    msg = AIMessage(content="", tool_calls=tool_call("read_file", {"path": "a"}, "c1"))
    calls = extract_tool_calls(msg)
    assert calls[0]["name"] == "read_file"
    assert calls[0]["args"] == {"path": "a"}
    assert calls[0]["id"] == "c1"


def test_extract_tool_calls_from_chunks():
    msg = AIMessage(
        content="",
        tool_call_chunks=[
            {
                "name": "read_file",
                "args": '{"path": "a.md"}',
                "id": "c1",
                "type": "tool_call_chunk",
            }
        ],
    )
    calls = extract_tool_calls(msg)
    assert calls[0]["name"] == "read_file"
    assert calls[0]["args"] == {"path": "a.md"}
    assert calls[0]["id"] == "c1"


def test_extract_tool_calls_bad_json_kept():
    msg = AIMessage(
        content="",
        tool_call_chunks=[
            {"name": "read_file", "args": "{bad json", "id": "c2", "type": "tool_call_chunk"}
        ],
    )
    calls = extract_tool_calls(msg)
    assert calls[0]["args"] == {"_raw": "{bad json"}


# ---------- AgentBudget 单元语义（对齐 Java AgentBudgetTest） ----------


def test_budget_none_means_unlimited():
    b = AgentBudget(token_budget=None)
    b.record_tokens(10**9, 10**9)
    assert b.check() is not ExitReason.TOKEN_BUDGET_EXCEEDED


def test_budget_stagnation_persists():
    b = AgentBudget(stagnation_window=2)
    call = [{"name": "x", "args": {"a": 1}}]
    b.record_tool_calls(call)
    b.record_tool_calls(call)
    assert b.check() is ExitReason.STAGNATION_DETECTED
    # 空轮次清空签名，但停滞状态保持（对齐 Java：一旦判定为停滞，状态会保持）
    b.record_tool_calls([])
    assert b.check() is ExitReason.STAGNATION_DETECTED


def test_budget_different_calls_no_stagnation():
    b = AgentBudget(stagnation_window=2)
    b.record_tool_calls([{"name": "x", "args": {"a": 1}}])
    b.record_tool_calls([{"name": "x", "args": {"a": 2}}])
    assert b.check() is not ExitReason.STAGNATION_DETECTED


def test_budget_validation():
    with pytest.raises(ValueError):
        AgentBudget(stagnation_window=1)
    with pytest.raises(ValueError):
        AgentBudget(hard_max_iterations=0)


def test_budget_describe_exit_messages():
    assert "0 / 100" in AgentBudget(token_budget=100).describe_exit(
        ExitReason.TOKEN_BUDGET_EXCEEDED
    )
    b = AgentBudget(token_budget=100, hard_max_iterations=7)
    assert "疑似死循环" in b.describe_exit(ExitReason.STAGNATION_DETECTED)
    assert "达到硬轮数上限（7）" in b.describe_exit(ExitReason.HARD_ITERATION_LIMIT)
    assert b.describe_exit(ExitReason.WITHIN_BUDGET) == "未触发兜底条件"


def test_budget_begin_iteration_increments():
    b = AgentBudget()
    assert b.begin_iteration() == 1
    assert b.begin_iteration() == 2
    assert b.iteration == 2


# ---------- 历史压缩接入（对齐 Java maybeCompactHistory） ----------


async def _fake_summarize(messages) -> str:
    return f"摘要了 {len(messages)} 条历史消息"


async def test_auto_compaction_in_agent_loop(project):
    """超阈值历史在第 1 轮 LLM 调用前被压缩；压缩后 ReAct 循环继续正常走完。

    trigger_tokens=1 是人造小阈值（生产环境压缩后即低于阈值不再重复触发），
    第 2 轮入口会再次压缩：摘要对 + 尾部 3 个 user 轮次继续工作，tool 对不被切断。
    """
    model = ScriptedModel(
        script=[
            AIMessage(content="", tool_calls=tool_call("read_file", {"path": "README.md"}, "call_1")),
            AIMessage(content="完成"),
        ]
    )
    guard = PathGuard(project)
    registry = ToolRegistry(settings=Settings(load_files=False))
    register_file_tools(registry, guard)
    renderer = RecordingRenderer()
    budget = AgentBudget(stagnation_window=3, hard_max_iterations=50)
    runtime = make_runtime(model)
    compactor = ConversationHistoryCompactor(summarize=_fake_summarize)
    graph = build_react_graph(
        runtime, registry, renderer, tool_mode="slim",
        compactor=compactor, trigger_tokens=1,
    )

    # 4 个 user 轮次（[S, H1, A1, ..., H4, A4]），retain=3 → 第 1 轮压缩保留后 3 轮
    history = [SystemMessage("system")]
    for i in range(1, 5):
        history.append(HumanMessage(f"问题 {i}"))
        history.append(AIMessage(f"回答 {i}"))

    result = await graph.ainvoke({"messages": history, "budget": budget})
    messages = result["messages"]

    # 第 1 轮压缩切在倒数第 3 个 user（H2），old=[H1,A1]；
    # 第 2 轮再次压缩（人造小阈值），old=[摘要对]；
    # 终态：[S, 摘要, 应答, H2, A2, H3, A3, H4, A4, AI(tc), ToolMessage, AI(完成)]
    assert len(messages) == 12
    assert isinstance(messages[0], SystemMessage)
    assert messages[1].content.startswith("[已压缩的历史对话摘要]\n摘要了 2 条历史消息")
    assert messages[2].content == "好的，我已了解之前的上下文，请继续。"
    assert messages[3].content == "问题 2"
    assert isinstance(messages[9], AIMessage) and messages[9].tool_calls
    assert isinstance(messages[10], ToolMessage)
    assert "hello world" in messages[10].content  # 工具真实执行，未被压缩破坏
    assert messages[11].content == "完成"
    # 最早轮次已从通道移除
    assert not any(getattr(m, "content", "") == "问题 1" for m in messages)
    # renderer 收到压缩提示（每轮一次）
    assert len(renderer.infos) == 2
    assert "已自动压缩历史对话" in renderer.infos[0]
    assert result.get("halt_reason") is None
    assert budget.iteration == 2


async def test_compaction_skipped_below_trigger(project):
    """未达阈值：历史原样保留，不触发摘要。"""
    model = ScriptedModel(script=[AIMessage(content="直接回答")])
    guard = PathGuard(project)
    registry = ToolRegistry(settings=Settings(load_files=False))
    register_file_tools(registry, guard)
    renderer = RecordingRenderer()
    runtime = make_runtime(model)
    compactor = ConversationHistoryCompactor(summarize=_fake_summarize)
    graph = build_react_graph(
        runtime, registry, renderer, tool_mode="slim",
        compactor=compactor, trigger_tokens=10**9,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage("hi")], "budget": AgentBudget()}
    )
    assert len(result["messages"]) == 2  # user + final，未插入摘要
    assert renderer.infos == []
