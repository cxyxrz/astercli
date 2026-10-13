"""LangGraph HITL 挂起恢复测试：interrupt 挂起 → 收集决策 → Command(resume) 恢复。

覆盖：approve 恢复后工具执行、reject 拒绝文本回灌 LLM、approve_all 二次
同工具调用免打断、非危险工具不触发 interrupt、无 checkpointer 时审批禁用。
ScriptedModel 驱动，不发真实请求。
"""

from __future__ import annotations

import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from astercli.agent.budget import AgentBudget
from astercli.agent.graph import ainvoke_with_hitl, build_react_graph, make_checkpointer
from astercli.config import Settings
from astercli.hitl import ApprovalSession
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.policy import PathGuard
from astercli.tools.builtin.files import register_file_tools
from astercli.tools.registry import ToolRegistry


class ScriptedModel(BaseChatModel):
    """按脚本顺序返回消息的假模型（同 test_graph.py，独立副本避免跨文件耦合）。"""

    script: list = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("测试只走 astream 路径")

    async def _astream(self, messages, stop=None, **kwargs):
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
        self.infos = []
        self.recorded_results = []

    def begin_thinking(self):
        pass

    def end_thinking(self):
        pass

    def stream_chunk(self, chunk):
        pass

    def take_reasoning(self):
        return ""

    def tool_calls(self, calls):
        pass

    def tool_results(self, calls, results):
        self.recorded_results.append((list(calls), list(results)))

    def halt(self, message):
        pass

    def info(self, message):
        self.infos.append(message)


def tool_call(name, args, id_):
    return [{"name": name, "args": args, "id": id_, "type": "tool_call"}]


def make_setup(project, model, *, approval, checkpointer):
    runtime = ProviderRuntime(
        client=model, provider="glm", model="scripted", spec=PROVIDER_SPECS["glm"]
    )
    registry = ToolRegistry(settings=Settings(load_files=False), approval=approval)
    register_file_tools(registry, PathGuard(project))
    renderer = RecordingRenderer()
    graph = build_react_graph(
        runtime,
        registry,
        renderer,
        tool_mode="slim",
        approval=approval,
        checkpointer=checkpointer,
    )
    return graph, renderer


def make_on_interrupt(record, action="approve", reason="no"):
    async def on_interrupt(interrupts):
        record.append(interrupts)
        payload = getattr(interrupts[0], "value", {})
        return {c["id"]: {"action": action, "reason": reason} for c in payload.get("calls", [])}

    return on_interrupt


async def run_turn(graph, *, on_interrupt=None):
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    budget = AgentBudget(stagnation_window=3, hard_max_iterations=10)
    return await ainvoke_with_hitl(
        graph,
        {"messages": [HumanMessage(content="任务")], "budget": budget, "halt_reason": None},
        config,
        on_interrupt,
    )


# ---------- 挂起 + approve 恢复 ----------


async def test_interrupt_suspends_and_approve_resumes(project):
    script = [
        AIMessage(
            content="",
            tool_calls=tool_call("write_file", {"path": "out.txt", "content": "hi"}, "c1"),
        ),
        AIMessage(content="写入完成"),
    ]
    record = []
    graph, _ = make_setup(
        project, ScriptedModel(script=script), approval=ApprovalSession(), checkpointer=make_checkpointer()
    )
    result = await run_turn(graph, on_interrupt=make_on_interrupt(record, action="approve"))

    # 发生过一次挂起，且载荷指向 write_file
    assert len(record) == 1
    payload = record[0][0].value
    assert payload["calls"][0]["name"] == "write_file"
    # 恢复后工具真实执行
    assert (project / "out.txt").read_text(encoding="utf-8") == "hi"
    messages = result["messages"]
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content == "文件已写入: out.txt"
    assert messages[3].content == "写入完成"
    assert result.get("halt_reason") is None


async def test_reject_decision_feeds_back_to_llm(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("write_file", {"path": "out.txt", "content": "x"}, "c1")),
        AIMessage(content="用户拒绝了写入，我改用只读方案完成任务"),
    ]
    record = []
    graph, _ = make_setup(
        project, ScriptedModel(script=script), approval=ApprovalSession(), checkpointer=make_checkpointer()
    )
    result = await run_turn(graph, on_interrupt=make_on_interrupt(record, action="reject", reason="别写这个文件"))

    messages = result["messages"]
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content == "[HITL] 操作已被拒绝：别写这个文件"
    # 工具没有真实执行
    assert not (project / "out.txt").exists()
    assert messages[3].content.startswith("用户拒绝")


async def test_approve_all_skips_second_interrupt(project):
    # 两次 write_file：第一次 approve_all，第二次不再挂起直接放行
    script = [
        AIMessage(content="", tool_calls=tool_call("write_file", {"path": "a.txt", "content": "1"}, "c1")),
        AIMessage(content="", tool_calls=tool_call("write_file", {"path": "b.txt", "content": "2"}, "c2")),
        AIMessage(content="两次都写完了"),
    ]
    record = []
    graph, _ = make_setup(
        project, ScriptedModel(script=script), approval=ApprovalSession(), checkpointer=make_checkpointer()
    )
    result = await run_turn(graph, on_interrupt=make_on_interrupt(record, action="approve_all"))

    # 只挂起一次；第二次调用经会话放行名单直接执行
    assert len(record) == 1
    assert (project / "a.txt").read_text(encoding="utf-8") == "1"
    assert (project / "b.txt").read_text(encoding="utf-8") == "2"
    assert result.get("halt_reason") is None


async def test_non_dangerous_tool_never_interrupts(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("read_file", {"path": "README.md"}, "c1")),
        AIMessage(content="读完了"),
    ]
    record = []

    async def on_interrupt(interrupts):
        record.append(interrupts)
        return {}

    graph, _ = make_setup(
        project, ScriptedModel(script=script), approval=ApprovalSession(), checkpointer=make_checkpointer()
    )
    result = await run_turn(graph, on_interrupt=on_interrupt)

    assert record == []  # 只读工具不触发审批
    messages = result["messages"]
    assert "hello world" in messages[2].content


async def test_no_on_interrupt_rejects_conservatively(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("write_file", {"path": "out.txt", "content": "x"}, "c1")),
        AIMessage(content="被拒绝了，停止写入"),
    ]
    graph, _ = make_setup(
        project, ScriptedModel(script=script), approval=ApprovalSession(), checkpointer=make_checkpointer()
    )
    result = await run_turn(graph, on_interrupt=None)

    messages = result["messages"]
    assert "保守拒绝" in messages[2].content
    assert not (project / "out.txt").exists()


# ---------- 审批禁用路径 ----------


async def test_no_checkpointer_disables_approval(project):
    script = [
        AIMessage(content="", tool_calls=tool_call("write_file", {"path": "out.txt", "content": "hi"}, "c1")),
        AIMessage(content="写入完成"),
    ]
    record = []
    # 传了 approval 但没传 checkpointer → 审批自动禁用，直接执行
    graph, _ = make_setup(
        project, ScriptedModel(script=script), approval=ApprovalSession(), checkpointer=None
    )
    result = await run_turn(graph, on_interrupt=make_on_interrupt(record))

    assert record == []
    assert (project / "out.txt").read_text(encoding="utf-8") == "hi"
    assert result.get("halt_reason") is None
