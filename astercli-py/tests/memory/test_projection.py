"""读时投影（第 3、4 层压缩）测试：微压缩、折叠、最旧轮次省略。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astercli.memory.projection import (
    MICRO_MARKER,
    project_messages,
)
from astercli.memory.tokens import estimate_messages_tokens

BIG = "x" * 4_000


def tool_history(tool_name: str, text: str, rounds_after: int) -> list:
    """[S, H1, AI(tool_call), Tool, AI, (H,A) * rounds_after]"""
    messages = [
        SystemMessage("system"),
        HumanMessage("第一轮"),
        AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content=text, tool_call_id="c1", name=tool_name),
        AIMessage("完成"),
    ]
    for i in range(rounds_after):
        messages.append(HumanMessage(f"后续 {i}"))
        messages.append(AIMessage("回答"))
    return messages


def test_旧轮次只读工具结果被清理():
    messages = tool_history("read_file", BIG, rounds_after=3)
    projected = project_messages(messages, keep_recent_rounds=3)
    assert projected[3].content.startswith("[micro-compact]")
    assert "read_file" in projected[3].content
    # tool_call_id 保留，协议不破坏
    assert projected[3].tool_call_id == "c1"


def test_最近轮次与副作用结果不清理():
    recent = tool_history("read_file", BIG, rounds_after=1)
    assert project_messages(recent, keep_recent_rounds=3)[3].content == BIG

    side_effect = tool_history("execute_command", BIG, rounds_after=3)
    assert project_messages(side_effect, keep_recent_rounds=3)[3].content == BIG


def test_短结果不清理():
    messages = tool_history("read_file", "短内容", rounds_after=3)
    assert project_messages(messages, keep_recent_rounds=3)[3].content == "短内容"


def test_不改原列表也不原地变异():
    messages = tool_history("read_file", BIG, rounds_after=3)
    projected = project_messages(messages, keep_recent_rounds=3)
    assert projected is not messages
    assert messages[3].content == BIG


def test_超预算先折叠旧正文():
    history = [SystemMessage("system")]
    for i in range(4):
        history.append(HumanMessage(f"问题 {i}"))
        history.append(AIMessage(BIG))
    before = estimate_messages_tokens(history)
    projected = project_messages(history, keep_recent_rounds=1, budget_tokens=800)
    assert estimate_messages_tokens(projected) < before
    assert any("context-collapse 省略" in m.content for m in projected)
    assert projected[-1] is history[-1]  # 最近轮次原文保留


def test_仍超预算时省略最旧轮次():
    history = [SystemMessage("system")]
    for i in range(6):
        history.append(HumanMessage(f"问题 {i}"))
        history.append(AIMessage(BIG))
    projected = project_messages(history, keep_recent_rounds=1, budget_tokens=100)
    assert len(projected) < len(history)
    assert isinstance(projected[1], HumanMessage)
    assert "已省略" in projected[1].content
    assert projected[-1] is history[-1]


def test_空历史():
    assert project_messages([]) == []


def test_marker_常量包含工具名占位():
    assert "{name}" in MICRO_MARKER
