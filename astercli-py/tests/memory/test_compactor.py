"""conversationHistory 压缩测试（对齐 Java ConversationHistoryCompactor 语义）。

覆盖：阈值跳过、user 轮次边界切割、tool 对不切断、摘要失败不破坏历史、
compact_now 手动模式、60k 截断、重建结构。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astercli.memory.compactor import (
    MAX_SUMMARY_INPUT_CHARS,
    SUMMARY_ACK,
    ConversationHistoryCompactor,
    _serialize_for_summary,
)


def make_history(rounds: int = 5) -> list:
    """[S, H1, A1, H2, A2, ...]：rounds 个 user 轮次。"""
    history = [SystemMessage("system")]
    for i in range(1, rounds + 1):
        history.append(HumanMessage(f"问题 {i}"))
        history.append(AIMessage(f"回答 {i}"))
    return history


async def fake_summarize(messages) -> str:
    return f"共 {len(messages)} 条消息的摘要"


def make_compactor(retain=3, summarize=fake_summarize, client=None) -> ConversationHistoryCompactor:
    return ConversationHistoryCompactor(client=client, retain_recent_rounds=retain, summarize=summarize)


# ---------- 阈值与轮次守卫 ----------


async def test_below_trigger_returns_none():
    history = make_history(5)
    compactor = make_compactor()
    assert await compactor.compact_if_needed(history, trigger_tokens=10**9) is None
    assert history  # 原历史不变


async def test_insufficient_user_rounds_returns_none():
    # user 轮次数 <= retain（3）时即使超阈值也不压缩
    history = make_history(3)
    compactor = make_compactor()
    assert await compactor.compact_if_needed(history, trigger_tokens=1) is None


async def test_empty_history_returns_none():
    compactor = make_compactor()
    assert await compactor.compact_if_needed([], trigger_tokens=1) is None
    assert await compactor.compact_now([]) is None


# ---------- 自动压缩 ----------


async def test_compact_if_needed_rebuild_structure():
    history = make_history(5)  # [S, H1..A5]，user 索引 [1,3,5,7,9]
    compactor = make_compactor(retain=3)
    rebuilt = await compactor.compact_if_needed(history, trigger_tokens=1)

    assert rebuilt is not None
    # retain=3 → 切分在倒数第 3 个 user（H3，索引 5）：old = [H1,A1,H2,A2]
    # 重建：[S, 摘要, 应答, H3, A3, H4, A4, H5, A5]
    assert len(rebuilt) == 9
    assert isinstance(rebuilt[0], SystemMessage)
    assert isinstance(rebuilt[1], HumanMessage)
    assert rebuilt[1].content.startswith("[已压缩的历史对话摘要]\n")
    assert "共 4 条消息的摘要" in rebuilt[1].content
    assert isinstance(rebuilt[2], AIMessage)
    assert rebuilt[2].content == SUMMARY_ACK
    assert rebuilt[3] is history[5]  # H3 原对象（保留最近 3 个 user 轮次）
    assert rebuilt[-1] is history[10]  # A5 原对象
    # 原历史不被破坏（返回新列表而非原地变异）
    assert len(history) == 11


async def test_compact_never_splits_tool_pair():
    # 轮次中间带 tool_call / tool_result：分割点必须落在 user 边界
    history = [
        SystemMessage("system"),
        HumanMessage("第一个任务"),
        AIMessage("第一轮回答"),
        HumanMessage("读文件"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "a.md"}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content="文件内容", tool_call_id="c1", name="read_file"),
        AIMessage("读到了"),
        HumanMessage("总结一下"),
        AIMessage("总结"),
    ]
    compactor = make_compactor(retain=1)
    rebuilt = await compactor.compact_if_needed(history, trigger_tokens=1)
    assert rebuilt is not None
    # 保留最近 1 个 user 轮次：尾部从"总结一下"开始，tool 对整体在被摘要区
    assert rebuilt[-2] is history[7]
    assert rebuilt[-1] is history[8]
    assert not any(isinstance(m, ToolMessage) for m in rebuilt)


async def test_summary_failure_keeps_history():
    async def boom(messages):
        raise RuntimeError("LLM down")

    history = make_history(5)
    compactor = make_compactor(summarize=boom)
    assert await compactor.compact_if_needed(history, trigger_tokens=1) is None
    assert len(history) == 11


async def test_empty_summary_keeps_history():
    async def empty(messages):
        return "   "

    history = make_history(5)
    compactor = make_compactor(summarize=empty)
    assert await compactor.compact_if_needed(history, trigger_tokens=1) is None


async def test_no_client_no_summarize_returns_none():
    history = make_history(5)
    compactor = ConversationHistoryCompactor()  # 无 client 无回调
    assert await compactor.compact_if_needed(history, trigger_tokens=1) is None


# ---------- compact_now（/compact 手动模式） ----------


async def test_compact_now_forces_and_retains_one_round():
    history = make_history(4)
    compactor = make_compactor(retain=3)  # compact_now 忽略 retain，固定 1
    rebuilt = await compactor.compact_now(history)
    assert rebuilt is not None
    # 尾部只保留最近 1 个 user 轮次：H4, A4
    assert rebuilt[-2] is history[7]
    assert rebuilt[-1] is history[8]
    assert len(rebuilt) == 5  # S + 摘要 + 应答 + H4 + A4


async def test_compact_now_insufficient_rounds_returns_none():
    history = make_history(1)
    compactor = make_compactor()
    assert await compactor.compact_now(history) is None


# ---------- 摘要序列化 ----------


def test_serialize_roles_and_tool_calls():
    messages = [
        HumanMessage("任务"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "a.md"}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content="结果", tool_call_id="c1", name="read_file"),
    ]
    text = _serialize_for_summary(messages)
    assert "USER: 任务" in text
    assert "ASSISTANT:" in text
    assert "TOOL_CALL read_file:" in text
    assert "TOOL: 结果" in text


def test_serialize_chunk_tool_calls_fallback():
    # 流式聚合的 AIMessageChunk：tool_calls 为空时从 tool_call_chunks 兜底
    msg = AIMessage(
        content="",
        tool_call_chunks=[
            {"name": "read_file", "args": '{"path": "a.md"}', "id": "c1", "type": "tool_call_chunk"}
        ],
    )
    text = _serialize_for_summary([msg])
    assert "TOOL_CALL read_file:" in text


def test_serialize_truncates_at_60k():
    messages = [HumanMessage("x" * 30_000), AIMessage("y" * 30_000), HumanMessage("z" * 30_000)]
    text = _serialize_for_summary(messages)
    assert len(text) <= MAX_SUMMARY_INPUT_CHARS + 200
    assert "超长内容已截断" in text
