"""读时投影（Context Collapse）+ 微压缩（Micro-Compact）。

对应五层压缩金字塔的第 3、4 层，共同点：**零 API 调用**，只在调用 LLM 前
生成一份投影列表，不改写 LangGraph state，也不删除消息（避免破坏
tool_call / tool_result 的成对协议）。

- Micro-Compact（时间衰减）：旧轮次里"可重新获取"的工具结果清成占位符。
  只清只读工具；write_file / execute_command 的结果是副作用记录，不能丢。
- Context Collapse（读时投影）：投影后仍超预算时，先截断旧消息正文
  （保留头尾），再从最旧的 user 轮次整体删除，插入一行省略标记。

两层都按 user 消息划分轮次，最近 keep_recent_rounds 轮永远原样保留。
"""

from __future__ import annotations

import copy
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .tokens import estimate_messages_tokens

MICRO_KEEP_RECENT_ROUNDS = 3
MICRO_CLEAR_MIN_CHARS = 600
COLLAPSE_MAX_CHARS = 1200
COLLAPSE_HEAD_CHARS = 800
COLLAPSE_TAIL_CHARS = 300

# 结果可重复获取（无副作用）的只读工具，才允许时间衰减清理
REFETCHABLE_TOOLS = frozenset(
    {
        "read_file",
        "read_skill_file",
        "list_dir",
        "glob_files",
        "grep_code",
        "search_code",
        "web_fetch",
        "web_search",
    }
)

MICRO_MARKER = "[micro-compact] 旧轮次结果已清理以节省上下文（需要时重新调用 {name}）"
COLLAPSE_TEMPLATE = "{head}\n...[context-collapse 省略 {omitted} 字符]...\n{tail}"
DROP_MARKER = "[context-collapse] 已省略 {count} 条较早消息（超出上下文预算）"


def project_messages(
    messages: List,
    *,
    keep_recent_rounds: int = MICRO_KEEP_RECENT_ROUNDS,
    budget_tokens: int = 0,
) -> List:
    """生成读时投影；未发生任何裁剪时返回新列表（不共享引用）。"""
    if not messages:
        return []
    projected = list(messages)
    system_end = 1 if isinstance(projected[0], SystemMessage) else 0
    boundary = _recent_boundary(projected, system_end, keep_recent_rounds)

    # 第 3 层：旧轮次可重新获取的工具结果 → 占位符
    for i in range(system_end, boundary):
        message = projected[i]
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) not in REFETCHABLE_TOOLS:
            continue
        text = _content_text(message)
        if len(text) <= MICRO_CLEAR_MIN_CHARS:
            continue
        marker = MICRO_MARKER.format(name=message.name)
        projected[i] = _with_content(message, marker)

    # 第 4 层：仍超预算 → 先截断旧正文，再从最旧轮次整段省略
    if budget_tokens > 0 and estimate_messages_tokens(projected) > budget_tokens:
        for i in range(system_end, boundary):
            text = _content_text(projected[i])
            if len(text) > COLLAPSE_MAX_CHARS:
                projected[i] = _with_content(projected[i], _collapse(text))

        if estimate_messages_tokens(projected) > budget_tokens:
            projected = _drop_oldest_rounds(projected, system_end, boundary, budget_tokens)
    return projected


def _recent_boundary(messages: List, system_end: int, keep_recent_rounds: int) -> int:
    """最近 keep_recent_rounds 个 user 轮的起始下标；轮次不足时返回 system_end。"""
    user_positions = [
        i for i in range(system_end, len(messages)) if isinstance(messages[i], HumanMessage)
    ]
    keep = max(1, keep_recent_rounds)
    if len(user_positions) <= keep:
        return system_end  # 轮次不足：全部算“最近”，没有可裁剪的旧区
    return user_positions[-keep]


def _drop_oldest_rounds(messages: List, system_end: int, boundary: int, budget: int) -> List:
    head = messages[:system_end]
    old = messages[system_end:boundary]
    tail = messages[boundary:]
    cuts = [i for i, m in enumerate(old) if isinstance(m, HumanMessage)][1:]
    kept = old
    dropped = 0
    for cut in cuts:
        if estimate_messages_tokens(head + kept + tail) <= budget:
            break
        kept = old[cut:]
        dropped = cut
    if dropped == 0:
        return messages
    marker = HumanMessage(DROP_MARKER.format(count=dropped))
    return head + [marker] + kept + tail


def _collapse(text: str) -> str:
    omitted = len(text) - COLLAPSE_HEAD_CHARS - COLLAPSE_TAIL_CHARS
    return COLLAPSE_TEMPLATE.format(
        head=text[:COLLAPSE_HEAD_CHARS],
        omitted=max(0, omitted),
        tail=text[-COLLAPSE_TAIL_CHARS:],
    )


def _content_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""


def _with_content(message, content: str):
    """复制消息并替换 content；langchain 消息是 pydantic 模型，优先 model_copy。"""
    try:
        return message.model_copy(update={"content": content})
    except Exception:  # noqa: BLE001 - 非 pydantic 实现走浅拷贝兜底
        clone = copy.copy(message)
        clone.content = content
        return clone
