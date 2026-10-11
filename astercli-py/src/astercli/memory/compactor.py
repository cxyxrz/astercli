"""conversationHistory 压缩（对齐 Java ConversationHistoryCompactor）。

压缩的是 Agent 真正发给 LLM 的消息列表——它才决定下一轮请求的 input token。
（Java 第 3 期的 shortTermMemory 压缩与本类度量错位，Python 版只保留这一道。）

算法（对齐 Java）：
1. 估算当前 token，未达 trigger 直接跳过
2. 以 user 消息为轮次边界，保留最近 retain_recent_rounds 个 user 起始的尾部
3. system 之后、分割点之前的消息喂给 LLM 摘要（60k 字符截断）
4. 重建：[system] + HumanMessage("[已压缩的历史对话摘要]") + AIMessage(应答) + 尾部

关键约束：分割点必然落在 user 边界，不会切断 tool_call / tool_result 的成对协议。
摘要失败（LLM 异常 / 空结果）返回 None，不破坏原历史。

返回新列表而非原地变异：调用方（agent 节点 / /compact 命令）决定如何应用。
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .tokens import estimate_messages_tokens

DEFAULT_RETAIN_RECENT_ROUNDS = 3
MAX_SUMMARY_INPUT_CHARS = 60_000

SUMMARY_PROMPT = """请把下面的对话历史压缩成简明摘要，保留：
1. 用户提出的关键诉求与目标
2. Agent 已经完成的关键操作（哪些工具调用了什么、返回了什么核心结果）
3. 已经达成的共识或结论
4. 仍未解决的问题或待办

不要复述每条原文，不要列举所有工具调用，不要保留无关闲聊。
输出 1-3 段中文，不要用列表，不要加任何前缀或元描述。

=== 待压缩的对话 ===
{dialog}
=== 待压缩的对话（结束）==="""

SUMMARY_SYSTEM = "你是一个对话摘要助手，只输出摘要本身，不输出元描述。"
SUMMARY_ACK = "好的，我已了解之前的上下文，请继续。"


class ConversationHistoryCompactor:
    """history 压缩器。

    :param client: LLM 客户端（生成摘要用）；注入 summarize 回调时可为 None
    :param retain_recent_rounds: 自动压缩保留的最近 user 轮次数
    :param summarize: 自定义摘要函数 ``async (messages) -> str``（测试注入）
    """

    def __init__(
        self,
        client=None,
        retain_recent_rounds: int = DEFAULT_RETAIN_RECENT_ROUNDS,
        summarize=None,
    ) -> None:
        self._client = client
        self._retain = max(1, retain_recent_rounds)
        self._summarize_fn = summarize

    @property
    def retain_recent_rounds(self) -> int:
        return self._retain

    async def compact_if_needed(self, messages: List, trigger_tokens: int) -> Optional[List]:
        """达到阈值才压缩；返回压缩后的新历史，未压缩返回 None。"""
        if not messages:
            return None
        if estimate_messages_tokens(messages) < trigger_tokens:
            return None
        return await self._compact(messages, retain_rounds=self._retain)

    async def compact_now(self, messages: List) -> Optional[List]:
        """手动压缩：跳过阈值判断，仅保留最近 1 个 user 轮次（/compact）。"""
        return await self._compact(messages, retain_rounds=1)

    async def _compact(self, messages: List, *, retain_rounds: int) -> Optional[List]:
        if not messages:
            return None
        system_end = 1 if isinstance(messages[0], SystemMessage) else 0
        user_indices = [
            i for i in range(system_end, len(messages)) if isinstance(messages[i], HumanMessage)
        ]
        # user 轮次不足时不压缩：宁可多花 token 也不为摘要制造过短的尾部
        if len(user_indices) <= retain_rounds:
            return None
        split = user_indices[-retain_rounds]
        if split <= system_end:
            return None
        old = messages[system_end:split]
        if not old:
            return None

        try:
            summary = await self._summarize(old)
        except Exception:
            # 摘要失败不破坏原历史（对齐 Java：warn 日志 + skip）
            return None
        if not summary or not summary.strip():
            return None

        rebuilt = list(messages[:system_end])
        rebuilt.append(HumanMessage("[已压缩的历史对话摘要]\n" + summary.strip()))
        rebuilt.append(AIMessage(SUMMARY_ACK))
        rebuilt.extend(messages[split:])
        return rebuilt

    async def _summarize(self, messages: List) -> Optional[str]:
        if self._summarize_fn is not None:
            return await self._summarize_fn(messages)
        if self._client is None:
            return None
        prompt = SUMMARY_PROMPT.format(dialog=_serialize_for_summary(messages))
        resp = await self._client.ainvoke(
            [SystemMessage(SUMMARY_SYSTEM), HumanMessage(prompt)]
        )
        content = getattr(resp, "content", None)
        return content if isinstance(content, str) else None


def _serialize_for_summary(messages: List) -> str:
    """把消息列表拼成摘要 prompt 的文本（超长截断）。"""
    parts: List[str] = []
    length = 0
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "USER"
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
        elif isinstance(m, ToolMessage):
            role = "TOOL"
        else:
            role = "SYSTEM"
        piece = f"{role}: {m.content if isinstance(m.content, str) else ''}\n"
        piece += _format_tool_calls(m)
        parts.append(piece + "\n")
        length += len(piece)
        if length > MAX_SUMMARY_INPUT_CHARS:
            parts.append("...(超长内容已截断)\n")
            break
    return "".join(parts)


def _format_tool_calls(message) -> str:
    """tool_calls 优先；流式聚合的 AIMessageChunk 从 tool_call_chunks 兜底。"""
    calls = list(getattr(message, "tool_calls", None) or [])
    if not calls:
        for chunk in getattr(message, "tool_call_chunks", None) or []:
            calls.append({"name": chunk.get("name") or "", "args": chunk.get("args") or ""})
    lines = []
    for tc in calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if not isinstance(args, dict):
            args = {"_raw": str(args)}
        lines.append(f"  TOOL_CALL {name}: {json.dumps(args, ensure_ascii=False, default=str)}\n")
    return "".join(lines)
