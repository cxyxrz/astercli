"""Token 估算与上下文阈值。

对齐 Java 侧三个来源：
- MemoryEntry.estimateTokens：中文约 1.5 字/token、其他约 4 字符/token
- TokenBudget.estimateMessagesTokens：content + tool_call args + 每条消息 4 token 开销
- ContextProfile.autoCompactTriggerTokens：window - 摘要预留(≤20k) - 压缩缓冲(≤13k)

与 Java 的已知差异：reasoning_content 随历史回传（DeepSeek/Kimi），同样计入
估算（Java 未计，属低估）。
"""

from __future__ import annotations

import json
import math
from typing import Iterable

from langchain_core.messages import AnyMessage

# 对齐 Java ContextProfile 常量
MAX_SUMMARY_OUTPUT_RESERVE_TOKENS = 20_000
AUTOCOMPACT_BUFFER_TOKENS = 13_000
MIN_COMPRESSION_TRIGGER_RATIO = 0.50
MIN_WINDOW = 8_000


def estimate_tokens(text) -> int:
    """粗略估算 token 数：中文约 1.5 字/token，英文约 4 字符/token。"""
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - chinese
    return math.ceil(chinese / 1.5 + other / 4.0)


def estimate_message_tokens(message: AnyMessage) -> int:
    total = 0
    content = getattr(message, "content", None)
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                total += estimate_tokens(block["text"])
    # DeepSeek/Kimi 的 reasoning_content 会随历史回传，同样占用请求 token
    reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
    if isinstance(reasoning, str):
        total += estimate_tokens(reasoning)
    for tc in getattr(message, "tool_calls", None) or []:
        args = tc.get("args") if isinstance(tc, dict) else None
        total += estimate_tokens(json.dumps(args or {}, ensure_ascii=False, default=str))
    # 每条消息的 role / 分隔符等固定开销
    return total + 4


def estimate_messages_tokens(messages: Iterable[AnyMessage]) -> int:
    return sum(estimate_message_tokens(m) for m in messages or [])


def auto_compact_trigger_tokens(window: int) -> int:
    """自动压缩触发阈值：window - 摘要输出预留 - 安全缓冲（小窗口按比例缩小预留）。"""
    safe = max(MIN_WINDOW, int(window))
    summary_reserve = min(MAX_SUMMARY_OUTPUT_RESERVE_TOKENS, max(1_000, safe // 4))
    buffer = min(AUTOCOMPACT_BUFFER_TOKENS, max(1_000, safe // 8))
    trigger = safe - summary_reserve - buffer
    return max(1_000, min(safe - 1, trigger))


def compression_trigger_ratio(window: int) -> float:
    """触发阈值占窗口的比例（仅供状态展示）。"""
    safe = max(MIN_WINDOW, int(window))
    return max(MIN_COMPRESSION_TRIGGER_RATIO, min(0.99, auto_compact_trigger_tokens(window) / safe))


def memory_context_tokens(window: int) -> int:
    """注入 system prompt 的相关长期记忆 token 上限：window × 0.5%，封顶 5000。"""
    safe = max(MIN_WINDOW, int(window))
    return max(500, min(5_000, safe // 200))
