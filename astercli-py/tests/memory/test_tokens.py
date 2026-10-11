"""token 估算与阈值函数测试（对齐 Java MemoryEntry / TokenBudget / ContextProfile）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from astercli.memory.tokens import (
    auto_compact_trigger_tokens,
    compression_trigger_ratio,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    memory_context_tokens,
)


# ---------- estimate_tokens ----------


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_chinese():
    # 中文约 1.5 字/token：2 字 → ceil(1.33) = 2
    assert estimate_tokens("你好") == 2
    # 3 字 → ceil(2.0) = 2
    assert estimate_tokens("你好吗") == 2


def test_estimate_tokens_english():
    # 英文约 4 字符/token：8 字符 → 2
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_mixed():
    # 2 中文 + 4 英文 → ceil(2/1.5 + 4/4) = ceil(2.33) = 3
    assert estimate_tokens("你好abcd") == 3


# ---------- estimate_messages_tokens ----------


def test_estimate_message_tokens_includes_per_message_overhead():
    assert estimate_message_tokens(HumanMessage("abcd")) == 1 + 4


def test_estimate_messages_tokens_sums():
    msgs = [SystemMessage("sys"), HumanMessage("abcd")]
    total = estimate_messages_tokens(msgs)
    assert total == (1 + 4) + (1 + 4)


def test_estimate_message_tokens_counts_tool_call_args():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "c1", "type": "tool_call"}],
    )
    # args JSON（约 17 字符 → 5 tokens）+ 空内容 + 4 开销
    assert estimate_message_tokens(msg) >= 4


def test_estimate_message_tokens_counts_reasoning():
    msg = AIMessage("答案", additional_kwargs={"reasoning_content": "思考过程"})
    base = estimate_message_tokens(AIMessage("答案"))
    assert estimate_message_tokens(msg) > base


def test_estimate_message_tokens_multimodal_content_blocks():
    msg = HumanMessage(
        content=[{"type": "text", "text": "abcdefgh"}, {"type": "image_url", "url": "x"}]
    )
    # text block 计入（2 tokens）+ 4 开销；图片 block Python 版暂不计
    assert estimate_message_tokens(msg) >= 6


# ---------- 阈值函数（对齐 Java ContextProfile） ----------


def test_auto_compact_trigger_tokens_large_window():
    # 200k：window - 20k - 13k = 167k（AGENTS.md 约定值）
    assert auto_compact_trigger_tokens(200_000) == 167_000
    # 1M：window - 20k - 13k = 967k
    assert auto_compact_trigger_tokens(1_000_000) == 967_000


def test_auto_compact_trigger_tokens_small_window():
    # 8000：预留 min(20000, max(1000, 2000))=2000，缓冲 min(13000, max(1000, 1000))=1000
    assert auto_compact_trigger_tokens(8_000) == 5_000
    # 低于 MIN_WINDOW 的输入按 8000 处理
    assert auto_compact_trigger_tokens(1_000) == 5_000


def test_compression_trigger_ratio_bounded():
    large = compression_trigger_ratio(200_000)
    assert 0.5 <= large <= 0.99
    assert large > 0.8  # 167k/200k = 0.835
    assert compression_trigger_ratio(8_000) == 0.625


def test_memory_context_tokens():
    # window × 0.5%，[500, 5000]
    assert memory_context_tokens(200_000) == 1_000
    assert memory_context_tokens(1_000_000) == 5_000
    assert memory_context_tokens(128_000) == 640
    assert memory_context_tokens(8_000) == 500
