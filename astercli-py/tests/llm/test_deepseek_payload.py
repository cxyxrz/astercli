"""DeepSeek 三约束的 MockTransport 实证测试（最高风险项）：

1. 输出方向：SSE delta.reasoning_content 提取到 AIMessageChunk.additional_kwargs
2. 输入方向：历史 assistant 消息的 reasoning_content 随请求回传
3. 图片降级：历史 image_url block 替换为省略提示
4. HTTP/1.1 强制

请求/响应全部走 httpx.MockTransport，不发真实网络请求。
"""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from astercli.llm.deepseek import DeepSeekClient
from astercli.llm.openai_compat import OpenAiCompatClient


def _sse_chunk(delta, finish=None):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse_bytes(*chunks):
    lines = ["data: " + json.dumps(c, ensure_ascii=False) for c in chunks]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


REASONING_STREAM = (
    _sse_chunk({"role": "assistant"}),
    _sse_chunk({"reasoning_content": "先分析需求"}),
    _sse_chunk({"reasoning_content": "，再决定工具"}),
    _sse_chunk({"content": "你好"}),
    _sse_chunk({}, finish="stop"),
)


def make_client(captured, *, stream_chunks=REASONING_STREAM, cls=DeepSeekClient, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse_bytes(*stream_chunks),
            headers={"content-type": "text/event-stream"},
        )

    common = dict(
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return cls(**{**common, **kwargs})


async def collect(client, messages):
    chunks = [c async for c in client.astream(messages)]
    final = None
    for c in chunks:
        if final is None:
            final = c
        else:
            final = final + c
    return final


# ---------- 1. 输出方向：reasoning 提取 ----------


async def test_stream_reasoning_content_extracted():
    captured = []
    client = make_client(captured)
    final = await collect(client, [HumanMessage(content="hi")])

    assert final is not None
    assert final.content == "你好"
    # 两个 reasoning delta 都应累积到 additional_kwargs
    assert final.additional_kwargs["reasoning_content"] == "先分析需求，再决定工具"


async def test_stream_reasoning_key_alias_supported():
    """部分 provider 用 `reasoning` 字段名，同样应被提取。"""
    captured = []
    client = make_client(
        captured,
        stream_chunks=(
            _sse_chunk({"role": "assistant"}),
            _sse_chunk({"reasoning": "thinking via alias"}),
            _sse_chunk({"content": "ok"}),
            _sse_chunk({}, finish="stop"),
        ),
    )
    final = await collect(client, [HumanMessage(content="hi")])
    assert final.additional_kwargs["reasoning_content"] == "thinking via alias"


async def test_base_client_does_not_extract_reasoning():
    """非 DeepSeek/Kimi 的基类同样提取展示用 reasoning（灰字渲染所需），
    但不回传历史——回传只由 reasoning_roundtrip 控制，见下。"""
    captured = []
    client = make_client(captured, cls=OpenAiCompatClient)
    final = await collect(client, [HumanMessage(content="hi")])
    assert final.additional_kwargs["reasoning_content"] == "先分析需求，再决定工具"


# ---------- 2. 输入方向：reasoning 回传 ----------


async def test_reasoning_roundtrip_in_request():
    captured = []
    client = make_client(captured)
    history = [
        HumanMessage(content="读一下 README"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "思考过程：需要 read_file"},
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="文件内容: ...", tool_call_id="call_1"),
    ]
    async for _ in client.astream(history):
        pass

    assert len(captured) == 1
    req = captured[0]
    assert req["stream"] is True
    assistant = req["messages"][1]
    assert assistant["role"] == "assistant"
    # 核心断言：reasoning_content 随历史 assistant 消息回传
    assert assistant["reasoning_content"] == "思考过程：需要 read_file"
    # 工具调用结构保持
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"
    # ToolMessage 正常回灌
    assert req["messages"][2]["role"] == "tool"


async def test_reasoning_not_roundtripped_for_base_client():
    """OpenAiCompatClient（reasoning_roundtrip=False）不应回传 reasoning_content。"""
    captured = []
    client = make_client(captured, cls=OpenAiCompatClient)
    history = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "不应回传"},
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ok", tool_call_id="call_1"),
    ]
    async for _ in client.astream(history):
        pass
    assistant = captured[0]["messages"][1]
    assert "reasoning_content" not in assistant


async def test_reasoning_roundtrip_skips_user_and_tool_messages():
    captured = []
    client = make_client(captured)
    history = [
        HumanMessage(content="hi"),
        AIMessage(content="", additional_kwargs={"reasoning_content": "r"}),
        HumanMessage(content="again"),
    ]
    async for _ in client.astream(history):
        pass
    messages = captured[0]["messages"]
    assert messages[1].get("reasoning_content") == "r"
    assert "reasoning_content" not in messages[0]
    assert "reasoning_content" not in messages[2]


# ---------- 3. 图片降级 ----------


async def test_history_image_degraded_for_deepseek():
    captured = []
    client = make_client(captured)
    history = [
        HumanMessage(
            content=[
                {"type": "text", "text": "看这张图"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ]
        ),
    ]
    async for _ in client.astream(history):
        pass
    user = captured[0]["messages"][0]
    assert isinstance(user["content"], list)
    types = [b["type"] for b in user["content"]]
    assert "image_url" not in types
    assert types == ["text", "text"]
    assert "历史图片附件已省略 1 张" in user["content"][-1]["text"]


async def test_image_not_degraded_for_base_client():
    captured = []
    client = make_client(captured, cls=OpenAiCompatClient)
    history = [
        HumanMessage(
            content=[
                {"type": "text", "text": "看这张图"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ]
        ),
    ]
    async for _ in client.astream(history):
        pass
    user = captured[0]["messages"][0]
    types = [b["type"] for b in user["content"]]
    assert "image_url" in types


def test_degrade_image_blocks_count():
    from astercli.llm.openai_compat import _degrade_image_blocks

    blocks = [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "u1"}},
        {"type": "text", "text": "b"},
        {"type": "image_url", "image_url": {"url": "u2"}},
    ]
    out = _degrade_image_blocks(blocks)
    assert [b["type"] for b in out] == ["text", "text", "text"]
    assert "已省略 2 张" in out[-1]["text"]


# ---------- 4. HTTP/1.1 强制 ----------


def test_deepseek_enforces_http1(monkeypatch):
    seen = []

    def spy_async(*args, **kwargs):
        seen.append(("async", kwargs))
        return None

    def spy_sync(*args, **kwargs):
        seen.append(("sync", kwargs))
        return None

    monkeypatch.setattr(httpx, "AsyncClient", spy_async)
    monkeypatch.setattr(httpx, "Client", spy_sync)

    DeepSeekClient(model="m", api_key="k", base_url="http://localhost:1")

    assert seen, "DeepSeekClient 应创建 httpx 客户端"
    for _, kwargs in seen:
        assert kwargs.get("http1") is True
        assert kwargs.get("http2") is False


def test_deepseek_injected_client_not_replaced():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    injected = httpx.AsyncClient(transport=transport)
    client = DeepSeekClient(
        model="m", api_key="k", base_url="http://localhost:1", http_async_client=injected
    )
    assert client.http_async_client is injected
