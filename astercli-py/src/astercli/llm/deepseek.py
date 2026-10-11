"""DeepSeek 定制客户端。

对齐 Java DeepSeekClient 的三条行为约束（AGENTS.md）：
1. V4 thinking 模式下 reasoning_content 必须随历史 assistant 消息回传；
2. SSE 调用强制 HTTP/1.1，避免部分网络/网关下 HTTP/2 长流被远端重置成
   "stream was reset: INTERNAL_ERROR"；
3. 按文本 provider 处理：不支持图片输入，历史图片降级为文本提示。
"""

from __future__ import annotations

import httpx

from .openai_compat import OpenAiCompatClient


class DeepSeekClient(OpenAiCompatClient):
    reasoning_roundtrip = True
    degrade_image_history = True

    def __init__(self, **kwargs) -> None:
        # http_async_client 支持外部注入（测试用 MockTransport），缺省时强制 HTTP/1.1
        kwargs.setdefault(
            "http_async_client",
            httpx.AsyncClient(http1=True, http2=False),
        )
        kwargs.setdefault("http_client", httpx.Client(http1=True, http2=False))
        super().__init__(**kwargs)
