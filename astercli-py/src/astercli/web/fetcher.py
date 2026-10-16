"""HTTP 抓取器（对齐 Java web.WebFetcher）：URL → 字节流（上限截断）→ 字符串。

边界（对齐 Java）：
- 5MB 响应体上限，流式读避免 OOM
- 30s 整体超时
- 4xx / 5xx 抛 RuntimeError，由调用方决定呈现方式
- 不处理 JS 渲染、登录态（Java 侧走 Chrome DevTools MCP，Python 版暂不迁移）

字符集：优先 Content-Type charset，失败回退 UTF-8；HTML meta 嗅探在
extractor 里做。Python 版用 httpx.AsyncClient 替代 OkHttp。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
FETCH_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (compatible; astercli-web-fetch/1.0)",
}


@dataclass(frozen=True)
class RawResponse:
    url: str
    body: str
    content_type: str
    charset: str
    truncated: bool


class WebFetcher:
    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES,
                 http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._max_bytes = max_bytes
        self._client = http_client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    async def fetch(self, url: str) -> RawResponse:
        chunks: list = []
        total = 0
        truncated = False
        async with self._client.stream("GET", url, headers=FETCH_HEADERS) as response:
            if response.status_code != 200:
                body = (await response.aread())[:200].decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code} {response.reason_phrase} {body}".strip())
            async for chunk in response.aiter_bytes():
                remaining = self._max_bytes - total
                if remaining <= 0:
                    truncated = True
                    break
                piece = chunk[:remaining]
                chunks.append(piece)
                total += len(piece)
                if total >= self._max_bytes:
                    truncated = True
                    break
        content_type = response.headers.get("Content-Type", "")
        charset = _charset_from_content_type(content_type) or "utf-8"
        text = b"".join(chunks).decode(charset, errors="replace")
        return RawResponse(
            url=url,
            body=text,
            content_type=content_type,
            charset=charset,
            truncated=truncated,
        )


def _charset_from_content_type(content_type: str) -> Optional[str]:
    """从 Content-Type 头解析 charset 参数（如 text/html; charset=gbk）。"""
    for part in content_type.split(";"):
        kv = part.strip().split("=", 1)
        if len(kv) == 2 and kv[0].lower() == "charset" and kv[1].strip():
            return kv[1].strip().strip("'\"")
    return None
