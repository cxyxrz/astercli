"""web 联网工具（P7）：web_search 多 provider 搜索 + web_fetch 抓取转 Markdown。

对齐 Java ToolRegistry.registerWebTools 的两个工具：
- web_search(query, top_k=5)：SearchProvider 抽象（zhipu / serpapi / searxng）
- web_fetch(url, max_chars=8000)：NetworkPolicy 围栏 → WebFetcher → HtmlExtractor

已知差异：Java 在 step-3.7-flash 模型下会优先转调 StepSearch MCP，Python 版
不实现该偏好（配置了搜索 Key 时直接走原生 provider）。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..config import Settings
from ..tools.registry import Param, ToolRegistry, create_parameters
from .extractor import Extracted, HtmlExtractor
from .fetcher import RawResponse, WebFetcher
from .network_policy import NetworkPolicy
from .search import (
    SearchProvider,
    SearchResult,
    SerpApiSearchProvider,
    SearxngSearchProvider,
    ZhipuSearchProvider,
    create_search_provider,
    format_search_results,
)

DEFAULT_FETCH_MAX_CHARS = 8000

__all__ = [
    "Extracted",
    "HtmlExtractor",
    "NetworkPolicy",
    "RawResponse",
    "SearchProvider",
    "SearchResult",
    "SerpApiSearchProvider",
    "SearxngSearchProvider",
    "ZhipuSearchProvider",
    "WebFetcher",
    "create_search_provider",
    "format_search_results",
    "register_web_tools",
]


def _parse_int(raw, default: int) -> int:
    try:
        value = int(raw)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def register_web_tools(registry: ToolRegistry, settings: Settings) -> None:
    """注册 web_search / web_fetch（均只读，不走 HITL 审批）。"""

    state = {"provider": None, "fetcher": None, "extractor": None, "policy": None}

    def provider() -> SearchProvider:
        if state["provider"] is None:
            state["provider"] = create_search_provider(settings)
        return state["provider"]

    def fetcher() -> WebFetcher:
        if state["fetcher"] is None:
            state["fetcher"] = WebFetcher()
        return state["fetcher"]

    def extractor() -> HtmlExtractor:
        if state["extractor"] is None:
            state["extractor"] = HtmlExtractor()
        return state["extractor"]

    def policy() -> NetworkPolicy:
        if state["policy"] is None:
            state["policy"] = NetworkPolicy()
        return state["policy"]

    async def web_search(args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "搜索关键词不能为空"
        top_k = _parse_int(args.get("top_k"), 5)
        search = provider()
        if not search.is_ready():
            return "⚠️ " + search.unavailable_hint()
        try:
            results = await search.search(query, top_k)
        except Exception as e:  # noqa: BLE001 —— 对齐 Java：失败转文本回给 LLM
            return f"搜索失败 ({search.name()}): {e}"
        return format_search_results(search.name(), query, results)

    async def web_fetch(args: dict) -> str:
        url = str(args.get("url") or "").strip()
        if not url:
            return "URL 不能为空"
        max_chars = _parse_int(args.get("max_chars"), DEFAULT_FETCH_MAX_CHARS)
        guard = policy()
        # DNS 解析是阻塞调用，放线程池避免卡住事件循环
        deny = await asyncio.to_thread(guard.check_url, url)
        if deny:
            return f"❌ 网络访问被拒绝: {deny}"
        rate = guard.acquire()
        if rate:
            return f"❌ {rate}"
        try:
            raw: RawResponse = await fetcher().fetch(url)
            extracted = extractor().extract(raw.body, raw.url)
        except Exception as e:  # noqa: BLE001
            return f"抓取失败: {e}"
        return _format_fetch_result(raw.url, extracted, max_chars)

    registry.register(
        "web_search",
        "搜索互联网，获取实时信息（最新版本、官方文档、技术资讯等）。"
        "支持 zhipu（GLM Key 共用）、serpapi、searxng 三种 provider，"
        "由 SEARCH_PROVIDER 环境变量切换。",
        create_parameters(
            Param("query", "string", "搜索关键词，例如'Java 21 新特性'、'Spring Boot 3.3 release notes'", True),
            Param("top_k", "integer", "返回结果数量（默认5）", False),
        ),
        web_search,
    )
    registry.register(
        "web_fetch",
        "抓取指定 URL，提取正文转 Markdown。"
        "适用静态 / SSR 页面（博客、文档、官网）；JS 渲染或防爬站会返回空正文，本期不重试。",
        create_parameters(
            Param("url", "string", "完整 URL，需 http 或 https 协议", True),
            Param("max_chars", "integer", f"返回 Markdown 最大字符数（默认 {DEFAULT_FETCH_MAX_CHARS}，超出截断）", False),
        ),
        web_fetch,
    )


def _format_fetch_result(url: str, extracted: Extracted, max_chars: int) -> str:
    """对齐 Java ToolRegistry.formatFetchResult 的输出格式。"""
    lines = [f"🌐 抓取: {url}"]
    if extracted.title:
        lines.append(f"📄 标题: {extracted.title}")
    markdown = extracted.markdown
    if not markdown.strip():
        lines.append("")
        lines.append("⚠️ 页面无可用正文（可能是 JS 渲染的 SPA 或防爬站点），建议改用 web_search 换个入口。")
        return "\n".join(lines)
    original_length = len(markdown)
    truncated = False
    if max_chars > 0 and original_length > max_chars:
        markdown = markdown[:max_chars]
        truncated = True
    lines.append(f"📏 正文 {original_length} 字符" + ("（已截断）" if truncated else ""))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(markdown)
    return "\n".join(lines)
