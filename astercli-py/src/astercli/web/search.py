"""搜索 provider 抽象与三个实现（对齐 Java web.SearchProvider 家族）。

- ZhipuSearchProvider：智谱 Web Search，与 GLM 推理共用 GLM_API_KEY，国内首选
- SerpApiSearchProvider：商业聚合 API，需 SERPAPI_KEY
- SearxngSearchProvider：开源元搜索引擎，需自托管实例（SEARXNG_URL）

选择优先级（对齐 Java SearchProviderFactory.pickProvider）：
显式 SEARCH_PROVIDER > 有 GLM_API_KEY → zhipu > 有 SERPAPI_KEY → serpapi
> 有 SEARXNG_URL → searxng > 默认占位 zhipu（is_ready() 为 False）。

Java 版由 ToolRegistry 在 step-3.7-flash 模型下转调 StepSearch MCP 的
偏好逻辑不在 Python 版实现（已知差异，配置了搜索 Key 时直接走原生 provider）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import httpx

from ..config import Settings

ZHIPU_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/web_search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
ZHIPU_ALLOWED_ENGINES = ("search_std", "search_pro", "search_pro_sogou", "search_pro_quark")
ZHIPU_DEFAULT_ENGINE = "search_std"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class SearchResult:
    position: int
    title: str
    url: str = ""
    snippet: str = ""
    source: str = ""


class SearchProvider(ABC):
    """搜索引擎抽象：新增 Brave / Tavily 等实现时只需继续实现本接口。"""

    @abstractmethod
    def name(self) -> str:
        """provider 名称（如 "zhipu"），用于错误信息和日志。"""

    @abstractmethod
    def is_ready(self) -> bool:
        """是否可用（必要 API Key 已配置 / 服务地址合法）。"""

    @abstractmethod
    def unavailable_hint(self) -> str:
        """is_ready() 为 False 时给用户的配置提示。"""

    @abstractmethod
    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        """执行搜索；实现可酌情截断结果数量。"""


def _normalize_engine(raw: Optional[str]) -> str:
    if not raw or not raw.strip():
        return ZHIPU_DEFAULT_ENGINE
    trimmed = raw.strip()
    return trimmed if trimmed in ZHIPU_ALLOWED_ENGINES else ZHIPU_DEFAULT_ENGINE


class ZhipuSearchProvider(SearchProvider):
    def __init__(self, api_key: Optional[str], search_engine: Optional[str] = None,
                 http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._api_key = (api_key or "").strip()
        self._engine = _normalize_engine(search_engine)
        self._client = http_client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    def name(self) -> str:
        return "zhipu"

    def is_ready(self) -> bool:
        return bool(self._api_key)

    def unavailable_hint(self) -> str:
        return (
            "智谱 Web Search 未配置 API Key。请在 .env 中设置 GLM_API_KEY（与 GLM 推理共用），"
            "或显式 SEARCH_PROVIDER=zhipu。"
        )

    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self.is_ready():
            raise RuntimeError(self.unavailable_hint())
        count = min(top_k, 50) if top_k > 0 else 10
        payload = {
            "search_engine": self._engine,
            "search_query": query,
            "count": count,
            "content_size": "medium",
        }
        response = await self._client.post(
            ZHIPU_ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code == 401:
            raise RuntimeError("智谱 API Key 无效或已过期")
        if response.status_code != 200:
            raise RuntimeError(
                f"智谱搜索请求失败 (HTTP {response.status_code}): {response.text[:200]}"
            )
        return _parse_zhipu(response.json(), count)


def _parse_zhipu(data: dict, max_results: int) -> List[SearchResult]:
    out: List[SearchResult] = []
    for node in data.get("search_result") or []:
        if len(out) >= max_results:
            break
        title = str(node.get("title") or "")
        link = str(node.get("link") or "")
        content = str(node.get("content") or "")
        if not title.strip() and not content.strip():
            continue
        out.append(SearchResult(position=len(out) + 1, title=title, url=link, snippet=content))
    return out


class SerpApiSearchProvider(SearchProvider):
    def __init__(self, api_key: Optional[str], http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._api_key = (api_key or "").strip()
        self._client = http_client or httpx.AsyncClient(timeout=15.0)

    def name(self) -> str:
        return "serpapi"

    def is_ready(self) -> bool:
        return bool(self._api_key)

    def unavailable_hint(self) -> str:
        return (
            "Web 搜索未配置 SerpAPI Key。请在 .env 中设置 SERPAPI_KEY，"
            "或 SEARCH_PROVIDER=searxng 切换到自托管 SearXNG。"
            "申请 SerpAPI Key：https://serpapi.com/manage-api-key"
        )

    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self.is_ready():
            raise RuntimeError(self.unavailable_hint())
        max_results = min(top_k, 10) if top_k > 0 else DEFAULT_TOP_K
        response = await self._client.get(
            SERPAPI_ENDPOINT,
            params={
                "q": query,
                "api_key": self._api_key,
                "num": str(max_results),
                "hl": "zh-cn",
            },
        )
        if response.status_code == 401:
            raise RuntimeError("SerpAPI Key 无效或已过期")
        if response.status_code != 200:
            raise RuntimeError(f"SerpAPI 请求失败 (HTTP {response.status_code})")
        return _parse_serpapi(response.json(), max_results)


def _parse_serpapi(data: dict, max_results: int) -> List[SearchResult]:
    out: List[SearchResult] = []
    for node in data.get("organic_results") or []:
        if len(out) >= max_results:
            break
        title = str(node.get("title") or "")
        link = str(node.get("link") or "")
        snippet = str(node.get("snippet") or "")
        if not title.strip() and not snippet.strip():
            continue
        out.append(SearchResult(position=len(out) + 1, title=title, url=link, snippet=snippet))
    if not out:
        # 没有 organic 结果时，尝试 answer_box 兜底（对齐 Java）
        box = data.get("answer_box") or {}
        answer = str(box.get("snippet") or box.get("answer") or "")
        if answer.strip():
            out.append(SearchResult(position=1, title="Google 精选摘要", snippet=answer))
    return out


class SearxngSearchProvider(SearchProvider):
    def __init__(self, base_url: Optional[str], http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=15.0)

    def name(self) -> str:
        return "searxng"

    def is_ready(self) -> bool:
        return bool(self._base_url) and self._base_url.startswith(("http://", "https://"))

    def unavailable_hint(self) -> str:
        return (
            "SearXNG 未配置或地址非法。请在 .env 中设置 SEARXNG_URL=http://localhost:8888，"
            "并确保实例已启动。"
        )

    async def search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self.is_ready():
            raise RuntimeError(self.unavailable_hint())
        max_results = min(top_k, 10) if top_k > 0 else DEFAULT_TOP_K
        response = await self._client.get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json", "language": "zh"},
            headers={"Accept": "application/json", "User-Agent": "astercli-web-search/1.0"},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"SearXNG 请求失败 (HTTP {response.status_code})。"
                "某些公共实例禁用了 JSON API，建议自托管 docker 实例。"
            )
        return _parse_searxng(response.json(), max_results)


def _parse_searxng(data: dict, max_results: int) -> List[SearchResult]:
    out: List[SearchResult] = []
    for node in data.get("results") or []:
        if len(out) >= max_results:
            break
        title = str(node.get("title") or "")
        link = str(node.get("url") or "")
        snippet = str(node.get("content") or "")
        if not title.strip() and not snippet.strip():
            continue
        out.append(SearchResult(position=len(out) + 1, title=title, url=link, snippet=snippet))
    return out


def pick_provider_name(
    explicit: Optional[str], glm_key: Optional[str], serp_key: Optional[str], searxng_url: Optional[str]
) -> str:
    """自动选择逻辑（对齐 Java pickProvider），便于单测锁死优先级。"""
    if explicit and explicit.strip():
        return explicit.strip().lower()
    if glm_key and glm_key.strip():
        return "zhipu"
    if serp_key and serp_key.strip():
        return "serpapi"
    if searxng_url and searxng_url.strip():
        return "searxng"
    return "zhipu"  # 默认占位（对齐 Java），is_ready() 会为 False


def create_search_provider(settings: Settings) -> SearchProvider:
    explicit = settings.get("SEARCH_PROVIDER")
    glm_key = settings.get("GLM_API_KEY")
    serp_key = settings.get("SERPAPI_KEY")
    searxng_url = settings.get("SEARXNG_URL")
    chosen = pick_provider_name(explicit, glm_key, serp_key, searxng_url)
    if chosen == "serpapi":
        return SerpApiSearchProvider(serp_key)
    if chosen == "searxng":
        return SearxngSearchProvider(searxng_url)
    return ZhipuSearchProvider(glm_key, settings.get("ZHIPU_SEARCH_ENGINE"))


def format_search_results(provider_name: str, query: str, results: List[SearchResult]) -> str:
    """搜索结果格式化（对齐 Java ToolRegistry.formatSearchResults）。"""
    if not results:
        return f"🔍 [{provider_name}] {query}\n\n未找到相关结果。"
    lines = [f"🔍 [{provider_name}] {query}", ""]
    for r in results:
        lines.append(f"{r.position}. {r.title}")
        if r.snippet.strip():
            snippet = r.snippet if len(r.snippet) <= 200 else r.snippet[:200] + "..."
            lines.append(f"   {snippet}")
        if r.url.strip():
            line = f"   🔗 {r.url}"
            if r.source.strip():
                line += f"  ({r.source})"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip()
