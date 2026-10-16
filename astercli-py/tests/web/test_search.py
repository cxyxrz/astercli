"""搜索 provider 测试：解析、自动选择优先级、结果格式化（MockTransport，零网络）。"""

from __future__ import annotations

import httpx
import pytest

from astercli.config import Settings
from astercli.web.search import (
    SearxngSearchProvider,
    SerpApiSearchProvider,
    ZhipuSearchProvider,
    _normalize_engine,
    create_search_provider,
    format_search_results,
    pick_provider_name,
)


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def zhipu_response():
    return httpx.Response(
        200,
        json={
            "search_result": [
                {"title": "Java 21 新特性", "link": "https://example.com/java21", "content": "虚拟线程正式转正"},
                {"title": "  ", "link": "", "content": "  "},  # 全空条目应被跳过
                {"title": "Spring Boot 3.3", "link": "https://example.com/sb", "content": "Release notes"},
            ]
        },
    )


class TestZhipu:
    async def test_未配置_key_不可用(self):
        provider = ZhipuSearchProvider(None)
        assert provider.is_ready() is False
        assert "GLM_API_KEY" in provider.unavailable_hint()

    async def test_搜索解析与裁剪(self):
        provider = ZhipuSearchProvider("test-key", http_client=mock_client(lambda req: zhipu_response()))
        results = await provider.search("Java 新特性", 1)
        assert len(results) == 1  # top_k=1 截断
        assert results[0].title == "Java 21 新特性"
        assert results[0].url == "https://example.com/java21"

    async def test_401_报_key_无效(self):
        provider = ZhipuSearchProvider(
            "bad-key", http_client=mock_client(lambda req: httpx.Response(401, text="unauthorized"))
        )
        with pytest.raises(RuntimeError, match="无效或已过期"):
            await provider.search("q", 5)

    async def test_引擎归一(self):
        assert _normalize_engine(None) == "search_std"
        assert _normalize_engine("search_pro") == "search_pro"
        assert _normalize_engine("baidu") == "search_std"  # 未知引擎回退默认


class TestSerpApi:
    async def test_未配置_key_不可用(self):
        provider = SerpApiSearchProvider(None)
        assert provider.is_ready() is False

    async def test_organic_解析与_answerbox_兜底(self):
        def handler(req):
            url = str(req.url)
            if "serpapi.com" in url:
                return httpx.Response(200, json={"organic_results": [
                    {"title": "Result A", "link": "https://a.com", "snippet": "snippet a"},
                ]})
            return httpx.Response(500)

        provider = SerpApiSearchProvider("k", http_client=mock_client(handler))
        results = await provider.search("q", 5)
        assert results[0].title == "Result A"

        def empty_handler(req):
            return httpx.Response(200, json={"answer_box": {"snippet": "直接答案"}})

        provider2 = SerpApiSearchProvider("k", http_client=mock_client(empty_handler))
        results2 = await provider2.search("q", 5)
        assert results2[0].title == "Google 精选摘要"


class TestSearxng:
    async def test_未配置_url_不可用(self):
        provider = SearxngSearchProvider(None)
        assert provider.is_ready() is False
        assert "SEARXNG_URL" in provider.unavailable_hint()

    async def test_json_解析(self):
        def handler(req):
            assert "format=json" in str(req.url)
            return httpx.Response(200, json={"results": [
                {"title": "Searx 结果", "url": "https://s.com", "content": "内容"},
            ]})

        provider = SearxngSearchProvider("http://localhost:8888", http_client=mock_client(handler))
        results = await provider.search("q", 5)
        assert results[0].title == "Searx 结果"


class TestPickProvider:
    def test_显式配置最优先(self):
        assert pick_provider_name("searxng", "glm-key", "serp-key", "http://x") == "searxng"
        assert pick_provider_name("SERPAPI", "glm-key", None, None) == "serpapi"

    def test_有_glm_key_选_zhipu(self):
        assert pick_provider_name(None, "glm-key", "serp-key", "url") == "zhipu"

    def test_有_serpapi_选_serpapi(self):
        assert pick_provider_name(None, None, "serp-key", "url") == "serpapi"

    def test_有_searxng_选_searxng(self):
        assert pick_provider_name(None, None, None, "http://x:8888") == "searxng"

    def test_全空_默认占位_zhipu(self):
        assert pick_provider_name(None, None, None, None) == "zhipu"

    def test_工厂_按_settings_构造(self):
        settings = Settings(env={"SEARCH_PROVIDER": "serpapi", "SERPAPI_KEY": "k"}, load_files=False)
        assert isinstance(create_search_provider(settings), SerpApiSearchProvider)

        settings2 = Settings(env={"SEARXNG_URL": "http://localhost:8888"}, load_files=False)
        assert isinstance(create_search_provider(settings2), SearxngSearchProvider)

        settings3 = Settings(env={"GLM_API_KEY": "g"}, load_files=False)
        assert isinstance(create_search_provider(settings3), ZhipuSearchProvider)


class TestFormat:
    def test_空结果(self):
        out = format_search_results("zhipu", "query", [])
        assert "未找到相关结果" in out

    def test_结果_带序号_链接与截断(self):
        from astercli.web.search import SearchResult

        results = [
            SearchResult(position=1, title="标题", url="https://a.com", snippet="s" * 300),
        ]
        out = format_search_results("zhipu", "查询词", results)
        assert "1. 标题" in out
        assert "🔗 https://a.com" in out
        assert "..." in out  # 长摘要截断
        assert out.count("s" * 200) == 1
