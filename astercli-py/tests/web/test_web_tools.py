"""web 工具注册表接线测试：web_search / web_fetch（零网络）。"""

from __future__ import annotations

import socket

import pytest

from astercli.config import Settings
from astercli.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    settings = Settings(env={}, load_files=False)
    registry = ToolRegistry(settings=settings)
    from astercli.web import register_web_tools

    register_web_tools(registry, settings)
    return registry


class TestWebSearchTool:
    async def test_未配置_key_返回提示_不抛错(self, registry):
        tool = registry.get("web_search")
        out = await tool.func({"query": "Java 新特性"})
        assert "⚠️" in out
        assert "GLM_API_KEY" in out

    async def test_空_query_拒绝(self, registry):
        tool = registry.get("web_search")
        assert "不能为空" in await tool.func({"query": ""})

    async def test_完整流程_mock_provider(self, registry, monkeypatch):
        from astercli.web.search import SearchResult

        class FakeProvider:
            def name(self):
                return "zhipu"

            def is_ready(self):
                return True

            def unavailable_hint(self):
                return ""

            async def search(self, query, top_k):
                return [SearchResult(position=1, title="结果", url="https://a.com", snippet="摘要")]

        import astercli.web as web_pkg

        # provider 惰性构建：首次调用时才查模块级 create_search_provider → patch 生效
        monkeypatch.setattr(web_pkg, "create_search_provider", lambda s: FakeProvider())
        tool = registry.get("web_search")
        out = await tool.func({"query": "测试", "top_k": 3})
        assert "🔍 [zhipu] 测试" in out
        assert "1. 结果" in out


class TestWebFetchTool:
    async def test_空_url_拒绝(self, registry):
        tool = registry.get("web_fetch")
        assert "不能为空" in await tool.func({"url": ""})

    async def test_非法_scheme_拒绝(self, registry):
        tool = registry.get("web_fetch")
        out = await tool.func({"url": "file:///etc/passwd"})
        assert "网络访问被拒绝" in out

    async def test_内网地址_被_网络策略拒绝(self, registry, monkeypatch):
        def fake_resolve(host, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 0))]

        monkeypatch.setattr("astercli.web.network_policy.socket.getaddrinfo", fake_resolve)
        tool = registry.get("web_fetch")
        out = await tool.func({"url": "http://intranet.local/x"})
        assert "网络访问被拒绝" in out
        assert "内网" in out

    async def test_正常抓取_转_markdown(self, registry, monkeypatch):
        import httpx

        def fake_resolve(host, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("astercli.web.network_policy.socket.getaddrinfo", fake_resolve)

        class FakeRaw:
            url = "https://example.com/doc"
            body = "<html><head><title>文档</title></head><body><main><p>正文内容</p></main></body></html>"
            content_type = "text/html"
            charset = "utf-8"
            truncated = False

        class FakeFetcher:
            async def fetch(self, url):
                return FakeRaw()

        # 替换模块级 fetcher 工厂惰性状态
        import astercli.web as web_pkg

        original_fetch = web_pkg.WebFetcher
        web_pkg.WebFetcher = FakeFetcher
        try:
            tool = registry.get("web_fetch")
            out = await tool.func({"url": "https://example.com/doc"})
        finally:
            web_pkg.WebFetcher = original_fetch
        assert "📄 标题: 文档" in out
        assert "正文内容" in out

    async def test_spa_空正文提示(self, registry, monkeypatch):
        def fake_resolve(host, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr("astercli.web.network_policy.socket.getaddrinfo", fake_resolve)

        class FakeRaw:
            url = "https://example.com/spa"
            body = "<html><body><div id=\"root\"></div></body></html>"
            content_type = "text/html"
            charset = "utf-8"
            truncated = False

        class FakeFetcher:
            async def fetch(self, url):
                return FakeRaw()

        import astercli.web as web_pkg

        original = web_pkg.WebFetcher
        web_pkg.WebFetcher = FakeFetcher
        try:
            tool = registry.get("web_fetch")
            out = await tool.func({"url": "https://example.com/spa"})
        finally:
            web_pkg.WebFetcher = original
        assert "无可用正文" in out
        assert "web_search" in out  # 引导换入口


class TestRegistration:
    def test_两个工具_只读_不审批(self, registry):
        assert registry.get("web_search") is not None
        assert registry.get("web_fetch") is not None
        assert registry.get("web_search").dangerous is False
        assert registry.get("web_fetch").dangerous is False

    def test_参数_required_正确(self, registry):
        assert "query" in registry.get("web_search").parameters.get("required", [])
        assert "url" in registry.get("web_fetch").parameters.get("required", [])
