"""WebFetcher + HtmlExtractor 测试（MockTransport，零网络）。"""

from __future__ import annotations

import httpx
import pytest

from astercli.web.extractor import HtmlExtractor
from astercli.web.fetcher import WebFetcher, _charset_from_content_type


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


PAGE = """<!DOCTYPE html>
<html><head><title>示例页面</title></head>
<body>
<nav><a href="/nav">导航链接</a></nav>
<div class="ads">广告内容广告内容广告内容广告内容广告内容</div>
<div class="content-wrap">
  <h1>主标题</h1>
  <p>第一段正文，介绍 <a href="/docs">文档</a> 的入口。</p>
  <ul><li>要点一</li><li>要点二</li></ul>
  <pre><code>print("hello")</code></pre>
  <blockquote>引用内容</blockquote>
</div>
<footer>页脚页脚页脚页脚页脚页脚页脚页脚页脚页脚</footer>
</body></html>"""


class TestWebFetcher:
    async def test_200_正常抓取(self):
        fetcher = WebFetcher(http_client=mock_client(lambda req: httpx.Response(200, text=PAGE)))
        raw = await fetcher.fetch("https://example.com/page")
        assert "主标题" in raw.body
        assert raw.truncated is False
        assert raw.charset == "utf-8"

    async def test_非_200_抛错(self):
        fetcher = WebFetcher(http_client=mock_client(lambda req: httpx.Response(404, text="Not Found")))
        with pytest.raises(RuntimeError, match="HTTP 404"):
            await fetcher.fetch("https://example.com/missing")

    async def test_超限截断(self):
        fetcher = WebFetcher(max_bytes=1024, http_client=mock_client(lambda req: httpx.Response(200, text="x" * 2048)))
        raw = await fetcher.fetch("https://example.com/big")
        assert raw.truncated is True
        assert len(raw.body.encode()) <= 1024

    async def test_content_type_charset_优先(self):
        handler = lambda req: httpx.Response(  # noqa: E731
            200, headers={"Content-Type": "text/html; charset=gbk"},
            content="中文".encode("gbk"),
        )
        fetcher = WebFetcher(http_client=mock_client(handler))
        raw = await fetcher.fetch("https://example.com/gbk")
        assert raw.body == "中文"
        assert raw.charset == "gbk"

    def test_charset_解析(self):
        assert _charset_from_content_type("text/html; charset=utf-8") == "utf-8"
        assert _charset_from_content_type("text/html; charset=\"gbk\"") == "gbk"
        assert _charset_from_content_type("text/html") is None


class TestHtmlExtractor:
    def setup_method(self):
        self.extractor = HtmlExtractor()

    def test_标题_正文_链接_列表_代码_引用(self):
        out = self.extractor.extract(PAGE, "https://example.com/")
        assert out.title == "示例页面"
        assert "# 主标题" in out.markdown
        assert "第一段正文" in out.markdown
        assert "[文档](https://example.com/docs)" in out.markdown  # 相对链接补全
        assert "- 要点一" in out.markdown and "- 要点二" in out.markdown
        assert "```" in out.markdown and 'print("hello")' in out.markdown
        assert "> 引用内容" in out.markdown

    def test_噪声标签被清理(self):
        out = self.extractor.extract(PAGE)
        assert "广告内容" not in out.markdown
        assert "页脚" not in out.markdown
        assert "导航链接" not in out.markdown

    def test_spa_空正文(self):
        out = self.extractor.extract("<html><body><div id=\"root\"></div></body></html>")
        assert out.markdown == ""

    def test_语义容器优先(self):
        html = """<html><body>
        <div>外围噪声文本</div>
        <article><h2>文章标题</h2><p>{}</p></article>
        </body></html>""".format("正文" * 100)
        out = self.extractor.extract(html)
        assert "文章标题" in out.markdown
        assert "外围噪声文本" not in out.markdown

    def test_表格转_markdown(self):
        html = """<html><body><table>
        <tr><th>名称</th><th>值</th></tr>
        <tr><td>a|b</td><td>1</td></tr>
        </table></body></html>"""
        out = self.extractor.extract(html)
        assert "| 名称 | 值 |" in out.markdown
        assert "| a\\|b | 1 |" in out.markdown  # 竖线转义
        assert "| --- | --- |" in out.markdown  # 表头分隔行

    def test_链接密度惩罚导航页(self):
        # 链接占比极高的列表页不应被选为主容器
        html = "<html><body><div>" + "".join(
            f'<a href="/l{i}">链接{i}</a>' for i in range(50)
        ) + '</div><div>' + "纯正文" * 200 + "</div></body></html>"
        out = self.extractor.extract(html)
        assert "纯正文" in out.markdown
        assert out.markdown.count("链接") < 5
