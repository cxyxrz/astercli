"""MCP mention 测试：@server:uri 解析（引号跳过）与 resource 块展开。"""

from astercli.mcp.mention import expand_mentions, parse_at_mentions


class FakeManager:
    """mention 展开的 manager 替身：预设读取结果。"""

    def __init__(self, results=None):
        # results: {(server, uri): (ok, text, mime, error)}
        self.results = results or {}
        self.calls = []

    def read_resource_for_mention(self, server, uri):
        self.calls.append((server, uri))
        return self.results.get((server, uri), (False, "", "", "MCP server 不存在"))


class TestParse:
    def test_basic(self):
        assert parse_at_mentions("看 @docs:https://example.com/a.md 这份文档") == [
            ("docs", "https://example.com/a.md", "@docs:https://example.com/a.md")
        ]

    def test_multiple_mentions(self):
        mentions = parse_at_mentions("@a:https://x.io/1 和 @b:file:///tmp/2.txt")
        assert [(s, u) for s, u, _ in mentions] == [
            ("a", "https://x.io/1"),
            ("b", "file:///tmp/2.txt"),
        ]

    def test_skips_double_quoted(self):
        assert parse_at_mentions('引号内 "@docs:https://x.io/a" 不展开') == []

    def test_skips_single_quoted(self):
        assert parse_at_mentions("引号内 '@docs:https://x.io/a' 不展开") == []

    def test_quoted_outside_still_parsed(self):
        mentions = parse_at_mentions('"@a:https://x.io/1" 外面 @b:https://x.io/2')
        assert [(s, u) for s, u, _ in mentions] == [("b", "https://x.io/2")]

    def test_requires_scheme(self):
        # 无 scheme:// 的不视为 mention（避免误伤邮箱、本地 @path 引用）
        assert parse_at_mentions("@docs:plain/path") == []

    def test_email_not_matched(self):
        assert parse_at_mentions("联系 user@host:443 端口") == []

    def test_server_must_start_with_letter(self):
        assert parse_at_mentions("@1bad:https://x.io/a") == []

    def test_server_allows_digits_underscore_dash(self):
        assert parse_at_mentions("@step_search-1:https://x.io/a") != []


class TestExpand:
    def test_success_block_with_mime(self):
        manager = FakeManager({("docs", "https://e.com/a.md"): (True, "内容", "text/markdown", "")})
        out = expand_mentions("读 @docs:https://e.com/a.md", manager)
        assert '<resource server="docs" uri="https://e.com/a.md" mimeType="text/markdown">' in out
        assert "内容" in out
        assert "</resource>" in out

    def test_success_block_without_mime(self):
        manager = FakeManager({("docs", "https://e.com/a"): (True, "x", "", "")})
        out = expand_mentions("读 @docs:https://e.com/a", manager)
        assert "mimeType" not in out

    def test_error_block_on_failure(self):
        manager = FakeManager({("gone", "https://e.com/a"): (False, "", "", "server 未就绪")})
        out = expand_mentions("读 @gone:https://e.com/a", manager)
        assert '<resource_error server="gone" uri="https://e.com/a">server 未就绪</resource_error>' in out

    def test_missing_server_degrades_to_error(self):
        manager = FakeManager()
        out = expand_mentions("读 @nope:https://e.com/a", manager)
        assert "<resource_error" in out

    def test_quoted_not_expanded(self):
        manager = FakeManager()
        text = '引用 "@docs:https://e.com/a" 保持原样'
        assert expand_mentions(text, manager) == text
        assert manager.calls == []

    def test_plain_text_unchanged(self):
        manager = FakeManager()
        assert expand_mentions("没有 at 的普通文本", manager) == "没有 at 的普通文本"

    def test_multiple_mixed(self):
        manager = FakeManager(
            {
                ("a", "https://x.io/1"): (True, "AAA", "text/plain", ""),
                ("b", "https://x.io/2"): (False, "", "", "超时"),
            }
        )
        out = expand_mentions("x @a:https://x.io/1 y @b:https://x.io/2 z", manager)
        assert "<resource " in out and "<resource_error " in out
        assert "AAA" in out
