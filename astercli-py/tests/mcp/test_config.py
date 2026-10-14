"""MCP 配置层测试：合并优先级 / ${VAR} 展开 / transport 校验 / 损坏容错。"""

import json

import pytest

from astercli.config import Settings
from astercli.mcp import McpConfigError
from astercli.mcp.config import load_merged_config, prepare_server


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """把用户级配置路径指到临时目录，避免读到真实 ~/.astercli-py。"""
    user_path = tmp_path / "home" / ".astercli-py" / "mcp.json"
    monkeypatch.setattr("astercli.mcp.config.user_config_path", lambda: user_path)
    return user_path


def make_settings(env=None):
    return Settings(env=env or {}, load_files=False)


class TestLoadMergedConfig:
    def test_no_config_files(self, isolated_home, tmp_path):
        assert load_merged_config(make_settings(), tmp_path) == {}

    def test_user_and_project_merge(self, isolated_home, tmp_path):
        write_json(isolated_home, {"mcpServers": {"user-server": {"command": "u"}}})
        write_json(tmp_path / ".astercli" / "mcp.json", {"mcpServers": {"proj-server": {"command": "p"}}})
        configs = load_merged_config(make_settings(), tmp_path)
        assert set(configs) == {"user-server", "proj-server"}

    def test_project_overrides_same_name(self, isolated_home, tmp_path):
        write_json(isolated_home, {"mcpServers": {"dup": {"command": "user-cmd"}}})
        write_json(tmp_path / ".astercli" / "mcp.json", {"mcpServers": {"dup": {"command": "proj-cmd"}}})
        configs = load_merged_config(make_settings(), tmp_path)
        assert configs["dup"].command == "proj-cmd"

    def test_broken_json_raises(self, isolated_home, tmp_path):
        path = tmp_path / ".astercli" / "mcp.json"
        path.parent.mkdir(parents=True)
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(McpConfigError, match="配置文件"):
            load_merged_config(make_settings(), tmp_path)

    def test_top_level_must_be_object(self, isolated_home, tmp_path):
        write_json(tmp_path / ".astercli" / "mcp.json", ["not", "object"])
        with pytest.raises(McpConfigError, match="顶层必须是对象"):
            load_merged_config(make_settings(), tmp_path)

    def test_mcp_servers_must_be_object(self, isolated_home, tmp_path):
        write_json(tmp_path / ".astercli" / "mcp.json", {"mcpServers": ["list"]})
        with pytest.raises(McpConfigError, match="必须是对象"):
            load_merged_config(make_settings(), tmp_path)

    def test_server_entry_must_be_object(self, isolated_home, tmp_path):
        write_json(tmp_path / ".astercli" / "mcp.json", {"mcpServers": {"bad": "string"}})
        with pytest.raises(McpConfigError, match="条目必须是对象"):
            load_merged_config(make_settings(), tmp_path)

    def test_disabled_preserved(self, isolated_home, tmp_path):
        write_json(
            tmp_path / ".astercli" / "mcp.json",
            {"mcpServers": {"off": {"command": "x", "disabled": True}}},
        )
        configs = load_merged_config(make_settings(), tmp_path)
        assert configs["off"].disabled is True


class TestPrepareServer:
    def test_stdio_basic(self):
        cfg = prepare_server("s", {"command": "npx", "args": ["-y", "pkg"], "env": {"A": "1"}}, make_settings())
        assert cfg.transport == "stdio"
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "pkg"]
        assert cfg.env == {"A": "1"}

    def test_http_from_url(self):
        cfg = prepare_server("s", {"url": "https://example.com/mcp"}, make_settings())
        assert cfg.transport == "http"
        assert cfg.is_remote

    def test_sse_type(self):
        cfg = prepare_server("s", {"url": "https://e.com/sse", "type": "sse"}, make_settings())
        assert cfg.transport == "sse"

    def test_stdio_type_explicit_ok(self):
        cfg = prepare_server("s", {"command": "x", "type": "stdio"}, make_settings())
        assert cfg.transport == "stdio"

    def test_type_conflict_with_command(self):
        with pytest.raises(McpConfigError, match="type 只能是 stdio"):
            prepare_server("s", {"command": "x", "type": "http"}, make_settings())

    def test_unknown_type(self):
        with pytest.raises(McpConfigError, match="未知 type"):
            prepare_server("s", {"url": "https://e.com", "type": "grpc"}, make_settings())

    def test_command_and_url_conflict(self):
        with pytest.raises(McpConfigError, match="互斥"):
            prepare_server("s", {"command": "x", "url": "https://e.com"}, make_settings())

    def test_missing_both(self):
        with pytest.raises(McpConfigError, match="必须设置 command"):
            prepare_server("s", {"args": ["x"]}, make_settings())

    def test_args_must_be_list(self):
        with pytest.raises(McpConfigError, match="args 必须是数组"):
            prepare_server("s", {"command": "x", "args": "not-list"}, make_settings())

    def test_env_must_be_object(self):
        with pytest.raises(McpConfigError, match="env 必须是对象"):
            prepare_server("s", {"command": "x", "env": ["a"]}, make_settings())


class TestVarExpansion:
    def test_expand_from_injected_env(self):
        settings = make_settings({"MY_CMD": "uvx"})
        cfg = prepare_server("s", {"command": "${MY_CMD}", "args": ["${MY_CMD}-arg"]}, settings)
        assert cfg.command == "uvx"
        assert cfg.args == ["uvx-arg"]

    def test_expand_url_headers_env_values(self):
        settings = make_settings({"TOKEN": "t1", "BASE": "https://e.com"})
        cfg = prepare_server(
            "s",
            {
                "url": "${BASE}/mcp",
                "headers": {"Authorization": "Bearer ${TOKEN}"},
                "env": {"K": "${TOKEN}"},
                "type": "http",
            },
            settings,
        )
        assert cfg.url == "https://e.com/mcp"
        assert cfg.headers == {"Authorization": "Bearer t1"}
        assert cfg.env == {"K": "t1"}

    def test_undefined_var_raises_with_name(self):
        with pytest.raises(McpConfigError, match="未设置的环境变量 \$\\{MISSING_KEY\}"):
            prepare_server("s", {"command": "${MISSING_KEY}"}, make_settings())

    def test_partial_expansion_keeps_literals(self):
        cfg = prepare_server("s", {"command": "plain-cmd"}, make_settings())
        assert cfg.command == "plain-cmd"

    def test_server_name_in_error(self):
        with pytest.raises(McpConfigError, match="server 'my-server'"):
            prepare_server("my-server", {"command": "${NOPE}"}, make_settings())
