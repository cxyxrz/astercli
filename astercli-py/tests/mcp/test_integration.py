"""MCP 端到端集成测试：真实 MCPServer stdio 子进程 + 真实 McpRuntime 后台 loop。

验收对齐 P5 计划：配一个真实 MCP server 并成功调用（工具调用 / resources /
mention / HITL 审批全链路）。
"""

import asyncio
import json
import sys

from astercli.config import Settings
from astercli.hitl import ApprovalSession
from astercli.mcp import McpServerManager
from astercli.tools import ToolRegistry

SERVER_SCRIPT = '''
"""最小测试 MCP server：一个工具 + 一个 resource。"""
from mcp.server.mcpserver import MCPServer

server = MCPServer("test-server")


@server.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b


@server.resource("config://app")
def app_config() -> str:
    return "hello from resource"


if __name__ == "__main__":
    server.run("stdio")
'''


def _setup(tmp_path, monkeypatch, *, approval=True):
    script = tmp_path / "serve.py"
    script.write_text(SERVER_SCRIPT, encoding="utf-8")
    (tmp_path / ".astercli").mkdir()
    (tmp_path / ".astercli" / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"test": {"command": sys.executable, "args": [str(script)]}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "astercli.mcp.config.user_config_path", lambda: tmp_path / "home" / "mcp.json"
    )
    monkeypatch.setenv("ASTERCLI_AUDIT_DIR", str(tmp_path / "audit"))
    settings = Settings(env={}, load_files=False)
    registry = ToolRegistry(settings=settings, approval=ApprovalSession() if approval else None)
    manager = McpServerManager(registry, settings, root=tmp_path)
    return manager, registry


def test_stdio_server_end_to_end(tmp_path, monkeypatch):
    """P5 验收主链路：startup → 工具注册 → HITL 审批放行 → 调用成功 → mention 读取。"""
    manager, registry = _setup(tmp_path, monkeypatch)
    try:
        manager.startup()
        assert manager._servers["test"].status == "READY", manager._servers["test"].error

        # 动态工具 + resources 虚拟工具已注册，且标记 dangerous（进 HITL）
        add_tool = registry.get("mcp__test__add")
        assert add_tool is not None and add_tool.dangerous
        assert registry.get("mcp__test__list_resources") is not None
        assert registry.get("mcp__test__read_resource") is not None

        # 审批放行后真实调用
        calls = [{"name": "mcp__test__add", "args": {"a": 3, "b": 5}, "id": "c1"}]
        messages = asyncio.run(registry.execute_all(calls, {"c1": {"action": "approve"}}))
        assert "8" in messages[0].content, messages[0].content

        # 拒绝路径：决策 reject 回灌拒绝文本
        messages = asyncio.run(registry.execute_all(calls, {"c1": {"action": "reject"}}))
        assert "[HITL]" in messages[0].content

        # 虚拟工具：list + read
        listing = asyncio.run(manager.list_resources_async("test"))
        assert "config://app" in listing
        content = asyncio.run(manager.read_resource_async("test", "config://app"))
        assert content == "hello from resource"

        # mention 同步读取 + 审计
        ok, text, mime, error = manager.read_resource_for_mention("test", "config://app")
        assert ok and text == "hello from resource", error

        # mention 展开为 <resource> 块
        from astercli.mcp import expand_mentions

        out = expand_mentions("参考 @test:config://app 说明", manager)
        assert '<resource server="test" uri="config://app"' in out
        assert "hello from resource" in out
    finally:
        manager.shutdown()


def test_stdio_server_without_hitl_direct_call(tmp_path, monkeypatch):
    """approval=None（无 checkpointer 场景同义）时 MCP 工具直接执行，不需要决策。"""
    manager, registry = _setup(tmp_path, monkeypatch, approval=False)
    try:
        manager.startup()
        assert manager._servers["test"].status == "READY"
        calls = [{"name": "mcp__test__add", "args": {"a": 2, "b": 4}, "id": "c1"}]
        messages = asyncio.run(registry.execute_all(calls, None))
        assert "6" in messages[0].content
    finally:
        manager.shutdown()
