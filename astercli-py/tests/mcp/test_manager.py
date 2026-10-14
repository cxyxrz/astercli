"""McpServerManager 单测：注册 / 状态机 / HITL 联动 / mention / replace 语义。

用 FakeRuntime + FakeGroup + FakeSession 走真实 _connect 注册路径
（不依赖真实 server 子进程；端到端见 test_integration.py）。
"""

import asyncio
import concurrent.futures
import json
from types import SimpleNamespace

import pytest

from astercli.cli.renderer import NullRenderer
from astercli.config import Settings
from astercli.hitl import ApprovalSession
from astercli.mcp import McpServerManager, namespaced_tool
from astercli.mcp.manager import (
    format_resource_list,
    format_tool_result,
    extract_resource_text,
)
from astercli.tools import ToolRegistry


def make_settings(env=None):
    return Settings(env=env or {}, load_files=False)


def write_project_config(tmp_path, servers, monkeypatch=None):
    (tmp_path / ".astercli").mkdir(exist_ok=True)
    (tmp_path / ".astercli" / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}, ensure_ascii=False), encoding="utf-8"
    )


# ---------- Fake 层：手动驱动协程（无真 IO，单次 send 推进到底） ----------


class FakeGroup:
    def __init__(self, queue):
        self._queue = list(queue)

    async def connect_to_server(self, params):
        if not self._queue:
            raise RuntimeError("fake 队列为空")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeRuntime:
    """替身：submit 立即完成（手动驱动协程），无后台线程。"""

    def __init__(self, sessions):
        self.group = FakeGroup(sessions)
        self.closed = False

    def start(self, *, startup_timeout=10.0):
        return True

    def is_running(self):
        return not self.closed

    def shutdown(self, timeout=5.0):
        self.closed = True

    def submit(self, coro_fn):
        fut = concurrent.futures.Future()
        coro = coro_fn(self.group)
        try:
            coro.send(None)
        except StopIteration as stop:
            fut.set_result(stop.value)
        except BaseException as exc:  # noqa: BLE001 —— future 携带异常
            fut.set_exception(exc)
        else:
            fut.set_exception(RuntimeError("fake 协程意外挂起（存在真 IO）"))
        finally:
            coro.close()
        return fut

    def submit_lifecycle(self, coro_fn):
        # fake：无 anyio scope，直接同 submit 立即完成
        return self.submit(coro_fn)

    async def call_async(self, coro_fn, timeout=None):
        return self.submit(coro_fn).result(timeout)

    def call_sync(self, coro_fn, timeout=None):
        return self.submit(coro_fn).result(timeout)


class PendingRuntime:
    """submit 返回永不完成的 future：测 startup 超时保留 STARTING。"""

    def start(self, *, startup_timeout=10.0):
        return True

    def is_running(self):
        return True

    def shutdown(self, timeout=5.0):
        pass

    def submit(self, coro_fn):
        return concurrent.futures.Future()

    def submit_lifecycle(self, coro_fn):
        return concurrent.futures.Future()


class FakeSession:
    def __init__(self, *, tool_text="8", tool_error=False, resources_supported=True):
        self.tool_text = tool_text
        self.tool_error = tool_error
        self.resources_supported = resources_supported

    async def list_tools(self, params=None):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="add",
                    description="两数相加",
                    inputSchema={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                )
            ]
        )

    async def list_resources(self, params=None):
        if not self.resources_supported:
            raise RuntimeError("server 不支持 resources")
        return SimpleNamespace(
            resources=[
                SimpleNamespace(
                    uri="config://app", name="app-config", mimeType="text/plain", description="应用配置"
                )
            ]
        )

    async def read_resource(self, uri, **kwargs):
        return SimpleNamespace(
            contents=[SimpleNamespace(text="hello resource", mimeType="text/plain")]
        )

    async def call_tool(self, name, arguments=None, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.tool_text)],
            isError=self.tool_error,
        )


def make_manager(tmp_path, monkeypatch, sessions, *, approval=None, env=None):
    """构建 manager 并注入 FakeRuntime；写好最小项目级配置。"""
    monkeypatch.setattr(
        "astercli.mcp.config.user_config_path",
        lambda: tmp_path / "home" / "mcp.json",
    )
    write_project_config(tmp_path, {"srv": {"command": "fake-cmd"}})
    settings = make_settings(env)
    registry = ToolRegistry(settings=settings, approval=approval)
    manager = McpServerManager(registry, settings, root=tmp_path, renderer=NullRenderer())
    manager._runtime = FakeRuntime(sessions)
    return manager, registry


# ---------- 注册与状态机 ----------


class TestStartupAndRegister:
    def test_ready_and_tool_registered(self, tmp_path, monkeypatch):
        manager, registry = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        state = manager._servers["srv"]
        assert state.status == "READY"
        tool = registry.get("mcp__srv__add")
        assert tool is not None
        assert tool.dangerous is True
        assert tool.description.startswith("[MCP:srv]")
        assert "两数相加" in tool.description
        assert tool.parameters["required"] == ["a", "b"]

    def test_resources_virtual_tools_registered(self, tmp_path, monkeypatch):
        manager, registry = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        assert registry.get("mcp__srv__list_resources") is not None
        read_tool = registry.get("mcp__srv__read_resource")
        assert read_tool is not None
        assert read_tool.parameters["required"] == ["uri"]

    def test_no_resources_capability_no_virtual_tools(self, tmp_path, monkeypatch):
        manager, registry = make_manager(
            tmp_path, monkeypatch, [FakeSession(resources_supported=False)]
        )
        manager.startup()
        assert registry.get("mcp__srv__list_resources") is None
        assert registry.get("mcp__srv__read_resource") is None

    def test_connect_failure_marks_error(self, tmp_path, monkeypatch):
        manager, _ = make_manager(tmp_path, monkeypatch, [RuntimeError("connection refused")])
        manager.startup()
        state = manager._servers["srv"]
        assert state.status == "ERROR"
        assert "connection refused" in state.error

    def test_startup_timeout_keeps_starting(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "astercli.mcp.config.user_config_path",
            lambda: tmp_path / "home" / "mcp.json",
        )
        write_project_config(tmp_path, {"srv": {"command": "x"}})
        settings = make_settings({"ASTERCLI_MCP_STARTUP_WAIT_SECONDS": "1"})
        registry = ToolRegistry(settings=settings)
        manager = McpServerManager(registry, settings, root=tmp_path, renderer=NullRenderer())
        manager._runtime = PendingRuntime()
        manager.startup()
        assert manager._servers["srv"].status == "STARTING"

    def test_no_config_is_silent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "astercli.mcp.config.user_config_path",
            lambda: tmp_path / "home" / "mcp.json",
        )
        settings = make_settings()
        registry = ToolRegistry(settings=settings)
        manager = McpServerManager(registry, settings, root=tmp_path, renderer=NullRenderer())
        manager.startup()
        assert manager._servers == {}

    def test_disabled_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "astercli.mcp.config.user_config_path",
            lambda: tmp_path / "home" / "mcp.json",
        )
        write_project_config(tmp_path, {"srv": {"command": "x", "disabled": True}})
        settings = make_settings()
        registry = ToolRegistry(settings=settings)
        manager = McpServerManager(registry, settings, root=tmp_path, renderer=NullRenderer())
        manager._runtime = FakeRuntime([FakeSession()])
        manager.startup()
        assert manager._servers["srv"].status == "DISABLED"
        assert registry.get("mcp__srv__add") is None


class TestLifecycleCommands:
    def test_restart_replaces_tools(self, tmp_path, monkeypatch):
        manager, registry = make_manager(
            tmp_path, monkeypatch, [FakeSession(tool_text="1"), FakeSession(tool_text="2")]
        )
        manager.startup()
        assert registry.get("mcp__srv__add") is not None
        assert manager.restart("srv") is True
        # replace 语义：旧工具先注销再注册，注册名不变
        assert registry.get("mcp__srv__add") is not None
        assert manager.restart("nope") is False

    def test_disable_and_enable(self, tmp_path, monkeypatch):
        manager, registry = make_manager(
            tmp_path, monkeypatch, [FakeSession(), FakeSession()]
        )
        manager.startup()
        assert manager.disable("srv") is True
        assert registry.get("mcp__srv__add") is None
        assert manager._servers["srv"].status == "DISABLED"
        assert manager.enable("srv") is True
        # fake runtime 同步完成：enable 后直接 READY（真实 runtime 下为 STARTING，
        # 状态机契约由 test_startup_timeout_keeps_starting 覆盖）
        assert manager._servers["srv"].status == "READY"

    def test_unregister_server_removes_tools(self, tmp_path, monkeypatch):
        manager, registry = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        manager.unregister_server("srv")
        assert registry.get("mcp__srv__add") is None
        assert registry.get("mcp__srv__list_resources") is None

    def test_status_lines(self, tmp_path, monkeypatch):
        manager, _ = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        lines = manager.status_lines()
        assert any("srv [READY] stdio" in line for line in lines)


# ---------- 工具执行 + HITL 联动 ----------


class TestCallToolWithHitl:
    def test_approved_call_returns_text(self, tmp_path, monkeypatch):
        approval = ApprovalSession()
        manager, registry = make_manager(
            tmp_path, monkeypatch, [FakeSession(tool_text="8")], approval=approval
        )
        manager.startup()
        calls = [{"name": "mcp__srv__add", "args": {"a": 3, "b": 5}, "id": "c1"}]
        decisions = {"c1": {"action": "approve"}}
        messages = asyncio.run(registry.execute_all(calls, decisions))
        assert "8" in messages[0].content

    def test_rejected_by_decision(self, tmp_path, monkeypatch):
        approval = ApprovalSession()
        manager, registry = make_manager(tmp_path, monkeypatch, [FakeSession()], approval=approval)
        manager.startup()
        calls = [{"name": "mcp__srv__add", "args": {}, "id": "c1"}]
        messages = asyncio.run(registry.execute_all(calls, {"c1": {"action": "reject", "reason": "不想执行"}}))
        assert "[HITL] 操作已被拒绝" in messages[0].content

    def test_missing_decision_fail_safe(self, tmp_path, monkeypatch):
        approval = ApprovalSession()
        manager, registry = make_manager(tmp_path, monkeypatch, [FakeSession()], approval=approval)
        manager.startup()
        calls = [{"name": "mcp__srv__add", "args": {}, "id": "c1"}]
        messages = asyncio.run(registry.execute_all(calls, None))
        assert "[HITL] 未收到审批" in messages[0].content

    async def test_server_not_ready_hint(self, tmp_path, monkeypatch):
        manager, _ = make_manager(tmp_path, monkeypatch, [])
        result = await manager.call_tool_async("srv", "add", {})
        assert "未就绪" in result
        result = await manager.call_tool_async("nope", "add", {})
        assert "未就绪" in result

    async def test_read_resource_and_list(self, tmp_path, monkeypatch):
        manager, _ = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        text = await manager.read_resource_async("srv", "config://app")
        assert text == "hello resource"
        listing = await manager.list_resources_async("srv")
        assert "config://app" in listing


# ---------- mention 读取 + 审计 ----------


class TestMentionRead:
    def test_success_and_audit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASTERCLI_AUDIT_DIR", str(tmp_path / "audit"))
        manager, _ = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        ok, text, mime, error = manager.read_resource_for_mention("srv", "config://app")
        assert ok is True
        assert text == "hello resource"
        assert mime == "text/plain"
        entries = _read_audit(tmp_path)
        assert any(e["tool"] == "mcp__srv__mention" and e["allowed"] for e in entries)

    def test_not_ready_rejected_audit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASTERCLI_AUDIT_DIR", str(tmp_path / "audit"))
        manager, _ = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()
        ok, text, _mime, error = manager.read_resource_for_mention("nope", "config://app")
        assert ok is False
        assert "不存在或未就绪" in error
        entries = _read_audit(tmp_path)
        assert any(e["tool"] == "mcp__nope__mention" and not e["allowed"] for e in entries)

    def test_truncates_huge_resource(self, tmp_path, monkeypatch):
        manager, _ = make_manager(tmp_path, monkeypatch, [FakeSession()])
        manager.startup()

        class BigSession(FakeSession):
            async def read_resource(self, uri, **kwargs):
                return SimpleNamespace(
                    contents=[SimpleNamespace(text="x" * 250_000, mimeType="text/plain")]
                )

        manager._servers["srv"].session = BigSession()
        ok, text, _mime, _ = manager.read_resource_for_mention("srv", "config://app")
        assert ok is True
        # 截断到 20 万预算 + 少量提示字符，远小于原始 25 万
        assert len(text) < 200_100
        assert text.endswith("⚠️ 资源内容已截断（>200000 字符）")


def _read_audit(tmp_path):
    import datetime
    import pathlib

    audit_dir = tmp_path / "audit"
    today = datetime.date.today().strftime("%Y-%m-%d")
    path = pathlib.Path(audit_dir) / f"audit-{today}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------- 纯函数 ----------


class TestFormatters:
    def test_namespaced_tool(self):
        assert namespaced_tool("srv", "add") == "mcp__srv__add"
        assert namespaced_tool("step search", "web_search") == "mcp__step_search__web_search"

    def test_format_tool_result_text(self):
        result = SimpleNamespace(content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")], isError=False)
        assert format_tool_result(result) == "a\nb"

    def test_format_tool_result_is_error(self):
        result = SimpleNamespace(content=[SimpleNamespace(text="boom")], isError=True)
        assert format_tool_result(result) == "MCP 工具返回错误: boom"

    def test_format_tool_result_empty(self):
        assert format_tool_result(SimpleNamespace(content=[], isError=False)) == "(MCP 工具无文本输出)"

    def test_format_resource_list(self):
        result = SimpleNamespace(
            resources=[SimpleNamespace(uri="u://1", name="n", mimeType="text/plain", description="d")]
        )
        listing = format_resource_list(result)
        assert "u://1" in listing and "d" in listing

    def test_extract_resource_text_blob(self):
        result = SimpleNamespace(
            contents=[SimpleNamespace(blob="AAAA", mimeType="image/png")]
        )
        text, mime = extract_resource_text(result)
        assert "二进制资源" in text and "image/png" in text
