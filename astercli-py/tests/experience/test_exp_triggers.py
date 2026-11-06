"""反思触发接线测试：工具失败 / HITL 拒绝 / 策略拒绝 / 超时 / LSP error。"""

from __future__ import annotations

import asyncio

from astercli.config import Settings
from astercli.experience import (
    TRIGGER_HITL_REJECT,
    TRIGGER_LSP_ERROR,
    TRIGGER_TOOL_FAILURE,
    ReflectionCollector,
)
from astercli.lsp import LspManager
from astercli.policy import PolicyException
from astercli.tools.registry import ToolRegistry

SETTINGS = Settings(load_files=False)


async def test_工具异常触发反思信号():
    collector = ReflectionCollector()
    registry = ToolRegistry(settings=SETTINGS, reflection=collector)

    async def boom(args):
        raise RuntimeError("磁盘只读")

    registry.register("boom", "失败工具", {"type": "object"}, boom)
    messages = await registry.execute_all([{"id": "c1", "name": "boom", "args": {"path": "a.py"}}])
    assert "工具执行失败" in messages[0].content
    triggers = collector.drain()
    assert len(triggers) == 1
    assert triggers[0].kind == TRIGGER_TOOL_FAILURE
    assert triggers[0].tool_name == "boom"
    assert "磁盘只读" in triggers[0].error


async def test_工具超时触发反思信号():
    collector = ReflectionCollector()
    registry = ToolRegistry(settings=SETTINGS, reflection=collector)
    registry._default_timeout = 0.05

    async def slow(args):
        await asyncio.sleep(0.3)
        return "done"

    registry.register("slow", "慢工具", {"type": "object"}, slow)
    messages = await registry.execute_all([{"id": "c1", "name": "slow", "args": {}}])
    assert "工具执行超时" in messages[0].content
    triggers = collector.drain()
    assert triggers and triggers[0].kind == TRIGGER_TOOL_FAILURE


async def test_HITL拒绝触发反思信号():
    collector = ReflectionCollector()
    registry = ToolRegistry(settings=SETTINGS, approval=True, reflection=collector)

    async def write(args):
        return "已写入"

    registry.register("write_file", "写文件", {"type": "object"}, write, dangerous=True)
    messages = await registry.execute_all(
        [{"id": "c1", "name": "write_file", "args": {"path": "a.py"}}],
        decisions={"c1": {"action": "reject", "reason": "不要动这个文件"}},
    )
    assert "[HITL] 操作已被拒绝" in messages[0].content
    triggers = collector.drain()
    assert len(triggers) == 1
    assert triggers[0].kind == TRIGGER_HITL_REJECT
    assert "不要动这个文件" in triggers[0].reason


async def test_策略拒绝按工具失败采集():
    collector = ReflectionCollector()
    registry = ToolRegistry(settings=SETTINGS, reflection=collector)

    async def guarded(args):
        raise PolicyException("路径不在项目根内")

    registry.register("guarded", "受限工具", {"type": "object"}, guarded)
    messages = await registry.execute_all([{"id": "c1", "name": "guarded", "args": {}}])
    assert "🛡️" in messages[0].content
    triggers = collector.drain()
    assert triggers and triggers[0].kind == TRIGGER_TOOL_FAILURE


async def test_未挂采集器时工具失败不影响主路径():
    registry = ToolRegistry(settings=SETTINGS)

    async def boom(args):
        raise RuntimeError("失败")

    registry.register("boom", "失败工具", {"type": "object"}, boom)
    messages = await registry.execute_all([{"id": "c1", "name": "boom", "args": {}}])
    assert "工具执行失败" in messages[0].content


def test_LSP错误通过回调触发反思(tmp_path):
    collector = ReflectionCollector()
    manager = LspManager(tmp_path, SETTINGS, on_diagnostics=collector.lsp_error)
    target = tmp_path / "broken.py"
    target.write_text("def f(:\n    pass\n", encoding="utf-8")

    manager.run_post_edit_lsp_hook("broken.py", target)

    triggers = collector.drain()
    assert len(triggers) == 1
    assert triggers[0].kind == TRIGGER_LSP_ERROR
    assert triggers[0].file_path == "broken.py"
    assert triggers[0].line >= 1


def test_LSP正常文件不触发反思(tmp_path):
    collector = ReflectionCollector()
    manager = LspManager(tmp_path, SETTINGS, on_diagnostics=collector.lsp_error)
    target = tmp_path / "ok.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    manager.run_post_edit_lsp_hook("ok.py", target)
    assert collector.drain() == []
