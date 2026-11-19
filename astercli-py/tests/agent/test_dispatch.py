"""Explore 子 agent 派遣（dispatch_agent）测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from astercli.agent.dispatch import (
    DISPATCH_TIMEOUT_SECONDS,
    DISPATCH_TOOL_NAME,
    MAX_EXPLORE_STEPS,
    ExploreRunner,
    readonly_command_reason,
    register_dispatch_tool,
)
from astercli.config import Settings
from astercli.tools.registry import ToolRegistry, create_parameters


class _FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        self.schemas = None

    def bind_tools(self, schemas):
        self.schemas = schemas
        return self

    async def ainvoke(self, messages):
        self.requests.append(list(messages))
        return self._replies.pop(0)


class _FakeRuntime:
    supports_tools = True

    def __init__(self, client):
        self.client = client


class TestExploreDispatch:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(self._tmp.name)
        (self.project / "a.txt").write_text("x", encoding="utf-8")
        self.settings = Settings(env={}, load_files=False)
        self.registry = ToolRegistry(settings=self.settings)
        self.registry.register("peek", "只读工具", create_parameters(), lambda a: "PEEK-OK")
        self.registry.register(
            "write_thing", "危险工具", create_parameters(), lambda a: "X", dangerous=True
        )
        self.registry.register("load_skill", "加载 skill", create_parameters(), lambda a: "L")
        self.registry.register(
            "run_skill_task", "skill fork", create_parameters(), lambda a: "F"
        )

    def teardown_method(self):
        self._tmp.cleanup()

    def _runner(self, replies):
        client = _FakeClient(replies)
        runner = ExploreRunner(
            _FakeRuntime(client), self.registry, self.settings, self.project
        )
        return runner, client

    async def test_子上下文不带主历史且工具池只读(self):
        runner, client = self._runner([AIMessage(content="结论A")])
        out = await runner.run("这个功能在哪实现", "定位功能")
        assert "结论A" in out
        assert "中间过程未进入主上下文" in out
        assert "dispatch_agent: 定位功能" in out

        request = client.requests[0]
        assert len(request) == 2
        assert isinstance(request[0], SystemMessage)
        assert isinstance(request[1], HumanMessage)
        assert request[1].content == "这个功能在哪实现"

        names = {item["function"]["name"] for item in client.schemas}
        assert "peek" in names
        assert "run_readonly_command" in names
        assert "write_thing" not in names
        assert "load_skill" not in names
        assert "run_skill_task" not in names
        assert DISPATCH_TOOL_NAME not in names

    async def test_工具调用后返回结论(self):
        replies = [
            AIMessage(content="", tool_calls=[{"name": "peek", "args": {}, "id": "c1"}]),
            AIMessage(content="最终结论"),
        ]
        runner, client = self._runner(replies)
        out = await runner.run("任务")
        assert "最终结论" in out
        assert "peek" in out
        assert any(getattr(m, "content", "") == "PEEK-OK" for m in client.requests[1])

    async def test_达到步数上限(self):
        replies = [
            AIMessage(content="", tool_calls=[{"name": "peek", "args": {}, "id": f"c{i}"}])
            for i in range(MAX_EXPLORE_STEPS)
        ]
        runner, _ = self._runner(replies)
        out = await runner.run("任务")
        assert f"{MAX_EXPLORE_STEPS} 步" in out
        assert "未收敛" in out

    def test_只读命令白名单(self):
        assert readonly_command_reason("grep -n foo src") == ""
        assert readonly_command_reason("git status") == ""
        assert readonly_command_reason("git log --oneline -5") == ""
        assert readonly_command_reason("git branch -a") == ""
        assert readonly_command_reason("git remote -v") == ""
        assert "白名单" in readonly_command_reason("rm -rf /")
        assert "白名单" in readonly_command_reason("python x.py")
        assert "只读子命令" in readonly_command_reason("git push")
        assert "只读查询选项" in readonly_command_reason("git branch -D feature")
        assert "只读查询选项" in readonly_command_reason("git branch new-branch")
        assert "远端配置" in readonly_command_reason("git remote add origin x")
        assert "管道" in readonly_command_reason("cat a | grep b")
        assert "管道" in readonly_command_reason("cat a > b")
        assert "参数" in readonly_command_reason("find . -delete")
        assert "参数" in readonly_command_reason("find . -fprint out.txt")
        assert "参数" in readonly_command_reason("rg --pre cat foo")

    async def test_注册工具可被主_agent_调用(self):
        client = _FakeClient([AIMessage(content="主结论")])
        register_dispatch_tool(
            self.registry, _FakeRuntime(client), self.settings, self.project
        )
        tool = self.registry.get(DISPATCH_TOOL_NAME)
        assert tool is not None
        assert tool.dangerous is False
        assert tool.timeout_seconds == DISPATCH_TIMEOUT_SECONDS
        out = await tool.func({"prompt": "任务", "description": "调研"})
        assert "主结论" in out

    async def test_参数校验(self):
        client = _FakeClient([])
        register_dispatch_tool(
            self.registry, _FakeRuntime(client), self.settings, self.project
        )
        tool = self.registry.get(DISPATCH_TOOL_NAME)
        assert "prompt 不能为空" in await tool.func({"prompt": " "})
        assert "不支持的 subagent_type" in await tool.func(
            {"prompt": "任务", "subagent_type": "general"}
        )

    async def test_只读命令真实执行(self):
        runner, _ = self._runner([])
        child = runner._child_registry()
        tool = child.get("run_readonly_command")
        out = tool.func({"command": "pwd"})
        assert str(self.project) in out
        denied = tool.func({"command": "rm -rf ."})
        assert "只读命令被拒绝" in denied
