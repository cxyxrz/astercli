"""Skill 子 agent 隔离（run_skill_task）测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from astercli.config import Settings
from astercli.skill.fork import (
    FORK_TIMEOUT_SECONDS,
    FORK_TOOL_NAME,
    MAX_FORK_STEPS,
    SkillForkRunner,
    register_skill_fork_tool,
)
from astercli.skill.registry import SkillRegistry
from astercli.skill.state_store import SkillStateStore
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


class TestSkillForkIsolation:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.project = base / "project"
        skill_dir = self.project / "demo"
        (skill_dir / "references").mkdir(parents=True)
        self.big_body = "# 长正文\n" + ("规则行\n" * 2000)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo\ndescription: 演示 skill\n---\n" + self.big_body,
            encoding="utf-8",
        )
        (skill_dir / "references" / "a.md").write_text("细则A\n", encoding="utf-8")
        self.settings = Settings(env={}, load_files=False)
        self.registry = ToolRegistry(settings=self.settings)
        self.registry.register("peek", "只读工具", create_parameters(), lambda a: "PEEK-OK")
        self.registry.register(
            "delete_thing", "危险工具", create_parameters(), lambda a: "X", dangerous=True
        )
        self.skill_registry = SkillRegistry(
            user_skills_dir=None,
            project_skills_dir=self.project,
            state_store=SkillStateStore(base / "skills.json"),
        )
        self.skill_registry.reload()
        self.skill = self.skill_registry.find_skill("demo")

    def teardown_method(self):
        self._tmp.cleanup()

    def _runner(self, replies):
        client = _FakeClient(replies)
        return SkillForkRunner(_FakeRuntime(client), self.registry, self.settings), client

    async def test_子上下文不带主历史且只复制只读工具(self):
        runner, client = self._runner([AIMessage(content="结论A")])
        out = await runner.run(self.skill, "只回答 OK")
        assert "结论A" in out
        assert "中间过程未进入主上下文" in out

        request = client.requests[0]
        assert len(request) == 2
        assert isinstance(request[0], SystemMessage)
        assert isinstance(request[1], HumanMessage)
        assert request[1].content == "只回答 OK"

        names = {item["function"]["name"] for item in client.schemas}
        assert "peek" in names
        assert "read_skill_file" in names
        assert "delete_thing" not in names
        assert FORK_TOOL_NAME not in names
        assert "load_skill" not in names

    async def test_长_body_完整进入子上下文(self):
        runner, client = self._runner([AIMessage(content="ok")])
        await runner.run(self.skill, "任务")
        system = client.requests[0][0].content
        assert self.big_body.strip() in system
        assert len(system) > 8000
        assert "skill_dir:" in system

    async def test_工具调用后返回结论(self):
        replies = [
            AIMessage(content="", tool_calls=[{"name": "peek", "args": {}, "id": "c1"}]),
            AIMessage(content="最终结论"),
        ]
        runner, client = self._runner(replies)
        out = await runner.run(self.skill, "任务")
        assert "最终结论" in out
        assert "peek" in out
        assert any(getattr(m, "content", "") == "PEEK-OK" for m in client.requests[1])

    async def test_达到步数上限(self):
        replies = [
            AIMessage(content="", tool_calls=[{"name": "peek", "args": {}, "id": f"c{i}"}])
            for i in range(MAX_FORK_STEPS)
        ]
        runner, _ = self._runner(replies)
        out = await runner.run(self.skill, "任务")
        assert f"{MAX_FORK_STEPS} 步" in out
        assert "未收敛" in out

    async def test_注册工具可被主_agent_调用(self):
        client = _FakeClient([AIMessage(content="主结论")])
        register_skill_fork_tool(
            self.registry, self.skill_registry, _FakeRuntime(client), self.settings
        )
        tool = self.registry.get(FORK_TOOL_NAME)
        assert tool is not None
        assert tool.dangerous is False
        assert tool.timeout_seconds == FORK_TIMEOUT_SECONDS
        out = await tool.func({"skill": "demo", "task": "任务"})
        assert "主结论" in out

    async def test_禁用与未知_skill_都被拒绝(self):
        client = _FakeClient([])
        register_skill_fork_tool(
            self.registry, self.skill_registry, _FakeRuntime(client), self.settings
        )
        tool = self.registry.get(FORK_TOOL_NAME)
        self.skill_registry.state_store().set_disabled(["demo"])
        assert "已被禁用" in await tool.func({"skill": "demo", "task": "任务"})
        assert "未找到" in await tool.func({"skill": "nope", "task": "任务"})
        assert "task 不能为空" in await tool.func({"skill": "nope", "task": ""})
