"""SkillContextBuffer / index_formatter / load_skill 工具测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from astercli.config import Settings
from astercli.skill import SkillContextBuffer, format_index, register_skill_tool
from astercli.skill.index_formatter import MAX_INDEX_CHARS
from astercli.skill.registry import SkillRegistry
from astercli.skill.skill import Skill, Source
from astercli.skill.state_store import SkillStateStore
from astercli.tools.registry import ToolRegistry


class TestBuffer:
    def test_空_buffer_drain_返回空串(self):
        buffer = SkillContextBuffer()
        assert buffer.drain() == ""
        assert buffer.is_empty()

    def test_push_drain_一次性消费(self):
        buffer = SkillContextBuffer()
        buffer.push("s", "技能正文")
        out = buffer.drain()
        assert "## 已加载 Skill：s" in out
        assert "技能正文" in out
        assert out.endswith("---\n")
        # 二次 drain 为空（防重复注入）
        assert buffer.drain() == ""

    def test_lru_淘汰_超过三个(self):
        buffer = SkillContextBuffer()
        buffer.push("a", "A")
        buffer.push("b", "B")
        buffer.push("c", "C")
        buffer.push("d", "D")  # 淘汰 a
        out = buffer.drain()
        assert "Skill：a" not in out
        assert "Skill：b" in out and "Skill：c" in out and "Skill：d" in out

    def test_同名重复_push_替换并置顶(self):
        buffer = SkillContextBuffer()
        buffer.push("a", "旧正文")
        buffer.push("b", "B")
        buffer.push("a", "新正文")  # 替换 + 移到末尾（LRU 最新）→ 顺序 b, a
        buffer.push("c", "C")
        buffer.push("d", "D")  # 超过 3 个 → 淘汰最旧的 b
        out = buffer.drain()
        assert "新正文" in out
        assert "旧正文" not in out
        assert "Skill：b" not in out
        assert "Skill：a" in out and "Skill：c" in out and "Skill：d" in out
        assert buffer.size() == 0

    def test_clear(self):
        buffer = SkillContextBuffer()
        buffer.push("a", "A")
        buffer.clear()
        assert buffer.is_empty()
        assert buffer.drain() == ""

    def test_非法入参_忽略(self):
        buffer = SkillContextBuffer()
        buffer.push("", "body")
        buffer.push("  ", "body")
        buffer.push("name", None)
        assert buffer.is_empty()


class TestIndexFormatter:
    def make_skill(self, name, desc):
        return Skill(name=name, description=desc)

    def test_空列表_空串(self):
        assert format_index([]) == ""

    def test_正常渲染(self):
        out = format_index([self.make_skill("java-style", "Java 风格约定")])
        assert "## 可用 Skills" in out
        assert "- **java-style**：Java 风格约定" in out
        assert "load_skill" in out

    def test_超_20_个截断(self):
        skills = [self.make_skill(f"s{i:02d}", "d") for i in range(25)]
        out = format_index(skills)
        assert "s00" in out  # 字典序保留前 20
        assert "s24" not in out

    def test_总长_超_4kb_截断(self):
        skills = [self.make_skill(f"skill-{i}", "长描述" * 200) for i in range(20)]
        out = format_index(skills)
        assert len(out) <= MAX_INDEX_CHARS + 100  # 截断提示行容差
        assert "索引段被截断" in out


class TestLoadSkillTool:
    def setup_method(self):
        # 隔离状态文件：禁用列表写到临时目录，绝不碰真实 ~/.astercli-py
        self._tmp = tempfile.TemporaryDirectory()
        state_store = SkillStateStore(Path(self._tmp.name) / "skills.json")
        self.settings = Settings(env={}, load_files=False)
        self.registry = ToolRegistry(settings=self.settings)
        self.skill_registry = SkillRegistry(
            user_skills_dir=None,
            project_skills_dir=None,
            state_store=state_store,
        )
        # 直接注入内存 skill（绕过文件加载）
        self.skill_registry._skills["demo"] = Skill(
            name="demo", description="演示", body="技能正文", source=Source.PROJECT
        )
        self.buffer = SkillContextBuffer()
        register_skill_tool(self.registry, self.skill_registry, self.buffer)

    def teardown_method(self):
        self._tmp.cleanup()

    def test_注册_只读(self):
        tool = self.registry.get("load_skill")
        assert tool is not None
        assert tool.dangerous is False

    async def test_加载成功_注入_buffer(self):
        tool = self.registry.get("load_skill")
        out = tool.func({"name": "demo"})
        assert "已加载 skill 'demo'" in out
        assert "下一轮" in out
        drained = self.buffer.drain()
        assert "## 已加载 Skill：demo" in drained
        assert "技能正文" in drained

    async def test_未知名(self):
        tool = self.registry.get("load_skill")
        assert "未找到" in tool.func({"name": "nope"})

    async def test_空名(self):
        tool = self.registry.get("load_skill")
        assert "不能为空" in tool.func({"name": ""})

    async def test_超长_body_截断入_buffer(self):
        self.skill_registry._skills["big"] = Skill(
            name="big", description="大", body="x" * (5 * 1024 + 100)
        )
        tool = self.registry.get("load_skill")
        out = tool.func({"name": "big"})
        assert "已截断" in out
        assert len(self.buffer.drain()) < 5 * 1024 + 100

    async def test_禁用_skill_提示启用路径(self):
        self.skill_registry.state_store().set_disabled(["demo"])
        tool = self.registry.get("load_skill")
        out = tool.func({"name": "demo"})
        assert "已被禁用" in out
        assert "/skill on demo" in out
