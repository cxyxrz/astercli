"""反思器测试：结构化解析、落盘、去重、配额、开关与 fail-safe（零网络）。"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from astercli.config import Settings
from astercli.experience import (
    CATEGORY_USER_CORRECTION,
    TRIGGER_HITL_REJECT,
    TRIGGER_LSP_ERROR,
    TRIGGER_TOOL_FAILURE,
    ExperienceStore,
    ReflectionCollector,
    ReflectionTrigger,
    Reflector,
    parse_reflection_output,
)


class FakeClient:
    """按脚本顺序返回响应的假 LLM（可注入异常）。"""

    def __init__(self, replies=None) -> None:
        self.replies = list(replies or [])
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else '{"lesson": ""}'
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


class FakeRenderer:
    def __init__(self) -> None:
        self.infos = []
        self.warns = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warn(self, message: str) -> None:
        self.warns.append(message)


def _settings(**env) -> Settings:
    return Settings(env=env, load_files=False)


def _store(tmp_path, **env) -> ExperienceStore:
    return ExperienceStore(
        tmp_path,
        _settings(**env),
        global_dir=tmp_path / "global",
        project_dir=tmp_path / "project",
    )


def _reflector(tmp_path, client, collector=None, renderer=None, **env) -> Reflector:
    settings = _settings(**env)
    return Reflector(
        SimpleNamespace(client=client),
        ExperienceStore(
            tmp_path, settings, global_dir=tmp_path / "global", project_dir=tmp_path / "project"
        ),
        renderer or FakeRenderer(),
        settings,
        collector or ReflectionCollector(),
        project_key=str(tmp_path),
    )


# ---------- 解析纯函数 ----------


def test_解析反思输出_容忍代码块与闲聊():
    text = '结果如下：\n```json\n{"lesson": "先检查目录", "correct_action": "glob_files"}\n```'
    data = parse_reflection_output(text)
    assert data is not None
    assert data["lesson"] == "先检查目录"


def test_解析反思输出_非法内容返回None():
    assert parse_reflection_output("") is None
    assert parse_reflection_output("不是 JSON") is None
    assert parse_reflection_output('{"lesson": }') is None


# ---------- 采集器 ----------


def test_采集器_三类触发与drain():
    collector = ReflectionCollector()
    collector.tool_failure("write_file", "磁盘只读", {"path": "a.py"})
    collector.hitl_reject("execute_command", "不允许删除文件", {"cmd": "rm -rf build"})
    collector.lsp_error(SimpleNamespace(severity="error", message="invalid syntax", file_path="a.py", line=3))
    assert collector.pending_count() == 3
    kinds = {t.kind for t in collector.drain()}
    assert kinds == {TRIGGER_TOOL_FAILURE, TRIGGER_HITL_REJECT, TRIGGER_LSP_ERROR}
    assert collector.pending_count() == 0


def test_采集器_忽略非error级LSP诊断():
    collector = ReflectionCollector()
    collector.lsp_error(SimpleNamespace(severity="warning", message="unused import", file_path="a.py", line=1))
    assert collector.pending_count() == 0


def test_采集器_参数骨架不含完整文件内容():
    collector = ReflectionCollector()
    collector.tool_failure("write_file", "失败", {"path": "a.py", "content": "x" * 1000})
    trigger = collector.drain()[0]
    assert "x" * 1000 not in trigger.args_skeleton


# ---------- 反思与落盘 ----------


async def test_反思成功落盘经验(tmp_path):
    client = FakeClient(
        ['{"lesson": "写文件前先确认父目录存在", "correct_action": "先用 glob_files 检查", '
         '"category": "error-lesson", "confidence": "high"}']
    )
    renderer = FakeRenderer()
    reflector = _reflector(tmp_path, client, renderer=renderer)
    reflector._collector.tool_failure("write_file", "No such file or directory", {"path": "a.py"})

    entries = await reflector.consume()
    assert len(entries) == 1
    assert entries[0].lesson == "写文件前先确认父目录存在"
    assert (tmp_path / "project" / f"{entries[0].id}.md").is_file()
    assert any("已沉淀经验" in info for info in renderer.infos)


async def test_同签名触发只反思一次(tmp_path):
    client = FakeClient(
        ['{"lesson": "先检查目录", "confidence": "high"}', '{"lesson": "第二条", "confidence": "high"}']
    )
    reflector = _reflector(tmp_path, client)
    for _ in range(2):
        reflector._collector.tool_failure("write_file", "No such file or directory", {"path": "a.py"})
    entries = await reflector.consume()
    assert len(entries) == 1
    assert len(client.calls) == 1


async def test_低置信度丢弃(tmp_path):
    client = FakeClient(['{"lesson": "可能吧", "confidence": "low"}'])
    reflector = _reflector(tmp_path, client)
    reflector._collector.tool_failure("write_file", "失败", {"path": "a.py"})
    assert await reflector.consume() == []
    assert reflector._store.list_entries() == []


async def test_空教训与非法JSON丢弃(tmp_path):
    client = FakeClient(['{"lesson": ""}', "不是 JSON"])
    reflector = _reflector(tmp_path, client)
    reflector._collector.tool_failure("write_file", "失败", {"path": "a.py"})
    reflector._collector.tool_failure("read_file", "失败", {"path": "b.py"})
    assert await reflector.consume() == []


async def test_LLM异常时fail_safe(tmp_path):
    client = FakeClient([RuntimeError("网络中断")])
    reflector = _reflector(tmp_path, client)
    reflector._collector.tool_failure("write_file", "失败", {"path": "a.py"})
    assert await reflector.consume() == []


async def test_会话反思次数上限(tmp_path):
    client = FakeClient(
        [
            '{"lesson": "教训一", "confidence": "high"}',
            '{"lesson": "教训二", "confidence": "high"}',
            '{"lesson": "教训三", "confidence": "high"}',
        ]
    )
    reflector = _reflector(tmp_path, client, ASTERCLI_REFLECTION_SESSION_LIMIT="2")
    for index in range(3):
        reflector._collector.tool_failure("tool", f"错误{index}", {"i": index})
    entries = await reflector.consume()
    assert len(entries) == 2
    assert len(client.calls) == 2


async def test_关闭开关时不反思且清空触发(tmp_path):
    client = FakeClient(['{"lesson": "不该出现"}'])
    reflector = _reflector(tmp_path, client)
    reflector.set_enabled(False)
    reflector._collector.tool_failure("write_file", "失败", {"path": "a.py"})
    assert await reflector.consume() == []
    assert client.calls == []


async def test_拒绝触发默认归类用户纠错(tmp_path):
    client = FakeClient(['{"lesson": "用户不希望直接删文件", "confidence": "high"}'])
    reflector = _reflector(tmp_path, client)
    reflector._collector.hitl_reject("execute_command", "不要删除", {"cmd": "rm"})
    entries = await reflector.consume()
    assert entries[0].category == CATEGORY_USER_CORRECTION
    assert entries[0].trigger == TRIGGER_HITL_REJECT


async def test_经验内容脱敏项目绝对路径(tmp_path):
    client = FakeClient(
        [f'{{"lesson": "{tmp_path}/src/app.py 不存在，先确认路径", "confidence": "high"}}']
    )
    reflector = _reflector(tmp_path, client)
    reflector._collector.tool_failure("read_file", "No such file", {"path": "a.py"})
    entries = await reflector.consume()
    assert "<project>" in entries[0].lesson
    assert str(tmp_path) not in entries[0].lesson


async def test_重复经验合并增加hits(tmp_path):
    reply = '{"lesson": "先确认目录", "confidence": "high"}'
    client = FakeClient([reply, reply])
    reflector = _reflector(tmp_path, client)
    reflector._collector.tool_failure("write_file", "错误A", {"path": "a.py"})
    await reflector.consume()
    reflector._collector.tool_failure("write_file", "错误B", {"path": "b.py"})
    entries = await reflector.consume()
    assert entries[0].hits == 2
    assert len(list((tmp_path / "project").glob("exp-*.md"))) == 1


def test_会话重置清空配额与去重(tmp_path):
    client = FakeClient()
    reflector = _reflector(tmp_path, client)
    reflector._collector.tool_failure("write_file", "失败", {"path": "a.py"})
    assert reflector.status().startswith("反思: 开启")
    reflector.reset_session()
    assert reflector.status().startswith("反思: 开启")
