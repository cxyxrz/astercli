"""Dream 整合器测试：素材拼装 / 预算裁剪 / 结构化提案 / 失败折叠。"""

from __future__ import annotations

import json
import os
import time

import pytest
from langchain_core.messages import AIMessage

from astercli.config import Settings
from astercli.dream import Consolidator, DreamPaths, DreamState, SessionLog
from astercli.experience import ExperienceEntry, ExperienceStore
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.memory import LongTermMemory

PROJECT = "/proj/demo"


class FakeClient:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.text)


def _fixture(tmp_path, **env):
    settings = Settings(env=env, load_files=False)
    paths = DreamPaths.for_project(
        PROJECT, dream_root=tmp_path / "dream", session_root=tmp_path / "sessions"
    )
    current = SessionLog(
        PROJECT, settings, base_dir=tmp_path / "sessions", session_id="20260110-000000-current"
    )
    old = SessionLog(
        PROJECT, settings, base_dir=tmp_path / "sessions", session_id="20260101-000000-old"
    )
    old.record_user("帮我看看项目怎么构建")
    old.record_tool("read_file")
    old.record_assistant("项目使用 Maven 构建")
    past = time.time() - 86400
    os.utime(old.path, (past, past))

    memory = LongTermMemory(storage_dir=tmp_path / "memory")
    entry = memory.store_fact("项目使用 Java 17", scope="project", project_key=PROJECT)

    experience = ExperienceStore(
        PROJECT,
        settings,
        global_dir=tmp_path / "global_exp",
        project_dir=tmp_path / "project_exp",
    )
    saved, _ = experience.save(
        ExperienceEntry(
            id="exp-cccc3333",
            lesson="写文件前先确认目录存在",
            correct_action="先用 glob_files 检查",
            category="workflow",
            trigger="tool_failure",
            project_key=PROJECT,
        )
    )
    return settings, paths, current, memory, experience, entry, saved


def _runtime(client) -> ProviderRuntime:
    return ProviderRuntime(
        client=client, provider="glm", model="glm-test", spec=PROVIDER_SPECS["glm"]
    )


def _consolidator(settings, paths, current, memory, experience, client) -> Consolidator:
    return Consolidator(
        _runtime(client),
        settings,
        paths,
        current,
        memory,
        experience,
        project_key=PROJECT,
    )


def test_素材包含记忆经验与历史会话(tmp_path):
    settings, paths, current, memory, experience, entry, saved = _fixture(tmp_path)
    consolidator = _consolidator(settings, paths, current, memory, experience, FakeClient())

    material = consolidator.collect(DreamState())
    assert material.session_count == 1
    assert material.memory_count == 1
    assert material.experience_count == 1
    assert "项目使用 Java 17" in material.text
    assert entry.id in material.text
    assert saved.id in material.text
    assert "read_file" in material.text
    assert "项目使用 Maven 构建" in material.text
    # 当前会话不参与整合
    assert "20260110-000000-current" not in material.text


def test_输入预算过小则丢弃会话(tmp_path):
    settings, paths, current, memory, experience, _, _ = _fixture(
        tmp_path, ASTERCLI_DREAM_MAX_INPUT_TOKENS="20"
    )
    consolidator = _consolidator(settings, paths, current, memory, experience, FakeClient())
    material = consolidator.collect(DreamState())
    assert material.session_count == 0


@pytest.mark.asyncio
async def test_合法输出生成并落盘提案(tmp_path):
    settings, paths, current, memory, experience, entry, saved = _fixture(tmp_path)
    payload = json.dumps(
        {
            "summary": "补充构建事实并沉淀工具经验",
            "ops": [
                {"type": "memory_add", "scope": "project", "content": "项目使用 Maven 构建"},
                {"type": "memory_update", "id": entry.id, "content": "项目使用 Java 17 + Maven"},
                {
                    "type": "experience_add",
                    "category": "workflow",
                    "lesson": "读代码前先用 glob_files 找入口",
                    "correct_action": "先列目录再读文件",
                },
                {"type": "experience_delete", "id": saved.id, "reason": "已被新经验覆盖"},
            ],
        },
        ensure_ascii=False,
    )
    client = FakeClient(payload)
    consolidator = _consolidator(settings, paths, current, memory, experience, client)

    proposal = await consolidator.consolidate(DreamState(), "dream-1")
    assert client.calls == 1
    assert proposal.summary.startswith("补充构建事实")
    assert proposal.op_count == 4
    assert proposal.errors == []
    assert (paths.proposal_dir("dream-1") / "ops.json").is_file()
    assert "补充构建事实" in (paths.proposal_dir("dream-1") / "proposal.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_LLM异常折叠为错误且不抛(tmp_path):
    settings, paths, current, memory, experience, _, _ = _fixture(tmp_path)
    client = FakeClient(error=RuntimeError("boom"))
    consolidator = _consolidator(settings, paths, current, memory, experience, client)

    proposal = await consolidator.consolidate(DreamState(), "dream-2")
    assert proposal.ops == []
    assert any("LLM 调用失败" in e for e in proposal.errors)


@pytest.mark.asyncio
async def test_非法JSON折叠为错误(tmp_path):
    settings, paths, current, memory, experience, _, _ = _fixture(tmp_path)
    client = FakeClient("这不是 JSON")
    consolidator = _consolidator(settings, paths, current, memory, experience, client)

    proposal = await consolidator.consolidate(DreamState(), "dream-3")
    assert proposal.ops == []
    assert any("不是合法 JSON" in e for e in proposal.errors)


@pytest.mark.asyncio
async def test_无新会话不调用LLM(tmp_path):
    settings, paths, current, memory, experience, _, _ = _fixture(tmp_path)
    client = FakeClient()
    consolidator = _consolidator(settings, paths, current, memory, experience, client)

    proposal = await consolidator.consolidate(DreamState(last_dream_at=time.time()), "dream-4")
    assert client.calls == 0
    assert proposal.ops == []
    assert any("没有可整合" in e for e in proposal.errors)
