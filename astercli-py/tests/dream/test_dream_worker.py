"""Dream 后台线程测试：启动门 / 守护线程 / 通知队列 / 手动触发。"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from langchain_core.messages import AIMessage

from astercli.config import Settings
from astercli.dream import DreamPaths, DreamRecord, DreamRunner, DreamWorker, SessionLog
from astercli.experience import ExperienceStore
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.memory import LongTermMemory

PROJECT = "/proj/demo"


class FakeClient:
    def __init__(self, payload: str = "") -> None:
        self.payload = payload
        self.calls = 0

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls += 1
        return AIMessage(content=self.payload)


class BoomClient(FakeClient):
    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls += 1
        raise RuntimeError("boom")


def _payload() -> str:
    return json.dumps(
        {
            "summary": "沉淀构建方式",
            "ops": [
                {"type": "memory_add", "scope": "project", "content": "项目使用 uv 构建"},
            ],
        },
        ensure_ascii=False,
    )


def _fixture(tmp_path, sessions: int = 5, **env):
    settings = Settings(env=env, load_files=False)
    paths = DreamPaths.for_project(
        PROJECT, dream_root=tmp_path / "dream", session_root=tmp_path / "sessions"
    )
    current = SessionLog(
        PROJECT, settings, base_dir=tmp_path / "sessions", session_id="20260201-000000-current"
    )
    base = time.time() - 3 * 86400
    for index in range(sessions):
        log = SessionLog(
            PROJECT,
            settings,
            base_dir=tmp_path / "sessions",
            session_id=f"2026010{index + 1}-000000-past{index}",
        )
        log.record_user(f"历史问题 {index}")
        log.record_assistant(f"历史回答 {index}")
        stamp = base + index * 60
        os.utime(log.path, (stamp, stamp))

    memory = LongTermMemory(storage_dir=tmp_path / "memory")
    experience = ExperienceStore(
        PROJECT,
        settings,
        global_dir=tmp_path / "global_exp",
        project_dir=tmp_path / "project_exp",
    )
    return settings, paths, current, memory, experience


def _worker(settings, paths, current, memory, experience, client, **kwargs) -> DreamWorker:
    runtime = ProviderRuntime(
        client=client, provider="glm", model="glm-test", spec=PROVIDER_SPECS["glm"]
    )
    return DreamWorker(
        DreamRunner(PROJECT, settings, paths=paths, session_log=current),
        settings,
        current,
        memory,
        experience,
        lambda: runtime,
        project_key=PROJECT,
        **kwargs,
    )


def test_启动后台整合并发出通知(tmp_path):
    settings, paths, current, memory, experience = _fixture(tmp_path)
    client = FakeClient(_payload())
    worker = _worker(settings, paths, current, memory, experience, client)

    assert worker.start() is None
    assert worker.wait(10) is True

    notices = worker.drain_notices()
    assert any("已整理记忆" in notice for notice in notices)
    assert worker.dirty is True
    assert client.calls == 1

    state = worker.runner.load_state()
    assert state.dream_count == 1
    assert state.history[0].status == "applied"
    assert state.history[0].ops == 1
    assert "项目使用 uv 构建" in {entry.content for entry in memory.get_all()}


def test_关闭时不启动(tmp_path):
    settings, paths, current, memory, experience = _fixture(tmp_path)
    worker = _worker(
        settings, paths, current, memory, experience, FakeClient(_payload()), enabled=False
    )

    assert worker.start() == "AutoDream 已关闭"
    assert worker.busy is False
    assert worker.drain_notices() == []


def test_新会话不足时不启动(tmp_path):
    settings, paths, current, memory, experience = _fixture(tmp_path, sessions=2)
    client = FakeClient(_payload())
    worker = _worker(settings, paths, current, memory, experience, client)

    reason = worker.start()
    assert reason is not None and "新增会话" in reason
    assert worker.busy is False
    assert client.calls == 0


def test_上次整合不久不重复启动(tmp_path):
    settings, paths, current, memory, experience = _fixture(tmp_path)
    worker = _worker(settings, paths, current, memory, experience, FakeClient(_payload()))
    worker.runner.record_result(
        DreamRecord(dream_id="dream-prev", status="applied", finished_at=time.time())
    )

    reason = worker.start()
    assert reason is not None and "距上次整合" in reason


def _touch_sessions(root, stamp: float) -> None:
    for path in root.rglob("*.jsonl"):
        os.utime(path, (stamp, stamp))


def test_手动触发忽略时间门(tmp_path):
    settings, paths, current, memory, experience = _fixture(tmp_path)
    client = FakeClient(_payload())
    worker = _worker(settings, paths, current, memory, experience, client)
    now = time.time()
    # 会话比上次 dream 新，但距上次 dream 只有 2 小时：自动门会拒绝，手动应执行
    _touch_sessions(tmp_path / "sessions", now - 3600)
    worker.runner.record_result(
        DreamRecord(dream_id="dream-prev", status="applied", finished_at=now - 7200)
    )
    assert worker.runner.check().allowed is False

    message = asyncio.run(worker.run_now())
    assert "已整理记忆" in message
    assert client.calls == 1
    assert worker.dirty is True


@pytest.mark.asyncio
async def test_LLM失败记录失败且不推进应用(tmp_path):
    settings, paths, current, memory, experience = _fixture(tmp_path)
    worker = _worker(settings, paths, current, memory, experience, BoomClient())

    message = await worker.run_now()
    assert "未完成" in message
    state = worker.runner.load_state()
    assert state.history[0].status == "failed"
    assert memory.get_all() == []
    assert worker.dirty is False
