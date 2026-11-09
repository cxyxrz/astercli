"""Dream 应用测试：事务写入 / 软删除归档 / 失败回滚 / 快照撤销。"""

from __future__ import annotations

import time

import pytest

from astercli.config import Settings
from astercli.dream import (
    Applier,
    DreamPaths,
    DreamRecord,
    DreamRunner,
    Proposal,
    SessionLog,
)
from astercli.experience import ExperienceEntry, ExperienceStore
from astercli.memory import LongTermMemory

PROJECT = "/proj/demo"


def _setup(tmp_path):
    settings = Settings(env={}, load_files=False)
    paths = DreamPaths.for_project(
        PROJECT, dream_root=tmp_path / "dream", session_root=tmp_path / "sessions"
    )
    memory = LongTermMemory(storage_dir=tmp_path / "memory")
    fact_a = memory.store_fact("旧事实 A", project_key=PROJECT)
    fact_b = memory.store_fact("旧事实 B", project_key=PROJECT)
    experience = ExperienceStore(
        PROJECT,
        settings,
        global_dir=tmp_path / "global_exp",
        project_dir=tmp_path / "project_exp",
    )
    kept, _ = experience.save(
        ExperienceEntry(
            id="exp-keep1234",
            lesson="保留的经验",
            category="workflow",
            project_key=PROJECT,
        )
    )
    gone, _ = experience.save(
        ExperienceEntry(
            id="exp-gone1234",
            lesson="将被整合归档的经验",
            category="workflow",
            project_key=PROJECT,
        )
    )
    return settings, paths, memory, experience, fact_a, fact_b, kept, gone


def _proposal(fact_a, fact_b, gone) -> Proposal:
    return Proposal(
        dream_id="dream-20260101-aaaaaa",
        summary="合并与清理",
        ops=[
            {"type": "memory_add", "scope": "project", "content": "新事实：项目使用 uv"},
            {"type": "memory_update", "id": fact_a.id, "content": "更新后的 A"},
            {"type": "memory_delete", "id": fact_b.id},
            {
                "type": "memory_merge",
                "ids": [],
                "content": "不会保留：空 ids",
            },
            {
                "type": "experience_add",
                "category": "workflow",
                "lesson": "dream 沉淀的经验",
                "correct_action": "先读文件再改",
            },
            {"type": "experience_delete", "id": gone.id, "reason": "已过时"},
        ],
    )


def test_应用全部操作并软删除(tmp_path):
    settings, paths, memory, experience, fact_a, fact_b, kept, gone = _setup(tmp_path)
    proposal = _proposal(fact_a, fact_b, gone)
    proposal.ops.pop(3)  # 去掉空 ids 的 merge
    applier = Applier(memory, experience, paths, project_key=PROJECT)

    result = applier.apply(proposal)

    assert result.rolled_back is False
    assert result.applied == 5
    assert result.ok is True

    contents = {entry.content for entry in memory.get_all()}
    assert "更新后的 A" in contents
    assert "旧事实 B" not in contents
    assert "新事实：项目使用 uv" in contents

    lessons = {entry.lesson for entry in experience.list_entries()}
    assert "保留的经验" in lessons
    assert "dream 沉淀的经验" in lessons
    assert "将被整合归档的经验" not in lessons

    # 删除是软删：归档文件仍在
    memory_archive = (
        paths.root / "archive" / proposal.dream_id / "memory" / f"{fact_b.id}.json"
    )
    exp_archive = (
        paths.root / "archive" / proposal.dream_id / "experience" / f"{gone.id}.md"
    )
    assert memory_archive.is_file()
    assert exp_archive.is_file()


def test_落盘失败整体回滚(tmp_path, monkeypatch):
    settings, paths, memory, experience, fact_a, fact_b, kept, gone = _setup(tmp_path)
    proposal = _proposal(fact_a, fact_b, gone)
    proposal.ops.pop(3)
    original = (tmp_path / "memory" / "long_term_memory.json").read_text(encoding="utf-8")

    monkeypatch.setattr(memory, "persist", lambda: False)
    result = Applier(memory, experience, paths, project_key=PROJECT).apply(proposal)

    assert result.rolled_back is True
    assert result.applied == 0
    # 磁盘上的长期记忆保持原样
    assert (tmp_path / "memory" / "long_term_memory.json").read_text(
        encoding="utf-8"
    ) == original
    # 经验库也回到应用前（新增被删除、被归档的恢复）
    reloaded = ExperienceStore(
        PROJECT,
        settings,
        global_dir=tmp_path / "global_exp",
        project_dir=tmp_path / "project_exp",
    )
    lessons = {entry.lesson for entry in reloaded.list_entries()}
    assert lessons == {"保留的经验", "将被整合归档的经验"}


def test_空提案不执行(tmp_path):
    _, paths, memory, experience, _, _, _, _ = _setup(tmp_path)
    result = Applier(memory, experience, paths, project_key=PROJECT).apply(
        Proposal(dream_id="dream-empty")
    )
    assert result.applied == 0
    assert result.ok is False
    assert any("没有可应用" in error for error in result.errors)


def test_撤销还原记忆与经验(tmp_path):
    settings, paths, memory, experience, fact_a, fact_b, kept, gone = _setup(tmp_path)
    proposal = _proposal(fact_a, fact_b, gone)
    proposal.ops.pop(3)
    result = Applier(memory, experience, paths, project_key=PROJECT).apply(proposal)
    assert result.ok is True

    log = SessionLog(PROJECT, settings, base_dir=tmp_path / "sessions")
    runner = DreamRunner(PROJECT, settings, paths=paths, session_log=log)
    runner.record_result(
        DreamRecord(
            dream_id=proposal.dream_id,
            status="applied",
            ops=result.applied,
            summary=proposal.summary,
            finished_at=time.time(),
        )
    )

    ok, message = runner.undo()
    assert ok is True
    assert proposal.dream_id in message

    restored = LongTermMemory(storage_dir=tmp_path / "memory")
    contents = {entry.content for entry in restored.get_all()}
    assert contents == {"旧事实 A", "旧事实 B"}

    restored_exp = ExperienceStore(
        PROJECT,
        settings,
        global_dir=tmp_path / "global_exp",
        project_dir=tmp_path / "project_exp",
    )
    assert {entry.lesson for entry in restored_exp.list_entries()} == {
        "保留的经验",
        "将被整合归档的经验",
    }
    state = runner.load_state()
    assert state.dream_count == 0
    assert state.history == []
    assert state.last_dream_at == 0


def test_撤销指定id与未知id(tmp_path):
    settings, paths, memory, experience, _, _, _, _ = _setup(tmp_path)
    log = SessionLog(PROJECT, settings, base_dir=tmp_path / "sessions")
    runner = DreamRunner(PROJECT, settings, paths=paths, session_log=log)

    ok, message = runner.undo()
    assert ok is False
    assert "未找到" in message

    runner.record_result(DreamRecord(dream_id="dream-x", status="failed"))
    ok, message = runner.undo("dream-missing")
    assert ok is False
    assert "dream-missing" in message
