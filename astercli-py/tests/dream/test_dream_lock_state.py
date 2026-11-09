"""AutoDream 锁与状态测试：PID 锁抢占 / 过期回收 / 状态原子读写 / Runner 门合并。"""

from __future__ import annotations

import json
import os
import time

from astercli.config import Settings
from astercli.dream import DreamLock, DreamPaths, DreamRecord, DreamRunner, DreamState, SessionLog
from astercli.dream.runner import pid_alive


def test_状态序列化往返():
    state = DreamState(last_scan_at=50.0)
    state.mark_dream(DreamRecord(dream_id="dream-1", status="applied"), now=100.0)
    state.mark_dream(
        DreamRecord(dream_id="dream-2", status="applied", ops=3, summary="合并 2 条"), now=200.0
    )

    restored = DreamState.from_dict(state.to_dict())
    assert restored.last_dream_at == 200.0
    assert restored.dream_count == 2
    assert restored.last_dream_id == "dream-2"
    assert restored.history[0].dream_id == "dream-2"
    assert restored.history[1].dream_id == "dream-1"


def test_状态历史上限():
    state = DreamState()
    for index in range(30):
        state.mark_dream(DreamRecord(dream_id=f"dream-{index}"), now=float(index))
    assert len(state.history) == 20
    assert state.history[0].dream_id == "dream-29"


def test_状态文件原子读写与损坏回退(tmp_path):
    path = tmp_path / "state.json"
    state = DreamState(dream_count=1, last_dream_id="dream-1")
    assert state.save(path) is True
    assert DreamState.load(path).last_dream_id == "dream-1"

    path.write_text("{broken", encoding="utf-8")
    assert DreamState.load(path).dream_count == 0
    assert DreamState.load(tmp_path / "missing.json").dream_count == 0


def test_锁获取与释放(tmp_path):
    path = tmp_path / ".lock"
    lock = DreamLock(path)
    assert lock.acquire() is True
    assert path.is_file()

    contender = DreamLock(path, pid=os.getpid() + 1)
    assert contender.acquire() is False  # 持有进程仍存活

    lock.release()
    assert not path.exists()
    assert contender.acquire() is True
    contender.release()


def test_过期锁可抢占(tmp_path):
    path = tmp_path / ".lock"
    path.write_text(
        json.dumps({"pid": os.getpid(), "created_at": time.time() - 7200}), encoding="utf-8"
    )
    lock = DreamLock(path, ttl_seconds=3600)
    assert lock.acquire() is True
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    lock.release()


def test_损坏锁视为过期(tmp_path):
    path = tmp_path / ".lock"
    path.write_text("garbage", encoding="utf-8")
    lock = DreamLock(path)
    assert lock.acquire() is True
    lock.release()


def test_释放不误删他人锁(tmp_path):
    path = tmp_path / ".lock"
    lock = DreamLock(path)
    assert lock.acquire() is True
    # 模拟被抢占：锁内容已属于另一个进程
    path.write_text(json.dumps({"pid": 999999, "created_at": time.time()}), encoding="utf-8")
    lock.release()
    assert path.exists()


def test_pid存活判断():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_runner合并前四道门(tmp_path):
    paths = DreamPaths.for_project(
        "/proj/demo", dream_root=tmp_path / "dream", session_root=tmp_path / "sessions"
    )
    log = SessionLog("/proj/demo", base_dir=tmp_path / "sessions", session_id="20260103-000000-cccccc")
    extra = SessionLog("/proj/demo", base_dir=tmp_path / "sessions", session_id="20260101-000000-aaaaaa")
    extra.record_user("历史会话")
    runner = DreamRunner("/proj/demo", Settings(env={}, load_files=False), paths=paths, session_log=log)

    result = runner.check()
    assert result.reason == "not_enough_sessions"

    # 把历史会话 mtime 拉远，并把门槛降到 1 个新会话
    old = time.time() - 3 * 86400
    os.utime(extra.path, (old, old))
    settings = Settings(env={"ASTERCLI_DREAM_MIN_SESSIONS": "1"}, load_files=False)
    runner = DreamRunner("/proj/demo", settings, paths=paths, session_log=log)
    assert runner.check().allowed is True

    state = runner.mark_scan()
    assert state.last_scan_at > 0
    assert runner.load_state().last_scan_at > 0

    recorded = runner.record_result(DreamRecord(dream_id="dream-1", status="applied"))
    assert recorded.dream_count == 1
    assert runner.load_state().last_dream_id == "dream-1"
