"""AutoDream 会话转写测试：字段最小化 / 截断 / 开关 / 会话枚举。"""

from __future__ import annotations

import json
import os

from astercli.config import Settings
from astercli.dream import DreamPaths, SessionLog
from astercli.dream.session_log import MAX_TEXT_CHARS, project_hash


def _log(tmp_path, **env) -> SessionLog:
    settings = Settings(env=env, load_files=False)
    return SessionLog("/proj/demo", settings, base_dir=tmp_path / "sessions")


def _read(log: SessionLog) -> list:
    lines = log.path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_记录用户工具与助手(tmp_path):
    log = _log(tmp_path)
    log.record_user("看看这个项目")
    log.record_tools(["read_file", "grep_code"])
    log.record_assistant("项目是 Java CLI")

    entries = _read(log)
    assert [e["role"] for e in entries] == ["user", "tool", "tool", "assistant"]
    assert entries[0]["turn"] == 1
    assert entries[0]["text"] == "看看这个项目"
    assert [e["name"] for e in entries[1:3]] == ["read_file", "grep_code"]
    # 只记工具名，不落参数与输出
    assert "args" not in entries[1] and "output" not in entries[1]
    assert entries[3]["text"] == "项目是 Java CLI"
    assert all(isinstance(e["ts"], float) for e in entries)


def test_每轮turn自增(tmp_path):
    log = _log(tmp_path)
    log.record_user("第一轮")
    log.record_assistant("回答一")
    log.record_user("第二轮")
    log.record_assistant("回答二")

    entries = _read(log)
    assert [e["turn"] for e in entries] == [1, 1, 2, 2]


def test_超长文本截断(tmp_path):
    log = _log(tmp_path)
    log.record_user("x" * (MAX_TEXT_CHARS + 500))
    entry = _read(log)[0]
    assert len(entry["text"]) == MAX_TEXT_CHARS + 1
    assert entry["text"].endswith("…")


def test_关闭开关不落盘(tmp_path):
    log = _log(tmp_path, ASTERCLI_SESSION_LOG="false")
    log.record_user("不应写入")
    log.record_assistant("也不写")
    assert log.enabled is False
    assert not log.path.exists()


def test_目录按项目哈希隔离(tmp_path):
    log = _log(tmp_path)
    log.record_user("hello")
    expected = tmp_path / "sessions" / project_hash("/proj/demo")
    assert log.directory == expected
    assert log.path.is_file()

    other = SessionLog("/proj/other", base_dir=tmp_path / "sessions")
    assert other.directory != log.directory


def test_会话枚举与新增计数(tmp_path):
    base = tmp_path / "sessions"
    first = SessionLog("/proj/demo", base_dir=base, session_id="20260101-000000-aaaaaa")
    first.record_user("旧会话")
    second = SessionLog("/proj/demo", base_dir=base, session_id="20260102-000000-bbbbbb")
    second.record_user("新会话")

    old_ts = 1_700_000_000.0
    os.utime(first.path, (old_ts, old_ts))
    new_ts = old_ts + 3600
    os.utime(second.path, (new_ts, new_ts))

    current = SessionLog("/proj/demo", base_dir=base, session_id="20260103-000000-cccccc")
    current.record_user("当前会话")

    sessions = current.list_sessions()
    assert [s.session_id for s in sessions][0] == "20260103-000000-cccccc"
    assert len(sessions) == 3

    # 默认排除当前会话，只统计 last_dream 之后有更新的会话
    newer = current.sessions_since(old_ts)
    assert [s.session_id for s in newer] == ["20260102-000000-bbbbbb"]
    assert current.sessions_since(new_ts) == []


def test_读取会话与非法id(tmp_path):
    log = _log(tmp_path)
    log.record_user("可读内容")
    assert len(log.read_session(log.session_id)) == 1
    assert log.read_session("../etc/passwd") == []
    assert log.read_session("no-such-session") == []


def test_损坏行被跳过(tmp_path):
    log = _log(tmp_path)
    log.record_user("正常一行")
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    entries = log.read_session(log.session_id)
    assert len(entries) == 1
    assert entries[0]["text"] == "正常一行"


def test_默认目录来自DreamPaths(tmp_path):
    paths = DreamPaths.for_project("/proj/demo", session_root=tmp_path / "sessions")
    log = SessionLog("/proj/demo", base_dir=tmp_path / "sessions")
    assert log.directory == paths.session_dir
