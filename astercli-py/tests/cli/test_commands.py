"""CLI 命令纯函数测试：/save 负载解析、长期记忆列表格式。"""

from __future__ import annotations

from astercli.cli.main import format_memory_entries, parse_save_payload
from astercli.memory.long_term import MemoryEntry


# ---------- parse_save_payload ----------


def test_parse_save_project_default():
    assert parse_save_payload("这个项目使用 Java 17") == ("project", "这个项目使用 Java 17")


def test_parse_save_global_flag():
    assert parse_save_payload("--global 默认用中文回答") == ("global", "默认用中文回答")


def test_parse_save_empty():
    assert parse_save_payload("") == ("project", "")
    assert parse_save_payload("   ") == ("project", "")


def test_parse_save_global_without_content():
    assert parse_save_payload("--global") == ("global", "")


# ---------- format_memory_entries ----------


def test_format_empty_entries():
    out = format_memory_entries("📋 长期记忆列表", [])
    assert "📭 没有长期记忆" in out


def test_format_entries_line():
    entry = MemoryEntry(
        id="fact-abcd1234",
        content="这个项目使用 Java 17",
        metadata={"scope": "project", "project": "/proj/a"},
        token_count=6,
    )
    out = format_memory_entries("📋 长期记忆列表", [entry])
    assert "fact-abcd1234" in out
    assert "(project)" in out
    assert "这个项目使用 Java 17" in out


def test_format_entries_truncates_long_content():
    entry = MemoryEntry(
        id="fact-long0001",
        content="x" * 200,
        metadata={"scope": "global"},
        token_count=50,
    )
    out = format_memory_entries("标题", [entry])
    assert "..." in out
    assert "x" * 100 not in out  # 截断到 80 + 省略号
