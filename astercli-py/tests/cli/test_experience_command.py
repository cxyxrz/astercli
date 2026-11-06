"""/experience 命令展示函数测试（纯函数，不启动 REPL）。"""

from __future__ import annotations

import time

from astercli.cli.main import (
    _parse_experience_scope,
    format_experience_detail,
    format_experience_entries,
)
from astercli.experience import ExperienceEntry


def _entry(lesson: str = "先确认目录", **kwargs) -> ExperienceEntry:
    defaults = dict(
        id="exp-abcdef12",
        lesson=lesson,
        correct_action="glob_files 检查目录",
        tool="write_file",
        created=time.time(),
    )
    defaults.update(kwargs)
    return ExperienceEntry(**defaults)


def test_空列表提示自动沉淀():
    text = format_experience_entries("列表", [])
    assert "没有经验记录" in text
    assert "自动沉淀" in text


def test_经验列表展示作用域分类与命中():
    entries = [_entry(), _entry("全局教训", id="exp-global01", scope="global", hits=3)]
    text = format_experience_entries("列表", entries)
    assert "[exp-abcdef12]" in text
    assert "hits=1" in text
    assert "scope=global" not in text  # 只展示 scope 值，不展示字段名
    assert "(global · error-lesson · tool_failure · hits=3" in text


def test_经验详情包含四要素():
    entry = _entry("写文件前先检查路径是否存在")
    text = format_experience_detail(entry, "/tmp/exp.md")
    assert "教训: 写文件前先检查路径是否存在" in text
    assert "正确做法: glob_files 检查目录" in text
    assert "错误教训" in text          # 分类中文标签
    assert "工具失败" in text          # 触发中文标签
    assert "/tmp/exp.md" in text


def test_解析作用域参数():
    assert _parse_experience_scope([]) == ""
    assert _parse_experience_scope(["--global"]) == "global"
    assert _parse_experience_scope(["--project"]) == "project"
    assert _parse_experience_scope(["global"]) == "global"
