"""内置工具测试：read_file / write_file / list_dir / glob_files / grep_code / execute_command。"""

from __future__ import annotations

from pathlib import Path

import pytest

from astercli.config import Settings
from astercli.policy import PathGuard
from astercli.tools.builtin.files import register_file_tools
from astercli.tools.builtin.shell import register_shell_tools
from astercli.tools.registry import ToolRegistry


@pytest.fixture
def registry(guard):
    r = ToolRegistry(settings=Settings(load_files=False), approval=None)
    register_file_tools(r, guard)
    register_shell_tools(r, guard)
    return r


def invoke(registry, name, args):
    tool = registry.get(name)
    assert tool is not None, f"工具未注册: {name}"
    return tool.func(args)


async def invoke_via_registry(registry, name, args):
    """走 execute_all 主路径：PolicyException 应转成 🛡️ 失败文本回给 LLM。"""
    results = await registry.execute_all([{"name": name, "args": args, "id": "call"}])
    return results[0].content


# ---------- read_file ----------


def test_read_file_full_content(registry, project):
    out = invoke(registry, "read_file", {"path": "README.md"})
    assert out.startswith("文件内容:")
    assert "hello world" in out


def test_read_file_by_line_range(registry, project):
    out = invoke(registry, "read_file", {"path": "src/Main.java", "offset": 2, "limit": 2})
    assert "lines 2-3 of 5" in out
    assert "    public static void main" in out
    assert "文件内容: Main.java" in out


def test_read_file_offset_out_of_range(registry, project):
    out = invoke(registry, "read_file", {"path": "src/Main.java", "offset": 99})
    assert "offset 超出范围" in out


def test_read_file_truncated_hint(registry, project):
    out = invoke(registry, "read_file", {"path": "src/Main.java", "limit": 2})
    assert "已截断" in out
    assert "offset=3" in out


def test_read_file_missing_returns_failure(registry, project):
    out = invoke(registry, "read_file", {"path": "nope.md"})
    assert "读取文件失败" in out


async def test_read_file_escape_rejected(registry, project):
    out = await invoke_via_registry(registry, "read_file", {"path": "../../etc/passwd"})
    assert out.startswith("🛡️")
    assert "路径越界" in out


# ---------- write_file ----------


def test_write_file_creates_and_confirms(registry, project):
    out = invoke(registry, "write_file", {"path": "docs/new.md", "content": "# hi"})
    assert out == "文件已写入: docs/new.md"
    assert (project / "docs" / "new.md").read_text(encoding="utf-8") == "# hi"


async def test_write_file_5mb_limit(registry, project):
    out = await invoke_via_registry(
        registry, "write_file", {"path": "big.txt", "content": "x" * (5 * 1024 * 1024 + 1)}
    )
    assert out.startswith("🛡️")
    assert "超过 5MB 上限" in out


async def test_write_file_escape_rejected(registry, project):
    out = await invoke_via_registry(registry, "write_file", {"path": "/tmp/evil.txt", "content": "x"})
    assert out.startswith("🛡️")
    assert "路径越界" in out


# ---------- list_dir ----------


def test_list_dir_format(registry, project):
    out = invoke(registry, "list_dir", {"path": "."})
    assert out.startswith("目录内容:")
    assert "[D] src" in out
    assert "[F] README.md" in out


def test_list_dir_missing(registry, project):
    out = invoke(registry, "list_dir", {"path": "no-such-dir"})
    assert "目录为空或不存在" in out


# ---------- glob_files ----------


def test_glob_files_recursive_pattern(registry, project):
    out = invoke(registry, "glob_files", {"pattern": "**/*.java"})
    assert "匹配文件 2 个" in out
    assert "src/Main.java" in out
    assert "src/Util.java" in out


def test_glob_files_name_match(registry, project):
    out = invoke(registry, "glob_files", {"pattern": "README.md"})
    assert "README.md" in out


def test_glob_files_no_match(registry, project):
    out = invoke(registry, "glob_files", {"pattern": "**/*.xyz"})
    assert "未找到匹配文件" in out


def test_glob_files_max_results_cap(registry, project):
    for i in range(5):
        (project / f"f{i}.txt").write_text("x", encoding="utf-8")
    out = invoke(registry, "glob_files", {"pattern": "f*.txt", "max_results": 3})
    assert "匹配文件 3 个" in out
    assert "已达到上限 3" in out


def test_glob_files_skips_excluded_dirs(registry, project):
    (project / "node_modules").mkdir()
    (project / "node_modules" / "skip.js").write_text("x", encoding="utf-8")
    out = invoke(registry, "glob_files", {"pattern": "**/*.js"})
    assert "skip.js" not in out


# ---------- grep_code ----------


def test_grep_literal_default(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "hello"})
    assert "(engine=python)" in out
    # README 第 3 行是 "hello world"；Main.java 第 3 行 println("hello")
    assert "README.md:3" in out
    assert "src/Main.java:3" in out
    assert "suggested_reads:" in out


def test_grep_regex_mode(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "System\\.out", "regex": True})
    assert "src/Main.java:3" in out


def test_grep_case_insensitive(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "HELLO", "case_sensitive": False})
    assert "README.md" in out


def test_grep_case_sensitive_no_hit(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "HELLO"})
    assert "未找到匹配内容" in out


def test_grep_glob_filter(registry, project):
    # "demo" 只在 README.md，被 *.java 过滤后应无命中
    out = invoke(registry, "grep_code", {"pattern": "demo", "glob": "**/*.java"})
    assert "未找到匹配内容" in out


def test_grep_context_lines(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "greet", "context_lines": 1})
    assert ">    2" in out or ">" in out
    assert "src/Util.java:2" in out


def test_grep_invalid_regex(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "(", "regex": True})
    assert "无效正则" in out


def test_grep_head_limit_partial(registry, project):
    # 单文件 6 行命中、head_limit=2 → partial 提示
    (project / "many.txt").write_text("hit\n" * 6, encoding="utf-8")
    out = invoke(registry, "grep_code", {"pattern": "hit", "head_limit": 2})
    assert "partial: true" in out
    assert "head_limit=2" in out


def test_grep_max_results_partial(registry, project):
    for i in range(4):
        (project / f"m{i}.txt").write_text("needle\n", encoding="utf-8")
    out = invoke(registry, "grep_code", {"pattern": "needle", "max_results": 2})
    assert "partial: true" in out
    assert "max_results=2" in out


def test_grep_suggested_reads_format(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "greet"})
    assert '- read_file {"path":"src/Util.java","offset":1,"limit":80}' in out


def test_grep_empty_pattern(registry, project):
    out = invoke(registry, "grep_code", {"pattern": "  "})
    assert "pattern 不能为空" in out


def test_grep_max_chars_truncation(registry, project):
    # 多条长命中行 + 小 max_chars → 字符预算截断提示
    (project / "wide.txt").write_text(
        "\n".join(f"needle {'x' * 100}" for _ in range(20)), encoding="utf-8"
    )
    out = invoke(registry, "grep_code", {"pattern": "needle", "max_chars": 1000})
    assert "partial: true" in out
    assert "max_chars=1000" in out


# ---------- execute_command ----------


def test_execute_command_echo(registry, project):
    out = invoke(registry, "execute_command", {"command": "echo hello-from-shell"})
    assert out.startswith("命令输出 (exit 0)")
    assert "hello-from-shell" in out


async def test_execute_command_blacklisted(registry, project):
    out = await invoke_via_registry(registry, "execute_command", {"command": "sudo rm -rf /"})
    assert out.startswith("🛡️ 命令被拒绝")
    assert "禁止" in out


def test_execute_command_nonzero_exit(registry, project):
    out = invoke(
        registry, "execute_command", {"command": f'exit 3'}
    )
    assert "exit 3" in out


def test_execute_command_runs_in_project_root(registry, project):
    out = invoke(registry, "execute_command", {"command": "basename \"$PWD\""})
    assert out.startswith("命令输出 (exit 0)")
    assert project.name in out


def test_execute_command_output_truncated():
    guard = PathGuard(Path(__file__).resolve().parents[1])
    r = ToolRegistry(settings=Settings(load_files=False))
    register_shell_tools(r, guard)
    out = r.get("execute_command").func({"command": "python3 -c \"print('z'*20000)\""})
    assert "⚠️ 输出超长已截断" in out
    assert len(out) < 20_000
