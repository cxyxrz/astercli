"""渲染基础块单测：CJK 宽度、折叠块、工具文案、行内 diff（P11）。"""

from astercli.cli.render_blocks import (
    FoldableBlock,
    build_tool_block,
    collapsed_header,
    compute_diff,
    display_width,
    expanded_lines,
    fit_to_columns,
    group_calls,
    group_hunks,
    key_param,
    render_diff,
    result_is_error,
    strip_ansi,
    tool_label,
    wrapped_rows,
)


def test_显示宽度按CJK双列():
    assert display_width("abc") == 3
    assert display_width("读取文件") == 8
    assert display_width("\x1b[31m红\x1b[0m") == 2
    assert strip_ansi("\x1b[1mok\x1b[0m") == "ok"


def test_按列宽截断():
    assert fit_to_columns("abcdef", 10) == "abcdef"
    truncated = fit_to_columns("读取文件路径", 6)
    assert display_width(truncated) <= 6
    assert truncated.endswith("…")


def test_估算物理行数():
    assert wrapped_rows("abc", 10) == 1
    assert wrapped_rows("abcdefghij", 5) == 2
    assert wrapped_rows("a\nb", 10) == 2


def test_折叠块切换():
    block = FoldableBlock("⏵ header", ["line-1", "line-2"])
    assert block.lines() == ["⏵ header"]
    block.toggle()
    assert block.lines()[0] == "line-1"
    assert block.lines()[-1].startswith("⏷")
    block.toggle()
    assert block.lines() == ["⏵ header"]


def test_工具标签与关键参数():
    assert tool_label("read_file", 3) == "📖 读取 3 个文件"
    assert tool_label("mcp__demo__add", 1) == "🔌 调用 MCP 工具 demo.add"
    assert key_param("read_file", {"path": "src/a.py"}) == "src/a.py"
    assert key_param("web_fetch", {"url": "https://example.com/a/"}) == "example.com/a"
    assert key_param("read_file", {"path": "x" * 200}).endswith("...")


def test_单次调用折叠标题带具名形态():
    grouped = group_calls([{"name": "read_file", "args": {"path": "src/a.py"}}])
    header = collapsed_header(grouped)
    assert "ReadFile(src/a.py)" in header
    assert "ctrl+o" in header


def test_多次调用折叠标题按组统计():
    grouped = group_calls(
        [
            {"name": "read_file", "args": {"path": "a"}},
            {"name": "read_file", "args": {"path": "b"}},
            {"name": "grep_code", "args": {"pattern": "foo"}},
        ]
    )
    header = collapsed_header(grouped)
    assert "2 组工具调用 / 3 次" in header
    lines = expanded_lines(grouped)
    assert any("读取 2 个文件" in line for line in lines)
    assert any("grep_code" in line or "搜索代码" in line for line in lines)


def test_空调用不生成折叠块():
    assert build_tool_block([]) is None


def test_结果错误前缀识别():
    assert result_is_error("工具执行失败: boom")
    assert result_is_error("参数校验失败: missing")
    assert not result_is_error("文件已写入: a.py")


def test_行内diff_新增文件():
    lines = render_diff("a.py", None, "x = 1\ny = 2\n")
    assert lines[0].text.startswith("📝 a.py")
    assert any(item.kind == "add" and item.text == "+x = 1" for item in lines)
    assert not any(item.kind == "delete" for item in lines)


def test_行内diff_同内容只有提示():
    lines = render_diff("a.py", "same\n", "same\n")
    assert [item.kind for item in lines] == ["header", "meta"]


def test_行内diff_修改生成hunk():
    before = "\n".join(f"line{i}" for i in range(10))
    after = before.replace("line4", "changed")
    lines = render_diff("a.py", before, after)
    assert any(item.kind == "hunk" and item.text.startswith("@@") for item in lines)
    assert any(item.kind == "delete" and item.text == "-line4" for item in lines)
    assert any(item.kind == "add" and item.text == "+changed" for item in lines)


def test_diff_hunk数量受上下文窗口约束():
    before = [f"old{i}" for i in range(20)]
    after = list(before)
    after[3] = "new3"
    after[15] = "new15"
    hunks = group_hunks(compute_diff(before, after))
    assert len(hunks) == 2
