"""渲染基础块单测：CJK 宽度、结果判定、行内 diff。"""

from astercli.cli.render_blocks import (
    compute_diff,
    display_width,
    fit_to_columns,
    group_hunks,
    render_diff,
    result_is_error,
    strip_ansi,
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
