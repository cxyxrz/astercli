"""底部状态栏单测（P11）：两行文案、上下文百分比、宽度裁剪、颜色片段。"""

from astercli.cli.status_bar import StatusState, format_status_lines, toolbar_fragments


def test_两行文案包含模式与模型():
    state = StatusState(
        model="glm/glm-5.1",
        phase="idle",
        mode="HITL 审批",
        mcp_summary="MCP 2/3",
        skill_summary="Skill 4",
        context_tokens=84000,
        context_window=200000,
        cwd="/tmp/demo",
    )
    first, second = format_status_lines(state, 100)
    assert "HITL 审批" in first
    assert "MCP 2/3" in first and "Skill 4" in first
    assert "glm-5.1" in second
    assert "ctx" in second
    assert "42%" in second
    assert "/tmp/demo" in second


def test_上下文百分比计算与边界():
    assert StatusState(context_tokens=50, context_window=0).context_percent() == 0
    assert StatusState(context_tokens=50, context_window=100).context_percent() == 50
    assert StatusState(context_tokens=300, context_window=100).context_percent() == 100


def test_token格式化与耗时():
    state = StatusState(
        input_tokens=1234,
        output_tokens=2_000_000,
        cached_tokens=600,
        elapsed_ms=72_000,
        context_window=1000,
        context_tokens=10,
    )
    second = format_status_lines(state, 200)[1]
    assert "in 1.2k" in second
    assert "out 2.0m" in second
    assert "cache 600" in second
    assert "1m12s" in second


def test_超宽裁剪到列宽():
    state = StatusState(model="很长的模型名字" * 20, cwd="/very/long/path/" * 10)
    first, second = format_status_lines(state, 40)
    from astercli.cli.render_blocks import display_width

    assert display_width(first) <= 40
    assert display_width(second) <= 40


def test_工具栏片段样式分段():
    state = StatusState(model="glm-5.1", phase="thinking", context_tokens=10, context_window=100)
    fragments = toolbar_fragments(state, 80)
    texts = [text for _, text in fragments]
    assert "\n" in texts
    assert any("glm-5.1" in text for text in texts)
    assert any("thinking" in text for text in texts)
    assert sum(len(text) for _, text in fragments) > 0
