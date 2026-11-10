"""ConsoleRenderer 行为单测：折叠块、代码围栏、diff、状态栏联动（P11）。"""

from io import StringIO

from langchain_core.messages import AIMessageChunk, ToolMessage
from rich.console import Console

from astercli.cli.renderer import ConsoleRenderer
from astercli.cli.status_bar import StatusState


def _renderer(status=None):
    stream = StringIO()
    console = Console(file=stream, width=100, force_terminal=False, highlight=False, markup=False)
    return ConsoleRenderer(console, status=status), stream


def _chunk(content="", reasoning=""):
    kwargs = {"reasoning_content": reasoning} if reasoning else {}
    return AIMessageChunk(content=content, additional_kwargs=kwargs)


def test_工具调用渲染折叠标题与结果行():
    renderer, stream = _renderer()
    renderer.begin_turn()
    renderer.tool_calls([{"name": "read_file", "args": {"path": "src/a.py"}}])
    renderer.tool_results(
        [], [ToolMessage(content="文件内容: ...", name="read_file", tool_call_id="1")]
    )
    output = stream.getvalue()
    assert "ReadFile(src/a.py)" in output
    assert "ctrl+o" in output
    assert "✓ read_file" in output


def test_工具失败结果用叉号():
    renderer, stream = _renderer()
    renderer.begin_turn()
    renderer.tool_results(
        [], [ToolMessage(content="工具执行失败: boom", name="write_file", tool_call_id="1")]
    )
    assert "✗ write_file" in stream.getvalue()


def test_代码围栏折叠成块():
    renderer, stream = _renderer()
    renderer.begin_turn()
    renderer.stream_chunk(_chunk("```python\nprint(1)\nprint(2)\n```\n"))
    output = stream.getvalue()
    assert "code: python (2 行" in output
    assert "print(1)" not in output


def test_未闭合围栏按原文输出():
    renderer, stream = _renderer()
    renderer.begin_turn()
    renderer.stream_chunk(_chunk("```python\nprint(1)\n"))
    renderer.end_thinking()
    output = stream.getvalue()
    assert "print(1)" in output


def test_非终端下推理落到引用块():
    renderer, stream = _renderer()
    renderer.begin_turn()
    renderer.begin_thinking()
    renderer.stream_chunk(_chunk(reasoning="先读文件"))
    renderer.end_thinking()
    assert "│ 先读文件" in stream.getvalue()


def test_diff渲染颜色标记():
    renderer, stream = _renderer()
    renderer.begin_turn()
    renderer.diff("a.py", "old\n", "new\n")
    output = stream.getvalue()
    assert "📝 a.py" in output
    assert "-old" in output
    assert "+new" in output


def test_usage写入状态栏而不打印():
    status = StatusState(model="glm/glm-5.1")
    renderer, stream = _renderer(status=status)

    class _Budget:
        total_input_tokens = 1200
        total_output_tokens = 340
        total_cached_input_tokens = 50
        iteration = 2

    renderer.usage(_Budget())
    assert status.input_tokens == 1200
    assert status.output_tokens == 340
    assert status.cached_tokens == 50
    assert stream.getvalue() == ""


def test_折叠块可展开重绘():
    renderer, _ = _renderer()
    renderer.begin_turn()
    renderer.tool_calls([{"name": "read_file", "args": {"path": "src/a.py"}}])
    assert renderer.toggle_last_block() is True
    block = renderer._blocks[-1]
    assert block.expanded is True
    assert any("└ src/a.py" in line for line in block.lines())


def test_无折叠块时toggle返回False():
    renderer, _ = _renderer()
    renderer.begin_turn()
    assert renderer.toggle_last_block() is False


def test_阶段流转写入状态栏():
    status = StatusState(model="x")
    renderer, _ = _renderer(status=status)
    renderer.begin_turn()
    assert status.phase == "thinking"
    renderer.tool_calls([{"name": "read_file", "args": {"path": "a"}}])
    assert status.phase == "tools"
    renderer.tool_results([], [ToolMessage(content="ok", name="read_file", tool_call_id="1")])
    assert status.phase == "thinking"
    renderer.end_thinking()
    assert status.phase in {"tools", "idle"}
