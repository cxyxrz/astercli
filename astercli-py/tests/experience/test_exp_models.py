"""经验数据模型测试：Markdown 往返、签名、脱敏工具。"""

from __future__ import annotations

from astercli.experience import (
    CATEGORY_ERROR_LESSON,
    CATEGORY_USER_CORRECTION,
    TRIGGER_HITL_REJECT,
    TRIGGER_LSP_ERROR,
    TRIGGER_TOOL_FAILURE,
    ExperienceEntry,
    ReflectionTrigger,
    args_skeleton,
    first_line,
)


# ---------- 文本工具 ----------


def test_first_line_取首行并限长():
    assert first_line("第一行\n第二行") == "第一行"
    assert first_line("") == ""
    long = "x" * 300
    out = first_line(long, limit=20)
    assert len(out) == 21  # 20 字符 + 省略号
    assert out.endswith("…")


def test_args_skeleton_截断长值且排序():
    skeleton = args_skeleton({"path": "src/Main.java", "content": "x" * 500})
    assert "content=" in skeleton
    assert "path=" in skeleton
    # 值被截断，不把整个文件内容带进反思提示词
    assert "x" * 500 not in skeleton
    assert len(skeleton) <= 121


def test_args_skeleton_非法输入返回空串():
    assert args_skeleton(None) == ""
    assert args_skeleton("not-a-dict") == ""


# ---------- 触发信号 ----------


def test_触发签名_同信号稳定_不同信号区分():
    a = ReflectionTrigger(kind=TRIGGER_TOOL_FAILURE, tool_name="read_file", error="文件不存在")
    b = ReflectionTrigger(kind=TRIGGER_TOOL_FAILURE, tool_name="read_file", error="文件不存在")
    c = ReflectionTrigger(kind=TRIGGER_TOOL_FAILURE, tool_name="write_file", error="文件不存在")
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_触发描述_包含关键信息():
    trigger = ReflectionTrigger(
        kind=TRIGGER_HITL_REJECT,
        tool_name="execute_command",
        reason="不想删除文件",
        args_skeleton="cmd='rm -rf x'",
    )
    text = trigger.describe()
    assert "hitl_reject" in text
    assert "execute_command" in text
    assert "不想删除文件" in text


def test_lsp_触发证据带文件与行号():
    trigger = ReflectionTrigger(
        kind=TRIGGER_LSP_ERROR, file_path="broken.py", line=3, message="invalid syntax"
    )
    assert "broken.py:3" in trigger.evidence()


# ---------- 经验条目 ----------


def _entry(**kwargs) -> ExperienceEntry:
    defaults = dict(
        id="exp-test0001",
        lesson="写文件前先确认父目录存在",
        correct_action="先用 glob_files 检查目录，再 write_file",
        category=CATEGORY_ERROR_LESSON,
        trigger=TRIGGER_TOOL_FAILURE,
        evidence="execute_command: no such file or directory",
        scope="project",
        tool="write_file",
    )
    defaults.update(kwargs)
    return ExperienceEntry(**defaults)


def test_经验条目_markdown往返():
    entry = _entry()
    text = entry.to_markdown()
    assert text.startswith("---\n")
    assert "## 教训" in text
    assert "## 正确做法" in text
    restored = ExperienceEntry.from_markdown(text)
    assert restored is not None
    assert restored.id == entry.id
    assert restored.lesson == entry.lesson
    assert restored.correct_action == entry.correct_action
    assert restored.category == entry.category
    assert restored.trigger == entry.trigger
    assert restored.evidence == entry.evidence
    assert restored.scope == "project"
    assert restored.tool == "write_file"


def test_经验条目_缺少教训段解析失败():
    text = "---\nid: exp-x\ncategory: error-lesson\ntrigger: tool_failure\n---\n\n# 空\n"
    assert ExperienceEntry.from_markdown(text) is None


def test_经验条目_非法分类回退到触发默认值():
    entry = _entry(category="unknown")
    assert entry.category == CATEGORY_ERROR_LESSON
    entry2 = ExperienceEntry(
        id="exp-test0002",
        lesson="用户偏好",
        trigger=TRIGGER_HITL_REJECT,
        category="unknown",
    )
    assert entry2.category == CATEGORY_USER_CORRECTION


def test_经验条目_内容签名与命中数无关():
    a = _entry(hits=1)
    b = _entry(hits=9)
    assert a.content_signature() == b.content_signature()


def test_经验条目_多行正文安全落盘():
    entry = _entry(lesson="第一句\n## 伪造标题\n第二句")
    restored = ExperienceEntry.from_markdown(entry.to_markdown())
    # 正文折叠单行落盘：用户内容里的二级标题不会破坏段落解析
    assert restored is not None
    assert restored.lesson == "第一句 ## 伪造标题 第二句"
