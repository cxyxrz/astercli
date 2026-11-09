"""经验条目与反思触发信号的数据模型（P9 层一 + 层二）。

设计要点：
- 反思触发三类：工具失败（tool_failure）/ HITL 用户拒绝（hitl_reject）/
  LSP 语法错误（lsp_error）
- 经验条目落盘为独立 Markdown：frontmatter 存元数据、正文四段存内容，
  便于人工阅读、git 共享与 /experience 精确删除
- 去重分两层：触发侧签名（kind + 工具名 + 错误首行 + 参数骨架）用于单会话
  去重；内容侧签名（category + 工具 + lesson）用于跨会话落盘合并（hits +1）

与 Java 版关系：Java 侧暂无经验子系统，本模块为 Python 版的独立设计，
数据模型与 Skill frontmatter 风格保持一致（不引入 PyYAML）。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------- 分类与触发类型 ----------

CATEGORY_ERROR_LESSON = "error-lesson"
CATEGORY_USER_CORRECTION = "user-correction"
CATEGORY_WORKFLOW = "workflow"
VALID_CATEGORIES = (CATEGORY_ERROR_LESSON, CATEGORY_USER_CORRECTION, CATEGORY_WORKFLOW)

CATEGORY_LABELS = {
    CATEGORY_ERROR_LESSON: "错误教训",
    CATEGORY_USER_CORRECTION: "用户纠错",
    CATEGORY_WORKFLOW: "工作流",
}

TRIGGER_TOOL_FAILURE = "tool_failure"
TRIGGER_HITL_REJECT = "hitl_reject"
TRIGGER_LSP_ERROR = "lsp_error"
TRIGGER_DREAM = "dream"
VALID_TRIGGERS = (TRIGGER_TOOL_FAILURE, TRIGGER_HITL_REJECT, TRIGGER_LSP_ERROR, TRIGGER_DREAM)

TRIGGER_LABELS = {
    TRIGGER_TOOL_FAILURE: "工具失败",
    TRIGGER_HITL_REJECT: "审批拒绝",
    TRIGGER_LSP_ERROR: "语法诊断",
    TRIGGER_DREAM: "后台整合",
}

DEFAULT_TRIGGER_CATEGORY = {
    TRIGGER_TOOL_FAILURE: CATEGORY_ERROR_LESSON,
    TRIGGER_LSP_ERROR: CATEGORY_ERROR_LESSON,
    TRIGGER_HITL_REJECT: CATEGORY_USER_CORRECTION,
    TRIGGER_DREAM: CATEGORY_WORKFLOW,
}


# ---------- 文本工具 ----------


def first_line(text: str, limit: int = 160) -> str:
    """取首行并限长（错误摘要只存首行，避免整段输出进经验库）。"""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""
    head = normalized.split("\n", 1)[0].strip()
    if len(head) > limit:
        head = head[:limit] + "…"
    return head


def args_skeleton(args: object, max_len: int = 120) -> str:
    """参数骨架：只保留 key + 截断值，避免把整个文件内容写进反思提示词。"""
    if not isinstance(args, dict) or not args:
        return ""
    parts = []
    for key in sorted(args, key=lambda k: str(k)):
        value = args[key]
        if isinstance(value, str):
            text = value if len(value) <= 24 else value[:24] + "…"
            parts.append(f"{key}={text!r}")
        else:
            text = str(value)
            if len(text) > 40:
                text = text[:40] + "…"
            parts.append(f"{key}={text}")
    joined = ", ".join(parts)
    if len(joined) > max_len:
        joined = joined[:max_len] + "…"
    return joined


def _sha1_hex(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


# ---------- 触发信号 ----------


@dataclass
class ReflectionTrigger:
    """一次值得复盘的事件（由工具层 / 审批层 / LSP 层埋点产生）。"""

    kind: str
    tool_name: str = ""
    error: str = ""
    args_skeleton: str = ""
    reason: str = ""
    file_path: str = ""
    line: int = 0
    message: str = ""

    def signature(self) -> str:
        """触发侧去重签名（同一工具、同一错误首行、同一参数骨架只反思一次）。"""
        return _sha1_hex(
            "|".join(
                [
                    self.kind,
                    self.tool_name,
                    first_line(self.error, limit=120),
                    self.args_skeleton,
                ]
            )
        )

    def describe(self) -> str:
        """供反思提示词使用的紧凑描述（已脱敏，不含完整工具输出）。"""
        lines = [f"触发类型: {self.kind}（{TRIGGER_LABELS.get(self.kind, self.kind)}）"]
        if self.tool_name:
            lines.append(f"工具: {self.tool_name}")
        if self.error:
            lines.append(f"错误摘要: {first_line(self.error)}")
        if self.reason:
            lines.append(f"用户拒绝原因: {first_line(self.reason)}")
        if self.file_path:
            loc = f"{self.file_path}:{self.line}" if self.line else self.file_path
            lines.append(f"文件: {loc}")
        if self.message:
            lines.append(f"诊断消息: {first_line(self.message)}")
        if self.args_skeleton:
            lines.append(f"参数骨架: {self.args_skeleton}")
        return "\n".join(lines)

    def evidence(self, limit: int = 240) -> str:
        """落盘证据：只保留首个有效信号行，不存完整输出。"""
        if self.kind == TRIGGER_HITL_REJECT and self.reason:
            return first_line(self.reason, limit=limit)
        if self.kind == TRIGGER_LSP_ERROR:
            loc = f"{self.file_path}:{self.line}" if self.line else self.file_path
            return first_line(f"{loc} {self.message}".strip(), limit=limit)
        return first_line(self.error or self.message, limit=limit)


# ---------- 经验条目 ----------


@dataclass
class ExperienceEntry:
    """一条持久化经验（对应一个 exp-<id>.md）。"""

    id: str
    lesson: str
    correct_action: str = ""
    category: str = CATEGORY_ERROR_LESSON
    trigger: str = TRIGGER_TOOL_FAILURE
    evidence: str = ""
    scope: str = "project"
    project_key: str = ""
    tool: str = ""
    created: float = field(default_factory=time.time)
    hits: int = 1
    signature: str = ""

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            self.category = DEFAULT_TRIGGER_CATEGORY.get(self.trigger, CATEGORY_ERROR_LESSON)
        if self.trigger not in VALID_TRIGGERS:
            self.trigger = TRIGGER_TOOL_FAILURE
        if self.scope not in ("project", "global"):
            self.scope = "project"
        self.hits = max(1, int(self.hits))
        if not self.created:
            self.created = time.time()
        if not self.signature:
            self.signature = self.content_signature()

    # ---------- 签名 ----------

    def content_signature(self) -> str:
        """内容侧签名：跨会话合并同一条经验（hits +1）的依据。"""
        return _sha1_hex(
            "|".join(
                [
                    self.category,
                    self.tool,
                    self.lesson.strip().lower(),
                ]
            )
        )

    # ---------- Markdown 序列化 ----------

    def to_markdown(self) -> str:
        frontmatter = [
            "---",
            f"id: {self.id}",
            f"category: {self.category}",
            f"trigger: {self.trigger}",
            f"scope: {self.scope}",
            f"created: {int(self.created)}",
            f"hits: {self.hits}",
            f"tool: {self.tool}",
            f"signature: {self.signature}",
            "---",
            "",
            f"# 经验：{first_line(self.lesson, limit=60)}",
            "",
            "## 教训",
            _flatten(self.lesson),
            "",
            "## 正确做法",
            _flatten(self.correct_action) or "（未提炼）",
            "",
            "## 触发信号",
            self.trigger,
            "",
            "## 证据",
            _flatten(self.evidence) or "（无）",
            "",
        ]
        return "\n".join(frontmatter)

    @staticmethod
    def from_markdown(text: str) -> Optional["ExperienceEntry"]:
        """解析 exp-*.md；frontmatter 复用 Skill 的零依赖解析器（延迟导入避免循环）。"""
        if not text:
            return None
        # 延迟导入：registry → experience.models 的模块加载链不能触达 skill 包
        from ..skill.frontmatter import parse

        result = parse(text)
        meta = result.frontmatter
        entry_id = str(meta.get("id") or "").strip()
        category = str(meta.get("category") or "").strip()
        trigger = str(meta.get("trigger") or "").strip()
        lesson = _body_section(result.body, "教训")
        if not entry_id or not category or not trigger or not lesson:
            return None
        try:
            created = float(meta.get("created") or 0)
        except (TypeError, ValueError):
            created = 0.0
        try:
            hits = int(float(meta.get("hits") or 1))
        except (TypeError, ValueError):
            hits = 1
        return ExperienceEntry(
            id=entry_id,
            lesson=lesson,
            correct_action=_body_section(result.body, "正确做法"),
            category=category,
            trigger=trigger,
            evidence=_body_section(result.body, "证据"),
            scope=str(meta.get("scope") or "project").strip() or "project",
            tool=str(meta.get("tool") or "").strip(),
            created=created or time.time(),
            hits=hits,
            signature=str(meta.get("signature") or "").strip(),
        )


def _body_section(body: str, title: str) -> str:
    """取正文中 ``## <title>`` 段内容（到下一个二级标题为止）。"""
    marker = f"## {title}"
    idx = body.find(marker)
    if idx < 0:
        return ""
    start = idx + len(marker)
    end = body.find("\n## ", start)
    chunk = body[start:] if end < 0 else body[start:end]
    return chunk.strip()


def _flatten(text: str) -> str:
    """经验正文按单行落盘：折叠换行，避免用户输入里的标题破坏段落解析。"""
    return " ".join(line.strip() for line in (text or "").splitlines() if line.strip())
