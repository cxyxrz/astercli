"""Dream 提案（P10 M2）：LLM 输出的 ops 结构校验 + 落盘。

ops 是 dream 与记忆库之间唯一的写入口，因此在这里做"代码侧 allowlist 校验"：
- 只接受 6 种已知类型
- 引用的记忆 / 经验 id 必须真实存在
- 每条 op 的文本长度有硬上限
- 单次 dream 最多 20 条 op（防止一次改写过多）

校验失败只记录错误、丢弃该条 op，不让整份提案失败；全部 op 都被丢弃时
调用方应放弃本次 apply。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..experience import VALID_CATEGORIES, sanitize_text

OP_MEMORY_ADD = "memory_add"
OP_MEMORY_UPDATE = "memory_update"
OP_MEMORY_DELETE = "memory_delete"
OP_MEMORY_MERGE = "memory_merge"
OP_EXPERIENCE_ADD = "experience_add"
OP_EXPERIENCE_DELETE = "experience_delete"

VALID_OPS = (
    OP_MEMORY_ADD,
    OP_MEMORY_UPDATE,
    OP_MEMORY_DELETE,
    OP_MEMORY_MERGE,
    OP_EXPERIENCE_ADD,
    OP_EXPERIENCE_DELETE,
)

OP_LABELS = {
    OP_MEMORY_ADD: "新增长期记忆",
    OP_MEMORY_UPDATE: "更新长期记忆",
    OP_MEMORY_DELETE: "归档长期记忆",
    OP_MEMORY_MERGE: "合并长期记忆",
    OP_EXPERIENCE_ADD: "新增经验",
    OP_EXPERIENCE_DELETE: "归档经验",
}

MAX_OPS = 20
MAX_FACT_CHARS = 400
MAX_LESSON_CHARS = 200
MAX_ACTION_CHARS = 300
MAX_REASON_CHARS = 200
MAX_MERGE_IDS = 8

PROPOSAL_FILE = "proposal.md"
OPS_FILE = "ops.json"


@dataclass
class Proposal:
    """一次 dream 的完整提案。"""

    dream_id: str
    summary: str = ""
    ops: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw: str = ""
    model: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def op_count(self) -> int:
        return len(self.ops)

    def to_dict(self) -> dict:
        return {
            "dream_id": self.dream_id,
            "summary": self.summary,
            "ops": self.ops,
            "errors": self.errors,
            "model": self.model,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Optional["Proposal"]:
        if not isinstance(data, dict):
            return None
        dream_id = str(data.get("dream_id") or "").strip()
        if not dream_id:
            return None
        ops = data.get("ops")
        errors = data.get("errors")
        return cls(
            dream_id=dream_id,
            summary=str(data.get("summary") or ""),
            ops=[op for op in ops if isinstance(op, dict)] if isinstance(ops, list) else [],
            errors=[str(e) for e in errors] if isinstance(errors, list) else [],
            model=str(data.get("model") or ""),
            created_at=_as_float(data.get("created_at")),
        )

    # ---------- 落盘 ----------

    def save(self, directory: Path) -> bool:
        """写 proposal.md（人读）+ ops.json（机器回放）。"""
        target = Path(directory)
        try:
            target.mkdir(parents=True, exist_ok=True)
            (target / OPS_FILE).write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (target / PROPOSAL_FILE).write_text(self.to_markdown(), encoding="utf-8")
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, directory: Path) -> Optional["Proposal"]:
        try:
            data = json.loads((Path(directory) / OPS_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))
        lines = [
            f"# Dream 提案 {self.dream_id}",
            "",
            f"- 生成时间: {when}",
            f"- 模型: {self.model or '（未记录）'}",
            f"- 操作数: {len(self.ops)}",
            "",
            "## 摘要",
            self.summary or "（无）",
            "",
            "## 操作",
        ]
        if not self.ops:
            lines.append("（无）")
        for index, op in enumerate(self.ops, 1):
            lines.append(f"{index}. {describe_op(op)}")
        if self.errors:
            lines.extend(["", "## 校验问题"])
            for error in self.errors:
                lines.append(f"- {error}")
        lines.append("")
        return "\n".join(lines)


def describe_op(op: Dict[str, Any]) -> str:
    """单条 op 的一行人读描述（CLI / proposal.md 共用）。"""
    kind = str(op.get("type") or "")
    label = OP_LABELS.get(kind, kind)
    if kind == OP_MEMORY_ADD:
        return f"{label}（{op.get('scope', 'project')}）: {op.get('content', '')}"
    if kind in (OP_MEMORY_UPDATE, OP_MEMORY_DELETE):
        return f"{label} [{op.get('id', '')}]: {op.get('content', '')}".rstrip(": ")
    if kind == OP_MEMORY_MERGE:
        return f"{label} {op.get('ids', [])} → {op.get('content', '')}"
    if kind == OP_EXPERIENCE_ADD:
        return f"{label}（{op.get('category', '')}）: {op.get('lesson', '')}"
    if kind == OP_EXPERIENCE_DELETE:
        return f"{label} [{op.get('id', '')}]: {op.get('reason', '')}".rstrip(": ")
    return json.dumps(op, ensure_ascii=False)


# ---------- 校验 ----------


def normalize_ops(
    raw_ops: Any,
    *,
    memory_ids: Iterable[str],
    experience_ids: Iterable[str],
    project_key: str = "",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """把 LLM 原始 ops 归一化成"只含合法字段"的字典列表。

    返回 ``(合法 ops, 错误列表)``；错误只用于人读日志，不阻断其余合法 op。
    """
    if not isinstance(raw_ops, list):
        return [], ["ops 不是数组"]

    memory_set: Set[str] = {str(i) for i in memory_ids}
    experience_set: Set[str] = {str(i) for i in experience_ids}
    ops: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_adds: Set[str] = set()

    for index, item in enumerate(raw_ops):
        if len(ops) >= MAX_OPS:
            errors.append(f"超过单次上限 {MAX_OPS} 条，丢弃第 {index + 1} 条及之后")
            break
        if not isinstance(item, dict):
            errors.append(f"第 {index + 1} 条不是对象，已丢弃")
            continue

        kind = str(item.get("type") or "").strip()
        if kind not in VALID_OPS:
            errors.append(f"第 {index + 1} 条类型未知: {kind or '（空）'}")
            continue

        if kind == OP_MEMORY_ADD:
            content = _text(item.get("content"), MAX_FACT_CHARS, project_key)
            if not content:
                errors.append(f"第 {index + 1} 条 memory_add 内容为空")
                continue
            if content in seen_adds:
                errors.append(f"第 {index + 1} 条 memory_add 与本次提案重复")
                continue
            seen_adds.add(content)
            ops.append(
                {
                    "type": kind,
                    "scope": _scope(item.get("scope")),
                    "content": content,
                }
            )

        elif kind == OP_MEMORY_UPDATE:
            entry_id = str(item.get("id") or "").strip()
            content = _text(item.get("content"), MAX_FACT_CHARS, project_key)
            if entry_id not in memory_set:
                errors.append(f"第 {index + 1} 条 memory_update 引用不存在的记忆: {entry_id}")
                continue
            if not content:
                errors.append(f"第 {index + 1} 条 memory_update 内容为空")
                continue
            ops.append({"type": kind, "id": entry_id, "content": content})

        elif kind == OP_MEMORY_DELETE:
            entry_id = str(item.get("id") or "").strip()
            if entry_id not in memory_set:
                errors.append(f"第 {index + 1} 条 memory_delete 引用不存在的记忆: {entry_id}")
                continue
            ops.append({"type": kind, "id": entry_id})

        elif kind == OP_MEMORY_MERGE:
            ids = _merge_ids(item.get("ids"), memory_set)
            content = _text(item.get("content"), MAX_FACT_CHARS, project_key)
            if len(ids) < 2:
                errors.append(f"第 {index + 1} 条 memory_merge 需要至少 2 个有效 id")
                continue
            if not content:
                errors.append(f"第 {index + 1} 条 memory_merge 内容为空")
                continue
            ops.append(
                {
                    "type": kind,
                    "ids": ids,
                    "scope": _scope(item.get("scope")),
                    "content": content,
                }
            )

        elif kind == OP_EXPERIENCE_ADD:
            lesson = _text(item.get("lesson"), MAX_LESSON_CHARS, project_key)
            category = str(item.get("category") or "").strip()
            if not lesson:
                errors.append(f"第 {index + 1} 条 experience_add 缺少 lesson")
                continue
            if category not in VALID_CATEGORIES:
                errors.append(f"第 {index + 1} 条 experience_add 分类未知: {category}")
                continue
            ops.append(
                {
                    "type": kind,
                    "category": category,
                    "lesson": lesson,
                    "correct_action": _text(
                        item.get("correct_action"), MAX_ACTION_CHARS, project_key
                    ),
                }
            )

        elif kind == OP_EXPERIENCE_DELETE:
            entry_id = str(item.get("id") or "").strip()
            if entry_id not in experience_set:
                errors.append(f"第 {index + 1} 条 experience_delete 引用不存在的经验: {entry_id}")
                continue
            ops.append(
                {
                    "type": kind,
                    "id": entry_id,
                    "reason": _text(item.get("reason"), MAX_REASON_CHARS, project_key),
                }
            )

    return ops, errors


def parse_proposal_output(text: str) -> Optional[dict]:
    """解析 LLM 返回的 JSON 对象（容忍 ```json 包裹与前后闲聊）。"""
    if not text or not text.strip():
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _text(value: Any, limit: int, project_key: str) -> str:
    if value is None:
        return ""
    return sanitize_text(str(value), project_key)[:limit].strip()


def _scope(value: Any) -> str:
    return "global" if str(value or "").strip().lower() == "global" else "project"


def _merge_ids(value: Any, memory_set: Set[str]) -> List[str]:
    if not isinstance(value, list):
        return []
    ids: List[str] = []
    for raw in value:
        entry_id = str(raw or "").strip()
        if entry_id in memory_set and entry_id not in ids:
            ids.append(entry_id)
        if len(ids) >= MAX_MERGE_IDS:
            break
    return ids


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
