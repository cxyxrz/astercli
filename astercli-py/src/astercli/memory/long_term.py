"""长期记忆（对齐 Java LongTermMemory + MemoryRetriever 检索语义）。

- 持久化到 ``~/.astercli-py/memory/long_term_memory.json``（``ASTERCLI_MEMORY_DIR`` 可覆盖）
- 内容完全相同去重；scope=global 全局可见，project（默认）按项目根过滤
- 检索：关键词命中率 + 24h 时间衰减加权；分词为轻量实现（英文整词 + 中文 2-gram），
  不引入 jieba 依赖

与 Java 的已知差异：timestamp 用 epoch 秒（Java 是 ISO Instant），JSON 字段名保持一致。
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .tokens import estimate_tokens

STORAGE_FILE = "long_term_memory.json"
DEFAULT_MEMORY_DIR = Path.home() / ".astercli-py" / "memory"

_WORD_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")


@dataclass
class MemoryEntry:
    id: str
    content: str
    type: str = "FACT"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tokenCount": self.token_count,
        }

    @staticmethod
    def from_dict(data: dict) -> Optional["MemoryEntry"]:
        try:
            return MemoryEntry(
                id=str(data["id"]),
                content=str(data["content"]),
                type=str(data.get("type", "FACT")),
                timestamp=float(data.get("timestamp", time.time())),
                metadata={str(k): str(v) for k, v in (data.get("metadata") or {}).items()},
                token_count=int(data.get("tokenCount") or 0),
            )
        except (KeyError, TypeError, ValueError):
            return None


def normalize_project_key(path) -> str:
    """项目根 key：realpath 优先（存在时），否则 resolve 后的字符串。"""
    p = Path(path).resolve()
    try:
        if p.exists():
            return str(p.resolve())
    except OSError:
        pass
    return str(p)


def tokenize(query: str) -> set:
    """检索分词：英文/数字整词（≥2 字符），中文段切 2-gram（单字过滤）。"""
    tokens: set = set()
    for m in _WORD_RE.finditer((query or "").lower()):
        seg = m.group(0)
        if seg[0].isascii():
            if len(seg) >= 2:
                tokens.add(seg)
        else:
            for i in range(len(seg) - 1):
                tokens.add(seg[i : i + 2])
    return tokens


def _matches(text: str, tokens: set) -> bool:
    if not text or not tokens:
        return False
    lower = text.lower()
    return any(t in lower for t in tokens)


def is_visible_in_project(entry: MemoryEntry, project_key: Optional[str]) -> bool:
    """global 全局可见；project 作用域仅当前项目可见。"""
    scope = (entry.metadata.get("scope") or "global").lower()
    if scope == "global":
        return True
    project = entry.metadata.get("project")
    return bool(project_key) and project == project_key


def _relevance(entry: MemoryEntry, query: str, tokens: set) -> float:
    """对齐 Java MemoryRetriever.computeRelevanceScore：精确子串 1.0，
    否则关键词命中率 × 24h 时间衰减（1.0 → 0.5）。"""
    lower = entry.content.lower()
    if query and query.lower() in lower:
        return 1.0
    if not tokens:
        return 0.0
    matched = sum(1 for t in tokens if t in lower)
    if matched == 0:
        return 0.0
    keyword_score = matched / len(tokens)
    age_hours = max(0.0, (time.time() - entry.timestamp) / 3600.0)
    decay = max(0.5, 1.0 - age_hours / 24.0)
    return keyword_score * decay


class LongTermMemory:
    def __init__(self, storage_dir=None) -> None:
        configured = storage_dir or os.environ.get("ASTERCLI_MEMORY_DIR")
        self._dir = Path(configured) if configured else DEFAULT_MEMORY_DIR
        self._file = self._dir / STORAGE_FILE
        self._entries: Dict[str, MemoryEntry] = {}
        self._load()

    # ---------- 存储 ----------

    def store_fact(
        self, fact: str, scope: str = "project", project_key: Optional[str] = None
    ) -> Optional[MemoryEntry]:
        """保存事实到长期记忆；内容完全相同则跳过（返回 None）。"""
        fact = (fact or "").strip()
        if not fact:
            return None
        normalized = "global" if (scope or "").strip().lower() == "global" else "project"
        if any(e.content == fact for e in self._entries.values()):
            return None
        metadata = {"source": "fact", "scope": normalized}
        if normalized == "project":
            metadata["project"] = project_key or ""
        entry = MemoryEntry(
            id=f"fact-{uuid.uuid4().hex[:8]}",
            content=fact,
            type="FACT",
            metadata=metadata,
            token_count=estimate_tokens(fact),
        )
        self._entries[entry.id] = entry
        self._save()
        return entry

    def delete(self, entry_id: str) -> bool:
        if self._entries.pop(entry_id, None) is None:
            return False
        self._save()
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    # ---------- 检索 ----------

    def get_all(self) -> List[MemoryEntry]:
        """/memory list 用：列出全部（不按项目过滤，对齐 Java）。"""
        return list(self._entries.values())

    def search(self, query: str, limit: int = 20, project_key: Optional[str] = None) -> List[MemoryEntry]:
        """关键词命中（content 或 metadata）+ 相关度排序。"""
        tokens = tokenize(query)
        if not tokens:
            return []
        scored = []
        for e in self._entries.values():
            if not is_visible_in_project(e, project_key):
                continue
            score = _relevance(e, query, tokens)
            if score <= 0 and _matches(" ".join(e.metadata.values()), tokens):
                score = 0.01  # 元数据命中保底（对齐 Java search 的 content-or-metadata）
            if score > 0:
                scored.append((score * 1.2, e))  # 长期记忆小加权（对齐 Java）
        scored.sort(key=lambda p: -p[0])
        return [e for _, e in scored[:limit]]

    def status_summary(self) -> str:
        tokens = sum(e.token_count for e in self._entries.values())
        return f"长期记忆: {len(self._entries)} 条 / {tokens} tokens"

    # ---------- 持久化 ----------

    def _save(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            data = [e.to_dict() for e in self._entries.values()]
            self._file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # 持久化失败静默，不阻断主流程

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else []:
                entry = MemoryEntry.from_dict(item)
                if entry:
                    self._entries[entry.id] = entry
        except (OSError, json.JSONDecodeError):
            pass


def build_memory_context(
    memory: LongTermMemory, query: str, max_tokens: int, project_key: Optional[str] = None
) -> str:
    """构建"## 相关长期记忆"段落（对齐 Java MemoryRetriever.buildContextForQuery）。

    当前轮 user 输入已在 history 里，只注入长期事实，不注入短期对话，
    避免模型把当前请求误读成历史事实。
    """
    relevant = memory.search(query, limit=10, project_key=project_key)
    if not relevant:
        return ""
    lines = ["## 相关长期记忆", ""]
    used = 0
    added = 0
    for e in relevant:
        if used + e.token_count > max_tokens:
            break
        lines.append(f"- [{e.type}] {e.content}")
        used += e.token_count
        added += 1
    if added == 0:
        return ""  # 首条即超预算：不注入空段落
    lines.append("")
    return "\n".join(lines)
