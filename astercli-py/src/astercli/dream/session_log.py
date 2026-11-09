"""会话转写（P10 M1）：每轮只记最小信号，供后台 AutoDream 消费。

记录粒度（刻意保守）：
- ``user``：用户本轮原始输入（截断）
- ``tool``：被调用的工具名，**不记录参数与完整输出**
- ``assistant``：最终回答文本（截断）

这样转写既能让 dream 复原"用户做过什么、用过哪些工具、得到什么结论"，
又不会把大段代码 / 命令输出落盘成第二份上下文。

写入失败（只读文件系统、磁盘满等）静默忽略，绝不影响主任务。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from .models import DreamPaths, project_hash

if TYPE_CHECKING:  # pragma: no cover —— 仅类型引用
    from ..config import Settings

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2_000
MAX_TOOL_NAME_CHARS = 80
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def new_session_id() -> str:
    """可排序的会话 id：时间前缀 + 随机后缀（同一秒多次启动也不冲突）。"""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class SessionInfo:
    """会话目录里的一个 jsonl 文件。"""

    session_id: str
    path: Path
    mtime: float
    size: int


class SessionLog:
    """单会话 jsonl 追加写入器 + 会话目录读取工具。"""

    def __init__(
        self,
        project_key: str,
        settings: Optional["Settings"] = None,
        *,
        base_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.project_key = project_key
        configured = base_dir
        if configured is None and settings is not None:
            raw = settings.get("ASTERCLI_SESSION_LOG_DIR")
            configured = Path(raw) if raw else None
        root = Path(configured).expanduser() if configured else DreamPaths.for_project(project_key).session_dir.parent
        self._dir = root / project_hash(project_key)
        self.session_id = session_id or new_session_id()
        self._file = self._dir / f"{self.session_id}.jsonl"
        if enabled is not None:
            self._enabled = bool(enabled)
        elif settings is not None:
            self._enabled = settings.session_log_enabled()
        else:
            self._enabled = True
        self._turn = 0

    # ---------- 状态 ----------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._file

    @property
    def directory(self) -> Path:
        return self._dir

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    # ---------- 写入 ----------

    def record_user(self, text: str) -> None:
        """一轮开始：turn 自增并记录用户原始输入。"""
        self._turn += 1
        self._append({"role": "user", "turn": self._turn, "text": _clip(text, MAX_TEXT_CHARS)})

    def record_assistant(self, text: str) -> None:
        self._append(
            {"role": "assistant", "turn": self._turn, "text": _clip(text, MAX_TEXT_CHARS)}
        )

    def record_tool(self, name: str) -> None:
        tool = _clip((name or "").strip(), MAX_TOOL_NAME_CHARS)
        if tool:
            self._append({"role": "tool", "turn": self._turn, "name": tool})

    def record_tools(self, names: Iterable[str]) -> None:
        for name in names:
            self.record_tool(name)

    def _append(self, entry: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        payload = {"ts": round(time.time(), 3), **entry}
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            log.debug("会话转写写入失败: %s", self._file, exc_info=True)

    # ---------- 读取 ----------

    def list_sessions(self) -> List[SessionInfo]:
        """当前项目全部会话，按修改时间倒序（最新在前）。"""
        try:
            paths = list(self._dir.glob("*.jsonl"))
        except OSError:
            return []
        sessions: List[SessionInfo] = []
        for path in paths:
            session_id = path.stem
            if not _SAFE_ID_RE.match(session_id):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            sessions.append(
                SessionInfo(session_id=session_id, path=path, mtime=stat.st_mtime, size=stat.st_size)
            )
        sessions.sort(key=lambda item: -item.mtime)
        return sessions

    def sessions_since(self, ts: float, *, exclude_current: bool = True) -> List[SessionInfo]:
        """上次 dream 之后有更新的会话（默认排除当前会话，避免把自己的半成品算进去）。"""
        return [
            session
            for session in self.list_sessions()
            if session.mtime > ts
            and not (exclude_current and session.session_id == self.session_id)
            and session.size > 0
        ]

    def read_session(self, session_id: str, *, max_entries: int = 400) -> List[dict]:
        """读取单个会话的转写条目；id 非法 / 文件不存在返回空列表。"""
        if not _SAFE_ID_RE.match((session_id or "").strip()):
            return []
        path = self._dir / f"{session_id}.jsonl"
        try:
            if not path.is_file():
                return []
            entries: List[dict] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(entries) >= max_entries:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        entries.append(item)
            return entries
        except OSError:
            return []


def _clip(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "…"
