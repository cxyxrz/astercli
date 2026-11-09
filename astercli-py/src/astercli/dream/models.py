"""AutoDream 数据模型（P10）：项目路径、整合状态、运行记录。

M1 只落地"门判定"需要的状态：上次 dream 时间 / 上次扫描时间 / 历史记录；
proposal / ops 等结构在 M2、M3 追加。

数据布局（与方案一致）::

    ~/.astercli-py/sessions/<project-hash>/<session-id>.jsonl
    ~/.astercli-py/dream/<project-hash>/state.json
    ~/.astercli-py/dream/<project-hash>/.lock
    ~/.astercli-py/dream/<project-hash>/backups/<dream-id>/
    ~/.astercli-py/dream/<project-hash>/proposals/<dream-id>/{proposal.md,ops.json}

``ASTERCLI_SESSION_LOG_DIR`` / ``ASTERCLI_DREAM_DIR`` 可覆盖两个根目录（测试与多机共用）。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DREAM_DIR = Path.home() / ".astercli-py" / "dream"
DEFAULT_SESSION_DIR = Path.home() / ".astercli-py" / "sessions"
STATE_FILE = "state.json"
LOCK_FILE = ".lock"
BACKUPS_DIR = "backups"
PROPOSALS_DIR = "proposals"
MAX_HISTORY = 20


def project_hash(project_key: str) -> str:
    """项目根 → 稳定短哈希（同一项目跨会话共享状态与会话转写）。"""
    return hashlib.sha1((project_key or "").encode("utf-8", "replace")).hexdigest()[:12]


@dataclass(frozen=True)
class DreamPaths:
    """一次项目对应的所有 AutoDream 落盘位置。"""

    project_key: str
    root: Path
    session_dir: Path

    @classmethod
    def for_project(
        cls,
        project_key: str,
        *,
        dream_root: Optional[Path] = None,
        session_root: Optional[Path] = None,
    ) -> "DreamPaths":
        key = project_hash(project_key)
        return cls(
            project_key=project_key,
            root=Path(dream_root or DEFAULT_DREAM_DIR).expanduser() / key,
            session_dir=Path(session_root or DEFAULT_SESSION_DIR).expanduser() / key,
        )

    @property
    def state_file(self) -> Path:
        return self.root / STATE_FILE

    @property
    def lock_file(self) -> Path:
        return self.root / LOCK_FILE

    @property
    def backups_dir(self) -> Path:
        return self.root / BACKUPS_DIR

    @property
    def proposals_dir(self) -> Path:
        return self.root / PROPOSALS_DIR

    def backup_dir(self, dream_id: str) -> Path:
        return self.backups_dir / dream_id

    def proposal_dir(self, dream_id: str) -> Path:
        return self.proposals_dir / dream_id


@dataclass
class DreamRecord:
    """一次 dream 运行的结果摘要（/dream log 展示用）。"""

    dream_id: str
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = ""  # applied / failed
    ops: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "ops": self.ops,
            "summary": self.summary,
        }

    @staticmethod
    def from_dict(data: dict) -> Optional["DreamRecord"]:
        if not isinstance(data, dict):
            return None
        dream_id = str(data.get("dream_id") or "").strip()
        if not dream_id:
            return None
        return DreamRecord(
            dream_id=dream_id,
            started_at=_as_float(data.get("started_at")),
            finished_at=_as_float(data.get("finished_at")),
            status=str(data.get("status") or ""),
            ops=_as_int(data.get("ops")),
            summary=str(data.get("summary") or ""),
        )


@dataclass
class DreamState:
    """项目级 AutoDream 状态（state.json）。"""

    last_dream_at: float = 0.0
    last_scan_at: float = 0.0
    dream_count: int = 0
    last_dream_id: str = ""
    history: List[DreamRecord] = field(default_factory=list)

    # ---------- 变更 ----------

    def mark_scan(self, now: Optional[float] = None) -> None:
        """记录一次扫描开始（扫描节流门依赖该时间）。"""
        self.last_scan_at = time.time() if now is None else float(now)

    def mark_dream(self, record: DreamRecord, now: Optional[float] = None) -> None:
        """记录一次成功应用的 dream（时间门与新会话门依赖该时间）。"""
        finished = time.time() if now is None else float(now)
        if not record.finished_at:
            record.finished_at = finished
        self.last_dream_at = record.finished_at
        self.dream_count += 1
        self.last_dream_id = record.dream_id
        self.history.insert(0, record)
        del self.history[MAX_HISTORY:]

    def latest(self) -> Optional[DreamRecord]:
        return self.history[0] if self.history else None

    # ---------- 序列化 ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "last_dream_at": self.last_dream_at,
            "last_scan_at": self.last_scan_at,
            "dream_count": self.dream_count,
            "last_dream_id": self.last_dream_id,
            "history": [record.to_dict() for record in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DreamState":
        if not isinstance(data, dict):
            return cls()
        history: List[DreamRecord] = []
        raw = data.get("history")
        for item in raw if isinstance(raw, list) else []:
            record = DreamRecord.from_dict(item)
            if record is not None:
                history.append(record)
        return cls(
            last_dream_at=_as_float(data.get("last_dream_at")),
            last_scan_at=_as_float(data.get("last_scan_at")),
            dream_count=_as_int(data.get("dream_count")),
            last_dream_id=str(data.get("last_dream_id") or ""),
            history=history[:MAX_HISTORY],
        )

    @classmethod
    def load(cls, path: Path) -> "DreamState":
        """读取 state.json；缺失 / 损坏 / IO 异常一律回退空状态（不阻断启动）。"""
        try:
            if not Path(path).is_file():
                return cls()
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path) -> bool:
        """原子写入（临时文件 + os.replace），避免中断留下半个 JSON。"""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, target)
            return True
        except OSError:
            return False


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
