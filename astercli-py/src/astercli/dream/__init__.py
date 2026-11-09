"""AutoDream（P10）：启动期后台记忆整合。

Claude Code 式 auto-dream 的 Python 版实现：累积分轮会话转写，
在启动期通过五道门判断是否值得在后台做一次"离线整合"，
把零散事实合并进长期记忆与经验库，并保留可回滚快照。

M1 已交付：会话转写（session_log）、五道门（gates）、状态与 PID 锁（runner）。
M2 已交付：快照（snapshot）、整合器（consolidator）、提案与 ops 校验（proposal）。
M3 已交付：事务 apply / 软删除归档 / 快照撤销（applier + runner.undo）。
M4 追加：启动触发、守护线程、完成通知与 /dream 全部子命令。
"""

from __future__ import annotations

from .applier import Applier, ApplyResult, load_snapshot
from .consolidator import Consolidator, Material
from .gates import (
    GATE_DISABLED,
    GATE_NOT_ENOUGH_SESSIONS,
    GATE_OK,
    GATE_SCAN_THROTTLED,
    GATE_TOO_SOON,
    GateResult,
    evaluate,
)
from .models import DreamPaths, DreamRecord, DreamState, project_hash
from .proposal import (
    OPS_FILE,
    PROPOSAL_FILE,
    VALID_OPS,
    Proposal,
    describe_op,
    normalize_ops,
    parse_proposal_output,
)
from .runner import DreamLock, DreamRunner, DreamWorker, pid_alive
from .session_log import SessionInfo, SessionLog, new_session_id
from .snapshot import Snapshot, SnapshotEntry, take_snapshot

__all__ = [
    "Applier",
    "ApplyResult",
    "Consolidator",
    "DreamLock",
    "DreamPaths",
    "DreamRecord",
    "DreamRunner",
    "DreamState",
    "DreamWorker",
    "GATE_DISABLED",
    "GATE_NOT_ENOUGH_SESSIONS",
    "GATE_OK",
    "GATE_SCAN_THROTTLED",
    "GATE_TOO_SOON",
    "GateResult",
    "Material",
    "OPS_FILE",
    "PROPOSAL_FILE",
    "Proposal",
    "SessionInfo",
    "SessionLog",
    "Snapshot",
    "SnapshotEntry",
    "VALID_OPS",
    "describe_op",
    "evaluate",
    "load_snapshot",
    "new_session_id",
    "normalize_ops",
    "parse_proposal_output",
    "pid_alive",
    "project_hash",
    "take_snapshot",
]
