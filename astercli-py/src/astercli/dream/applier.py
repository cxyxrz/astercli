"""Dream 事务应用（P10 M3）：快照 → 短临界区应用 → 失败整体回滚。

三条安全承诺都在这里落地：

1. **先备份**：apply 前对长期记忆 JSON 与全部经验文件做快照；
2. **删除只软删**：记忆条目写入 ``archive/<dream-id>/memory/``，
   经验文件移动到 ``archive/<dream-id>/experience/``，不硬删；
3. **失败整体回滚**：任一持久化失败 → 用快照还原全部文件，本次 dream 标记失败，
   状态（state.json）不推进。

并发：主线程的 ``/save`` / ``/memory`` 与 dream 线程共享同一把
``threading.RLock``（由 CLI 注入），应用阶段是纯内存 + 少量文件 IO 的短临界区。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..experience import TRIGGER_DREAM, ExperienceEntry, ExperienceStore
from ..memory import LongTermMemory
from .models import DreamPaths
from .proposal import (
    OP_EXPERIENCE_ADD,
    OP_EXPERIENCE_DELETE,
    OP_MEMORY_ADD,
    OP_MEMORY_DELETE,
    OP_MEMORY_MERGE,
    OP_MEMORY_UPDATE,
    Proposal,
)
from .snapshot import Snapshot, SnapshotEntry, take_snapshot


class ApplyError(RuntimeError):
    """应用阶段的致命错误（触发整体回滚）。"""


@dataclass
class ApplyResult:
    dream_id: str
    applied: int = 0
    rolled_back: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rolled_back and self.applied > 0


class Applier:
    """把校验后的 Proposal 一次性写进记忆库。"""

    def __init__(
        self,
        long_term: LongTermMemory,
        experience: ExperienceStore,
        paths: DreamPaths,
        *,
        project_key: str,
        lock: Optional["threading.RLock"] = None,
    ) -> None:
        self._long_term = long_term
        self._experience = experience
        self._paths = paths
        self._project_key = project_key
        self._lock = lock

    # ---------- 目标文件 ----------

    def targets(self) -> List[Path]:
        return [self._long_term.storage_file, *self._experience.entry_paths()]

    # ---------- 应用 ----------

    def apply(self, proposal: Proposal) -> ApplyResult:
        result = ApplyResult(dream_id=proposal.dream_id)
        if not proposal.ops:
            result.errors.append("提案没有可应用的操作")
            return result

        guard = self._lock if self._lock is not None else nullcontext()
        with guard:
            try:
                snapshot = take_snapshot(
                    proposal.dream_id, self.targets(), self._paths.backups_dir
                )
            except OSError as exc:
                result.errors.append(f"快照失败，已放弃本次整合: {exc}")
                return result

            created: List[Path] = []
            try:
                for op in proposal.ops:
                    self._apply_op(op, proposal.dream_id, result, created)
                # 新建文件先记进快照（existed=False）：即便下一步落盘失败，
                # 回滚也能把这些文件删掉
                for path in created:
                    snapshot.entries.append(
                        SnapshotEntry(original=str(path), backup="", existed=False)
                    )
                if not self._long_term.persist():
                    raise ApplyError("长期记忆落盘失败")
                snapshot.save()
            except Exception as exc:  # noqa: BLE001 —— 任何异常都要回滚
                restored = snapshot.restore()
                result.rolled_back = True
                result.applied = 0
                result.errors.append(f"应用失败已整体回滚（还原 {restored} 个文件）: {exc}")
        return result

    def _apply_op(
        self,
        op: Dict[str, Any],
        dream_id: str,
        result: ApplyResult,
        created: List[Path],
    ) -> None:
        kind = str(op.get("type") or "")
        if kind == OP_MEMORY_ADD:
            entry = self._long_term.add_fact(
                str(op.get("content") or ""),
                scope=str(op.get("scope") or "project"),
                project_key=self._project_key,
            )
            if entry is None:
                result.errors.append("memory_add 跳过：内容已存在")
            else:
                _mark_dream_origin(entry, dream_id)
                result.applied += 1

        elif kind == OP_MEMORY_UPDATE:
            entry = self._long_term.update_content(
                str(op.get("id") or ""), str(op.get("content") or "")
            )
            if entry is None:
                result.errors.append(f"memory_update 跳过：未找到 {op.get('id')}")
            else:
                result.applied += 1

        elif kind == OP_MEMORY_DELETE:
            entry_id = str(op.get("id") or "")
            entry = self._long_term.pop(entry_id)
            if entry is None:
                result.errors.append(f"memory_delete 跳过：未找到 {entry_id}")
            else:
                self._archive_memory(entry, dream_id)
                result.applied += 1

        elif kind == OP_MEMORY_MERGE:
            merged: List[str] = []
            for entry_id in op.get("ids") or []:
                entry = self._long_term.pop(str(entry_id))
                if entry is not None:
                    self._archive_memory(entry, dream_id)
                    merged.append(entry.id)
            new_entry = self._long_term.add_fact(
                str(op.get("content") or ""),
                scope=str(op.get("scope") or "project"),
                project_key=self._project_key,
            )
            if new_entry is None:
                result.errors.append("memory_merge 跳过：合并内容已存在")
            else:
                _mark_dream_origin(new_entry, dream_id)
                result.applied += 1

        elif kind == OP_EXPERIENCE_ADD:
            entry = ExperienceEntry(
                id=f"exp-{uuid.uuid4().hex[:8]}",
                lesson=str(op.get("lesson") or ""),
                correct_action=str(op.get("correct_action") or ""),
                category=str(op.get("category") or ""),
                trigger=TRIGGER_DREAM,
                scope="project",
                project_key=self._project_key,
            )
            saved, is_new = self._experience.save(entry)
            if saved is None:
                raise ApplyError("经验写入失败")
            if is_new:
                path = self._experience.path_of(saved)
                if path is not None:
                    created.append(path)
            result.applied += 1

        elif kind == OP_EXPERIENCE_DELETE:
            entry_id = str(op.get("id") or "")
            archive_dir = self._paths.root / "archive" / dream_id / "experience"
            if self._experience.archive(entry_id, archive_dir):
                result.applied += 1
            else:
                result.errors.append(f"experience_delete 跳过：未找到 {entry_id}")

        else:
            result.errors.append(f"未知 op 类型，已跳过: {kind}")

    # ---------- 软删除归档 ----------

    def _archive_memory(self, entry, dream_id: str) -> None:
        directory = self._paths.root / "archive" / dream_id / "memory"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "dream_id": dream_id,
                "deleted_at": time.time(),
                "entry": entry.to_dict(),
            }
            (directory / f"{entry.id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            # 归档是审计辅助，失败不阻断事务（快照仍可完整撤销）
            pass


def load_snapshot(paths: DreamPaths, dream_id: str) -> Optional[Snapshot]:
    """按 dream-id 读取快照（``/dream undo`` 用）。"""
    return Snapshot.load(paths.backup_dir(dream_id), dream_id)


def _mark_dream_origin(entry, dream_id: str) -> None:
    """标记自动写入来源，便于 /memory list 审计与后续冲突判断。"""
    entry.metadata["source"] = "dream"
    entry.metadata["dream_id"] = dream_id
