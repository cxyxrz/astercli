"""Dream 前快照（P10 M2）：先备份再改写，供 ``/dream undo`` 精确还原。

快照只覆盖"本次 dream 可能改写的文件"：长期记忆 JSON、全部经验 Markdown。
``manifest.json`` 记录原路径、备份相对路径与"原本是否存在"三要素：

- 原本存在 → undo 时把备份复制回原位置（覆盖）
- 原本不存在 → undo 时删除新生成的文件

备份目录：``~/.astercli-py/dream/<hash>/backups/<dream-id>/``。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MANIFEST_FILE = "manifest.json"
FILES_DIR = "files"


@dataclass(frozen=True)
class SnapshotEntry:
    original: str
    backup: str  # 相对快照目录；existed=False 时为空
    existed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"original": self.original, "backup": self.backup, "existed": self.existed}

    @staticmethod
    def from_dict(data: dict) -> Optional["SnapshotEntry"]:
        if not isinstance(data, dict):
            return None
        original = str(data.get("original") or "").strip()
        if not original:
            return None
        return SnapshotEntry(
            original=original,
            backup=str(data.get("backup") or ""),
            existed=bool(data.get("existed")),
        )


@dataclass
class Snapshot:
    dream_id: str
    root: Path
    entries: List[SnapshotEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def save(self) -> bool:
        payload = {
            "dream_id": self.dream_id,
            "created_at": self.created_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / MANIFEST_FILE).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, root: Path, dream_id: str = "") -> Optional["Snapshot"]:
        root = Path(root)
        try:
            data = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        entries: List[SnapshotEntry] = []
        raw = data.get("entries")
        for item in raw if isinstance(raw, list) else []:
            entry = SnapshotEntry.from_dict(item)
            if entry is not None:
                entries.append(entry)
        return cls(
            dream_id=dream_id or str(data.get("dream_id") or ""),
            root=root,
            entries=entries,
            created_at=_as_float(data.get("created_at")),
        )

    def restore(self) -> int:
        """按 manifest 还原；返回成功处理的文件数（失败条目跳过）。"""
        restored = 0
        for entry in self.entries:
            original = Path(entry.original)
            if not entry.existed:
                try:
                    if original.is_file():
                        original.unlink()
                        restored += 1
                except OSError:
                    continue
                continue
            backup = self.root / entry.backup if entry.backup else None
            if backup is None or not backup.is_file():
                continue
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, original)
                restored += 1
            except OSError:
                continue
        return restored


def take_snapshot(
    dream_id: str,
    targets: Iterable[Path],
    backups_dir: Path,
) -> Snapshot:
    """复制 targets（文件）到 ``backups/<dream_id>``，返回可还原的快照。

    只处理已存在的普通文件；目录与不存在的路径同样记录 ``existed=False``，
    以便 undo 时删除 dream 期间新建的文件。
    """
    root = Path(backups_dir) / dream_id
    files_dir = root / FILES_DIR
    try:
        files_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise

    entries: List[SnapshotEntry] = []
    seen = set()
    for index, target in enumerate(targets):
        original = Path(target).expanduser()
        key = str(original)
        if key in seen:
            continue
        seen.add(key)
        try:
            exists = original.is_file()
        except OSError:
            exists = False
        if not exists:
            entries.append(SnapshotEntry(original=key, backup="", existed=False))
            continue
        backup_name = f"{index:03d}-{original.name}"
        shutil.copy2(original, files_dir / backup_name)
        entries.append(
            SnapshotEntry(
                original=key, backup=f"{FILES_DIR}/{backup_name}", existed=True
            )
        )

    snapshot = Snapshot(dream_id=dream_id, root=root, entries=entries)
    if not snapshot.save():
        raise OSError(f"快照清单写入失败: {root}")
    return snapshot


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
