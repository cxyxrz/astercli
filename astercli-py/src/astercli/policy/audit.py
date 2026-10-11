"""危险工具调用审计（对齐 Java AuditLog 按天 JSONL 语义）。

目录默认 ~/.astercli-py/audit，与 Java 版 ~/.astercli 分开，避免双版本混写。
写入失败静默吞掉——审计不能影响工具主路径。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def _audit_dir() -> Path:
    configured = os.environ.get("ASTERCLI_AUDIT_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path.home() / ".astercli-py" / "audit"


def record_audit(tool: str, detail: str, *, allowed: bool) -> None:
    try:
        path = _audit_dir() / f"audit-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": tool,
            "allowed": allowed,
            "detail": detail[:500],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
