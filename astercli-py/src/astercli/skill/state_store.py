"""Skill 启用状态持久化（对齐 Java SkillStateStore）。

disabled 列表存 ``{"disabled": [name, ...]}``，路径默认
``~/.astercli-py/skills.json``（用户级目录与 Java 版 ~/.astercli 隔离，
沿用 P5 MCP 的 ~/.astercli-py 约定）。读写失败不阻塞加载，仅丢失持久化。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set


class SkillStateStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else Path.home() / ".astercli-py" / "skills.json"

    def disabled(self) -> Set[str]:
        try:
            if not self._path.is_file():
                return set()
            data = json.loads(self._path.read_text(encoding="utf-8"))
            names = data.get("disabled")
            if isinstance(names, list):
                return {str(n) for n in names}
        except (OSError, ValueError):
            pass
        return set()

    def set_disabled(self, names: List[str]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"disabled": sorted(set(names))}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
