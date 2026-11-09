"""经验持久化（P9 层二）：一经验一 Markdown + 目录索引 + 淘汰。

存储位置（两级独立，互不覆盖）：
- 项目级 ``<项目根>/.astercli/experience/``：可随 git 提交，团队共享
- 全局级 ``~/.astercli-py/experience/``：个人跨项目经验，
  ``ASTERCLI_EXPERIENCE_DIR`` 可覆盖

每个条目文件名 ``exp-<id>.md``（frontmatter + 四段正文）；每个目录维护一份
``index.md`` 供人浏览（LLM 侧索引由 :meth:`index_text` 现算，避免读旧文件）。
目录条目超过上限（默认 200）时按 ``hits × 时效`` 淘汰最低分条目。
"""

from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..memory.long_term import tokenize
from .models import ExperienceEntry

if TYPE_CHECKING:  # pragma: no cover —— 仅类型引用，避免运行期循环依赖
    from ..config import Settings

log = logging.getLogger(__name__)

INDEX_FILE = "index.md"
ENTRY_GLOB = "exp-*.md"
DEFAULT_GLOBAL_DIR = Path.home() / ".astercli-py" / "experience"
MAX_INDEX_PROMPT_CHARS = 4_000
MAX_CONTEXT_PROMPT_CHARS = 1_200
_SAFE_ID_RE = re.compile(r"^exp-[A-Za-z0-9_-]{1,40}$")


class ExperienceStore:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        settings: Optional["Settings"] = None,
        *,
        global_dir: Optional[Path] = None,
        project_dir: Optional[Path] = None,
    ) -> None:
        self._root = Path(project_root).resolve() if project_root else None
        configured = global_dir
        if configured is None and settings is not None:
            raw = settings.get("ASTERCLI_EXPERIENCE_DIR")
            configured = Path(raw).expanduser() if raw else None
        self._global_dir = Path(configured) if configured else DEFAULT_GLOBAL_DIR
        self._project_dir = (
            Path(project_dir)
            if project_dir
            else (self._root / ".astercli" / "experience" if self._root else None)
        )
        self._max_entries = settings.experience_max_entries() if settings is not None else 200

    # ---------- 目录 ----------

    def _dir(self, scope: str) -> Optional[Path]:
        return self._global_dir if scope == "global" else self._project_dir

    def _scopes(self) -> List[str]:
        return ["project", "global"] if self._project_dir is not None else ["global"]

    def path_of(self, entry: ExperienceEntry) -> Optional[Path]:
        directory = self._dir(entry.scope)
        return directory / f"{entry.id}.md" if directory is not None else None

    # ---------- 写 ----------

    def save(self, entry: ExperienceEntry, *, dedupe: bool = True) -> Tuple[Optional[ExperienceEntry], bool]:
        """落盘一条经验，返回 ``(条目, 是否新建)``。

        - 同目录内内容签名相同 → 合并为 hits + 1（不新增文件）
        - 写入失败（IO 异常）静默返回 ``(None, False)``，不影响主任务
        """
        scope = entry.scope if entry.scope in ("project", "global") else "project"
        directory = self._dir(scope)
        if directory is None:
            return None, False
        entry.scope = scope
        entry.signature = entry.signature or entry.content_signature()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.debug("经验目录创建失败: %s", directory, exc_info=True)
            return None, False

        if dedupe:
            signature = entry.content_signature()
            for existing in self._scan(scope, directory):
                if existing.content_signature() == signature:
                    existing.hits += 1
                    existing.signature = existing.content_signature()
                    if not self._write(directory, existing):
                        return None, False
                    self._rebuild_index(scope, directory)
                    return existing, False

        if not _SAFE_ID_RE.match(entry.id or ""):
            entry.id = f"exp-{uuid.uuid4().hex[:8]}"
        entry.signature = entry.content_signature()
        if not self._write(directory, entry):
            return None, False
        self._enforce_cap(scope, directory)
        self._rebuild_index(scope, directory)
        return entry, True

    def delete(self, entry_id: str) -> bool:
        entry_id = (entry_id or "").strip()
        for scope in self._scopes():
            directory = self._dir(scope)
            if directory is None:
                continue
            path = directory / f"{entry_id}.md"
            try:
                if path.is_file():
                    path.unlink()
                    self._rebuild_index(scope, directory)
                    return True
            except OSError:
                return False
        return False

    def archive(self, entry_id: str, archive_dir: Path) -> bool:
        """软删除：把 ``exp-<id>.md`` 移到归档目录（AutoDream 可回滚删除用）。

        归档不是硬删，``/dream undo`` 可以按快照把文件放回原位置。
        """
        entry_id = (entry_id or "").strip()
        for scope in self._scopes():
            directory = self._dir(scope)
            if directory is None:
                continue
            path = directory / f"{entry_id}.md"
            try:
                if not path.is_file():
                    continue
                Path(archive_dir).mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(Path(archive_dir) / path.name))
                self._rebuild_index(scope, directory)
                return True
            except OSError:
                return False
        return False

    def entry_paths(self) -> List[Path]:
        """全部经验文件（AutoDream 快照用）。"""
        paths: List[Path] = []
        for scope in self._scopes():
            directory = self._dir(scope)
            if directory is None:
                continue
            paths.extend(self._entry_paths(directory))
        return paths

    def clear(self, scope: Optional[str] = None) -> int:
        """清空指定作用域（None 表示全部），返回删除条目数。"""
        removed = 0
        for current in [scope] if scope in ("project", "global") else self._scopes():
            directory = self._dir(current)
            if directory is None or not directory.is_dir():
                continue
            for path in self._entry_paths(directory):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
            self._rebuild_index(current, directory)
        return removed

    # ---------- 读 ----------

    def list_entries(self, scope: Optional[str] = None, category: Optional[str] = None) -> List[ExperienceEntry]:
        scopes = [scope] if scope in ("project", "global") else self._scopes()
        entries: List[ExperienceEntry] = []
        for current in scopes:
            directory = self._dir(current)
            if directory is None:
                continue
            entries.extend(self._scan(current, directory))
        if category:
            entries = [e for e in entries if e.category == category]
        entries.sort(key=lambda e: (-e.created, -e.hits))
        return entries

    def get(self, entry_id: str) -> Optional[ExperienceEntry]:
        entry_id = (entry_id or "").strip()
        if not entry_id:
            return None
        for scope in self._scopes():
            directory = self._dir(scope)
            if directory is None:
                continue
            for entry in self._scan(scope, directory):
                if entry.id == entry_id:
                    return entry
        return None

    def search(self, query: str, limit: int = 20) -> List[ExperienceEntry]:
        """关键词检索（复用长期记忆的英文整词 / 中文 2-gram 分词）。"""
        tokens = tokenize(query)
        if not tokens:
            return []
        scored = []
        for entry in self.list_entries():
            text = " ".join(
                [entry.lesson, entry.correct_action, entry.evidence, entry.category, entry.trigger]
            ).lower()
            if query and query.lower() in text:
                score = 1.0
            else:
                matched = sum(1 for t in tokens if t in text)
                if matched == 0:
                    continue
                score = matched / len(tokens)
            age_days = max(0.0, (time.time() - entry.created) / 86400.0)
            score *= 2.0 if age_days <= 7 else (1.0 if age_days <= 30 else 0.6)
            score += min(entry.hits, 5) * 0.05
            scored.append((score, entry))
        scored.sort(key=lambda pair: -pair[0])
        return [entry for _, entry in scored[:limit]]

    def status_summary(self) -> str:
        entries = self.list_entries()
        project = sum(1 for e in entries if e.scope == "project")
        return (
            f"经验库: {len(entries)} 条（项目 {project} / 全局 {len(entries) - project}）"
            f" · 目录 {self._project_dir or '（未设置项目级）'} / {self._global_dir}"
        )

    # ---------- 提示词注入 ----------

    def index_text(self, max_entries: int = 20, max_chars: int = MAX_INDEX_PROMPT_CHARS) -> str:
        """system prompt 索引段（会话启动时注入一次，Claude Code 极简索引路线）。"""
        entries = self.list_entries()
        if not entries:
            return ""
        ranked = sorted(entries, key=lambda e: (-e.hits, -e.created))
        lines = [
            "## 可用经验（历史失败复盘）",
            "",
            "以下经验由系统在工具失败 / 审批拒绝 / 语法错误后自动沉淀，执行相关操作前先参考：",
        ]
        for entry in ranked[:max_entries]:
            lines.append(f"- [{entry.id}] ({entry.category} · {entry.trigger} · hits={entry.hits}) {_clip(entry.lesson, 90)}")
        if len(ranked) > max_entries:
            lines.append(f"- …（共 {len(ranked)} 条，其余用 /experience list 查看）")
        return _clip("\n".join(lines), max_chars)

    def context_for_query(
        self,
        query: str,
        limit: int = 3,
        max_chars: int = MAX_CONTEXT_PROMPT_CHARS,
    ) -> str:
        """任务级检索注入：只取与当前输入相关的少量经验（进本轮 user message）。"""
        matched = self.search(query, limit=limit)
        if not matched:
            return ""
        lines = ["## 相关经验（历史复盘）", ""]
        for entry in matched:
            line = f"- ({entry.category}) {entry.lesson}"
            if entry.correct_action:
                line += f"\n  正确做法: {entry.correct_action}"
            lines.append(line)
        return _clip("\n".join(lines), max_chars)

    # ---------- 内部 ----------

    def _entry_paths(self, directory: Path) -> List[Path]:
        try:
            return sorted(directory.glob(ENTRY_GLOB))
        except OSError:
            return []

    def _scan(self, scope: str, directory: Path) -> List[ExperienceEntry]:
        entries: List[ExperienceEntry] = []
        for path in self._entry_paths(directory):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            entry = ExperienceEntry.from_markdown(text)
            if entry is None:
                continue
            entry.scope = scope
            if path.stem != entry.id:
                entry.id = path.stem  # 文件名为准，避免 frontmatter 手改后错位
            entries.append(entry)
        return entries

    def _write(self, directory: Path, entry: ExperienceEntry) -> bool:
        try:
            (directory / f"{entry.id}.md").write_text(entry.to_markdown(), encoding="utf-8")
            return True
        except OSError:
            log.debug("经验写入失败: %s", directory, exc_info=True)
            return False

    def _enforce_cap(self, scope: str, directory: Path) -> None:
        entries = self._scan(scope, directory)
        overflow = len(entries) - self._max_entries
        if overflow <= 0:
            return
        now = time.time()

        def score(entry: ExperienceEntry) -> float:
            age_days = max(0.0, (now - entry.created) / 86400.0)
            decay = 2.0 if age_days <= 7 else (1.0 if age_days <= 30 else 0.5)
            return entry.hits * decay

        entries.sort(key=lambda entry: (score(entry), entry.created))
        for entry in entries[:overflow]:
            try:
                (directory / f"{entry.id}.md").unlink()
            except OSError:
                continue

    def _rebuild_index(self, scope: str, directory: Path) -> None:
        """维护目录 index.md（人工浏览用；失败静默）。"""
        entries = self._scan(scope, directory)
        label = "项目级" if scope == "project" else "全局"
        lines = [
            f"# 经验索引（{label}）",
            "",
            f"共 {len(entries)} 条 · 更新于 {time.strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        ranked = sorted(entries, key=lambda e: (-e.hits, -e.created))
        for entry in ranked:
            when = time.strftime("%Y-%m-%d", time.localtime(entry.created))
            lines.append(
                f"- [{entry.id}] ({entry.category} · {entry.trigger} · hits={entry.hits} · {when}) "
                f"{_clip(entry.lesson, 80)}"
            )
        if not ranked:
            lines.append("- （空）")
        lines.append("")
        try:
            (directory / INDEX_FILE).write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            log.debug("经验索引写入失败: %s", directory, exc_info=True)


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n…（已按预算截断）"
