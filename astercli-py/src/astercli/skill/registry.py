"""Skill 加载与运行时维护（对齐 Java SkillRegistry）。

三层目录扫描顺序（后者整体覆盖前者同名 skill）：
  1. builtin：包内 ``skill/builtin/<name>/SKILL.md``（web-research / codebase-onboarding）
  2. user：~/.astercli-py/skills/<name>/SKILL.md（与 Java ~/.astercli 隔离）
  3. project：<projectDir>/.astercli/skills/<name>/SKILL.md

启用状态由 SkillStateStore 的 disabled 列表过滤。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

from .frontmatter import parse
from .skill import Skill, Source
from .state_store import SkillStateStore

# 描述冲突检测：归一化后按字符二元组做廉价预筛，再用 SequenceMatcher 精确打分。
# 阈值 0.75 是"换个说法描述同一件事"的经验分界；低于它的通常只是同领域。
CONFLICT_SIMILARITY = 0.75
_CONFLICT_PREFILTER = 0.25
_CONFLICT_MAX_CHARS = 300
_NON_WORD_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalize_description(text: str) -> str:
    return _NON_WORD_RE.sub("", (text or "").lower())[:_CONFLICT_MAX_CHARS]


def _bigrams(text: str) -> set:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def description_similarity(a: str, b: str) -> float:
    """两条 description 的相似度（0~1）；对中文换词与英文改写都适用。"""
    na, nb = _normalize_description(a), _normalize_description(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ga, gb = _bigrams(na), _bigrams(nb)
    overlap = len(ga & gb) / max(1, min(len(ga), len(gb)))
    if overlap < _CONFLICT_PREFILTER:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


class SkillRegistry:
    def __init__(
        self,
        user_skills_dir: Path,
        project_skills_dir: Path,
        state_store: Optional[SkillStateStore] = None,
        *,
        builtin_dir: Optional[Path] = None,
    ) -> None:
        self._builtin_dir = builtin_dir
        self._user_dir = user_skills_dir
        self._project_dir = project_skills_dir
        self._state_store = state_store or SkillStateStore()
        self._skills: dict = {}
        self._warnings: List[str] = []

    # ---------- 加载 ----------

    def reload(self) -> None:
        self._skills.clear()
        self._warnings.clear()
        if self._builtin_dir is not None:
            self._load_directory(self._builtin_dir, Source.BUILTIN)
        self._load_directory(self._user_dir, Source.USER)
        self._load_directory(self._project_dir, Source.PROJECT)

    def _load_directory(self, directory: Path, source: Source) -> None:
        if directory is None or not directory.is_dir():
            return
        try:
            entries = sorted(p for p in directory.iterdir() if p.is_dir())
        except OSError as e:
            self._warnings.append(f"扫描 skill 目录失败 {directory}: {e}")
            return
        for entry in entries:
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            skill = self._parse_skill(entry, skill_md, source)
            if skill is not None:
                self._skills[skill.name] = skill

    def _parse_skill(self, skill_dir: Path, skill_md: Path, source: Source) -> Optional[Skill]:
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            self._warnings.append(f"读取 SKILL.md 失败 {skill_md}: {e}")
            return None

        parsed = parse(content)
        for warning in parsed.warnings:
            self._warnings.append(f"{skill_md}: {warning}")

        fm = parsed.frontmatter
        raw_name = fm.get("name")
        name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else skill_dir.name
        raw_desc = fm.get("description")
        description = str(raw_desc) if isinstance(raw_desc, str) else ""
        raw_version = fm.get("version")
        version = str(raw_version) if isinstance(raw_version, str) and raw_version.strip() else None
        raw_author = fm.get("author")
        author = str(raw_author) if isinstance(raw_author, str) and raw_author.strip() else None
        raw_tags = fm.get("tags")
        tags = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []

        references_dir = skill_dir / "references"
        if not references_dir.is_dir():
            references_dir = None

        return Skill(
            name=name,
            description=description,
            version=version,
            author=author,
            tags=tags,
            source=source,
            body=parsed.body,
            skill_md_path=skill_md,
            references_dir=references_dir,
        )

    # ---------- 查询 ----------

    def all_skills(self) -> List[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def enabled_skills(self) -> List[Skill]:
        disabled = self._state_store.disabled()
        return [s for s in self.all_skills() if s.name not in disabled]

    def find_skill(self, name: Optional[str]) -> Optional[Skill]:
        """按名查找启用 skill；被禁用的返回 None。"""
        if not name:
            return None
        skill = self._skills.get(name)
        if skill is None:
            return None
        if name in self._state_store.disabled():
            return None
        return skill

    def find_any_skill(self, name: Optional[str]) -> Optional[Skill]:
        """按名查找（含被禁用），/skill show 用。"""
        if not name:
            return None
        return self._skills.get(name)

    def resolve(self, name: Optional[str]):
        """按名取启用 skill，返回 ``(skill, error_text)``；供 load_skill / read_skill_file / fork 共用。"""
        if not name:
            return None, "skill 名称不能为空"
        skill = self.find_skill(name)
        if skill is not None:
            return skill, ""
        if self.find_any_skill(name) is not None:
            return None, f"Skill '{name}' 已被禁用，可用 /skill on {name} 启用"
        return None, f"Skill '{name}' 未找到，可用 /skill list 查看可用 skill"

    def warnings(self) -> List[str]:
        return list(self._warnings)

    def conflicts(self) -> List[str]:
        """描述高度相似的启用 skill 对；按需计算，`/skill off` 后立即消失。"""
        skills = [s for s in self.enabled_skills() if (s.description or "").strip()]
        found: List[str] = []
        for i, left in enumerate(skills):
            for right in skills[i + 1 :]:
                score = description_similarity(left.description, right.description)
                if score >= CONFLICT_SIMILARITY:
                    found.append(
                        f"描述高度相似（{score:.0%}）: '{left.name}' 与 '{right.name}'，"
                        "模型可能随机选一个；建议改写 description 或写清适用场景差异"
                    )
        return found

    def state_store(self) -> SkillStateStore:
        return self._state_store
