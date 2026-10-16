"""Skill 数据模型（对齐 Java skill.Skill）。

一个 Skill 是沉淀决策与经验的复用单元：frontmatter 决定索引段元数据，
body 在 LLM 调 load_skill 时经 SkillContextBuffer 注入下一轮 user message。
source 标记加载来源，用于 /skill list 展示与三层覆盖的可观测性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Source(Enum):
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: Source = Source.PROJECT
    body: str = ""
    skill_md_path: Optional[Path] = None
    references_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Skill name 不能为空")

    def display_source(self) -> str:
        return self.source.value
