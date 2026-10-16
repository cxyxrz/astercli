"""Skill 系统（P7）：三层目录加载 + load_skill 工具 + SkillContextBuffer 注入。

对齐 Java skill 包：
- SkillRegistry：user（~/.astercli-py/skills）+ project（.astercli/skills）扫描
- load_skill 工具：LLM 按索引段触发场景调用，body 经 buffer 前置注入下一轮 user message
- SkillIndexFormatter：system prompt 索引段（≤20 个 / ≤4KB）

已知差异：Java 有 builtin（jar 内置 skill 解压）层，Python 版暂无内置资产，
仅保留参数位。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..tools.registry import Param, ToolRegistry, create_parameters
from .buffer import SkillContextBuffer
from .index_formatter import format_index
from .registry import SkillRegistry
from .skill import Skill, Source
from .state_store import SkillStateStore

LOAD_SKILL_BODY_MAX_CHARS = 5 * 1024

__all__ = [
    "Skill",
    "Source",
    "SkillContextBuffer",
    "SkillRegistry",
    "SkillStateStore",
    "format_index",
    "register_skill_tool",
    "default_skill_registry",
]


def default_skill_registry(project_root: Path) -> SkillRegistry:
    """构建默认三层注册表（user ~/.astercli-py/skills + project .astercli/skills）。"""
    registry = SkillRegistry(
        user_skills_dir=Path.home() / ".astercli-py" / "skills",
        project_skills_dir=project_root / ".astercli" / "skills",
        state_store=SkillStateStore(),
    )
    registry.reload()
    return registry


def register_skill_tool(
    registry: ToolRegistry, skill_registry: SkillRegistry, skill_buffer: SkillContextBuffer
) -> None:
    """注册 load_skill 工具（只读，不走 HITL 审批；对齐 Java registerSkillTools）。"""

    def load_skill(args: dict) -> str:
        name = str(args.get("name") or "").strip()
        if not name:
            return "load_skill 失败: name 不能为空"
        skill = skill_registry.find_skill(name)
        if skill is None:
            any_skill = skill_registry.find_any_skill(name)
            if any_skill is None:
                return f"Skill '{name}' 未找到，可用 /skill list 查看可用 skill"
            return f"Skill '{name}' 已被禁用，可用 /skill on {name} 启用"
        body = skill.body or ""
        original_len = len(body)
        injected = body
        truncated_note = ""
        if len(injected) > LOAD_SKILL_BODY_MAX_CHARS:
            injected = (
                injected[:LOAD_SKILL_BODY_MAX_CHARS]
                + f"\n\n...(skill body 已截断，完整内容可用 /skill show {name} 查看)"
            )
            truncated_note = "，超长部分已截断"
        skill_buffer.push(name, injected)
        return (
            f"已加载 skill '{name}' 的完整指引（{original_len} 字符{truncated_note}），"
            f"将在下一轮上下文中以 \"## 已加载 Skill：{name}\" 段出现。"
        )

    registry.register(
        "load_skill",
        "Load full SKILL.md instructions for a skill the system has indexed (see the \"可用 Skills\" "
        "section in this system prompt). Call this when a skill's description matches the current task. "
        "Pass the exact kebab-case skill name. The full body will appear at the start of your next "
        "user message under \"## 已加载 Skill：<name>\". Don't reload the same skill twice in one session.",
        create_parameters(
            Param("name", "string", "the exact kebab-case skill name (e.g. web-access)", True)
        ),
        load_skill,
    )
