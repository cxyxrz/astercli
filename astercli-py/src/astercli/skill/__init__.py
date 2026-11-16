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
from .reader import DEFAULT_READ_LINES, SKILL_MD, list_skill_files, read_skill_text, skill_root
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


def _resolve_skill(skill_registry: SkillRegistry, name: str):
    """按名取启用 skill，返回 (skill, error_text)。"""
    skill = skill_registry.find_skill(name)
    if skill is not None:
        return skill, ""
    if skill_registry.find_any_skill(name) is not None:
        return None, f"Skill '{name}' 已被禁用，可用 /skill on {name} 启用"
    return None, f"Skill '{name}' 未找到，可用 /skill list 查看可用 skill"


def _injection_header(skill: Skill) -> str:
    """注入段头部：把 skill 目录与附属文件暴露给 LLM，第三层（references/）才可寻址。"""
    root = skill_root(skill)
    if root is None:
        return ""
    lines = [f"> skill_dir: {root}"]
    files = list_skill_files(skill)
    if files:
        lines.append("> 附属文件: " + ", ".join(files))
        lines.append(
            '> 读取方式: read_skill_file(name="'
            + skill.name
            + '", path="references/xxx.md")；只在正文要求查阅细则时才读取'
        )
    return "\n".join(lines) + "\n\n"


def register_skill_tool(
    registry: ToolRegistry, skill_registry: SkillRegistry, skill_buffer: SkillContextBuffer
) -> None:
    """注册 skill 工具（只读，不走 HITL 审批；对齐 Java registerSkillTools）。

    两个工具构成渐进披露：load_skill 注入 SKILL.md 正文（第二层），
    read_skill_file 按需分页读取 skill 目录内的 references/ 等附属文件（第三层）。
    """

    def load_skill(args: dict) -> str:
        name = str(args.get("name") or "").strip()
        if not name:
            return "load_skill 失败: name 不能为空"
        skill, error = _resolve_skill(skill_registry, name)
        if error:
            return error
        body = skill.body or ""
        original_len = len(body)
        injected = body
        truncated_note = ""
        if len(body) > LOAD_SKILL_BODY_MAX_CHARS:
            # 在最后一次换行处切，保证 offset 续读不会从半行开始
            cut = body.rfind("\n", 0, LOAD_SKILL_BODY_MAX_CHARS)
            if cut <= 0:
                cut = LOAD_SKILL_BODY_MAX_CHARS
            consumed = body[:cut].count("\n")
            injected = body[:cut] + (
                "\n\n...(skill body 已截断，剩余内容请调用 "
                f'read_skill_file(name="{name}", path="{SKILL_MD}", offset={consumed + 1}) 续读)'
            )
            truncated_note = "，超长部分可续读"
        skill_buffer.push(name, _injection_header(skill) + injected)
        root = skill_root(skill)
        location = f"；skill_dir={root}" if root is not None else ""
        return (
            f"已加载 skill '{name}' 的完整指引（{original_len} 字符{truncated_note}）{location}，"
            f"将在下一轮上下文中以 \"## 已加载 Skill：{name}\" 段出现。"
        )

    def read_skill_file(args: dict) -> str:
        name = str(args.get("name") or "").strip()
        if not name:
            return "read_skill_file 失败: name 不能为空"
        skill, error = _resolve_skill(skill_registry, name)
        if error:
            return error
        return read_skill_text(
            skill,
            str(args.get("path") or SKILL_MD),
            args.get("offset") or 1,
            args.get("limit") or DEFAULT_READ_LINES,
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
    registry.register(
        "read_skill_file",
        "读取某个 skill 目录内的文件（只读、仅限该 skill 自己的目录）。"
        "load_skill 注入的正文被截断时用它按 offset 续读；或当正文要求查阅 references/ 下的细则时"
        "按需读取（例如 references/forms.md）。path=SKILL.md 返回去 frontmatter 的正文，"
        "其余文件按行返回且带行号。",
        create_parameters(
            Param("name", "string", "skill 名称，与 load_skill 相同", True),
            Param(
                "path",
                "string",
                "相对 skill 目录的文件路径，默认 SKILL.md；例如 references/forms.md",
                False,
            ),
            Param("offset", "integer", "起始行号，1 表示第一行", False),
            Param("limit", "integer", f"最多读取行数，默认 {DEFAULT_READ_LINES}", False),
        ),
        read_skill_file,
    )
