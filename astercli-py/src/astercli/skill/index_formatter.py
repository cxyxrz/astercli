"""把启用 skill 渲染成 system prompt 索引段（对齐 Java SkillIndexFormatter）。

预算约束（命中即截断）：
- 单条 description ≤ 500 字符
- 启用 skill 数 ≤ 20（按 name 字典序保留前 20）
- 总段大小 ≤ 4096 字符

注入位置：主 Agent system prompt 末尾与 /team 三个角色提示词，独立段。
"""

from __future__ import annotations

from typing import List

from .skill import Skill

MAX_DESCRIPTION_CHARS = 500
MAX_ENABLED_SKILLS = 20
MAX_INDEX_CHARS = 4096


def format_index(enabled: List[Skill]) -> str:
    if not enabled:
        return ""

    effective = enabled
    if len(enabled) > MAX_ENABLED_SKILLS:
        effective = sorted(enabled, key=lambda s: s.name)[:MAX_ENABLED_SKILLS]

    lines = ["## 可用 Skills（按需调用 load_skill 加载完整指引）", ""]
    for skill in effective:
        desc = skill.description.strip()[:MAX_DESCRIPTION_CHARS]
        lines.append(f"- **{skill.name}**：{desc}")
    lines.append("")
    lines.append(
        "判断准则：当任务描述匹配某个 skill 的触发场景时，调用 load_skill(name) 加载完整指引；"
        "已加载的 skill 会在下一轮以 \"## 已加载 Skill\" 段落出现在你的 user message 中。"
        "不要重复加载同一 skill；同一会话内一次足够。"
    )
    text = "\n".join(lines)
    if len(text) > MAX_INDEX_CHARS:
        text = text[:MAX_INDEX_CHARS] + "\n...(skill 索引段被截断)\n"
    return text
