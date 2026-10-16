"""Skill 注入缓冲区（对齐 Java SkillContextBuffer）。

生命周期：LLM 调 load_skill → push 到 buffer → 下一轮构造 user message 时
drain → 拼到原内容前。

关键约束：
- drain 是一次性消费（防止跨轮重复注入）
- 同一会话内最多保留 3 个 skill body，超出 LRU 淘汰最旧
- 同一 skill 重复 push 会替换旧 body 并刷新到末尾，避免重复
- /clear 命令调 clear() 复位

Java 注释说明 Multi-Agent 三角色共享同一 buffer 是简化实现；Python 版
延续同一决策（主 Agent 与 /team 各 SubAgent 共用 ReplApp 级 buffer）。
"""

from __future__ import annotations


class SkillContextBuffer:
    MAX_SKILLS = 3

    def __init__(self) -> None:
        self._entries: dict = {}

    def push(self, skill_name: str, body: str) -> None:
        if not skill_name or not skill_name.strip() or body is None:
            return
        self._entries.pop(skill_name, None)
        self._entries[skill_name] = body
        while len(self._entries) > self.MAX_SKILLS:
            oldest = next(iter(self._entries))
            del self._entries[oldest]

    def drain(self) -> str:
        """取出全部已积累 skill body 并清空，返回可直接前置到 user message 的段。"""
        if not self._entries:
            return ""
        snapshot = list(self._entries.items())
        self._entries.clear()
        parts = []
        for name, body in snapshot:
            parts.append(f"## 已加载 Skill：{name}\n{body.strip()}\n\n")
        parts.append("---\n")
        return "".join(parts)

    def is_empty(self) -> bool:
        return not self._entries

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
