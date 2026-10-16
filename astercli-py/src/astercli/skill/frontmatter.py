"""SKILL.md frontmatter 解析器（直译 Java SkillFrontmatterParser，不引入 PyYAML）。

支持的语法（覆盖 95% 实际写法）：
- 单行 key: value（引号自动剥离）
- 多行 key: |（以首行缩进推断，空白折叠为单行）
- 行内数组 key: [a, b, c]

不支持（命中即 warnings 并跳过该字段，不阻塞整个 skill 加载）：
- 嵌套对象 key: { nested: ... }
- YAML anchor / alias / merge key
- 复杂 YAML 类型（!!str 等）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ParseResult:
    frontmatter: Dict[str, object] = field(default_factory=dict)
    body: str = ""
    warnings: List[str] = field(default_factory=list)


def parse(full_text: str) -> ParseResult:
    if full_text is None:
        return ParseResult(warnings=["SKILL.md 内容为 null"])
    normalized = full_text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return ParseResult(frontmatter={}, body=normalized, warnings=["缺少 frontmatter 起始标记 ---"])

    end_idx = _find_frontmatter_end(normalized)
    if end_idx < 0:
        return ParseResult(frontmatter={}, body=normalized, warnings=["缺少 frontmatter 结束标记 ---"])

    frontmatter_text = normalized[4:end_idx]
    body = normalized[end_idx + 4:]
    if body.startswith("\n"):
        body = body[1:]

    warnings: List[str] = []
    frontmatter = _parse_frontmatter(frontmatter_text, warnings)
    return ParseResult(frontmatter=frontmatter, body=body, warnings=warnings)


def _find_frontmatter_end(text: str) -> int:
    idx = 4
    while idx < len(text):
        line_end = text.find("\n", idx)
        if line_end < 0:
            return -1
        if text[idx:line_end] == "---":
            return idx
        idx = line_end + 1
    return -1


def _find_key_colon_index(line: str) -> int:
    """找 key 冒号位置；引号内的冒号不算（对齐 Java findKeyColonIndex）。"""
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            return i
    return -1


def _leading_spaces(s: str) -> int:
    i = 0
    while i < len(s) and s[i] == " ":
        i += 1
    return i


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_frontmatter(text: str, warnings: List[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        colon_idx = _find_key_colon_index(line)
        if colon_idx < 0:
            warnings.append(f"无法解析的 frontmatter 行: {line}")
            i += 1
            continue
        key = line[:colon_idx].strip()
        raw_value = line[colon_idx + 1:].strip()
        if not key:
            warnings.append(f"frontmatter 行缺少 key: {line}")
            i += 1
            continue
        if not raw_value:
            warnings.append(f"frontmatter 字段 '{key}' 缺少值或使用了不支持的嵌套结构")
            i += 1
            continue
        if raw_value.startswith("{"):
            warnings.append(f"frontmatter 字段 '{key}' 使用了不支持的嵌套对象语法")
            i += 1
            continue
        if raw_value == "|" or raw_value.startswith("|"):
            # 多行值：以首个非空行的缩进为基准，遇更浅缩进结束
            collected: List[str] = []
            i += 1
            base_indent = None
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    collected.append("")
                    i += 1
                    continue
                indent = _leading_spaces(nxt)
                if indent == 0:
                    break
                if base_indent is None:
                    base_indent = indent
                if indent < base_indent:
                    break
                collected.append(nxt[base_indent:])
                i += 1
            result[key] = " ".join(part for part in collected if part).strip()
            continue
        if raw_value.startswith("[") and raw_value.endswith("]"):
            inner = raw_value[1:-1].strip()
            items: List[str] = []
            if inner:
                for part in inner.split(","):
                    trimmed = _unquote(part.strip())
                    if trimmed:
                        items.append(trimmed)
            result[key] = items
            i += 1
            continue
        result[key] = _unquote(raw_value)
        i += 1
    return result
