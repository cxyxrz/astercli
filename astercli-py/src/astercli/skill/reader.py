"""Skill 目录内文件读取（渐进披露第三层）。

``load_skill`` 只把 SKILL.md 正文推进上下文；skill 目录下的 ``references/``
等附属文件保持"只在磁盘上"，由 LLM 通过 ``read_skill_file`` 按需分页读取。

与 ``read_file`` 的差异只在根：这里以 skill 目录为根，而不是项目根——用户级
skill 位于 ``~/.astercli-py/skills/``，不在项目根内，``read_file`` 会被
PathGuard 直接拒绝。路径围栏复用同一个 PathGuard（含符号链接展开），
所以 ``../`` 与绝对路径逃逸依旧拦得住。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..policy import PathGuard, PolicyException
from .skill import Skill

SKILL_MD = "SKILL.md"
DEFAULT_READ_LINES = 400
MAX_READ_LINES = 2000
MAX_READ_CHARS = 24_000
MAX_LISTED_ENTRIES = 200


def skill_root(skill: Skill) -> Optional[Path]:
    """skill 目录（SKILL.md 的父目录）；内存注册的 skill 没有目录。"""
    if skill.skill_md_path is None:
        return None
    return Path(skill.skill_md_path).parent


def list_skill_files(skill: Skill, limit: int = 20) -> List[str]:
    """列出 skill 目录下最多两层附属文件/目录（跳过 SKILL.md 自身），供注入段展示。"""
    root = skill_root(skill)
    if root is None or not root.is_dir():
        return []
    found: List[str] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.name == SKILL_MD:
            continue
        found.append(entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir():
            try:
                for sub in sorted(entry.iterdir()):
                    found.append(f"{entry.name}/{sub.name}" + ("/" if sub.is_dir() else ""))
            except OSError:
                pass
        if len(found) >= limit:
            return found[:limit] + [f"…（更多请用 read_skill_file 列出 {entry.name}/）"]
    return found


def read_skill_text(
    skill: Skill,
    relative_path: str = SKILL_MD,
    offset: int = 1,
    limit: int = DEFAULT_READ_LINES,
) -> str:
    """读取 skill 目录内文件，返回带行号的可续读文本。

    ``path=SKILL.md`` 返回**去掉 frontmatter 的正文**，与 ``load_skill``
    截断时给出的 offset 语义一致；其他文件原样按行返回。
    """
    root = skill_root(skill)
    if root is None:
        return "读取 skill 文件失败: 该 skill 没有可定位的目录"
    raw = (relative_path or "").strip() or SKILL_MD
    try:
        guard = PathGuard(root)
        safe = guard.resolve_safe(raw)
    except PolicyException as e:
        return f"读取 skill 文件失败: {e}"
    except (OSError, ValueError) as e:
        return f"读取 skill 文件失败: {e}"

    if safe.is_dir():
        return _list_directory(safe, raw)
    if not safe.is_file():
        return f"读取 skill 文件失败: {raw} 不存在（可用 read_skill_file 不带 path 查看 SKILL.md）"

    if safe == guard.root_path / SKILL_MD:
        lines = (skill.body or "").splitlines()
        label = "SKILL.md 正文（frontmatter 已剥离）"
    else:
        try:
            lines = safe.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return f"读取 skill 文件失败: {e}"
        label = str(safe.relative_to(guard.root_path))
    return _slice(lines, offset, limit, label)


def _list_directory(directory: Path, display: str) -> str:
    try:
        entries = sorted(directory.iterdir())
    except OSError as e:
        return f"读取 skill 文件失败: {e}"
    if not entries:
        return f"目录内容: {display}（空）"
    out = [f"目录内容: {display}"]
    for entry in entries[:MAX_LISTED_ENTRIES]:
        out.append(f"- {entry.name}" + ("/" if entry.is_dir() else ""))
    if len(entries) > MAX_LISTED_ENTRIES:
        out.append(f"...（仅列出前 {MAX_LISTED_ENTRIES} 项）")
    return "\n".join(out)


def _slice(lines: List[str], offset: int, limit: int, label: str) -> str:
    total = len(lines)
    if total == 0:
        return f"{label}: 文件为空"
    offset = _to_int(offset, 1)
    if offset > total:
        return f"{label}: 共 {total} 行，offset 超出范围"
    limit = max(1, min(_to_int(limit, DEFAULT_READ_LINES), MAX_READ_LINES))
    start = offset - 1
    end = min(start + limit, total)
    out = [f"{label} (lines {offset}-{end} of {total})"]
    budget = MAX_READ_CHARS
    for i in range(start, end):
        line = f"{i + 1:5d} | {lines[i]}"
        if budget - len(line) < 0:
            out.append(f"...(已达单次字符上限，可用 offset={i + 1} 继续读取)")
            return "\n".join(out)
        budget -= len(line) + 1
        out.append(line)
    if end < total:
        out.append(f"...(已截断，可用 offset={end + 1} 继续读取)")
    return "\n".join(out).rstrip("\n")


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
