"""文件类内置工具：read_file / write_file / list_dir / glob_files / grep_code。

参数定义与输出格式对齐 Java ToolRegistry（含提示文案），
所有路径先经 PathGuard.resolveSafe。grep_code 暂为纯 Python 扫描引擎
（Java 版优先 ripgrep，性能优化放到后续分期）。
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath
from typing import Optional

from ...policy import PathGuard
from ..registry import Param, ToolRegistry, create_parameters

# 对齐 Java SEARCH_EXCLUDED_DIRS 的常用忽略集合
EXCLUDED_DIRS = {
    ".git", "target", "node_modules", "dist", "build", ".idea",
    ".venv", "venv", "__pycache__", ".astercli", ".paicli",
    "snapshots", ".gradle", ".mvn", ".pytest_cache",
}

MAX_WRITE_FILE_BYTES = 5 * 1024 * 1024   # 单文件 5MB 上限
MAX_READ_FILE_LINES = 2000
MAX_GREP_RESULTS = 200
MAX_GREP_CONTEXT_LINES = 5
DEFAULT_GREP_HEAD_LIMIT = 20
DEFAULT_GREP_MAX_CHARS = 24_000


def register_file_tools(registry: ToolRegistry, path_guard: PathGuard) -> None:
    registry.register(
        "read_file",
        "读取文件内容（仅限项目根目录之内）；可用 offset/limit 按行读取，避免把大文件整段塞进上下文",
        create_parameters(
            Param("path", "string", "文件路径", True),
            Param("offset", "integer", "起始行号，1 表示第一行；省略时读取全文", False),
            Param("limit", "integer", "最多读取多少行；省略时读取全文，最大 2000 行", False),
        ),
        _read_file(path_guard),
    )
    registry.register(
        "write_file",
        "写入文件内容（仅限项目根目录之内，单文件 5MB 上限）",
        create_parameters(
            Param("path", "string", "文件路径", True),
            Param("content", "string", "文件内容", True),
        ),
        _write_file(path_guard, registry),
        dangerous=True,
    )
    registry.register(
        "list_dir",
        "列出目录内容（仅限项目根目录之内）",
        create_parameters(Param("path", "string", "目录路径", True)),
        _list_dir(path_guard),
    )
    registry.register(
        "glob_files",
        "按文件名 glob 查找项目内文件（只读、实时、尊重常见忽略目录）；适合先定位候选文件，例如 **/*Service.java",
        create_parameters(
            Param("pattern", "string", "glob 模式，例如 **/*.java、**/*Controller*、README.md", True),
            Param("path", "string", "搜索起始目录，默认 .", False),
            Param("max_results", "integer", "最多返回结果数，默认 50，上限 200", False),
        ),
        _glob_files(path_guard),
    )
    registry.register(
        "grep_code",
        "在项目内按关键字或正则实时搜索代码（只读、返回文件和行号）；适合精确符号/字符串定位，找到后再 read_file 读取上下文",
        create_parameters(
            Param("pattern", "string", "要搜索的关键字或正则", True),
            Param("path", "string", "搜索起始目录，默认 .", False),
            Param("glob", "string", "可选文件 glob 过滤，例如 **/*.java", False),
            Param("regex", "boolean", "是否按正则解释 pattern，默认 false 表示字面量搜索", False),
            Param("case_sensitive", "boolean", "是否大小写敏感，默认 true", False),
            Param("context_lines", "integer", "每条命中前后上下文行数，默认 0，上限 5", False),
            Param("max_results", "integer", "最多返回命中数，默认 50，上限 200", False),
            Param("head_limit", "integer", "单个文件最多返回多少条命中，默认 20，上限 50", False),
            Param("max_chars", "integer", "单次工具结果字符预算，默认 24000，上限 60000", False),
        ),
        _grep_code(path_guard),
    )


# ---------- read_file ----------


def _read_file(path_guard: PathGuard):
    def read_file(args: dict) -> str:
        safe = path_guard.resolve_safe(args["path"])
        if not safe.is_file():
            return "读取文件失败: 不是普通文件"
        ranged = "offset" in args or "limit" in args
        try:
            if not ranged:
                return "文件内容:\n" + safe.read_text(encoding="utf-8", errors="replace")
            offset = max(1, _parse_int(args.get("offset"), 1))
            limit = max(1, min(_parse_int(args.get("limit"), 200), MAX_READ_FILE_LINES))
            lines = safe.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            if offset > total:
                return f"文件内容: {safe.name} 共 {total} 行，offset 超出范围"
            frm = offset - 1
            to = min(frm + limit, total)
            out = [f"文件内容: {safe.name} (lines {offset}-{to} of {total})"]
            for i in range(frm, to):
                out.append(f"{i + 1:5d} | {lines[i]}")
            if to < total:
                out.append(f"...(已截断，可用 offset={to + 1} 继续读取)")
            return "\n".join(out).rstrip("\n")
        except Exception as e:  # noqa: BLE001 —— 对齐 Java：读文件异常转失败文本
            return f"读取文件失败: {e}"

    return read_file


# ---------- write_file ----------


def _write_file(path_guard: PathGuard, registry: Optional[object] = None):
    def write_file(args: dict) -> str:
        content = args.get("content") or ""
        content_bytes = len(content.encode("utf-8"))
        if content_bytes > MAX_WRITE_FILE_BYTES:
            from ...policy.exceptions import PolicyException

            raise PolicyException(
                f"写入内容 {content_bytes} 字节超过 {MAX_WRITE_FILE_BYTES // 1024 // 1024}MB 上限"
            )
        safe = path_guard.resolve_safe(args["path"])
        try:
            parent = safe.parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            safe.write_text(content, encoding="utf-8")
            # LSP 诊断出口钩子（P8，对齐 Java runPostEditLspHook）：
            # 内部吞异常，失败不影响写入主结果；registry 缺省时 no-op
            if registry is not None:
                registry.run_post_edit_lsp_hook(args["path"], safe)
            return f"文件已写入: {args['path']}"
        except Exception as e:  # noqa: BLE001
            return f"写入文件失败: {e}"

    return write_file


# ---------- list_dir ----------


def _list_dir(path_guard: PathGuard):
    def list_dir(args: dict) -> str:
        safe = path_guard.resolve_safe(args["path"])
        try:
            if not safe.is_dir():
                return "目录为空或不存在"
            entries = sorted(safe.iterdir(), key=lambda p: p.name)
            out = ["目录内容:"]
            for entry in entries:
                out.append(("[D] " if entry.is_dir() else "[F] ") + entry.name)
            return "\n".join(out)
        except Exception as e:  # noqa: BLE001
            return f"列出目录失败: {e}"

    return list_dir


# ---------- glob_files ----------


def _glob_files(path_guard: PathGuard):
    def glob_files(args: dict) -> str:
        pattern = args.get("pattern")
        if not pattern or not pattern.strip():
            return "文件匹配失败: pattern 不能为空"
        root = path_guard.resolve_safe(args.get("path") or ".")
        max_results = _clamp(_parse_int(args.get("max_results"), 50), 1, MAX_GREP_RESULTS)
        project_root = path_guard.root_path

        pattern_parts = PurePosixPath(_normalize_glob(pattern)).parts
        file_name_pattern = PurePosixPath(_normalize_glob(pattern)).name

        matches: list = []
        partial = False
        for path in _walk_files(root):
            rel = path.relative_to(project_root)
            rel_parts = tuple(rel.parts)
            if _glob_match(pattern_parts, rel_parts) or fnmatch.fnmatchcase(
                path.name, file_name_pattern
            ):
                matches.append(rel.as_posix())
                if len(matches) >= max_results:
                    partial = True
                    break

        if not matches:
            return f"未找到匹配文件: {pattern}"
        header = f"匹配文件 {len(matches)} 个"
        if partial:
            header += f"（已达到上限 {max_results}）"
        out = [header + ":"]
        for i, m in enumerate(matches):
            out.append(f"{i + 1}. {m}")
        return "\n".join(out)

    return glob_files


def _normalize_glob(pattern: str) -> str:
    trimmed = pattern.strip()
    while trimmed.startswith("./"):
        trimmed = trimmed[2:]
    return trimmed


def _glob_match(pattern_parts: tuple, parts: tuple) -> bool:
    """支持 ** 的 glob 匹配：** 可匹配任意层级（含 0 层）；其余段用 fnmatch 逐段匹配。"""
    if not pattern_parts:
        return not parts
    head = pattern_parts[0]
    if head == "**":
        return any(
            _glob_match(pattern_parts[1:], parts[i:]) for i in range(len(parts) + 1)
        )
    if not parts:
        return False
    if fnmatch.fnmatchcase(parts[0], head):
        return _glob_match(pattern_parts[1:], parts[1:])
    return False


def _walk_files(root: Path):
    """忽略 EXCLUDED_DIRS 的文件遍历（对齐 Java SearchFileVisitor 的目录裁剪语义）。"""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for fname in filenames:
            yield Path(dirpath) / fname


# ---------- grep_code ----------


def _grep_code(path_guard: PathGuard):
    def grep_code(args: dict) -> str:
        query = args.get("pattern")
        if not query or not query.strip():
            return "代码搜索失败: pattern 不能为空"
        root = path_guard.resolve_safe(args.get("path") or ".")
        project_root = path_guard.root_path
        max_results = _clamp(_parse_int(args.get("max_results"), 50), 1, MAX_GREP_RESULTS)
        context_lines = _clamp(
            _parse_int(args.get("context_lines"), 0), 0, MAX_GREP_CONTEXT_LINES
        )
        regex = _parse_bool(args.get("regex"), False)
        case_sensitive = _parse_bool(args.get("case_sensitive"), True)
        head_limit = _clamp(
            _parse_int(args.get("head_limit"), DEFAULT_GREP_HEAD_LIMIT), 1, 50
        )
        max_chars = _clamp(
            _parse_int(args.get("max_chars"), DEFAULT_GREP_MAX_CHARS), 1_000, 60_000
        )
        glob_filter = args.get("glob")

        try:
            if regex:
                pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(query), 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return f"代码搜索失败: 无效正则: {e}"

        pattern_parts = (
            PurePosixPath(_normalize_glob(glob_filter)).parts if glob_filter else None
        )

        matches: list = []
        partial_reason = ""
        for path in _walk_files(root):
            if len(matches) >= max_results:
                partial_reason = f"匹配数达到 max_results={max_results} 上限"
                break
            rel = path.relative_to(project_root).as_posix()
            rel_parts = tuple(path.relative_to(project_root).parts)
            if pattern_parts and not _glob_match(pattern_parts, rel_parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            file_hits = 0
            for idx, line in enumerate(lines):
                if pattern.search(line):
                    file_hits += 1
                    matches.append((rel, idx + 1, lines))
                    if len(matches) >= max_results:
                        break
                    if file_hits >= head_limit:
                        partial_reason = partial_reason or (
                            f"{rel} 命中数达到 head_limit={head_limit}"
                        )
                        break

        if not matches:
            return f"未找到匹配内容: {query}"

        out = [f"匹配结果 {len(matches)} 条 (engine=python)"]
        if partial_reason:
            out[-1] += f"（partial: {partial_reason}）"
        out[-1] += ":"
        truncated_by_chars = False
        rendered = 0
        for i, (file, line_no, file_lines) in enumerate(matches):
            header = f"{i + 1}. {file}:{line_no}"
            if len("\n".join(out)) + len(header) > max_chars:
                truncated_by_chars = True
                break
            out.append(header)
            ctx_from = max(0, line_no - 1 - context_lines)
            ctx_to = min(len(file_lines), line_no + context_lines)
            for ln in range(ctx_from, ctx_to):
                marker = ">" if ln == line_no - 1 else " "
                ctx_line = f"   {marker}{ln + 1:5d} | {file_lines[ln]}"
                if len("\n".join(out)) + len(ctx_line) > max_chars:
                    truncated_by_chars = True
                    break
                out.append(ctx_line)
            rendered += 1
            if truncated_by_chars:
                break
        if truncated_by_chars:
            out.append(
                f"\npartial: true（已达到 max_chars={max_chars}，"
                "请缩小 path/glob/pattern 或提高 offset 后 read_file）"
            )
        elif partial_reason:
            out.append(f"\npartial: true（{partial_reason}，请缩小 path/glob/pattern 继续搜索）")

        out.append(_suggested_reads(matches[:rendered]))
        return "\n".join(out).rstrip("\n")

    return grep_code


def _suggested_reads(matches: list) -> str:
    if not matches:
        return ""
    seen: set = set()
    lines = ["", "suggested_reads:"]
    for file, line_no, _ in matches:
        if len(seen) >= 3 or file in seen:
            continue
        seen.add(file)
        offset = max(1, line_no - 20)
        escaped = file.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'- read_file {{"path":"{escaped}","offset":{offset},"limit":80}}')
    if len(lines) == 2:
        return ""
    return "\n".join(lines)


# ---------- 解析辅助 ----------


def _parse_int(raw, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool(raw, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return default


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))
