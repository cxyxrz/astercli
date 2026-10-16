"""代码分块器：tree-sitter AST 符号级分块 + 文本分段兜底。

对齐 Java CodeChunker 的块语义，符号提取从 JavaParser 换成 tree-sitter：
- class 块：声明头前 5 行（浓缩声明信息，便于命中后跳读）
- method 块：完整函数体文本
- file 块：无符号声明文件（配置/文档/文本）的整文件兜底，超 2000 字符按行边界分段

语言覆盖：python / java / javascript / typescript(tsx) / go / rust / c / cpp。
对应语言的 tree-sitter grammar 未安装或解析失败时，回退整文件文本分段，
保证索引永远可用（对齐 Java"索引失败不致命"的工程姿态）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..tools.builtin.files import EXCLUDED_DIRS

# 对齐 Java CodeChunker.MAX_CHUNK_CHARS
MAX_CHUNK_CHARS = 2000
# class 块取声明头行数（对齐 Java 类块"浓缩信息"语义）
CLASS_HEADER_LINES = 5
# 单次索引文件数上限，防误索引巨型 monorepo / 误传家目录
MAX_INDEX_FILES = 20000
MAX_FILE_BYTES = 512 * 1024

# 支持索引的扩展名（对齐 Java collectFiles 的常见集合）
SUPPORTED_EXTS = {
    ".java", ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".go", ".rs", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".md", ".json", ".xml", ".yml", ".yaml", ".sql", ".html", ".css",
    ".txt", ".properties", ".toml", ".cfg", ".ini",
}

# RAG 额外排除目录（在 tools 通用排除集合之外）
RAG_EXCLUDED_DIRS = EXCLUDED_DIRS | {
    "out", "coverage", ".next", ".nuxt", ".tox", ".mypy_cache", "site-packages",
}


@dataclass
class CodeChunk:
    """一个可被嵌入与检索的代码块。

    file_path 存相对项目根的 posix 路径（对齐 Java 相对路径展示习惯）。
    """

    file_path: str
    chunk_type: str          # file | class | method
    name: str
    content: str
    start_line: int
    end_line: int

    def to_embed_text(self) -> str:
        """嵌入文本 = "[类型:名字] 内容"（对齐 Java toEmbeddingText，让向量携带符号语义）。"""
        return f"[{self.chunk_type}:{self.name}] {self.content}"


class _LangSpec:
    """一种语言的 AST 声明节点映射。

    loader 延迟 import grammar 包（未安装时 chunk_file 自动回退文本分段）。
    declarator_types 里的节点声明名在 name 字段之外（C/C++ 的函数名藏在
    declarator 子树里），需要专用提取逻辑。
    """

    def __init__(self, loader: Callable, class_types: tuple, method_types: tuple,
                 declarator_types: tuple = ()) -> None:
        self.loader = loader
        self.class_types = class_types
        self.method_types = method_types
        self.declarator_types = declarator_types


def _load_ts_python():
    import tree_sitter_python
    return tree_sitter_python.language()


def _load_ts_java():
    import tree_sitter_java
    return tree_sitter_java.language()


def _load_ts_javascript():
    import tree_sitter_javascript
    return tree_sitter_javascript.language()


def _load_ts_typescript():
    import tree_sitter_typescript
    return tree_sitter_typescript.language_typescript()


def _load_ts_tsx():
    import tree_sitter_typescript
    return tree_sitter_typescript.language_tsx()


def _load_ts_go():
    import tree_sitter_go
    return tree_sitter_go.language()


def _load_ts_rust():
    import tree_sitter_rust
    return tree_sitter_rust.language()


def _load_ts_c():
    import tree_sitter_c
    return tree_sitter_c.language()


def _load_ts_cpp():
    import tree_sitter_cpp
    return tree_sitter_cpp.language()


_LANG_SPECS = {
    ".py": _LangSpec(
        _load_ts_python,
        class_types=("class_definition",),
        method_types=("function_definition",),
    ),
    ".java": _LangSpec(
        _load_ts_java,
        class_types=("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"),
        method_types=("method_declaration",),
    ),
    ".js": _LangSpec(
        _load_ts_javascript,
        class_types=("class_declaration",),
        method_types=("function_declaration", "method_definition"),
    ),
    ".ts": _LangSpec(
        _load_ts_typescript,
        class_types=("class_declaration",),
        method_types=("function_declaration", "method_definition"),
    ),
    ".go": _LangSpec(
        _load_ts_go,
        class_types=("type_spec",),
        method_types=("function_declaration", "method_declaration"),
    ),
    ".rs": _LangSpec(
        _load_ts_rust,
        class_types=("struct_item", "enum_item", "trait_item", "impl_item"),
        method_types=("function_item",),
    ),
    ".c": _LangSpec(
        _load_ts_c,
        class_types=(),
        method_types=("function_definition",),
        declarator_types=("function_definition",),
    ),
    ".cpp": _LangSpec(
        _load_ts_cpp,
        class_types=("class_specifier",),
        method_types=("function_definition",),
        declarator_types=("function_definition",),
    ),
}
# 语法别名
_LANG_SPECS[".jsx"] = _LANG_SPECS[".js"]
_LANG_SPECS[".mjs"] = _LANG_SPECS[".js"]
_LANG_SPECS[".cjs"] = _LANG_SPECS[".js"]
_LANG_SPECS[".tsx"] = _LANG_SPECS[".c"]  # 占位，下一行覆盖
_LANG_SPECS[".tsx"] = _LangSpec(
    _load_ts_tsx,
    class_types=("class_declaration",),
    method_types=("function_declaration", "method_definition"),
)
_LANG_SPECS[".cc"] = _LANG_SPECS[".cpp"]
_LANG_SPECS[".cxx"] = _LANG_SPECS[".cpp"]
_LANG_SPECS[".hpp"] = _LANG_SPECS[".cpp"]
_LANG_SPECS[".hh"] = _LANG_SPECS[".cpp"]
_LANG_SPECS[".h"] = _LANG_SPECS[".c"]

_PARSER_CACHE: dict = {}


def _get_parser(ext: str):
    """按扩展名取缓存的 tree-sitter Parser；语言不受支持或 grammar 缺失返回 None。"""
    if ext in _PARSER_CACHE:
        return _PARSER_CACHE[ext]
    spec = _LANG_SPECS.get(ext)
    if spec is None:
        _PARSER_CACHE[ext] = None
        return None
    try:
        from tree_sitter import Language, Parser

        parser = Parser(Language(spec.loader()))
    except Exception:
        parser = None
    _PARSER_CACHE[ext] = parser
    return parser


def _node_name(node, source: bytes, spec: _LangSpec) -> Optional[str]:
    """提取声明节点名称；C/C++ 函数定义从 declarator 子树找标识符。"""
    if node.type in spec.declarator_types:
        declarator = node.child_by_field_name("declarator")
        if declarator is None:
            return None
        stack = [declarator]
        while stack:
            cur = stack.pop(0)
            if cur.type in ("identifier", "qualified_identifier", "field_identifier") and cur.child_count == 0:
                return source[cur.start_byte:cur.end_byte].decode("utf-8", errors="replace")
            stack.extend(cur.children)
        return None
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")


def _header_text(node, source: bytes, text_lines: list) -> tuple:
    """class 块取声明头前 5 行（对齐 Java 类块浓缩语义）。"""
    start = node.start_point[0]
    end = min(node.end_point[0] + 1, start + CLASS_HEADER_LINES)
    content = "\n".join(text_lines[start:end])
    return content, start + 1, end


def _full_text(node, source: bytes, text_lines: list) -> tuple:
    start = node.start_point[0]
    end = node.end_point[0] + 1
    content = "\n".join(text_lines[start:end])
    return content, start + 1, end


def _split_text(content: str, name: str, file_path: str, start_line: int) -> list:
    """超长文本按行边界分段（对齐 Java chunkLargeText 语义）。

    行累积超过 MAX_CHUNK_CHARS 切段；单行本身超限则硬切。
    """
    if len(content) <= MAX_CHUNK_CHARS:
        return [CodeChunk(file_path, "file", name, content, start_line,
                          start_line + content.count("\n"))]
    chunks: list = []
    buf: list = []
    buf_chars = 0
    line_no = start_line
    seg_start = start_line
    for line in content.split("\n"):
        line_len = len(line) + 1
        if buf and buf_chars + line_len > MAX_CHUNK_CHARS:
            text = "\n".join(buf)
            suffix = "" if len(chunks) == 0 else f" (part {len(chunks) + 1})"
            chunks.append(CodeChunk(file_path, "file", name + suffix, text,
                                    seg_start, line_no - 1))
            buf, buf_chars, seg_start = [], 0, line_no
        if line_len > MAX_CHUNK_CHARS:
            for i in range(0, len(line), MAX_CHUNK_CHARS):
                buf.append(line[i:i + MAX_CHUNK_CHARS])
                buf_chars = MAX_CHUNK_CHARS
                text = "\n".join(buf)
                suffix = "" if len(chunks) == 0 else f" (part {len(chunks) + 1})"
                chunks.append(CodeChunk(file_path, "file", name + suffix, text,
                                        seg_start, line_no))
                buf, buf_chars, seg_start = [], 0, line_no
        else:
            buf.append(line)
            buf_chars += line_len
        line_no += 1
    if buf:
        text = "\n".join(buf)
        suffix = "" if len(chunks) == 0 else f" (part {len(chunks) + 1})"
        chunks.append(CodeChunk(file_path, "file", name + suffix, text,
                                seg_start, line_no - 1))
    return chunks


def chunk_file(abs_path: Path, rel_path: str, text: str) -> list:
    """单个文件 → CodeChunk 列表。

    支持语言走 AST 符号分块；解析无符号 / 不支持 / 异常一律回退
    整文件 file 块（超 2000 字符自动分段）。
    """
    ext = abs_path.suffix.lower()
    parser = _get_parser(ext)
    if parser is not None:
        try:
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            tree = None
        if tree is not None:
            spec = _LANG_SPECS[ext]
            source = text.encode("utf-8")
            text_lines = text.split("\n")
            chunks: list = []
            # 显式栈遍历，避免深文件递归爆栈
            stack = [tree.root_node]
            while stack:
                node = stack.pop()
                if node.type in spec.class_types:
                    nm = _node_name(node, source, spec)
                    if nm:
                        content, s, e = _header_text(node, source, text_lines)
                        chunks.append(CodeChunk(rel_path, "class", nm, content, s, e))
                elif node.type in spec.method_types:
                    nm = _node_name(node, source, spec)
                    if nm:
                        content, s, e = _full_text(node, source, text_lines)
                        chunks.append(CodeChunk(rel_path, "method", nm, content, s, e))
                stack.extend(reversed(node.children))
            if chunks:
                return chunks
    # 兜底：整文件 file 块
    return _split_text(text, abs_path.name, rel_path, 1)


def collect_files(root: Path) -> list:
    """收集项目内可索引文件（扩展名白名单 + 排除目录 + 大小/数量预算）。"""
    results: list = []
    for path in sorted(root.rglob("*")):
        if len(results) >= MAX_INDEX_FILES:
            break
        if not path.is_file():
            continue
        # 相对路径任一段命中排除目录即跳过（rglob 无法剪枝，靠路径判断）
        rel = path.relative_to(root)
        if any(part in RAG_EXCLUDED_DIRS or part.startswith(".")
               for part in rel.parts[:-1]):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        results.append(path)
    return results
