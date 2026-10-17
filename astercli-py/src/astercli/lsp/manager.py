"""进程内解析诊断管理器（对齐 Java lsp.LspManager 的契约与降级语义）。

Java 版用 JavaParser 对 .java 做语法级诊断（轻量形态，非真实 LSP server）；
Python 版按生态对位重设计：
- `.py` → 内置 ast（消息质量最高的核心路径，source="ast"）
- 其余语言 → 复用 P6 RAG 的 tree-sitter grammar（8 语言 + 别名）收集
  ERROR / MISSING 节点（消息较笼统但行列定位可用，source="tree-sitter"）

契约对齐：
- run_post_edit_lsp_hook：write_file 出口钩子；非普通文件 / 不受支持的后缀
  → 清除该文件旧诊断；任何异常静默，不影响写入主路径
- flush_pending_diagnostics：取走全部 pending（severity → 文件 → 行 → 列
  排序）并清空；下一轮 ReAct 调 LLM 前作为 user message 注入
- 开关 `ASTERCLI_LSP_ENABLED` 默认开（仅显式 false 关闭，对齐 Java
  Boolean.parseBoolean 语义）；上限走 formatter 读 `ASTERCLI_LSP_MAX_DIAGNOSTICS`
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import Settings
from .diagnostics import LspDiagnostic, LspDiagnosticReport, LspSeverity
from .formatter import format_diagnostics

log = logging.getLogger(__name__)


class LspManager:
    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or Settings(load_files=False)
        self._pending: Dict[Path, List[LspDiagnostic]] = {}
        self.set_project_path(project_root)

    # ---------- 对齐 Java setProjectPath ----------

    def set_project_path(self, project_root: Optional[Union[str, Path]]) -> None:
        root = project_root or Path.cwd()
        self._project_root = Path(root).resolve()

    # ---------- 对齐 Java runPostEditLspHook ----------

    @property
    def enabled(self) -> bool:
        return self._settings.lsp_enabled()

    def run_post_edit_lsp_hook(
        self, display_path: str, edited_file: Optional[Path]
    ) -> None:
        """write_file 出口钩子：收集或清除单个文件的诊断（异常一律吞掉）。"""
        if not self.enabled or edited_file is None:
            return
        try:
            file = Path(edited_file).resolve()
            if not file.is_file():
                self._clear(file)
                return
            if file.suffix == ".py":
                diagnostics = _diagnose_python(display_path, file, self._project_root)
            else:
                parser = _tree_sitter_parser(file.suffix)
                if parser is None:
                    # 不受支持的后缀（含非代码文件）→ 清除旧诊断，对齐 Java isJavaFile 分支
                    self._clear(file)
                    return
                diagnostics = _diagnose_tree_sitter(display_path, parser, file, self._project_root)
            self._replace(file, diagnostics)
            log.debug("LSP post-edit diagnostics for %s: %s", file, len(diagnostics))
        except Exception:  # noqa: BLE001 —— 对齐 Java：诊断失败不影响主路径
            log.debug("LSP post-edit hook failed for %s", edited_file, exc_info=True)

    # ---------- 对齐 Java flushPendingDiagnostics ----------

    def flush_pending_diagnostics(self) -> LspDiagnosticReport:
        if not self._pending:
            return LspDiagnosticReport.empty()
        diagnostics = sorted(
            (d for ds in self._pending.values() for d in ds),
            key=lambda d: (d.severity.rank(), d.file_path, d.line, d.column),
        )
        self._pending.clear()
        return format_diagnostics(diagnostics, self._settings)

    def pending_diagnostics_snapshot(self) -> List[LspDiagnostic]:
        """pending 只读快照（测试与排障用，不取走）。"""
        return [d for ds in self._pending.values() for d in ds]

    # ---------- 内部 ----------

    def _replace(self, file: Path, diagnostics: List[LspDiagnostic]) -> None:
        if not diagnostics:
            self._pending.pop(file, None)
        else:
            self._pending[file] = list(diagnostics)

    def _clear(self, file: Optional[Path]) -> None:
        if file is not None:
            self._pending.pop(Path(file).resolve(), None)


# ---------- 诊断引擎 ----------


def _diagnose_python(display_path: str, file: Path, project_root: Path) -> List[LspDiagnostic]:
    """内置 ast 解析：SyntaxError 只报首个（ast 语义），消息/行列质量最高。"""
    text = file.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(text)
        return []
    except SyntaxError as e:
        return [
            LspDiagnostic(
                severity=LspSeverity.ERROR,
                file_path=_display_path(display_path, project_root),
                line=max(1, e.lineno or 1),
                column=max(1, e.offset or 1),
                message=clean_message(e.msg or "Python parse error"),
                source="ast",
            )
        ]


def _diagnose_tree_sitter(display_path: str, parser, file: Path, project_root: Path) -> List[LspDiagnostic]:
    """tree-sitter 解析：收集 ERROR / MISSING 节点；不深入问题节点子树，
    避免一处语法错报一串。行列为 0-based，输出转 1-based。"""
    text = file.read_text(encoding="utf-8", errors="replace")
    tree = parser.parse(text.encode("utf-8"))
    path_out = _display_path(display_path, project_root)
    out: List[LspDiagnostic] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            out.append(
                LspDiagnostic(
                    severity=LspSeverity.ERROR,
                    file_path=path_out,
                    line=node.start_point.row + 1,
                    column=node.start_point.column + 1,
                    message="Syntax error",
                    source="tree-sitter",
                )
            )
            continue
        if node.is_missing:
            out.append(
                LspDiagnostic(
                    severity=LspSeverity.ERROR,
                    file_path=path_out,
                    line=node.start_point.row + 1,
                    column=node.start_point.column + 1,
                    message=f"Missing {node.type}",
                    source="tree-sitter",
                )
            )
            continue
        stack.extend(node.children)
    return out


def _tree_sitter_parser(suffix: str):
    """按后缀取缓存的 tree-sitter Parser（复用 P6 chunker 的语言表/加载/缓存）；
    grammar 未安装或语言不受支持返回 None。延迟导入：.py 主路径零 rag 耦合。"""
    try:
        from ..rag.chunker import _get_parser
    except Exception:  # noqa: BLE001 —— rag 子系统不可用时按不支持处理
        return None
    try:
        return _get_parser(suffix)
    except Exception:  # noqa: BLE001
        return None


def _display_path(display_path: Optional[str], project_root: Path) -> str:
    """对齐 Java normalizeDisplayPath：display 缺省回退项目根。"""
    if display_path and display_path.strip():
        return display_path.replace("\\", "/")
    return str(project_root).replace("\\", "/")


def clean_message(message: Optional[str]) -> str:
    """对齐 Java cleanMessage：规范化换行、取首行；空白回退占位提示。"""
    if not message or not message.strip():
        return "Parse error"
    normalized = message.replace("\r\n", "\n").replace("\r", "\n").strip()
    first = normalized.split("\n", 1)[0].strip()
    return first or "Parse error"
