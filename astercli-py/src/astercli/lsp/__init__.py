"""LSP 诊断（P8）：write_file 出口钩子收集 + ReAct 循环调 LLM 前注入。

对齐 Java lsp 包（Phase 17 轻量形态，进程内解析，非真实 LSP server）。
"""

from .diagnostics import EMPTY_REPORT, LspDiagnostic, LspDiagnosticReport, LspSeverity
from .formatter import DEFAULT_MAX_DIAGNOSTICS, format_diagnostics, max_diagnostics
from .manager import LspManager

__all__ = [
    "EMPTY_REPORT",
    "DEFAULT_MAX_DIAGNOSTICS",
    "LspManager",
    "LspDiagnostic",
    "LspDiagnosticReport",
    "LspSeverity",
    "format_diagnostics",
    "max_diagnostics",
]
