"""诊断格式化（对齐 Java LspDiagnosticFormatter）。

双通道输出：prompt_text 注入 LLM 上下文、display_text 进终端；文案与行格式
对齐 Java（标题 `[LSP 诊断注入]` + 修复指引 + `- [severity] path:line:col message (source)`）。
上限默认 20，`ASTERCLI_LSP_MAX_DIAGNOSTICS` 可调（非正数回退默认）。
"""

from __future__ import annotations

from typing import List, Optional

from ..config import Settings
from .diagnostics import LspDiagnostic, LspDiagnosticReport

DEFAULT_MAX_DIAGNOSTICS = 20


def format_diagnostics(
    diagnostics: List[LspDiagnostic],
    settings: Optional[Settings] = None,
) -> LspDiagnosticReport:
    if not diagnostics:
        return LspDiagnosticReport.empty()
    max_count = max_diagnostics(settings)
    count = min(max_count, len(diagnostics))
    omitted = len(diagnostics) - count

    prompt_lines = [
        "[LSP 诊断注入]",
        "Agent 刚修改代码后，系统收集到以下诊断。请优先修复 error，再处理 warning；"
        "不要原样重复同一处错误写法。",
        "",
    ]
    display_lines = ["🔎 LSP 诊断"]

    for d in diagnostics[:count]:
        line = f"- [{d.severity.value}] {d.file_path}:{d.line}:{d.column} {d.message} ({d.source})"
        prompt_lines.append(line)
        display_lines.append(line)

    if omitted > 0:
        line = f"... 还有 {omitted} 条诊断未注入（已达到上限 {max_count}）"
        prompt_lines.append(line)
        display_lines.append(line)

    return LspDiagnosticReport(
        prompt_text="\n".join(prompt_lines).strip(),
        display_text="\n".join(display_lines).strip(),
    )


def max_diagnostics(settings: Optional[Settings] = None) -> int:
    """注入上限：settings 缺省时读环境，非正数回退默认 20（对齐 Java maxDiagnostics）。"""
    if settings is None:
        settings = Settings(load_files=False)
    return settings.lsp_max_diagnostics()
