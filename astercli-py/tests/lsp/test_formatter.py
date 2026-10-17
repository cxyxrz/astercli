"""LSP 诊断格式化测试：双通道文案、上限截断、omitted 提示（对齐 Java LspDiagnosticFormatter）。"""

from __future__ import annotations

from astercli.config import Settings
from astercli.lsp import LspDiagnostic, LspSeverity, format_diagnostics


def _diagnostics(count, severity=LspSeverity.ERROR):
    return [
        LspDiagnostic(
            severity=severity,
            file_path=f"src/file{i}.py",
            line=i + 1,
            column=2,
            message="invalid syntax",
            source="ast",
        )
        for i in range(count)
    ]


def test_空列表返回空报告():
    report = format_diagnostics([])
    assert report.is_empty()


def test_格式对齐_java_契约():
    report = format_diagnostics(_diagnostics(1))
    assert report.prompt_text.startswith("[LSP 诊断注入]")
    assert "请优先修复 error" in report.prompt_text
    assert "- [error] src/file0.py:1:2 invalid syntax (ast)" in report.prompt_text
    assert "🔎 LSP 诊断" in report.display_text
    assert "- [error] src/file0.py:1:2 invalid syntax (ast)" in report.display_text


def test_上限截断与_omitted_提示():
    settings = Settings(env={"ASTERCLI_LSP_MAX_DIAGNOSTICS": "2"}, load_files=False)
    report = format_diagnostics(_diagnostics(5), settings)
    assert report.prompt_text.count("[error]") == 2
    assert "还有 3 条诊断未注入（已达到上限 2）" in report.prompt_text
    assert "还有 3 条诊断未注入（已达到上限 2）" in report.display_text


def test_上限默认_20_且非正数回退(monkeypatch):
    monkeypatch.delenv("ASTERCLI_LSP_MAX_DIAGNOSTICS", raising=False)
    settings = Settings(load_files=False)
    report = format_diagnostics(_diagnostics(25), settings)
    assert report.prompt_text.count("[error]") == 20
    assert "还有 5 条诊断未注入（已达到上限 20）" in report.prompt_text


def test_警告与信息级别标签():
    report = format_diagnostics(_diagnostics(1, severity=LspSeverity.WARNING))
    assert "- [warning]" in report.prompt_text
    report = format_diagnostics(_diagnostics(1, severity=LspSeverity.INFO))
    assert "- [info]" in report.prompt_text
