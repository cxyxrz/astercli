"""LSP 诊断数据模型测试：默认值规范化（对齐 Java record 紧凑构造器）与排序权重。"""

from __future__ import annotations

from astercli.lsp import EMPTY_REPORT, LspDiagnostic, LspDiagnosticReport, LspSeverity


def test_diagnostic_默认值规范化():
    d = LspDiagnostic(
        severity=None,
        file_path=None,
        line=-1,
        column=0,
        message=None,
        source="",
    )
    assert d.severity is LspSeverity.INFO
    assert d.file_path == ""
    assert d.line == 1
    assert d.column == 1
    assert d.message == ""
    assert d.source == "lsp"


def test_diagnostic_正常字段保留():
    d = LspDiagnostic(
        severity=LspSeverity.ERROR,
        file_path="src/broken.py",
        line=3,
        column=8,
        message="invalid syntax",
        source="ast",
    )
    assert d.severity is LspSeverity.ERROR
    assert d.file_path == "src/broken.py"
    assert d.line == 3
    assert d.column == 8
    assert d.message == "invalid syntax"
    assert d.source == "ast"


def test_severity_rank_排序权重():
    assert LspSeverity.ERROR.rank() < LspSeverity.WARNING.rank() < LspSeverity.INFO.rank()


def test_empty_report_判定():
    assert EMPTY_REPORT.is_empty()
    assert LspDiagnosticReport.empty().is_empty()
    assert not LspDiagnosticReport(prompt_text="[LSP 诊断注入]", display_text="x").is_empty()
    assert LspDiagnosticReport(prompt_text="   ", display_text="x").is_empty()
