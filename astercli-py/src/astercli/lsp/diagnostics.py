"""LSP 诊断数据模型（对齐 Java lsp.LspDiagnostic / LspSeverity / LspDiagnosticReport）。

对齐点：
- record 紧凑构造器的默认值规范化（None severity → INFO、行列为负 → 1、
  source 空白 → "lsp"）
- 报告双通道：prompt_text 进 LLM 上下文、display_text 进终端渲染
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LspSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def rank(self) -> int:
        """排序权重：error < warning < info（数字越小越靠前，对齐 Java severityRank）。"""
        return {LspSeverity.ERROR: 0, LspSeverity.WARNING: 1, LspSeverity.INFO: 2}[self]


@dataclass(frozen=True)
class LspDiagnostic:
    severity: LspSeverity
    file_path: str
    line: int
    column: int
    message: str
    source: str

    def __post_init__(self) -> None:
        # 对齐 Java record 紧凑构造器的默认值规范化
        object.__setattr__(self, "severity", self.severity or LspSeverity.INFO)
        object.__setattr__(self, "file_path", self.file_path or "")
        object.__setattr__(self, "line", max(1, self.line))
        object.__setattr__(self, "column", max(1, self.column))
        object.__setattr__(self, "message", self.message or "")
        object.__setattr__(self, "source", self.source or "lsp")


@dataclass(frozen=True)
class LspDiagnosticReport:
    prompt_text: str
    display_text: str

    @classmethod
    def empty(cls) -> "LspDiagnosticReport":
        return cls(prompt_text="", display_text="")

    def is_empty(self) -> bool:
        return not self.prompt_text or not self.prompt_text.strip()


# 对齐 Java LspDiagnosticReport.EMPTY 静态常量
EMPTY_REPORT = LspDiagnosticReport(prompt_text="", display_text="")
