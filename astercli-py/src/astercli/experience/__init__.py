"""经验系统（P9）：反思触发采集 + Markdown 归档 + 检索注入。

对齐设计（详见 .trae/documents/python-langchain-rewrite.md）：
- 层一：工具失败 / HITL 拒绝 / LSP error 三类触发 → 反思提示词 → 结构化经验
- 层二：一经验一 Markdown（``~/.astercli-py/experience`` + ``<项目根>/.astercli/experience``）
  + 目录 index.md + 容量淘汰 + ``/experience`` 管理命令
- 层三（P10）：完整的动态检索注入策略；P9 只做启动期索引段 + 任务级 top-K
"""

from __future__ import annotations

from .models import (
    CATEGORY_ERROR_LESSON,
    CATEGORY_LABELS,
    CATEGORY_USER_CORRECTION,
    CATEGORY_WORKFLOW,
    TRIGGER_HITL_REJECT,
    TRIGGER_LABELS,
    TRIGGER_LSP_ERROR,
    TRIGGER_TOOL_FAILURE,
    VALID_CATEGORIES,
    VALID_TRIGGERS,
    ExperienceEntry,
    ReflectionTrigger,
    args_skeleton,
    first_line,
)
from .reflector import ReflectionCollector, Reflector, parse_reflection_output
from .store import ExperienceStore

__all__ = [
    "CATEGORY_ERROR_LESSON",
    "CATEGORY_LABELS",
    "CATEGORY_USER_CORRECTION",
    "CATEGORY_WORKFLOW",
    "ExperienceEntry",
    "ExperienceStore",
    "ReflectionCollector",
    "ReflectionTrigger",
    "Reflector",
    "TRIGGER_HITL_REJECT",
    "TRIGGER_LABELS",
    "TRIGGER_LSP_ERROR",
    "TRIGGER_TOOL_FAILURE",
    "VALID_CATEGORIES",
    "VALID_TRIGGERS",
    "args_skeleton",
    "first_line",
    "parse_reflection_output",
]
