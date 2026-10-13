"""HITL 审批层（对齐 Java hitl 包）。

拦截顺序：HITL 审批 → ToolRegistry 执行 → PathGuard / CommandGuard。
Python 版审批经 LangGraph ``interrupt()`` 挂起图、``Command(resume=...)`` 恢复，
不依赖并行任务内的回调。
"""

from .approval import ApprovalSession, normalize_decision, parse_hitl_choice
from .policy import (
    DANGEROUS_TOOLS,
    danger_level,
    is_mcp_tool,
    mcp_server_name,
    requires_approval,
    risk_description,
)
from .terminal import HitlPrompter

__all__ = [
    "ApprovalSession",
    "DANGEROUS_TOOLS",
    "HitlPrompter",
    "danger_level",
    "is_mcp_tool",
    "mcp_server_name",
    "normalize_decision",
    "parse_hitl_choice",
    "requires_approval",
    "risk_description",
]
