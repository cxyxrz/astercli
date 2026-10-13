"""危险操作识别策略（对齐 Java hitl.ApprovalPolicy）。

设计原则：
- 读取类操作（read_file / list_dir / glob_files / grep_files）不需要确认，无副作用
- 写入/执行类操作（write_file / execute_command）需要确认，有潜在破坏性
- create_project 属于写入操作，默认需要确认（工具落地后自动纳入）
- revert_turn 会批量回写工作区文件，默认需要确认（工具落地后自动纳入）
- MCP 工具来自外部 server（P5），默认都需要确认

拦截顺序约定（对齐 Java）：HITL 审批 → ToolRegistry 执行 → PathGuard / CommandGuard。
"""

from __future__ import annotations

from typing import Optional

DANGEROUS_TOOLS = frozenset(
    {
        "write_file",
        "execute_command",
        "create_project",
        "revert_turn",
    }
)


def is_mcp_tool(tool_name: Optional[str]) -> bool:
    return bool(tool_name) and tool_name.startswith("mcp__")


def requires_approval(tool_name: Optional[str], tool_def=None) -> bool:
    """该工具调用是否需要人工确认。

    静态规则（对齐 Java ApprovalPolicy.requiresApproval）∪ 注册时声明的
    ``dangerous`` 标记，两个来源取并集，避免双轨维护。
    """
    if not tool_name:
        return False
    if tool_name in DANGEROUS_TOOLS or is_mcp_tool(tool_name):
        return True
    return bool(getattr(tool_def, "dangerous", False))


def danger_level(tool_name: Optional[str]) -> str:
    if tool_name == "execute_command" or tool_name == "revert_turn":
        return "🔴 高危"
    if tool_name == "write_file" or tool_name == "create_project":
        return "🟡 中危"
    if is_mcp_tool(tool_name):
        return "🟡 MCP"
    return "🟢 安全"


def risk_description(tool_name: Optional[str]) -> str:
    if tool_name == "execute_command":
        return "将在系统上执行 Shell 命令，可能修改文件、安装软件或影响系统状态"
    if tool_name == "revert_turn":
        return "将按 Side-Git 快照批量恢复工作区文件，可能覆盖当前未保存修改"
    if tool_name == "write_file":
        return "将写入或覆盖文件内容，原有内容将丢失"
    if tool_name == "create_project":
        return "将在磁盘上创建新目录和文件"
    if is_mcp_tool(tool_name):
        return "将调用外部 MCP server 提供的工具，可能访问网络、文件或第三方服务"
    return "安全的只读操作"


def mcp_server_name(tool_name: Optional[str]) -> Optional[str]:
    if not is_mcp_tool(tool_name):
        return None
    parts = tool_name.split("__", 3)
    return parts[1] if len(parts) >= 2 else None
