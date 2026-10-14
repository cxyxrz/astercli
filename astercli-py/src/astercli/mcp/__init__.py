"""MCP 子系统（P5）：官方 ``mcp`` SDK 2.x + ClientSessionGroup 会话池。

对齐 Java 侧拦截顺序约定：HITL 审批（所有 ``mcp__*`` 需审批）→
ToolRegistry 执行 → PathGuard / CommandGuard（MCP 工具不经过本地路径/
命令守卫，由 server 端自治）。

模块切分：
- config：配置加载合并 + ``${VAR}`` 展开 + transport 校验
- runtime：后台线程 event loop（会话跨 REPL 每轮 asyncio.run 存活）
- manager：状态机 + 启动续载 + 动态工具注册（``mcp__{server}__{tool}``）
- mention：``@server:uri`` 解析与 resource 块展开
"""

from .config import McpConfigError, McpServerConfig, load_merged_config
from .manager import McpServerManager, namespaced_tool
from .mention import expand_mentions, parse_at_mentions
from .runtime import McpRuntime, McpRuntimeError

__all__ = [
    "McpConfigError",
    "McpServerConfig",
    "McpServerManager",
    "McpRuntime",
    "McpRuntimeError",
    "expand_mentions",
    "load_merged_config",
    "namespaced_tool",
    "parse_at_mentions",
]
