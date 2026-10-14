"""MCP server 管理器（对齐 Java mcp.McpServerManager 的生命周期与注册语义）。

行为契约（与 Java 对齐）：
- 状态机（Java McpServerStatus）：``STARTING → READY / ERROR``；``DISABLED``
  来自配置项或运行期 ``/mcp disable``。
- 启动期不阻塞首屏：``startup()`` 同步最多等 ``ASTERCLI_MCP_STARTUP_WAIT_SECONDS``
  秒（默认 8），未完成的保留 STARTING 由后台续载完成注册，``/mcp`` 查看最新状态。
- 动态工具命名 ``mcp__{server}__{tool}``；注册 ``dangerous=True`` 进入 HITL 审批
  （policy.requires_approval 对 ``mcp__`` 前缀的静态规则是双保险）。
- resources 虚拟工具：server 支持 resources 时追加 ``list_resources`` /
  ``read_resource`` 两个虚拟工具（对齐 Java McpResourceTool）。
- 重复注册走 replace 语义（对齐 Java replaceMcpToolOutputsForServer）：
  ``/mcp restart`` 先注销该 server 全部旧工具再注册新集合。
- MCP 工具结果超 64k 由 ToolRegistry 统一出口截断兜底（MCP 层不重复截断）。
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..policy import record_audit
from ..tools import ToolRegistry
from .config import McpConfigError, McpServerConfig, load_merged_config
from .runtime import McpRuntime

STATUS_STARTING = "STARTING"
STATUS_READY = "READY"
STATUS_DISABLED = "DISABLED"
STATUS_ERROR = "ERROR"

#: mention / 资源读取的同步等待上限（秒）——REPL 同步展开不能无限挂起
_MENTION_TIMEOUT_SECONDS = 30.0

#: mention 展开的单资源截断预算（对齐 Java AtMentionExpander 的 20 万字符）
_MENTION_MAX_CHARS = 200_000


@dataclass
class McpServerState:
    config: McpServerConfig
    status: str = STATUS_STARTING
    session: object = None
    tool_names: list = field(default_factory=list)
    supports_resources: bool = False
    error: str = ""


def _safe_name(name: str) -> str:
    """server 名进入工具命名空间前做净化（工具名不能含空白等字符）。"""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def namespaced_tool(server: str, tool: str) -> str:
    return f"mcp__{_safe_name(server)}__{tool}"


class McpServerManager:
    def __init__(
        self,
        registry: ToolRegistry,
        settings: Settings,
        *,
        root: Optional[Path] = None,
        renderer=None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._root = (root or Path.cwd()).resolve()
        self._renderer = renderer  # 可选：启动摘要与错误提示；None 静默（测试路径）
        self._runtime = McpRuntime()
        self._servers: dict = {}          # name -> McpServerState
        self._registered: dict = {}       # name -> [已注册工具名]（replace 语义用）
        self._startup_done = False

    # ---------- 启动 ----------

    def startup(self) -> None:
        """加载配置并启动所有 server（同步等待有限时长，超时部分后台续载）。"""
        self._startup_done = True
        try:
            configs = load_merged_config(self._settings, self._root)
        except McpConfigError as e:
            self._warn(f"MCP 配置加载失败: {e}")
            return
        if not configs:
            return
        if not self._runtime.start():
            self._warn("MCP runtime 启动失败，本次会话 MCP 不可用。")
            return

        pending = []
        for name, config in configs.items():
            state = McpServerState(config=config)
            self._servers[name] = state
            if config.disabled:
                state.status = STATUS_DISABLED
                continue
            pending.append(self._launch_connect(name, state))

        if pending:
            wait_seconds = self._settings.mcp_startup_wait_seconds()
            # 对齐 Java：有限等待；未完成的保持 STARTING，done callback 后台续注册
            concurrent.futures.wait(pending, timeout=wait_seconds)

        if self._servers:
            self._info(self.startup_summary())

    def startup_summary(self) -> str:
        if not self._servers:
            return "MCP: 无配置 server。"
        ready = sum(1 for s in self._servers.values() if s.status == STATUS_READY)
        starting = sum(1 for s in self._servers.values() if s.status == STATUS_STARTING)
        error = sum(1 for s in self._servers.values() if s.status == STATUS_ERROR)
        disabled = sum(1 for s in self._servers.values() if s.status == STATUS_DISABLED)
        parts = [f"就绪 {ready}"]
        if starting:
            parts.append(f"启动中 {starting}")
        if error:
            parts.append(f"失败 {error}")
        if disabled:
            parts.append(f"禁用 {disabled}")
        tools = sum(len(s.tool_names) for s in self._servers.values())
        return (
            f"MCP: {len(self._servers)} 个 server（{' / '.join(parts)}），"
            f"工具 {tools} 个，/mcp 查看详情"
        )

    # ---------- 连接（后台 loop 协程） ----------

    def _launch_connect(self, name: str, state: McpServerState):
        config = state.config
        # 先置 STARTING 再 submit：同步完成的 fake / 极快连接下，
        # done callback 可能先于本函数返回执行，不能把已完成状态覆盖回启动中
        state.status = STATUS_STARTING
        state.error = ""
        # connect 属生命周期调用：必须在 runtime 常驻 _main 任务里执行
        # （anyio cancel scope 进出同 task），故走 submit_lifecycle 而非 submit
        future = self._runtime.submit_lifecycle(lambda group: self._connect(group, config))
        future.add_done_callback(lambda f, n=name: self._on_connect_done(n, f))
        return future

    async def _connect(self, group, config: McpServerConfig):
        params = self._build_params(config)
        session = await group.connect_to_server(params)
        tools = list((await session.list_tools()).tools)
        supports_resources = True
        try:
            await session.list_resources()
        except Exception:  # noqa: BLE001 —— server 不支持 resources 属正常路径
            supports_resources = False
        return session, tools, supports_resources

    def _build_params(self, config: McpServerConfig):
        if config.transport == "stdio":
            from mcp import StdioServerParameters

            return StdioServerParameters(
                command=config.command, args=list(config.args), env=dict(config.env)
            )
        if config.transport == "sse":
            from mcp.client.session_group import SseServerParameters

            return SseServerParameters(url=config.url, headers=dict(config.headers))
        from mcp.client.session_group import StreamableHttpParameters

        return StreamableHttpParameters(url=config.url, headers=dict(config.headers))

    def _on_connect_done(self, name: str, future) -> None:
        state = self._servers.get(name)
        if state is None:
            return
        exc = future.exception()
        if exc is not None:
            state.status = STATUS_ERROR
            state.error = str(exc)
            self._warn(f"MCP server '{name}' 连接失败: {exc}")
            return
        session, tools, supports_resources = future.result()
        state.session = session
        state.supports_resources = supports_resources
        state.tool_names = self._register_tools(name, session, tools, supports_resources)
        state.status = STATUS_READY
        state.error = ""

    # ---------- 工具注册（replace 语义） ----------

    def _register_tools(self, name: str, session, tools, supports_resources: bool) -> list:
        self.unregister_server(name)
        registered = []
        for tool in tools:
            schema = getattr(tool, "inputSchema", None)
            parameters = schema if isinstance(schema, dict) else {"type": "object"}
            description = (getattr(tool, "description", "") or "").strip()
            tool_name = namespaced_tool(name, tool.name)
            self._registry.register(
                tool_name,
                f"[MCP:{name}] {description or tool.name}",
                parameters,
                _make_tool_func(self, name, tool.name),
                dangerous=True,
            )
            registered.append(tool_name)
        if supports_resources:
            list_name = namespaced_tool(name, "list_resources")
            self._registry.register(
                list_name,
                f"[MCP:{name}] 列出该 MCP server 提供的资源（uri / 名称 / 类型）",
                {"type": "object"},
                _make_resource_list_func(self, name),
                dangerous=True,
            )
            read_name = namespaced_tool(name, "read_resource")
            self._registry.register(
                read_name,
                f"[MCP:{name}] 读取该 MCP server 上指定 uri 的资源内容",
                {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "资源 uri，先用 list_resources 获取可用列表",
                        }
                    },
                    "required": ["uri"],
                },
                _make_resource_read_func(self, name),
                dangerous=True,
            )
            registered.extend([list_name, read_name])
        self._registered[name] = registered
        return registered

    def unregister_server(self, name: str) -> None:
        """注销该 server 的全部旧工具（restart / disable 前；replace 语义）。"""
        for tool_name in self._registered.pop(name, []):
            self._registry.unregister(tool_name)
        state = self._servers.get(name)
        if state is not None:
            state.tool_names = []

    # ---------- 工具执行（runtime 桥接） ----------

    async def call_tool_async(self, server: str, tool: str, args: dict) -> str:
        result = await self._call_session(server, lambda session: session.call_tool(tool, args or {}))
        if result is None:
            return f"MCP server '{server}' 未就绪，无法调用 {tool}（/mcp 查看状态）"
        return format_tool_result(result)

    def _session_of(self, server: str) -> Optional[object]:
        state = self._servers.get(server)
        if state is None or state.status != STATUS_READY:
            return None
        return state.session

    async def _call_session(self, server, coro_factory):
        session = self._session_of(server)
        if session is None:
            return None

        def coro(_group):
            return coro_factory(session)

        return await self._runtime.call_async(coro)

    # ---------- resources / mention ----------

    async def list_resources_async(self, server: str) -> str:
        result = await self._call_session(server, lambda s: s.list_resources())
        if result is None:
            return f"MCP server '{server}' 未就绪（/mcp 查看状态）"
        return format_resource_list(result)

    async def read_resource_async(self, server: str, uri: str) -> str:
        result = await self._call_session(server, lambda s: s.read_resource(uri))
        if result is None:
            return f"MCP server '{server}' 未就绪（/mcp 查看状态）"
        return format_resource_contents(result)

    def read_resource_for_mention(self, server: str, uri: str):
        """mention 同步读取：返回 ``(ok, text, mime_type)``。

        对齐 Java readResourceForMention：server 不存在或未就绪直接失败；
        结果按 allowByMention 记审计（approver 语义即 mention）。
        """
        session = self._session_of(server)
        if session is None:
            record_audit(namespaced_tool(server, "mention"), uri, allowed=False)
            return False, "", "", f"MCP server '{server}' 不存在或未就绪"

        def coro(_group):
            return session.read_resource(uri)

        try:
            result = self._runtime.call_sync(coro, timeout=_MENTION_TIMEOUT_SECONDS)
        except Exception as e:  # noqa: BLE001 —— mention 失败不中断输入
            record_audit(namespaced_tool(server, "mention"), uri, allowed=False)
            return False, "", "", str(e)
        text, mime = extract_resource_text(result)
        if len(text) > _MENTION_MAX_CHARS:
            text = text[:_MENTION_MAX_CHARS] + f"\n\n⚠️ 资源内容已截断（>{_MENTION_MAX_CHARS} 字符）"
        record_audit(namespaced_tool(server, "mention"), uri, allowed=True)
        return True, text, mime, ""

    # ---------- /mcp 子命令数据 ----------

    def status_lines(self) -> list:
        if not self._servers:
            return ["MCP: 无配置 server（~/.astercli-py/mcp.json 或 .astercli/mcp.json）"]
        lines = []
        for name, state in self._servers.items():
            transport = state.config.transport
            extra = f" · {len(state.tool_names)} 工具" if state.tool_names else ""
            if state.supports_resources:
                extra += " · resources"
            if state.error:
                extra += f" · {state.error[:120]}"
            lines.append(f"  {name} [{state.status}] {transport}{extra}")
        return ["MCP servers:", *lines]

    def restart(self, name: str) -> bool:
        state = self._servers.get(name)
        if state is None:
            return False
        self.unregister_server(name)
        self._launch_connect(name, state)
        return True

    def disable(self, name: str) -> bool:
        state = self._servers.get(name)
        if state is None:
            return False
        self.unregister_server(name)
        state.config.disabled = True
        state.status = STATUS_DISABLED
        return True

    def enable(self, name: str) -> bool:
        state = self._servers.get(name)
        if state is None:
            return False
        state.config.disabled = False
        self._launch_connect(name, state)
        return True

    def shutdown(self) -> None:
        """进程退出前统一回收（关闭所有连接与后台线程）。"""
        for name in list(self._registered):
            self.unregister_server(name)
        self._runtime.shutdown()

    # ---------- 内部 ----------

    def _info(self, message: str) -> None:
        if self._renderer is not None:
            self._renderer.info(message)

    def _warn(self, message: str) -> None:
        if self._renderer is not None:
            self._renderer.warn(message)


# ---------- 工具 func 工厂（闭包捕获 server / tool 名） ----------


def _make_tool_func(manager: McpServerManager, server: str, tool: str):
    async def func(args: dict) -> str:
        return await manager.call_tool_async(server, tool, args or {})

    return func


def _make_resource_list_func(manager: McpServerManager, server: str):
    async def func(_args: dict) -> str:
        return await manager.list_resources_async(server)

    return func


def _make_resource_read_func(manager: McpServerManager, server: str):
    async def func(args: dict) -> str:
        uri = (args or {}).get("uri") or ""
        if not uri:
            return "缺少 uri 参数；先用 list_resources 获取可用资源列表"
        return await manager.read_resource_async(server, uri)

    return func


# ---------- 结果格式化（对齐 Java：MCP 原生 content block → 文本回灌 LLM） ----------


def format_tool_result(result) -> str:
    """CallToolResult → 文本。isError=True 时保留原文并标注，让 LLM 能调整重试。"""
    blocks = getattr(result, "content", None) or []
    parts = [b.text for b in blocks if hasattr(b, "text") and isinstance(b.text, str)]
    text = "\n".join(p for p in parts if p)
    if not text:
        text = "(MCP 工具无文本输出)"
    if getattr(result, "isError", False):
        return f"MCP 工具返回错误: {text}"
    return text


def format_resource_list(result) -> str:
    resources = getattr(result, "resources", None) or []
    if not resources:
        return "该 server 没有可用资源。"
    lines = []
    for r in resources:
        desc = (getattr(r, "description", "") or "").strip()
        mime = getattr(r, "mimeType", "") or "unknown"
        suffix = f" —— {desc}" if desc else ""
        lines.append(f"- {getattr(r, 'uri', '')} [{getattr(r, 'name', '')}, {mime}]{suffix}")
    return "\n".join(lines)


def extract_resource_text(result) -> tuple:
    """ReadResourceResult → (text, mime_type)；非文本资源降级为占位说明。"""
    contents = getattr(result, "contents", None) or []
    texts, mimes = [], []
    for c in contents:
        text = getattr(c, "text", None)
        if isinstance(text, str):
            texts.append(text)
            mimes.append(getattr(c, "mimeType", "") or "")
        else:
            blob = getattr(c, "blob", None)
            if blob:
                texts.append(f"(二进制资源 {getattr(c, 'mimeType', '') or 'unknown'}，长度 {len(blob)})")
    mime = mimes[0] if mimes else ""
    return "\n".join(texts), mime


def format_resource_contents(result) -> str:
    text, _ = extract_resource_text(result)
    return text or "(资源无文本内容)"
