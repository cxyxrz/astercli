"""MCP 配置加载与合并（对齐 Java mcp.config.McpConfigLoader / McpServerConfig）。

行为契约：
- 合并顺序：先用户级 ``~/.astercli-py/mcp.json``，再项目级 ``<root>/.astercli/mcp.json``
  覆盖同名 server（对齐 Java "先用户级后项目级"）。Python 版用户级目录为
  ``~/.astercli-py``，与 Java ``~/.astercli`` 隔离（与长期记忆/审计同一原则）；
  项目级 ``.astercli/mcp.json`` 是项目内容，两版本共享。
- server 条目字段：``command`` / ``args`` / ``env`` / ``url`` / ``headers`` /
  ``type`` / ``disabled``。``command`` 存在 → stdio；``url`` 存在 → HTTP
  （``type=sse`` 时走 SSE）；二者互斥且必设其一。
- ``${VAR}`` 展开顺序：系统环境变量 → .env 链（Settings.get：注入 env >
  系统环境变量 > 仓库根 .env > 运行目录 .env > ~/.env）。展开范围覆盖
  command / args / env 值 / url / headers 值；未定义变量报配置错误
  （对齐 Java "MCP 配置引用了未设置的环境变量"）。
- 单个 server 展开失败只把该 server 置错，不影响其余 server 加载。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import Settings

#: 顶层配置 key（对齐 Java McpConfigFile.mcpServers，亦是 MCP 生态通用约定）
TOP_LEVEL_KEY = "mcpServers"

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# type 字段允许值（http 家族统一走 streamable HTTP transport）
TYPE_HTTP = ("http", "https", "streamable-http", "streamable_http")
TYPE_SSE = "sse"


class McpConfigError(Exception):
    """MCP 配置错误（JSON 损坏 / 字段冲突 / ${VAR} 未定义等）。"""


@dataclass
class McpServerConfig:
    """单个 MCP server 的展开后配置（对齐 Java McpServerConfig）。"""

    name: str
    command: Optional[str] = None          # stdio 启动命令
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    url: Optional[str] = None              # http / sse 远端地址
    headers: dict = field(default_factory=dict)
    transport: str = "stdio"               # stdio / http / sse
    disabled: bool = False

    @property
    def is_remote(self) -> bool:
        return self.url is not None


def user_config_path() -> Path:
    return Path.home() / ".astercli-py" / "mcp.json"


def project_config_path(root: Path) -> Path:
    return root / ".astercli" / "mcp.json"


def load_merged_config(settings: Settings, root: Path) -> dict:
    """读取用户级 + 项目级配置并合并，返回 ``{server_name: McpServerConfig}``。

    同名 server 项目级整条覆盖（不深合并）。单个 server 条目损坏/展开失败
    抛 :class:`McpConfigError` 的粒度为单条——由调用方决定置 ERROR。
    """
    raw: dict = {}
    for path in (user_config_path(), project_config_path(root)):
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise McpConfigError(f"配置文件读取失败: {path}（{e}）") from e
        if not isinstance(data, dict):
            raise McpConfigError(f"配置文件顶层必须是对象: {path}")
        servers = data.get(TOP_LEVEL_KEY)
        if servers is None:
            continue
        if not isinstance(servers, dict):
            raise McpConfigError(f"{TOP_LEVEL_KEY} 必须是对象: {path}")
        raw.update(servers)  # 后读的项目级覆盖同名条目

    configs: dict = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise McpConfigError(f"server '{name}' 条目必须是对象")
        configs[name] = prepare_server(name, entry, settings)
    return configs


def prepare_server(name: str, entry: dict, settings: Settings) -> McpServerConfig:
    """校验 + ``${VAR}`` 展开单个 server 条目。"""
    command = _expand_str(_opt_str(entry, "command", name), settings, name)
    url = _expand_str(_opt_str(entry, "url", name), settings, name)
    if command and url:
        raise McpConfigError(f"server '{name}': command 与 url 互斥，只能设置其一")
    if not command and not url:
        raise McpConfigError(f"server '{name}': 必须设置 command（stdio）或 url（http/sse）之一")

    raw_type = str(entry.get("type") or "").strip().lower()
    if command:
        if raw_type and raw_type not in ("stdio",):
            raise McpConfigError(f"server '{name}': 配置了 command 时 type 只能是 stdio")
        transport = "stdio"
    elif raw_type == TYPE_SSE:
        transport = "sse"
    elif raw_type and raw_type not in TYPE_HTTP:
        raise McpConfigError(f"server '{name}': 未知 type '{raw_type}'（允许: stdio/http/sse）")
    else:
        transport = "http"

    args_raw = entry.get("args") or []
    if not isinstance(args_raw, list):
        raise McpConfigError(f"server '{name}': args 必须是数组")
    args = [_expand_str(str(a), settings, name) for a in args_raw]

    env_raw = entry.get("env") or {}
    if not isinstance(env_raw, dict):
        raise McpConfigError(f"server '{name}': env 必须是对象")
    env = {str(k): _expand_str(str(v), settings, name) for k, v in env_raw.items()}

    headers_raw = entry.get("headers") or {}
    if not isinstance(headers_raw, dict):
        raise McpConfigError(f"server '{name}': headers 必须是对象")
    headers = {str(k): _expand_str(str(v), settings, name) for k, v in headers_raw.items()}

    return McpServerConfig(
        name=name,
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
        transport=transport,
        disabled=bool(entry.get("disabled", False)),
    )


def _opt_str(entry: dict, key: str, server_name: str) -> Optional[str]:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise McpConfigError(f"server '{server_name}': {key} 必须是非空字符串")
    return value


def _expand_str(value: Optional[str], settings: Settings, server_name: str) -> Optional[str]:
    """展开 ``${VAR}`` 占位符；未定义变量按配置错误处理。"""
    if not value:
        return value

    def repl(match) -> str:
        var = match.group(1)
        resolved = settings.get(var)
        if not resolved:
            raise McpConfigError(
                f"server '{server_name}': 引用了未设置的环境变量 ${{{var}}}"
            )
        return resolved

    return _PLACEHOLDER.sub(repl, value)
