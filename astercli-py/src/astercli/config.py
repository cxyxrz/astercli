"""配置读取（对齐 Java AsterCliConfig 的环境变量语义）。

优先级：注入 env（测试用） > 系统环境变量 > 仓库根 .env > 运行目录 .env > ~/.env。

对齐点：
- provider 的 API Key / MODEL / BASE_URL 变量名及 MOONSHOT_*、XFYUN_* 兼容别名
- 工具精简模式开关 ASTERCLI_TOOL_MODE（默认 slim，schema 回退）
- 统一工具输出预算 ASTERCLI_TOOL_OUTPUT_MAX_CHARS（默认 64000）
- ReAct 兜底预算 ASTERCLI_REACT_*（token 预算默认不限，stagnation 3 轮，硬轮数 50）

Java 版 ~/.astercli/config.json 持久化暂不接入，/model 切换仅作用于当前会话。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

from dotenv import dotenv_values

from .llm.providers import PROVIDER_SPECS, ProviderSpec

TOOL_MODE_SLIM = "slim"
TOOL_MODE_SCHEMA = "schema"


def _env_files() -> list:
    """.env 查找顺序：仓库根（Java 版主配置位）> 运行目录 > 用户目录。"""
    here = Path(__file__).resolve()
    repo_root_env = None
    # src/astercli/config.py 向上三层是 astercli-py，再上是仓库根
    if len(here.parents) >= 4:
        repo_root_env = here.parents[3] / ".env"
    files = []
    if repo_root_env is not None:
        files.append(repo_root_env)
    files.append(Path.cwd() / ".env")
    files.append(Path.home() / ".env")
    return files


class Settings:
    def __init__(
        self, env: Optional[Mapping[str, str]] = None, *, load_files: bool = True
    ) -> None:
        self._extra: dict = dict(env or {})
        merged: dict = {}
        for f in (_env_files() if load_files else ()):
            try:
                if not f.is_file():
                    continue
            except OSError:
                continue
            for k, v in dotenv_values(f).items():
                if v is None:
                    continue
                v = v.strip()
                if v and k not in merged:
                    merged[k] = v
        self._file_env = merged

    def get(self, key: str) -> Optional[str]:
        if key in self._extra:
            v = self._extra[key].strip()
            if v:
                return v
        real = os.environ.get(key)
        if real and real.strip():
            return real.strip()
        return self._file_env.get(key)

    def _first(self, keys) -> Optional[str]:
        for k in keys:
            v = self.get(k)
            if v:
                return v
        return None

    # ---------- provider 配置 ----------

    def api_key(self, spec: ProviderSpec) -> Optional[str]:
        return self._first(spec.api_key_envs)

    def model(self, spec: ProviderSpec) -> str:
        return self._first(spec.model_envs) or spec.default_model

    def base_url(self, spec: ProviderSpec) -> Optional[str]:
        return self._first(spec.base_url_envs)

    def lora_id(self, spec: ProviderSpec) -> Optional[str]:
        return self._first(spec.lora_id_envs)

    def default_provider(self) -> str:
        return self.get("ASTERCLI_PROVIDER") or "glm"

    # ---------- 工具层 ----------

    def tool_mode(self) -> str:
        configured = self.get("ASTERCLI_TOOL_MODE")
        if configured and configured.strip().lower() == TOOL_MODE_SCHEMA:
            return TOOL_MODE_SCHEMA
        return TOOL_MODE_SLIM

    def tool_output_max_chars(self) -> int:
        return _positive_int(self.get("ASTERCLI_TOOL_OUTPUT_MAX_CHARS"), 64_000)

    def tool_timeout_seconds(self) -> int:
        """registry 级单工具执行兜底超时（execute_command 等工具自身超时优先生效）。"""
        return _positive_int(self.get("ASTERCLI_TOOL_TIMEOUT_SECONDS"), 120)

    # ---------- ReAct 兜底预算 ----------

    def react_token_budget(self) -> Optional[int]:
        """None 表示不限（对齐 Java 默认 Integer.MAX_VALUE 的"实质不限"语义）。"""
        raw = self.get("ASTERCLI_REACT_TOKEN_BUDGET")
        if raw is None or not raw.strip():
            return None
        try:
            parsed = int(raw.strip())
            return parsed if parsed > 0 else None
        except ValueError:
            return None

    def react_stagnation_window(self) -> int:
        return _positive_int(self.get("ASTERCLI_REACT_STAGNATION_WINDOW"), 3)

    def react_hard_max_iterations(self) -> int:
        return _positive_int(self.get("ASTERCLI_REACT_HARD_MAX_ITERATIONS"), 50)

    # ---------- MCP ----------

    def mcp_startup_wait_seconds(self) -> int:
        """启动期 MCP 最大同步等待（对齐 Java 默认 8 秒，超时后台续载）。"""
        return _positive_int(self.get("ASTERCLI_MCP_STARTUP_WAIT_SECONDS"), 8)

    # ---------- LSP 诊断（P8） ----------

    def lsp_enabled(self) -> bool:
        """对齐 Java enabled()：仅显式 false（忽略大小写）关闭，缺失/其他值默认开。"""
        raw = self.get("ASTERCLI_LSP_ENABLED")
        return (raw or "").strip().lower() != "false"

    def lsp_max_diagnostics(self) -> int:
        """注入上限（对齐 Java 默认 20；非正数回退默认）。"""
        return _positive_int(self.get("ASTERCLI_LSP_MAX_DIAGNOSTICS"), 20)


def _positive_int(raw: Optional[str], default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
        return parsed if parsed > 0 else default
    except ValueError:
        return default
