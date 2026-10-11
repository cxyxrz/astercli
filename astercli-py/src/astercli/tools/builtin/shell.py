"""execute_command：短时 Shell 命令工具（对齐 Java 语义）。

- 在项目根目录执行，默认 60 秒超时
- CommandGuard 黑名单 fast-fail（主防线仍是 HITL 审批 + 用户判断）
- 输出预算 8k 字符（stdout + stderr 合并截断）
"""

from __future__ import annotations

import subprocess
from typing import Callable, Optional

from ...policy import PathGuard, PolicyException, check_command
from ..registry import Param, ToolRegistry, create_parameters

COMMAND_TIMEOUT_SECONDS = 60
COMMAND_OUTPUT_MAX_CHARS = 8_192


def register_shell_tools(
    registry: ToolRegistry, path_guard: PathGuard, command_guard: Optional[Callable] = None
) -> None:
    guard = command_guard or check_command

    def execute_command(args: dict) -> str:
        command = args.get("command") or ""
        reason = guard(command)
        if reason:
            raise PolicyException(f"命令被拒绝: {reason}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(path_guard.root_path),
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"命令执行失败: 超时（{COMMAND_TIMEOUT_SECONDS} 秒），命令已终止"
        combined = proc.stdout or ""
        if proc.stderr:
            combined = (combined + "\n" + proc.stderr) if combined else proc.stderr
        if len(combined) > COMMAND_OUTPUT_MAX_CHARS:
            combined = (
                combined[:COMMAND_OUTPUT_MAX_CHARS]
                + f"\n⚠️ 输出超长已截断（>{COMMAND_OUTPUT_MAX_CHARS} 字符）"
            )
        return f"命令输出 (exit {proc.returncode}):\n{combined}"

    registry.register(
        "execute_command",
        "在当前项目目录中执行短时 Shell 命令（默认 60 秒超时，不允许全盘扫描）",
        create_parameters(Param("command", "string", "要执行的命令", True)),
        execute_command,
        dangerous=True,
    )
