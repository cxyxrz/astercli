"""Explore 子 agent 派遣（对齐 Claude Code 的 Agent / Explore agent）。

主 agent 调 ``dispatch_agent`` 派一个**独立运行实例**：它有自己的 messages、
自己的多轮工具循环，中间结果全部留在子上下文。主上下文只多一条结论。

隔离边界（有意为之，与 skill/fork.py 同源）：
- 子上下文 = 全新 system + 单条 prompt，不携带主对话历史
- 工具只复制父 registry 的**只读**工具；危险工具一律不进去，避免绕过 HITL
- 不复制 ``dispatch_agent`` 自己（防止层层递归），也不复制
  ``load_skill`` / ``run_skill_task`` / ``read_skill_file``（会污染主 buffer
  或绑定到具体 skill）
- 额外提供一个 ``run_readonly_command`` 代替 ``execute_command``：只允许
  白名单命令、禁止管道/重定向/命令串联，因此不需要审批也不会写盘
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import Settings
from ..tools.registry import Param, ToolRegistry, create_parameters, truncate_output

DISPATCH_TOOL_NAME = "dispatch_agent"
MAX_EXPLORE_STEPS = 12
MAX_EXPLORE_ANSWER_CHARS = 6000
DISPATCH_TIMEOUT_SECONDS = 600
READONLY_COMMAND_TIMEOUT_SECONDS = 60
READONLY_COMMAND_MAX_CHARS = 8_192

# 子 registry 的排除集：递归入口 + 会污染主上下文的工具
_CHILD_EXCLUDED_TOOLS = {
    DISPATCH_TOOL_NAME,
    "load_skill",
    "run_skill_task",
    "read_skill_file",
    "revert_turn",
}

# 只读命令白名单（按 argv[0] 匹配，禁止管道 / 重定向 / 串联）
READONLY_COMMANDS = {
    "cat", "head", "tail", "ls", "tree", "wc", "file", "stat",
    "find", "grep", "rg", "git", "pwd", "echo", "diff",
}
# git 只允许不改工作区的子命令
GIT_READONLY_SUBCOMMANDS = {
    "status", "log", "diff", "show", "blame", "branch", "remote",
    "rev-parse", "ls-files", "describe", "shortlog", "grep", "cat-file",
}
# 即使命令在白名单里，这些参数也可能写盘或执行子进程
_MUTATING_FLAGS = {
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprint0", "-fls",
    "--in-place", "--output", "--output-document",
    "--pre", "--pre-glob",
}
# branch / remote 默认带位置参数就会改仓库，只放行纯查询选项
_GIT_BRANCH_FLAGS = {
    "-a", "-r", "-v", "-vv", "--list", "--all", "--remotes", "--show-current",
    "--contains", "--merged", "--no-merged", "--points-at", "--format",
}
_GIT_REMOTE_FLAGS = {"-v", "--verbose", "show", "get-url"}
_SHELL_METACHARS = set(";|&<>`$()\n\r\\")


def readonly_command_reason(command: str) -> str:
    """返回拒绝原因；空字符串表示允许执行。"""
    text = (command or "").strip()
    if not text:
        return "命令为空"
    if any(ch in text for ch in _SHELL_METACHARS):
        return "只读模式不允许管道、重定向、命令串联或命令替换"
    try:
        argv = shlex.split(text)
    except ValueError as e:
        return f"命令解析失败: {e}"
    if not argv:
        return "命令为空"
    exe = os.path.basename(argv[0])
    if exe not in READONLY_COMMANDS:
        return f"{exe} 不在只读命令白名单内"
    if exe == "git":
        sub = argv[1] if len(argv) > 1 else ""
        if sub not in GIT_READONLY_SUBCOMMANDS:
            return f"git {sub or '(无子命令)'} 不在只读子命令白名单内"
        if sub == "branch":
            for token in argv[2:]:
                if token not in _GIT_BRANCH_FLAGS and not token.startswith("--format="):
                    return f"git branch {token} 不是只读查询选项"
        if sub == "remote":
            for token in argv[2:]:
                if token not in _GIT_REMOTE_FLAGS:
                    return f"git remote {token} 可能修改远端配置"
    for token in argv[1:]:
        bare = token.split("=", 1)[0]
        if bare in _MUTATING_FLAGS:
            return f"参数 {bare} 可能写盘或执行子进程"
    return ""


class ExploreRunner:
    def __init__(
        self,
        runtime,
        registry: ToolRegistry,
        settings: Settings,
        project_root: Path,
        *,
        renderer=None,
    ) -> None:
        self.runtime = runtime
        self._registry = registry
        self._settings = settings
        self._root = Path(project_root).resolve()
        self._renderer = renderer

    def notify(self, text: str) -> None:
        """进度提示（渲染器缺失或异常一律静默，不影响子任务）。"""
        if self._renderer is None:
            return
        try:
            self._renderer.info(text)
        except Exception:  # noqa: BLE001
            pass

    async def run(self, prompt: str, description: str = "") -> str:
        child = self._child_registry()
        messages: List = [
            SystemMessage(content=self._child_system_prompt()),
            HumanMessage(content=prompt),
        ]
        used: List[str] = []
        label = description.strip() or "explore"
        for step in range(1, MAX_EXPLORE_STEPS + 1):
            response = await self._call(messages, child)
            messages.append(response)
            calls = list(getattr(response, "tool_calls", None) or [])
            if not calls:
                answer = str(getattr(response, "content", "") or "").strip()
                if not answer:
                    return f"[{DISPATCH_TOOL_NAME}: {label}] 子 agent 未给出结论"
                used_note = f"，工具: {', '.join(sorted(set(used)))}" if used else ""
                return (
                    f"[{DISPATCH_TOOL_NAME}: {label}] 子 agent 结论（{step} 步{used_note}；"
                    "中间过程未进入主上下文）\n\n"
                    + truncate_output(answer, MAX_EXPLORE_ANSWER_CHARS)
                )
            used.extend(str(c.get("name") or "") for c in calls)
            messages.extend(await child.execute_all(calls, decisions=None))
        return (
            f"[{DISPATCH_TOOL_NAME}: {label}] 子 agent 达到 {MAX_EXPLORE_STEPS} 步仍未收敛，"
            "未返回结论；建议把问题拆小或用更具体的文件/符号线索重试。"
        )

    async def _call(self, messages: List, child: ToolRegistry) -> AIMessage:
        client = self.runtime.client
        if getattr(self.runtime, "supports_tools", True) and child.names():
            client = client.bind_tools(child.schemas())
        return await client.ainvoke(messages)

    def _child_registry(self) -> ToolRegistry:
        child = ToolRegistry(settings=self._settings)
        for name in self._registry.names():
            if name in _CHILD_EXCLUDED_TOOLS:
                continue
            tool = self._registry.get(name)
            if tool is None or tool.dangerous:
                continue
            child.register(name, tool.description, tool.parameters, tool.func)
        child.register(
            "run_readonly_command",
            "在项目根执行只读 Shell 命令（替代 execute_command）：仅允许 "
            + " / ".join(sorted(READONLY_COMMANDS))
            + " 等命令，git 仅限状态/日志/差异类子命令；禁止管道、重定向、命令串联与写盘参数。",
            create_parameters(Param("command", "string", "要执行的只读命令", True)),
            _readonly_command(self._root),
        )
        return child

    def _child_system_prompt(self) -> str:
        return (
            "你是一个只读的 Explore 子 agent，只为下面这一个调研任务运行。你的完整上下文"
            "（工具输出、推理过程）不会进入主对话，主 agent 只会看到你最后一条结论。\n\n"
            "## 可用工具\n"
            "- glob_files / grep_code / read_file / list_dir：定位并阅读代码\n"
            "- search_code：语义检索（索引存在时可用）\n"
            "- run_readonly_command：只读 shell，禁止管道与重定向\n"
            "- 你不能编辑或写入文件，也不能派遣其他子 agent\n\n"
            "## 执行要求\n"
            "- 先定位再读：用 glob_files 找候选，grep_code 精确定位，read_file 只读相关行段\n"
            "- 只做与任务直接相关的最小工作，不要复述工具输出\n"
            "- 结论必须自包含：给出答案、关键文件路径与行号（必要时附可复用的命令）\n"
            "- 无法确认时明确说明缺什么，不要编造；不要向用户提问\n"
        )


def _readonly_command(root: Path):
    def run_readonly_command(args: dict) -> str:
        command = str(args.get("command") or "")
        reason = readonly_command_reason(command)
        if reason:
            return f"只读命令被拒绝: {reason}"
        argv = shlex.split(command)
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=READONLY_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return (
                f"只读命令执行失败: 超时（{READONLY_COMMAND_TIMEOUT_SECONDS} 秒），命令已终止"
            )
        except OSError as e:
            return f"只读命令执行失败: {e}"
        combined = proc.stdout or ""
        if proc.stderr:
            combined = (combined + "\n" + proc.stderr) if combined else proc.stderr
        if len(combined) > READONLY_COMMAND_MAX_CHARS:
            combined = (
                combined[:READONLY_COMMAND_MAX_CHARS]
                + f"\n⚠️ 输出超长已截断（>{READONLY_COMMAND_MAX_CHARS} 字符）"
            )
        return f"只读命令输出 (exit {proc.returncode}):\n{combined}"

    return run_readonly_command


def register_dispatch_tool(
    registry: ToolRegistry,
    runtime,
    settings: Settings,
    project_root: Path,
    *,
    renderer=None,
) -> None:
    """注册 dispatch_agent：派只读 Explore 子 agent，只回结论。"""
    runner = ExploreRunner(runtime, registry, settings, project_root, renderer=renderer)

    async def dispatch_agent(args: dict) -> str:
        subagent_type = str(args.get("subagent_type") or "explore").strip()
        if subagent_type != "explore":
            return f"dispatch_agent 失败: 不支持的 subagent_type={subagent_type}（目前仅 explore）"
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return "dispatch_agent 失败: prompt 不能为空"
        if runner.runtime is None:
            return "dispatch_agent 失败: 当前没有可用的 LLM 客户端"
        description = str(args.get("description") or "").strip()
        runner.notify(f"Explore 子 agent 启动：{description or '调研任务'}")
        try:
            return await runner.run(prompt, description)
        finally:
            runner.notify(f"Explore 子 agent 结束：{description or '调研任务'}")

    registry.register(
        DISPATCH_TOOL_NAME,
        "派遣一个只读 Explore 子 agent，在独立上下文里多轮搜索与阅读；主上下文只收到最终结论。"
        "适合回答『这个功能在哪实现 / 调用链是什么 / 某模块怎么工作』这类需要扫很多文件的问题；"
        "已知文件的一两步读取直接 read_file 更省。子 agent 不能写文件、执行写命令或再派子 agent。",
        create_parameters(
            Param("description", "string", "3-5 个词的短任务名，用于进度显示", False),
            Param(
                "prompt",
                "string",
                "交给子 agent 的完整任务：目标、已知线索、期望产出（结论要含文件路径与行号）",
                True,
            ),
            Param("subagent_type", "string", "子 agent 类型，目前仅 explore", False),
        ),
        dispatch_agent,
        timeout_seconds=DISPATCH_TIMEOUT_SECONDS,
    )
