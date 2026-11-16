"""Skill 子 agent 隔离执行（渐进披露的上下文隔离层）。

主上下文只保留 skill 索引与结论：``load_skill`` 把正文塞进主历史（5KB 上限），
而 ``run_skill_task`` 把 skill 正文与全部中间步骤放进一个一次性子上下文，
用完即弃，主对话只收到一条结论。适合长正文 / 多 references / 会读很多文件的
调研型 skill。

隔离边界（有意为之）：
- 子上下文 = 全新 system + 单条 task，不携带主对话历史
- 工具只复制父 registry 的**只读**工具，危险工具（写文件 / execute_command /
  MCP 动态工具 / revert_turn）一律不进去，避免绕过 HITL 审批
- ``load_skill`` 与 ``run_skill_task`` 不复制：前者会污染主 buffer，后者会递归
- ``read_skill_file`` 重新注册为只能读当前这个 skill 的目录
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import Settings
from ..tools.registry import Param, ToolRegistry, create_parameters, truncate_output
from .reader import DEFAULT_READ_LINES, SKILL_MD, injection_header, read_skill_text
from .registry import SkillRegistry
from .skill import Skill

FORK_TOOL_NAME = "run_skill_task"
MAX_FORK_STEPS = 8
MAX_FORK_ANSWER_CHARS = 4000
FORK_TIMEOUT_SECONDS = 600
# load_skill 会写主 buffer；read_skill_file 要按 skill 重新绑定；其余见模块注释
_CHILD_EXCLUDED_TOOLS = {FORK_TOOL_NAME, "load_skill", "read_skill_file", "revert_turn"}


class SkillForkRunner:
    def __init__(
        self,
        runtime,
        registry: ToolRegistry,
        settings: Settings,
        *,
        renderer=None,
    ) -> None:
        self.runtime = runtime
        self._registry = registry
        self._settings = settings
        self._renderer = renderer

    def notify(self, text: str) -> None:
        """进度提示（渲染器缺失或异常一律静默，不影响子任务）。"""
        if self._renderer is None:
            return
        try:
            self._renderer.info(text)
        except Exception:  # noqa: BLE001
            pass

    async def run(self, skill: Skill, task: str) -> str:
        child = self._child_registry(skill)
        messages: List = [
            SystemMessage(content=self._child_system_prompt(skill)),
            HumanMessage(content=task),
        ]
        used: List[str] = []
        for step in range(1, MAX_FORK_STEPS + 1):
            response = await self._call(messages, child)
            messages.append(response)
            calls = list(getattr(response, "tool_calls", None) or [])
            if not calls:
                answer = str(getattr(response, "content", "") or "").strip()
                if not answer:
                    return f"[{FORK_TOOL_NAME}: {skill.name}] 子 agent 未给出结论"
                used_note = f"，工具: {', '.join(sorted(set(used)))}" if used else ""
                return (
                    f"[{FORK_TOOL_NAME}: {skill.name}] 子 agent 结论（{step} 步{used_note}；"
                    "中间过程未进入主上下文）\n\n"
                    + truncate_output(answer, MAX_FORK_ANSWER_CHARS)
                )
            used.extend(str(c.get("name") or "") for c in calls)
            messages.extend(await child.execute_all(calls, decisions=None))
        return (
            f"[{FORK_TOOL_NAME}: {skill.name}] 子 agent 达到 {MAX_FORK_STEPS} 步仍未收敛，"
            "未返回结论；建议把任务拆小后重试。"
        )

    async def _call(self, messages: List, child: ToolRegistry) -> AIMessage:
        client = self.runtime.client
        if getattr(self.runtime, "supports_tools", True) and child.names():
            client = client.bind_tools(child.schemas())
        return await client.ainvoke(messages)

    def _child_registry(self, skill: Skill) -> ToolRegistry:
        child = ToolRegistry(settings=self._settings)
        for name in self._registry.names():
            if name in _CHILD_EXCLUDED_TOOLS:
                continue
            tool = self._registry.get(name)
            if tool is None or tool.dangerous:
                continue
            child.register(name, tool.description, tool.parameters, tool.func)

        def read_skill_file(args: dict) -> str:
            return read_skill_text(
                skill,
                str(args.get("path") or SKILL_MD),
                args.get("offset") or 1,
                args.get("limit") or DEFAULT_READ_LINES,
            )

        child.register(
            "read_skill_file",
            "读取当前 skill 目录内的文件（只读、仅限本 skill）。"
            "path=SKILL.md 返回去 frontmatter 的正文，其余文件按行返回且带行号。",
            create_parameters(
                Param("path", "string", "相对 skill 目录的文件路径，默认 SKILL.md", False),
                Param("offset", "integer", "起始行号，1 表示第一行", False),
                Param("limit", "integer", f"最多读取行数，默认 {DEFAULT_READ_LINES}", False),
            ),
            read_skill_file,
        )
        return child

    def _child_system_prompt(self, skill: Skill) -> str:
        return (
            "你是一个隔离子 agent，只为下面这一个任务运行。你的完整上下文（skill 正文、工具结果、"
            "推理过程）不会进入主对话，主 agent 只会看到你最后一条结论。\n\n"
            f"## Skill: {skill.name}\n{skill.description}\n\n"
            f"{injection_header(skill)}"
            "## Skill 正文（完整，未截断）\n"
            f"{skill.body or ''}\n\n"
            "## 执行要求\n"
            "- 只做与任务直接相关的最小工作，不要复述 skill 正文或工具输出\n"
            "- 工具为只读；无法完成时明确说明缺什么，不要编造结果\n"
            "- 结论必须自包含：给出答案、关键文件路径与行号、可复用的命令\n"
            "- 不要向用户提问，直接给出最终结果\n"
        )


def register_skill_fork_tool(
    registry: ToolRegistry,
    skill_registry: SkillRegistry,
    runtime,
    settings: Settings,
    *,
    renderer=None,
) -> None:
    """注册 run_skill_task：把 skill 任务 fork 到一次性子上下文，只回结论。"""
    runner = SkillForkRunner(runtime, registry, settings, renderer=renderer)

    async def run_skill_task(args: dict) -> str:
        name = str(args.get("skill") or "").strip()
        task = str(args.get("task") or "").strip()
        if not name:
            return "run_skill_task 失败: skill 不能为空"
        if not task:
            return "run_skill_task 失败: task 不能为空"
        skill, error = skill_registry.resolve(name)
        if error:
            return error
        if runner.runtime is None:
            return "run_skill_task 失败: 当前没有可用的 LLM 客户端"
        runner.notify(f"子 agent 启动：{name}")
        try:
            return await runner.run(skill, task)
        finally:
            runner.notify(f"子 agent 结束：{name}")

    registry.register(
        FORK_TOOL_NAME,
        "在隔离子 agent 中执行一个 skill 任务：完整 skill 正文与所有中间步骤都留在子上下文，"
        "主上下文只收到一条结论。适合正文较长、需要查阅多个 references、或会读取大量文件的"
        "调研型任务；简单的一两步任务直接 load_skill 更省。传入 skill 名与具体任务，"
        "子 agent 内的工具为只读（不能写文件或执行命令）。",
        create_parameters(
            Param("skill", "string", "要使用的 skill 名称（与索引段中的名字一致）", True),
            Param("task", "string", "交给子 agent 的具体任务：目标、已知线索、期望产出", True),
        ),
        run_skill_task,
        timeout_seconds=FORK_TIMEOUT_SECONDS,
    )
