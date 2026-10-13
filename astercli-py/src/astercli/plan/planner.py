"""规划器（对齐 Java plan.Planner）：LLM 分解任务为执行计划。

- 简单任务快路径：不调 LLM，直接生成单任务计划
- LLM 计划 JSON 两遍解析：第一遍建任务并归一化 id（task_1..N），
  第二遍回填依赖（容忍前向引用），最后拓扑排序检测环
- replan：带着失败原因 + 已完成任务清单重新规划
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from ..prompt import PLANNER_PROMPT
from .models import ExecutionPlan, Task, TaskType

_SIMPLE_VERBS = ("列出", "查看", "读取", "显示", "执行", "运行", "搜索", "当前目录", "文件")
_MULTI_STEP_CUES = ("然后", "并且", "并", "再", "最后", "同时", "先", "之后", "接着", "以及")


def parse_task_type(type_str) -> TaskType:
    try:
        return TaskType(str(type_str or "").upper())
    except ValueError:
        return TaskType.ANALYSIS


def is_simple_goal(goal) -> bool:
    """对齐 Java Planner.isSimpleGoal：无多步线索、够短、含简单动词。"""
    if not goal:
        return False
    normalized = goal.strip()
    if not normalized:
        return False
    if any(cue in normalized for cue in _MULTI_STEP_CUES):
        return False
    if len(normalized) > 30:
        return False
    return any(verb in normalized for verb in _SIMPLE_VERBS)


def _infer_simple_task_type(goal: str) -> TaskType:
    if ("读取" in goal or "打开" in goal or "查看" in goal) and "文件" in goal:
        return TaskType.FILE_READ
    if "写入" in goal or "修改" in goal or "创建文件" in goal:
        return TaskType.FILE_WRITE
    if "分析" in goal or "总结" in goal or "解释" in goal:
        return TaskType.ANALYSIS
    if "验证" in goal or "检查" in goal:
        return TaskType.VERIFICATION
    return TaskType.COMMAND


def parse_plan(goal: str, plan_json: str) -> ExecutionPlan:
    """解析 LLM 生成的计划 JSON（纯函数，供测试直连）。

    :raises ValueError: JSON 非法 / 存在循环依赖
    """
    cleaned = re.sub(r"```json\s*", "", plan_json or "")
    cleaned = cleaned.replace("```", "").strip()
    try:
        root = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"计划 JSON 解析失败: {e}") from e
    if not isinstance(root, dict):
        raise ValueError("计划 JSON 必须是对象")

    plan = ExecutionPlan(goal=goal, summary=str(root.get("summary") or ""))
    tasks_node = root.get("tasks")
    if not isinstance(tasks_node, list):
        raise ValueError("计划 JSON 缺少 tasks 数组")

    # 第一遍：创建任务，id 统一归一化为 task_1..N（容忍 LLM 自造 id）
    id_mapping: dict = {}
    for index, node in enumerate(tasks_node, 1):
        if not isinstance(node, dict):
            continue
        original_id = str(node.get("id") or "")
        new_id = f"task_{index}"
        id_mapping[original_id] = new_id
        plan.add_task(Task(new_id, str(node.get("description") or ""), parse_task_type(node.get("type"))))

    # 第二遍：回填依赖（前向引用此时可解析；未知依赖丢弃）
    for index, node in enumerate(tasks_node, 1):
        if not isinstance(node, dict):
            continue
        task = plan.get_task(f"task_{index}")
        deps = node.get("dependencies")
        if not isinstance(deps, list):
            continue
        for dep in deps:
            new_dep = id_mapping.get(str(dep), str(dep))
            if plan.get_task(new_dep) is not None:
                task.add_dependency(new_dep)

    if not plan.compute_execution_order():
        raise ValueError("计划中存在循环依赖")
    return plan


class Planner:
    """LLM 规划器：create_plan / replan。"""

    def __init__(self, runtime, renderer=None) -> None:
        self.runtime = runtime
        self.renderer = renderer

    async def create_plan(self, goal: str) -> ExecutionPlan:
        if is_simple_goal(goal):
            return self._minimal_plan(goal)
        if self.renderer is not None:
            self.renderer.info(f"📋 正在规划任务: {goal}")
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content="请为以下任务制定执行计划：\n" + goal),
        ]
        content = await self._astream_content(messages)
        return parse_plan(goal, content)

    async def replan(self, failed_plan: ExecutionPlan, failure_reason: str) -> ExecutionPlan:
        if self.renderer is not None:
            self.renderer.info(f"🔄 重新规划，原因: {failure_reason}")
        lines = [
            f"原任务: {failed_plan.goal}",
            f"失败原因: {failure_reason}",
            "已完成的任务:",
        ]
        for task in failed_plan.all_tasks():
            if task.status.value == "COMPLETED":
                lines.append(f"- {task.id}: {task.description}")
        lines.append("")
        lines.append("请制定新的执行计划，避开之前的问题。")
        return await self.create_plan("\n".join(lines))

    # ---------- 内部 ----------

    def _minimal_plan(self, goal: str) -> ExecutionPlan:
        plan = ExecutionPlan(
            goal=goal,
            summary=f"直接执行简单任务：{goal.strip()}" if goal.strip() else "执行简单任务",
        )
        plan.add_task(Task("task_1", goal.strip(), _infer_simple_task_type(goal or "")))
        if not plan.compute_execution_order():
            raise ValueError("简单计划不应出现循环依赖")
        return plan

    async def _astream_content(self, messages) -> str:
        """纯对话流式调用（不 bind 工具），聚合出最终 content。"""
        final = None
        if self.renderer is not None:
            self.renderer.begin_thinking()
        try:
            async for chunk in self.runtime.client.astream(messages):
                if not isinstance(chunk, AIMessage):
                    continue
                if self.renderer is not None:
                    self.renderer.stream_chunk(chunk)
                if final is None:
                    final = chunk
                elif isinstance(final, AIMessageChunk) and isinstance(chunk, AIMessageChunk):
                    final = final + chunk
                else:
                    final = chunk
        finally:
            if self.renderer is not None:
                self.renderer.end_thinking()
        content = getattr(final, "content", "") if final is not None else ""
        return content if isinstance(content, str) else ""
