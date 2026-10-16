"""Multi-Agent 编排器（对齐 Java AgentOrchestrator 的主从架构）。

协作流程：
1. 用户提交任务 → 规划者拆解为 JSON 步骤计划（无工具，单轮流式）
2. 编排器解析计划，按依赖顺序执行步骤
3. 执行者（worker）跑 ReAct 图完成单步（复用 build_react_graph：工具 +
   budget 兜底 + reasoning 流式 + HITL 审批全量继承）
4. 检查者审查执行结果（无工具），不通过带反馈重试（每步最多 2 次）
5. 全部步骤完成后汇总返回

与 Java 的已知差异（延续 P4 决策）：
- Java 同一依赖批次内多步并行（Worker 池 + 缓冲流式输出），Python 版
  批次内顺序执行——避免并行 interrupt 审批与共享 renderer 的输出交错
- Java 每步保留独立 SubAgent 对话历史，Python 版 worker 每步开独立
  thread（messages 全新），planner/reviewer 无状态单轮调用
- Java 的 MCP resource 索引 externalContext 注入暂未迁移（后续分期）
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import Settings
from ..llm.factory import ProviderRuntime
from ..prompt import build_team_prompt
from ..skill import SkillRegistry, format_index
from ..tools.registry import ToolRegistry
from .budget import AgentBudget
from .graph import ainvoke_with_hitl, build_react_graph

MAX_RETRIES_PER_STEP = 2
RESULT_PREVIEW_CHARS = 500


class StepStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExecutionStep:
    id: str
    description: str
    type: str = "COMMAND"
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    status: StepStatus = StepStatus.PENDING

    def with_status(self, status: StepStatus, result: Optional[str] = None) -> "ExecutionStep":
        return replace(self, status=status, result=result if result is not None else self.result)


# ---------- 计划 / 审查解析（对齐 Java parsePlan / parseReviewApproval） ----------


def parse_plan(plan_text: str) -> List[ExecutionStep]:
    """解析规划者输出的 JSON 计划；steps / tasks 字段兼容，id 重编号为 step_N。"""
    if not plan_text or not plan_text.strip():
        return []
    cleaned = plan_text.replace("```json", "").replace("```", "").strip()
    try:
        root = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    steps_node = root.get("steps")
    if not isinstance(steps_node, list) or not steps_node:
        steps_node = root.get("tasks")  # 兼容 Plan-and-Execute 的格式
    if not isinstance(steps_node, list) or not steps_node:
        return []
    valid_nodes = [n for n in steps_node if isinstance(n, dict)]
    if not valid_nodes:
        return []

    # 第一遍：重编号 id（跳过的非字典元素不占号）
    id_mapping: Dict[str, str] = {}
    steps: List[ExecutionStep] = []
    for index, node in enumerate(valid_nodes, start=1):
        original_id = str(node.get("id") or "")
        new_id = f"step_{index}"
        id_mapping[original_id] = new_id
        steps.append(
            ExecutionStep(
                id=new_id,
                description=str(node.get("description") or ""),
                type=str(node.get("type") or "COMMAND"),
            )
        )

    # 第二遍：建立依赖（原始 id 映射到重编号 id，zip 保证不错位）
    mapped: List[ExecutionStep] = []
    for step, node in zip(steps, valid_nodes):
        deps_node = node.get("dependencies")
        deps: List[str] = []
        if isinstance(deps_node, list):
            deps = [id_mapping.get(str(dep), str(dep)) for dep in deps_node]
        mapped.append(replace(step, dependencies=deps))
    return mapped


def get_executable_steps(steps: List[ExecutionStep]) -> List[ExecutionStep]:
    """取当前可执行步骤（PENDING 且依赖全部 COMPLETED）。"""
    status_map = {s.id: s.status for s in steps}
    return [
        s
        for s in steps
        if s.status == StepStatus.PENDING
        and all(status_map.get(dep) == StepStatus.COMPLETED for dep in s.dependencies)
    ]


def parse_review_approval(review_content: Optional[str]) -> bool:
    """解析检查者审批结果；解析失败保守判为不通过（对齐 Java）。"""
    if not review_content or not review_content.strip():
        return False
    cleaned = review_content.replace("```json", "").replace("```", "").strip()
    try:
        root = json.loads(cleaned)
        if not isinstance(root, dict) or "approved" not in root or root["approved"] is None:
            return False
        return bool(root["approved"])
    except json.JSONDecodeError:
        lower = review_content.lower()
        has_negative = any(
            kw in lower
            for kw in ("未通过", "不通过", "不合格", "有问题", '"approved": false', '"approved":false')
        )
        has_positive = any(
            kw in lower for kw in ("通过", "合格", '"approved": true', '"approved":true')
        )
        if has_negative:
            return False
        return has_positive


def parse_review_issues(review_content: Optional[str]) -> str:
    """解析检查者反馈的问题列表；无结构化问题时给保守兜底文案。"""
    if not review_content or not review_content.strip():
        return "审查未通过，请改进执行结果"
    cleaned = review_content.replace("```json", "").replace("```", "").strip()
    try:
        root = json.loads(cleaned)
        if isinstance(root, dict):
            for key in ("issues", "suggestions"):
                items = root.get(key)
                if isinstance(items, list) and items:
                    return "\n".join(f"- {item}" for item in items).strip()
            summary = str(root.get("summary") or "")
            if summary.strip():
                return summary.strip()
    except json.JSONDecodeError:
        pass
    return "审查未通过，请改进执行结果"


def build_step_context(steps: List[ExecutionStep], current: ExecutionStep) -> str:
    """拼总任务上下文：已完成依赖步骤的描述 + 结果预览（对齐 Java 500 字符截断）。"""
    lines = ["总任务上下文："]
    for step in steps:
        if step.status == StepStatus.COMPLETED and step.id in current.dependencies:
            lines.append(f"已完成的依赖步骤 [{step.id}]: {step.description}")
            if step.result:
                preview = (
                    step.result[:RESULT_PREVIEW_CHARS] + "..."
                    if len(step.result) > RESULT_PREVIEW_CHARS
                    else step.result
                )
                lines.append(f"结果：{preview}")
            lines.append("")
    return "\n".join(lines)


def summarize_steps(steps: List[ExecutionStep]) -> str:
    lines = []
    for step in steps:
        marker = "✅" if step.status == StepStatus.COMPLETED else "⏳"
        deps = "无" if not step.dependencies else ", ".join(step.dependencies)
        lines.append(f"  {marker} [{step.id}] {step.description} (依赖: {deps})")
    return "\n".join(lines)


def build_final_result(steps: List[ExecutionStep]) -> str:
    """汇总：状态 + 简短预览（完整输出已在执行中流式打印，对齐 Java 不重复全文）。"""
    all_completed = all(s.status == StepStatus.COMPLETED for s in steps)
    has_failed = any(s.status == StepStatus.FAILED for s in steps)
    if all_completed:
        head = "✅ 多 Agent 协作任务完成！\n\n"
    elif has_failed:
        head = "⚠️ 多 Agent 协作任务未完全完成，存在失败步骤。\n\n"
    else:
        head = "⚠️ 多 Agent 协作任务部分完成，仍有未执行步骤。\n\n"
    lines = [head.rstrip("\n"), "", "📋 执行总结："]
    for step in steps:
        if step.status == StepStatus.COMPLETED:
            marker = "✅"
        elif step.status == StepStatus.FAILED:
            marker = "❌"
        else:
            marker = "⏳"
        lines.append(f"[{step.id}] {marker} {step.description}")
        if step.result and step.result.strip():
            preview = step.result[:120] + "..." if len(step.result) > 120 else step.result
            lines.append(f"   结果：{preview}")
    return "\n".join(lines)


class AgentOrchestrator:
    def __init__(
        self,
        runtime: ProviderRuntime,
        registry: ToolRegistry,
        renderer,
        *,
        project_root: Path,
        settings: Settings,
        skill_registry: Optional[SkillRegistry] = None,
        skill_buffer=None,
        approval=None,
        checkpointer=None,
        on_interrupt: Optional[Callable] = None,
        tool_mode: Optional[str] = None,
    ) -> None:
        self._runtime = runtime
        self._registry = registry
        self._renderer = renderer
        self._settings = settings
        self._skill_buffer = skill_buffer
        self._checkpointer = checkpointer
        self._on_interrupt = on_interrupt

        skill_index = (
            format_index(skill_registry.enabled_skills()) if skill_registry is not None else ""
        )
        self._planner_prompt = build_team_prompt("planner", project_root, skill_index)
        self._worker_prompt = build_team_prompt("worker", project_root, skill_index)
        self._reviewer_prompt = build_team_prompt("reviewer", project_root, skill_index)

        # worker 复用 ReAct 图：工具执行 / budget 兜底 / reasoning 流式 / HITL 审批全量继承
        self._worker_graph = build_react_graph(
            runtime,
            registry,
            renderer,
            tool_mode=tool_mode or settings.tool_mode(),
            approval=approval,
            checkpointer=checkpointer,
        )

    # ---------- 主流程 ----------

    async def run(self, user_input: str) -> str:
        retry_count: Dict[str, int] = {}

        # 1. 规划阶段
        self._renderer.info("📋 第一阶段：规划\n🧑‍💼 规划者正在分析任务...")
        plan_text = await self._run_role_stream(
            self._planner_prompt, f"请为以下任务制定执行计划：\n{user_input}"
        )
        steps = parse_plan(plan_text or "")
        if not steps:
            if not plan_text:
                return "❌ 规划失败：规划者未能生成有效计划"
            return "❌ 规划失败：无法解析执行计划\n原始输出:\n" + plan_text

        self._renderer.info("📋 执行计划\n" + summarize_steps(steps))

        # 2. 执行阶段：按依赖批次顺序执行（批次内不再并行，见模块注释）
        self._renderer.info("⚡ 第二阶段：执行")
        while True:
            executable = get_executable_steps(steps)
            if not executable:
                break
            for step in executable:
                steps = self._replace_step(steps, await self._run_step(step, steps, retry_count))

        # 3. 因前置失败被跳过的残留步骤
        for step in steps:
            if step.status == StepStatus.PENDING:
                self._renderer.info(f"⏭️ 步骤 [{step.id}] 因前置步骤失败被跳过: {step.description}")

        return build_final_result(steps)

    # ---------- 单步执行（worker + reviewer + 重试） ----------

    async def _run_step(
        self, step: ExecutionStep, steps: List[ExecutionStep], retry_count: Dict[str, int]
    ) -> ExecutionStep:
        renderer = self._renderer
        renderer.info(f"🛠️ worker 执行步骤 [{step.id}]: {step.description}")
        context = build_step_context(steps, step)

        result = await self._run_worker(step.description, context)
        if result is None:
            renderer.warn(f"❌ 步骤 [{step.id}] 执行失败：结果为空")
            return step.with_status(StepStatus.FAILED, "执行结果为空")

        renderer.info(f"🔍 reviewer 正在审查步骤 [{step.id}] 的结果...")
        review = await self._run_reviewer(step.description, result)
        if review is None:
            # 审查阶段 LLM 调用失败：保留当前执行结果（对齐 Java）
            renderer.warn(f"⚠️ 步骤 [{step.id}] 审查阶段 LLM 调用失败，保留当前执行结果")
            return step.with_status(StepStatus.COMPLETED, result)

        approved = parse_review_approval(review)
        accepted = result
        retries = retry_count.get(step.id, 0)
        issues = parse_review_issues(review)

        while not approved and retries < MAX_RETRIES_PER_STEP:
            retries += 1
            retry_count[step.id] = retries
            renderer.info(f"⚠️ 步骤 [{step.id}] 审查未通过，正在重新执行...\n   反馈: {issues}")
            feedback_context = context + "\n\n之前的执行结果被审查拒绝，原因：\n" + issues
            retry_result = await self._run_worker(step.description, feedback_context)
            if retry_result is None:
                issues = "重试时执行结果为空"
                continue
            accepted = retry_result
            retry_review = await self._run_reviewer(step.description, accepted)
            if retry_review is None:
                approved = True  # 审查失败兜底放行（对齐 Java）
                break
            approved = parse_review_approval(retry_review)
            issues = parse_review_issues(retry_review)

        if approved:
            renderer.info(f"✅ 步骤 [{step.id}] 审查通过")
        else:
            renderer.info(f"⚠️ 步骤 [{step.id}] 超过最大重试次数，保留当前结果")
        return step.with_status(StepStatus.COMPLETED, accepted)

    # ---------- 角色调用 ----------

    async def _run_worker(self, description: str, context: str) -> Optional[str]:
        """worker 走 ReAct 图（工具 + budget + HITL）；返回最终 content，失败返回 None。"""
        content = f"{context}\n\n当前任务：{description}" if context else description
        # Skill body 前置注入（对齐 Java SubAgent.prependSkillBodies）
        if self._skill_buffer is not None and not self._skill_buffer.is_empty():
            drained = self._skill_buffer.drain()
            if drained:
                content = drained + "\n" + content

        budget = AgentBudget(
            self._settings.react_token_budget(),
            self._settings.react_stagnation_window(),
            self._settings.react_hard_max_iterations(),
        )
        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = await ainvoke_with_hitl(
                self._worker_graph,
                {
                    "messages": [SystemMessage(self._worker_prompt), HumanMessage(content)],
                    "budget": budget,
                    "halt_reason": None,
                },
                config,
                self._on_interrupt,
            )
        finally:
            self._delete_thread(thread_id)

        if result.get("halt_reason"):
            return None  # halt 原因已由 renderer 打印
        for message in reversed(result.get("messages") or []):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                text = message.content
                return text if isinstance(text, str) else str(text)
        return None

    async def _run_role_stream(self, system_prompt: str, content: str) -> Optional[str]:
        """planner / reviewer 单轮流式调用（无工具）。"""
        messages = [SystemMessage(system_prompt), HumanMessage(content)]
        final = None
        self._renderer.begin_thinking()
        try:
            async for chunk in self._runtime.client.astream(messages):
                if isinstance(chunk, AIMessage):
                    self._renderer.stream_chunk(chunk)
                    final = chunk if final is None else _merge_chunk(final, chunk)
        finally:
            self._renderer.end_thinking()
        # reasoning 兜底清空（对齐 graph.py：渲染端累积为准）
        self._renderer.take_reasoning()
        if final is None or not (final.content or "").strip():
            return None
        return final.content if isinstance(final.content, str) else str(final.content)

    async def _run_reviewer(self, description: str, result: str) -> Optional[str]:
        review_input = f"原始任务：{description}\n\n执行结果：\n{result}"
        return await self._run_role_stream(self._reviewer_prompt, review_input)

    # ---------- 工具 ----------

    @staticmethod
    def _replace_step(steps: List[ExecutionStep], updated: ExecutionStep) -> List[ExecutionStep]:
        return [updated if s.id == updated.id else s for s in steps]

    def _delete_thread(self, thread_id: str) -> None:
        if self._checkpointer is None:
            return
        try:
            self._checkpointer.delete_thread(thread_id)
        except Exception:  # noqa: BLE001 —— 清理失败不影响任务结果
            pass


def _merge_chunk(current, chunk):
    """流式聚合：AIMessageChunk 相加；非 chunk 直接替换（对齐 graph.py 策略）。"""
    try:
        from langchain_core.messages import AIMessageChunk

        if isinstance(current, AIMessageChunk) and isinstance(chunk, AIMessageChunk):
            return current + chunk
    except ImportError:
        pass
    return chunk
