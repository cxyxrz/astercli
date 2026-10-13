"""Plan-and-Execute Agent（对齐 Java PlanExecuteAgent）。

与 Java 的已声明差异：
- 任务批次按拓扑序顺序执行（Java 4 线程并行 + 输出缓冲；P4 顺序执行，
  保证 HITL 审批与流式渲染不交错，任务级并行留后续分期）
- 每个任务复用同一个 ReAct 图（独立 thread + 独立 budget，
  hard_max_iterations = MAX_TASK_ITERATIONS），工具审批走同一条
  interrupt 通道，与 ReAct 主路径行为一致
- 最终结果保留叶子任务全文（Java 跳过已流式输出的任务；Python 版结果
  只进会话历史不重印，无需去重）
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..config import TOOL_MODE_SLIM
from ..plan.models import ExecutionPlan, Task
from ..plan.planner import Planner
from ..prompt import PLAN_TASK_PROMPT
from .budget import AgentBudget
from .graph import ainvoke_with_hitl, build_react_graph

MAX_TASK_ITERATIONS = 5


class PlanExecuteAgent:
    """先规划后执行：create_plan → 审阅循环 → 拓扑序执行。"""

    def __init__(
        self,
        runtime,
        registry,
        renderer,
        planner: Planner,
        review: Callable,
        *,
        tool_mode: str = TOOL_MODE_SLIM,
        compactor=None,
        trigger_tokens: Optional[int] = None,
        approval=None,
        checkpointer=None,
        on_interrupt=None,
        memory_provider: Optional[Callable[[str], str]] = None,
    ) -> None:
        """:param review: async (goal, plan) -> PlanReviewDecision
        :param on_interrupt: async (interrupts) -> {call_id: decision}，HITL 审批
        :param memory_provider: (task_description) -> 长期记忆上下文文本
        """
        self.runtime = runtime
        self.registry = registry
        self.renderer = renderer
        self.planner = planner
        self.review = review
        self.memory_provider = memory_provider
        self.graph = build_react_graph(
            runtime,
            registry,
            renderer,
            tool_mode=tool_mode,
            compactor=compactor,
            trigger_tokens=trigger_tokens,
            approval=approval,
            checkpointer=checkpointer,
        )
        self._on_interrupt = on_interrupt

    # ---------- 入口 ----------

    async def run(self, goal: str) -> str:
        plan = await self.planner.create_plan(goal)
        return await self._review_and_execute(plan)

    # ---------- 审阅循环 ----------

    async def _review_and_execute(self, plan: ExecutionPlan) -> str:
        while True:
            decision = await self.review(plan.goal, plan)
            if decision is None or decision.action == "execute":
                return await self._execute_plan(plan)
            if decision.action == "cancel":
                return "⏹️ 已取消本次计划执行。"
            feedback = (decision.feedback or "").strip()
            if not feedback:
                return await self._execute_plan(plan)
            self.renderer.info("📝 已收到补充要求，正在重新规划...")
            plan = await self.planner.create_plan(plan.goal + "\n补充要求：" + feedback)

    # ---------- 执行 ----------

    async def _execute_plan(self, plan: ExecutionPlan) -> str:
        self.renderer.info("🚀 开始执行计划...")
        plan.mark_started()
        failures: list = []

        while True:
            executable = [
                plan.get_task(tid)
                for tid in plan.execution_order
                if plan.get_task(tid) is not None and plan.get_task(tid).is_executable(plan.tasks)
            ]
            if not executable:
                break

            if len(executable) > 1:
                self.renderer.info(
                    "⚡ 本轮 "
                    + str(len(executable))
                    + " 个任务就绪，按拓扑序执行: "
                    + ", ".join(t.id for t in executable)
                )

            for task in executable:
                self.renderer.info(f"▶️ 执行任务 [{task.id}]: {task.description}")
                task.mark_started()
                try:
                    result = await self._execute_task(plan, task)
                except Exception as e:  # noqa: BLE001 —— 对齐 Java：任务失败进入重规划/汇总分支
                    task.mark_failed(str(e))
                    self.renderer.warn(f"❌ 失败 [{task.id}]: {e}")
                    if plan.progress() < 0.5:
                        self.renderer.info("🔄 尝试重新规划...")
                        replanned = await self.planner.replan(plan, str(e))
                        return await self._review_and_execute(replanned)
                    failures.append(f"任务 {task.id} 失败: {e}")
                    continue
                task.mark_completed(result)
                self.renderer.info(f"✅ 完成 [{task.id}]")

        if not plan.is_all_completed() and not plan.has_failed():
            plan.mark_failed()
            return "⚠️ 计划未能继续推进，存在未满足依赖的任务。"

        if plan.has_failed():
            plan.mark_failed()
            summary = "\n".join(failures) if failures else ""
            return "⚠️ 计划部分完成，有任务失败。\n" + summary if summary else "⚠️ 计划部分完成，有任务失败。"

        plan.mark_completed()
        summary = self._build_final_result(plan)
        return "✅ 计划执行完成！\n" + summary if summary else "✅ 计划执行完成！"

    # ---------- 单任务执行 ----------

    async def _execute_task(self, plan: ExecutionPlan, task: Task) -> str:
        budget = AgentBudget(stagnation_window=3, hard_max_iterations=MAX_TASK_ITERATIONS)
        task_input = self._build_task_input(plan, task)
        if self.memory_provider is not None:
            memory = self.memory_provider(task.description)
            if memory:
                task_input = task_input + "\n\n" + memory
        messages = [
            SystemMessage(content=PLAN_TASK_PROMPT),
            HumanMessage(content=task_input),
        ]
        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = await ainvoke_with_hitl(
                self.graph,
                {"messages": messages, "budget": budget, "halt_reason": None},
                config,
                self._on_interrupt,
            )
        finally:
            self._delete_thread(thread_id)

        msgs = result.get("messages") or []
        # 预算兜底触发时以已积累的工具结果作为产出（对齐 Java MAX_TASK_ITERATIONS 兜底）
        tool_texts = [
            m.content
            for m in msgs
            if isinstance(m, ToolMessage) and isinstance(m.content, str) and m.content.strip()
        ]
        if result.get("halt_reason"):
            return "\n".join(tool_texts) if tool_texts else ""
        last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        content = last_ai.content if last_ai is not None and isinstance(last_ai.content, str) else ""
        if content.strip():
            return content
        return "\n".join(tool_texts)

    def _build_task_input(self, plan: ExecutionPlan, task: Task) -> str:
        """任务上下文（对齐 Java buildTaskContext）。"""
        lines = [f"总目标：{plan.goal}", f"当前任务：{task.description}"]
        if not task.dependencies:
            lines.append("依赖任务：无")
        else:
            lines.append("依赖任务结果：")
            for dep_id in task.dependencies:
                dep = plan.get_task(dep_id)
                if dep is None:
                    continue
                lines.append(f"- {dep.id} / {dep.description} / 状态={dep.status.value}")
                if dep.result:
                    lines.append(dep.result)
        lines.append("请执行此任务。如果是 ANALYSIS 或 VERIFICATION 类型，请基于以上上下文直接给出结果。")
        return "\n".join(lines)

    def _build_final_result(self, plan: ExecutionPlan) -> str:
        """最终结果取叶子任务输出；无叶子输出时回退最后一个非空任务结果。"""
        leaf_tasks = [t for t in plan.all_tasks() if not t.dependents]
        parts = [f"[{t.id}] {t.result}" for t in leaf_tasks if (t.result or "").strip()]
        if parts:
            return "\n".join(parts)
        results = [t.result for t in plan.all_tasks() if (t.result or "").strip()]
        return results[-1] if results else ""

    def _delete_thread(self, thread_id: str) -> None:
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
            try:
                checkpointer.delete_thread(thread_id)
            except Exception:  # noqa: BLE001 —— 清理失败不影响任务结果
                pass
