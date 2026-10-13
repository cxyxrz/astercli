"""PlanExecuteAgent 测试：审阅循环 / 依赖顺序 / 失败重规划 / 叶子汇总 / 任务内审批。

假 planner + 假 review + ScriptedModel 驱动任务执行图，不发真实请求。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from astercli.agent.plan_execute import PlanExecuteAgent
from astercli.cli.plan_review import PlanReviewDecision
from astercli.config import Settings
from astercli.hitl import ApprovalSession
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.agent.graph import make_checkpointer
from astercli.plan import ExecutionPlan, Task, TaskStatus, TaskType
from astercli.plan.planner import Planner
from astercli.policy import PathGuard
from astercli.tools.builtin.files import register_file_tools
from astercli.tools.registry import ToolRegistry


class ScriptedModel(BaseChatModel):
    """任务执行图用的假模型：按脚本顺序返回消息（元素可为 str 或 AIMessage）。

    str 用于 Planner 规划输出（纯对话 content），AIMessage 用于任务执行图。
    """

    script: list = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("测试只走 astream 路径")

    async def _astream(self, messages, stop=None, **kwargs):
        idx = self.call_count
        self.call_count += 1
        msg = self.script[idx] if idx < len(self.script) else AIMessage(content="(end)")
        if isinstance(msg, str):
            msg = AIMessage(content=msg)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=msg.content, tool_calls=msg.tool_calls)
        )


class RecordingRenderer:
    def __init__(self):
        self.infos = []
        self.warns = []

    def begin_thinking(self):
        pass

    def end_thinking(self):
        pass

    def stream_chunk(self, chunk):
        pass

    def take_reasoning(self):
        return ""

    def tool_calls(self, calls):
        pass

    def tool_results(self, calls, results):
        pass

    def halt(self, message):
        pass

    def info(self, message):
        self.infos.append(message)

    def warn(self, message):
        self.warns.append(message)


def plan_of(goal, tasks_spec):
    """tasks_spec: [(id, desc, deps), ...] → 已拓扑排序的 ExecutionPlan。"""
    plan = ExecutionPlan(goal=goal)
    for tid, desc, deps in tasks_spec:
        t = Task(tid, desc, TaskType.ANALYSIS)
        for d in deps:
            t.add_dependency(d)
        plan.add_task(t)
    assert plan.compute_execution_order()
    return plan


def make_agent(project, model, *, planner, review, approval=None):
    runtime = ProviderRuntime(
        client=model, provider="glm", model="scripted", spec=PROVIDER_SPECS["glm"]
    )
    registry = ToolRegistry(settings=Settings(load_files=False), approval=approval)
    register_file_tools(registry, PathGuard(project))
    renderer = RecordingRenderer()
    return PlanExecuteAgent(
        runtime,
        registry,
        renderer,
        planner,
        review,
        tool_mode="slim",
        approval=approval,
        checkpointer=make_checkpointer() if approval is not None else None,
        on_interrupt=None,
    )


class FakePlanner:
    """按脚本顺序出计划；记录每次 create_plan 的 goal 供断言。"""

    def __init__(self, plans):
        self.plans = list(plans)
        self.goals = []

    async def create_plan(self, goal):
        self.goals.append(goal)
        return self.plans.pop(0)

    async def replan(self, failed_plan, reason):
        return await self.create_plan(f"{failed_plan.goal} | replan: {reason}")


class FakeReview:
    """按脚本顺序返回审阅决策。"""

    def __init__(self, decisions):
        self.decisions = list(decisions)

    async def __call__(self, goal, plan):
        return self.decisions.pop(0)


# ---------- 审阅循环 ----------


async def test_review_execute_runs_plan(project):
    plan = plan_of("目标", [("task_1", "第一步", []), ("task_2", "第二步", ["task_1"])])
    planner = FakePlanner([plan])
    review = FakeReview([PlanReviewDecision.execute()])
    model = ScriptedModel(script=[AIMessage(content="结果一"), AIMessage(content="结果二")])
    agent = make_agent(project, model, planner=planner, review=review)

    result = await agent.run("目标")
    assert result.startswith("✅ 计划执行完成！")
    assert plan.is_all_completed()


async def test_review_cancel_returns_cancel_message(project):
    plan = plan_of("目标", [("task_1", "第一步", [])])
    planner = FakePlanner([plan])
    review = FakeReview([PlanReviewDecision.cancel()])
    model = ScriptedModel(script=[])
    agent = make_agent(project, model, planner=planner, review=review)

    result = await agent.run("目标")
    assert "取消" in result
    assert plan.get_task("task_1").status == TaskStatus.PENDING  # 未执行


async def test_review_supplement_triggers_replan(project):
    first = plan_of("目标", [("task_1", "初版任务", [])])
    second = plan_of("目标", [("task_1", "修订任务", [])])
    planner = FakePlanner([first, second])
    review = FakeReview(
        [PlanReviewDecision.supplement("要求更严格"), PlanReviewDecision.execute()]
    )
    model = ScriptedModel(script=[AIMessage(content="修订结果")])
    agent = make_agent(project, model, planner=planner, review=review)

    result = await agent.run("目标")
    # 第一次审阅被补充打断，重新规划（goal 追加补充要求）后再执行
    assert "补充要求" in planner.goals[1]
    assert second.get_task("task_1").status == TaskStatus.COMPLETED
    assert result.startswith("✅")


# ---------- 依赖顺序与结果汇总 ----------


async def test_tasks_execute_in_dependency_order(project):
    plan = plan_of(
        "目标",
        [("task_1", "读取", []), ("task_2", "分析", ["task_1"]), ("task_3", "汇报", ["task_2"])],
    )
    planner = FakePlanner([plan])
    review = FakeReview([PlanReviewDecision.execute()])
    model = ScriptedModel(
        script=[AIMessage(content="读取结果"), AIMessage(content="分析结果"), AIMessage(content="汇报结果")]
    )
    agent = make_agent(project, model, planner=planner, review=review)

    result = await agent.run("目标")
    assert plan.execution_order == ["task_1", "task_2", "task_3"]
    # 每个任务拿到自己的模型输出（ScriptedModel 按任务独立 thread 逐轮消费）
    assert plan.get_task("task_1").result == "读取结果"
    assert plan.get_task("task_2").result == "分析结果"
    # 最终结果 = 叶子任务（task_3）输出
    assert "[task_3] 汇报结果" in result


async def test_final_result_is_single_task_output(project):
    plan = plan_of("目标", [("task_1", "唯一任务", [])])
    planner = FakePlanner([plan])
    review = FakeReview([PlanReviewDecision.execute()])
    model = ScriptedModel(script=[AIMessage(content="唯一结果")])
    agent = make_agent(project, model, planner=planner, review=review)

    result = await agent.run("目标")
    assert "[task_1] 唯一结果" in result


# ---------- 失败与重规划 ----------


async def test_early_failure_triggers_replan(project):
    first = plan_of("目标", [("task_1", "会失败的任务", []), ("task_2", "后续", ["task_1"])])
    second = plan_of("目标", [("task_1", "重试任务", [])])
    planner = FakePlanner([first, second])
    review = FakeReview([PlanReviewDecision.execute(), PlanReviewDecision.execute()])
    model = ScriptedModel(script=[AIMessage(content="重试成功")])
    agent = make_agent(project, model, planner=planner, review=review)

    # 首个计划的首个任务抛异常：progress = 0 < 0.5 → replan + 重新审阅
    original = agent._execute_task
    failed_ids = []

    async def failing_execute(p, task):
        if p is first:
            failed_ids.append(task.id)
            raise RuntimeError("命令超时")
        return await original(p, task)

    agent._execute_task = failing_execute
    result = await agent.run("目标")

    assert failed_ids == ["task_1"]
    # replan 请求带失败原因；新计划经重新审阅后执行完成
    assert "命令超时" in planner.goals[1]
    assert second.get_task("task_1").status == TaskStatus.COMPLETED
    assert result.startswith("✅")


async def test_late_failure_reports_partial_without_replan(project):
    # 4 个任务完成 3 个后最后一个失败：progress = 0.75 ≥ 0.5 → 不 replan
    plan = plan_of(
        "目标",
        [
            ("task_1", "一", []),
            ("task_2", "二", ["task_1"]),
            ("task_3", "三", ["task_2"]),
            ("task_4", "四", ["task_3"]),
        ],
    )
    planner = FakePlanner([plan])
    review = FakeReview([PlanReviewDecision.execute()])
    model = ScriptedModel(script=[AIMessage(content="ok")] * 3)
    agent = make_agent(project, model, planner=planner, review=review)

    original = agent._execute_task

    async def failing_execute(p, task):
        if task.id == "task_4":
            raise RuntimeError("最后一步失败")
        return await original(p, task)

    agent._execute_task = failing_execute
    result = await agent.run("目标")

    assert result.startswith("⚠️ 计划部分完成")
    assert "task_4" in result  # 失败信息带上任务 id
    assert len(planner.goals) == 1  # 没有触发 replan


# ---------- 任务内工具审批 ----------


async def test_task_tool_call_hitl_conservative_reject(project):
    plan = plan_of("目标", [("task_1", "写入文件", [])])
    planner = FakePlanner([plan])
    review = FakeReview([PlanReviewDecision.execute()])
    model = ScriptedModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "out.txt", "content": "hi"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="写入被拒绝，我调整方案直接汇报"),
        ]
    )
    approval = ApprovalSession()
    agent = make_agent(project, model, planner=planner, review=review, approval=approval)

    # on_interrupt=None → ainvoke_with_hitl 对挂起保守拒绝（fail-safe），
    # 拒绝文本作为 ToolMessage 回灌，任务以最终 content 收尾
    result = await agent.run("目标")
    assert not (project / "out.txt").exists()
    assert plan.get_task("task_1").status == TaskStatus.COMPLETED
    assert "[task_1] 写入被拒绝" in result


# ---------- 真实 Planner 输出格式集成 ----------


async def test_agent_with_real_planner_output(project):
    plan_json = (
        '{"summary": "两步", "tasks": ['
        '{"id": "task_1", "description": "读取 README", "type": "FILE_READ", "dependencies": []},'
        '{"id": "task_2", "description": "总结内容", "type": "ANALYSIS", "dependencies": ["task_1"]}]}'
    )
    model = ScriptedModel(
        script=[plan_json, AIMessage(content="读取结果"), AIMessage(content="总结结果")]
    )
    runtime = ProviderRuntime(
        client=model, provider="glm", model="scripted", spec=PROVIDER_SPECS["glm"]
    )
    registry = ToolRegistry(settings=Settings(load_files=False))
    register_file_tools(registry, PathGuard(project))
    renderer = RecordingRenderer()
    agent = PlanExecuteAgent(
        runtime,
        registry,
        renderer,
        Planner(runtime, renderer),
        FakeReview([PlanReviewDecision.execute()]),
        tool_mode="slim",
    )

    result = await agent.run("分析 README 并总结")
    assert result.startswith("✅")
    assert "[task_2] 总结结果" in result
