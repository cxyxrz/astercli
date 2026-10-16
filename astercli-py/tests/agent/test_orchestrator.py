"""Multi-Agent 编排测试：纯函数解析 + FakeTeamModel 全流程（零网络）。

FakeTeamModel 按系统提示词分派角色：
- planner（规划者）→ 步骤计划 JSON
- worker（执行者）→ ReAct：先调 echo 工具，再出汇报文本
- reviewer（检查者）→ 审查 JSON（队列控制 approve/reject 序列）
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from astercli.agent.budget import AgentBudget
from astercli.agent.orchestrator import (
    MAX_RETRIES_PER_STEP,
    ExecutionStep,
    StepStatus,
    build_final_result,
    build_step_context,
    get_executable_steps,
    parse_plan,
    parse_review_approval,
    parse_review_issues,
    summarize_steps,
)
from astercli.agent.orchestrator import AgentOrchestrator as _Orchestrator
from astercli.config import Settings
from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.tools.registry import Param, ToolRegistry, create_parameters


# ---------- 纯函数 ----------


class TestParsePlan:
    def test_标准_json(self):
        steps = parse_plan(
            """```json
            {"summary": "s", "steps": [
                {"id": "a", "description": "第一步", "type": "FILE_READ", "dependencies": []},
                {"id": "b", "description": "第二步", "type": "FILE_WRITE", "dependencies": ["a"]}
            ]}
            ```"""
        )
        assert [s.id for s in steps] == ["step_1", "step_2"]
        assert steps[1].dependencies == ["step_1"]  # 依赖映射到重编号 id

    def test_tasks_字段_兼容(self):
        steps = parse_plan('{"tasks": [{"id": "t1", "description": "唯一步骤"}]}')
        assert len(steps) == 1
        assert steps[0].id == "step_1"

    def test_空_与_非法输入(self):
        assert parse_plan("") == []
        assert parse_plan("不是 JSON") == []
        assert parse_plan('{"steps": []}') == []
        assert parse_plan('{"steps": "不是列表"}') == []

    def test_非字典_元素_跳过(self):
        steps = parse_plan('{"steps": ["纯字符串", {"description": "有效"}]}')
        assert len(steps) == 1


class TestExecutableSteps:
    def make_steps(self):
        return [
            ExecutionStep(id="step_1", description="一", dependencies=[]),
            ExecutionStep(id="step_2", description="二", dependencies=["step_1"]),
            ExecutionStep(id="step_3", description="三", dependencies=["step_1"]),
            ExecutionStep(id="step_4", description="四", dependencies=["step_2", "step_3"]),
        ]

    def test_初始只有_无依赖步(self):
        runnable = get_executable_steps(self.make_steps())
        assert [s.id for s in runnable] == ["step_1"]

    def test_依赖完成后_分支解锁(self):
        steps = [s.with_status(StepStatus.COMPLETED) if s.id == "step_1" else s for s in self.make_steps()]
        runnable = get_executable_steps(steps)
        assert [s.id for s in runnable] == ["step_2", "step_3"]

    def test_依赖失败_后续_永不解锁(self):
        steps = [s.with_status(StepStatus.FAILED) if s.id == "step_1" else s for s in self.make_steps()]
        assert get_executable_steps(steps) == []
        # step_2/3 PENDING，step_4 依赖它们 → 主循环结束后按"跳过"处理

    def test_全链完成_无_可执行(self):
        steps = [s.with_status(StepStatus.COMPLETED) for s in self.make_steps()]
        assert get_executable_steps(steps) == []


class TestReviewParsing:
    def test_approved_json(self):
        assert parse_review_approval('{"approved": true, "issues": []}') is True
        assert parse_review_approval('{"approved": false}') is False

    def test_空_与_解析失败_保守拒绝(self):
        assert parse_review_approval(None) is False
        assert parse_review_approval("") is False
        assert parse_review_approval("垃圾内容") is False
        assert parse_review_approval('{"no_approved_field": 1}') is False

    def test_自由文本_启发式(self):
        assert parse_review_approval("结果正确，通过") is True
        assert parse_review_approval("审查未通过，存在遗漏") is False

    def test_issues_提取(self):
        review = '{"approved": false, "issues": ["问题甲", "问题乙"], "suggestions": ["建议一"]}'
        issues = parse_review_issues(review)
        assert "- 问题甲" in issues and "- 问题乙" in issues

    def test_issues_无结构_兜底文案(self):
        assert "改进" in parse_review_issues("随便说的")
        assert "改进" in parse_review_issues(None)


class TestSummaries:
    def test_step_context_只含_已完成依赖(self):
        steps = [
            ExecutionStep(id="step_1", description="读取", dependencies=[]).with_status(
                StepStatus.COMPLETED, "读取结果"
            ),
            ExecutionStep(id="step_2", description="独立", dependencies=[]),
            ExecutionStep(id="step_3", description="依赖一", dependencies=["step_1"]),
        ]
        ctx = build_step_context(steps, steps[2])
        assert "已完成的依赖步骤 [step_1]" in ctx
        assert "读取结果" in ctx
        assert "step_2" not in ctx

    def test_final_result_状态汇总(self):
        steps = [
            ExecutionStep(id="step_1", description="ok", dependencies=[]).with_status(
                StepStatus.COMPLETED, "完成"
            ),
            ExecutionStep(id="step_2", description="fail", dependencies=["step_1"]).with_status(
                StepStatus.FAILED, "失败"
            ),
            ExecutionStep(id="step_3", description="skip", dependencies=["step_2"]),
        ]
        out = build_final_result(steps)
        assert "未完全完成" in out
        assert "✅" in out and "❌" in out and "⏳" in out

    def test_summarize_steps_显示依赖(self):
        steps = [ExecutionStep(id="step_1", description="d", dependencies=["a"])]
        assert "(依赖: a)" in summarize_steps(steps)


# ---------- 全流程（FakeTeamModel） ----------


def tool_call_msg(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_, "type": "tool_call"}])


class FakeTeamModel(BaseChatModel):
    """按 system prompt 分派角色脚本的假模型。

    - planner: 返回 plan_script
    - worker: 消息末尾是 ToolMessage → 返回 worker_final；否则返回 worker_first
    - reviewer: 依次弹出 review_script 队列
    """

    plan_script: str = ""
    worker_first: object = None
    worker_final: str = "步骤执行完毕：已生成结果"
    review_script: list = []
    reviewer_calls: int = 0
    worker_calls: int = 0

    @property
    def _llm_type(self):
        return "fake-team"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("测试只走 astream 路径")

    @staticmethod
    def _to_chunk(msg):
        return AIMessageChunk(
            content=msg.content,
            tool_calls=getattr(msg, "tool_calls", None) or [],
        )

    async def _astream(self, messages, stop=None, **kwargs):
        system = messages[0] if messages else None
        role = ""
        if isinstance(system, SystemMessage):
            # 用角色括号词分派：planner 提示词正文也含"执行者"字样，中文词不可靠
            if "（planner）" in system.content:
                role = "planner"
            elif "（worker）" in system.content:
                role = "worker"
            elif "（reviewer）" in system.content:
                role = "reviewer"
        if role == "planner":
            msg = AIMessage(content=self.plan_script)
        elif role == "worker":
            self.worker_calls += 1
            last = messages[-1] if messages else None
            if isinstance(last, ToolMessage):
                msg = AIMessage(content=self.worker_final)
            else:
                msg = self.worker_first or AIMessage(content="直接完成，无需工具")
        elif role == "reviewer":
            idx = min(self.reviewer_calls, len(self.review_script) - 1) if self.review_script else 0
            self.reviewer_calls += 1
            script = self.review_script[idx] if self.review_script else '{"approved": true}'
            msg = AIMessage(content=script)
        else:
            msg = AIMessage(content="未知角色")
        yield ChatGenerationChunk(message=self._to_chunk(msg))


class RecordingRenderer:
    def __init__(self):
        self.infos = []
        self.warns = []
        self.halts = []

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

    def info(self, message):
        self.infos.append(message)

    def warn(self, message):
        self.warns.append(message)

    def halt(self, message):
        self.halts.append(message)


def make_registry():
    settings = Settings(env={}, load_files=False)
    registry = ToolRegistry(settings=settings)
    registry.register(
        "echo",
        "回显工具",
        create_parameters(Param("text", "string", "文本", True)),
        lambda args: args.get("text", ""),
    )
    return registry


def make_orchestrator(model, project_root):
    renderer = RecordingRenderer()
    runtime = ProviderRuntime(
        client=model, provider="glm", model="fake", spec=PROVIDER_SPECS["glm"]
    )
    registry = make_registry()
    settings = Settings(env={}, load_files=False)
    orch = _Orchestrator(
        runtime,
        registry,
        renderer,
        project_root=project_root,
        settings=settings,
    )
    return orch, renderer


PLAN_ONE_STEP = (
    '{"summary": "一步计划", "steps": ['
    '{"id": "s1", "description": "生成示例文件", "type": "COMMAND", "dependencies": []}]}'
)


class TestOrchestratorFlow:
    async def test_单步_一次通过(self, tmp_path):
        model = FakeTeamModel()
        model.plan_script = PLAN_ONE_STEP
        model.worker_first = tool_call_msg("echo", {"text": "hi"}, "call-1")
        model.review_script = ['{"approved": true, "summary": "通过"}']
        orch, renderer = make_orchestrator(model, tmp_path)
        result = await orch.run("做一个演示任务")
        assert "✅ 多 Agent 协作任务完成" in result
        # worker_calls 计 astream 轮数：1 次 worker × 2 轮（tool_call → final）
        assert model.worker_calls == 2
        assert model.reviewer_calls == 1
        # 执行与审查阶段都有渲染输出
        assert any("worker 执行步骤" in m for m in renderer.infos)
        assert any("reviewer 正在审查" in m for m in renderer.infos)

    async def test_审查拒绝_重试后通过(self, tmp_path):
        model = FakeTeamModel()
        model.plan_script = PLAN_ONE_STEP
        model.worker_first = tool_call_msg("echo", {"text": "v1"}, "c1")
        model.worker_final = "改进后的结果"
        model.review_script = [
            '{"approved": false, "issues": ["缺少测试"], "suggestions": ["补测试"]}',
            '{"approved": true, "summary": "通过"}',
        ]
        orch, renderer = make_orchestrator(model, tmp_path)
        result = await orch.run("任务")
        assert "✅" in result
        assert model.worker_calls == 4  # 2 次 worker × 2 轮
        assert model.reviewer_calls == 2
        assert any("审查未通过" in m for m in renderer.infos)

    async def test_审查一直拒绝_保留结果并_警示(self, tmp_path):
        model = FakeTeamModel()
        model.plan_script = PLAN_ONE_STEP
        model.worker_first = tool_call_msg("echo", {"text": "x"}, "c1")
        model.review_script = ['{"approved": false, "issues": ["仍不合格"]}']
        orch, renderer = make_orchestrator(model, tmp_path)
        result = await orch.run("任务")
        # 达到最大重试后保留结果，仍标记 COMPLETED（对齐 Java）
        assert "✅" in result
        assert model.worker_calls == (1 + MAX_RETRIES_PER_STEP) * 2  # 3 次 worker × 2 轮
        assert any("超过最大重试次数" in m for m in renderer.infos)

    async def test_规划失败_返回_原始输出(self, tmp_path):
        model = FakeTeamModel()
        model.plan_script = "我认为无法规划"
        orch, _ = make_orchestrator(model, tmp_path)
        result = await orch.run("任务")
        assert "❌ 规划失败" in result
        assert "我认为无法规划" in result

    async def test_多步依赖_按序执行(self, tmp_path):
        model = FakeTeamModel()
        model.plan_script = (
            '{"summary": "两步", "steps": ['
            '{"id": "a", "description": "第一步", "type": "COMMAND", "dependencies": []},'
            '{"id": "b", "description": "第二步", "type": "COMMAND", "dependencies": ["a"]}]}'
        )
        model.worker_first = tool_call_msg("echo", {"text": "t"}, "c1")
        model.review_script = ['{"approved": true}']
        orch, renderer = make_orchestrator(model, tmp_path)
        result = await orch.run("任务")
        assert "✅" in result
        assert model.worker_calls == 4  # 2 步 × 2 轮
        # 第二步的输入包含第一步结果
        infos_text = "\n".join(renderer.infos)
        assert "已完成的依赖步骤 [step_1]" not in infos_text  # 上下文进消息不进渲染
        assert "[step_2]" in infos_text

    async def test_前置失败_后续步骤_被跳过(self, tmp_path):
        model = FakeTeamModel()
        model.plan_script = (
            '{"summary": "两步", "steps": ['
            '{"id": "a", "description": "会失败的步骤", "dependencies": []},'
            '{"id": "b", "description": "依赖它的步骤", "dependencies": ["a"]}]}'
        )
        # worker 不带 tool_call 且返回空内容 → halt → step 失败
        model.worker_first = AIMessage(content="")
        orch, renderer = make_orchestrator(model, tmp_path)
        result = await orch.run("任务")
        assert "未完全完成" in result
        assert any("被跳过" in m for m in renderer.infos)


class TestBudgetWiring:
    def test_worker_budget_默认不限(self):
        settings = Settings(env={}, load_files=False)
        budget = AgentBudget(
            settings.react_token_budget(),
            settings.react_stagnation_window(),
            settings.react_hard_max_iterations(),
        )
        assert budget.token_budget is None
