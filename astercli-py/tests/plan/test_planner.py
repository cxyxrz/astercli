"""规划器测试：计划 JSON 解析（纯函数）/ 简单任务快路径 / LLM 规划 / replan。

ScriptedModel 驱动 LLM 路径，不发真实请求。
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from astercli.llm.factory import ProviderRuntime
from astercli.llm.providers import PROVIDER_SPECS
from astercli.plan import ExecutionPlan, Task
from astercli.plan.planner import Planner, is_simple_goal, parse_plan


# ---------- parse_plan（纯函数） ----------


def test_parse_plan_basic_json():
    plan_json = """
    ```json
    {
      "summary": "两步计划",
      "tasks": [
        {"id": "task_1", "description": "读取配置", "type": "FILE_READ", "dependencies": []},
        {"id": "task_2", "description": "分析结果", "type": "ANALYSIS", "dependencies": ["task_1"]}
      ]
    }
    ```
    """
    plan = parse_plan("目标", plan_json)
    assert plan.summary == "两步计划"
    assert list(plan.tasks) == ["task_1", "task_2"]
    assert plan.get_task("task_2").dependencies == ["task_1"]
    assert plan.execution_order == ["task_1", "task_2"]


def test_parse_plan_normalizes_llm_custom_ids():
    # LLM 自造 id → 统一归一化为 task_1..N，依赖同步映射
    plan_json = """
    {
      "tasks": [
        {"id": "step-a", "description": "先做", "type": "FILE_READ", "dependencies": []},
        {"id": "step-b", "description": "后做", "type": "ANALYSIS", "dependencies": ["step-a"]}
      ]
    }
    """
    plan = parse_plan("目标", plan_json)
    assert list(plan.tasks) == ["task_1", "task_2"]
    assert plan.get_task("task_2").dependencies == ["task_1"]


def test_parse_plan_forward_reference_dependencies():
    # 前向引用（依赖声明出现在被依赖任务之前）：第二遍回填时解析
    plan_json = """
    {
      "tasks": [
        {"id": "t1", "description": "第一步", "type": "COMMAND", "dependencies": ["t2"]},
        {"id": "t2", "description": "第二步", "type": "COMMAND", "dependencies": []}
      ]
    }
    """
    plan = parse_plan("目标", plan_json)
    assert plan.get_task("task_1").dependencies == ["task_2"]
    assert plan.execution_order == ["task_2", "task_1"]


def test_parse_plan_unknown_dependency_dropped():
    plan_json = """
    {
      "tasks": [
        {"id": "t1", "description": "x", "type": "COMMAND", "dependencies": ["ghost"]},
        {"id": "t2", "description": "y", "type": "COMMAND", "dependencies": []}
      ]
    }
    """
    plan = parse_plan("目标", plan_json)
    assert plan.get_task("task_1").dependencies == []


def test_parse_plan_invalid_type_falls_back_to_analysis():
    plan_json = '{"tasks": [{"id": "t1", "description": "x", "type": "不认识", "dependencies": []}]}'
    plan = parse_plan("目标", plan_json)
    assert plan.get_task("task_1").type.value == "ANALYSIS"


def test_parse_plan_rejects_cycle():
    plan_json = """
    {
      "tasks": [
        {"id": "t1", "description": "x", "type": "COMMAND", "dependencies": ["t2"]},
        {"id": "t2", "description": "y", "type": "COMMAND", "dependencies": ["t1"]}
      ]
    }
    """
    with pytest.raises(ValueError, match="循环依赖"):
        parse_plan("目标", plan_json)


def test_parse_plan_rejects_bad_json():
    with pytest.raises(ValueError, match="解析失败"):
        parse_plan("目标", "{not json")


def test_parse_plan_rejects_non_object_root():
    with pytest.raises(ValueError, match="对象"):
        parse_plan("目标", "[1, 2]")


def test_parse_plan_rejects_missing_tasks():
    with pytest.raises(ValueError, match="tasks"):
        parse_plan("目标", '{"summary": "空"}')


# ---------- is_simple_goal ----------


def test_simple_goal_with_verb():
    assert is_simple_goal("查看当前目录") is True
    assert is_simple_goal("列出文件") is True


def test_multi_step_cue_is_not_simple():
    assert is_simple_goal("读取 README 然后总结") is False
    assert is_simple_goal("先查看再分析") is False


def test_long_goal_is_not_simple():
    assert is_simple_goal("查看" + "特别" * 15 + "长的目标内容") is False


def test_goal_without_verb_is_not_simple():
    assert is_simple_goal("你好世界") is False


# ---------- Planner（ScriptedModel 驱动） ----------


class ScriptedModel(BaseChatModel):
    script: list = []
    outputs: list = []
    requests: list = []  # 每次 astream 收到的 messages

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("测试只走 astream 路径")

    async def _astream(self, messages, stop=None, **kwargs):
        self.requests.append(list(messages))
        idx = len(self.outputs)
        text = self.script[idx] if idx < len(self.script) else "(end)"
        self.outputs.append(text)
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))


def make_planner(model):
    return Planner(ProviderRuntime(client=model, provider="glm", model="scripted", spec=PROVIDER_SPECS["glm"]))


async def test_create_plan_simple_goal_skips_llm():
    model = ScriptedModel(script=["不应被调用"])
    planner = make_planner(model)
    plan = await planner.create_plan("读取文件")
    assert model.outputs == []  # 快路径不调 LLM
    assert list(plan.tasks) == ["task_1"]
    assert plan.get_task("task_1").type.value == "FILE_READ"


async def test_create_plan_uses_llm_json():
    plan_json = (
        '{"summary": "两步", "tasks": ['
        '{"id": "task_1", "description": "读", "type": "FILE_READ", "dependencies": []},'
        '{"id": "task_2", "description": "析", "type": "ANALYSIS", "dependencies": ["task_1"]}'
        "]}"
    )
    model = ScriptedModel(script=[plan_json])
    planner = make_planner(model)
    plan = await planner.create_plan("分析项目结构并给出总结报告")
    assert model.outputs == [plan_json]
    assert plan.execution_order == ["task_1", "task_2"]


async def test_replan_carries_failure_context():
    plan_json = (
        '{"summary": "重试", "tasks": ['
        '{"id": "task_1", "description": "重试命令", "type": "COMMAND", "dependencies": []}]}'
    )
    model = ScriptedModel(script=[plan_json])
    planner = make_planner(model)
    failed = ExecutionPlan(goal="原目标")
    completed = Task("task_1", "已完成的任务")
    completed.mark_completed("ok")
    failed.add_task(completed)
    plan = await planner.replan(failed, "命令超时")

    # replan 请求里带原目标 + 失败原因 + 已完成任务清单
    request_text = "\n".join(str(m.content) for m in model.requests[0])
    assert "原目标" in request_text
    assert "命令超时" in request_text
    assert "已完成的任务" in request_text
    assert list(plan.tasks) == ["task_1"]
    assert plan.get_task("task_1").description == "重试命令"
