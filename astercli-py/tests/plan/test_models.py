"""计划数据模型测试：依赖图 / 拓扑排序 / 批次划分 / 状态流转 / 可视化。"""

from __future__ import annotations

from astercli.plan import ExecutionPlan, Task, TaskStatus, TaskType


def make_task(task_id, deps=()):
    t = Task(task_id, f"任务 {task_id}", TaskType.ANALYSIS)
    for d in deps:
        t.add_dependency(d)
    return t


def make_plan():
    plan = ExecutionPlan(goal="完成某复杂目标", summary="演示计划")
    plan.add_task(make_task("task_1"))
    plan.add_task(make_task("task_2", deps=("task_1",)))
    plan.add_task(make_task("task_3", deps=("task_1",)))
    plan.add_task(make_task("task_4", deps=("task_2", "task_3")))
    return plan


# ---------- 结构 ----------


def test_add_task_backfills_dependents():
    plan = make_plan()
    assert plan.get_task("task_1").dependents == ["task_2", "task_3"]
    assert plan.get_task("task_2").dependents == ["task_4"]
    assert plan.get_task("task_4").dependents == []


def test_compute_execution_order_is_topological():
    plan = make_plan()
    assert plan.compute_execution_order() is True
    order = plan.execution_order
    assert set(order) == {"task_1", "task_2", "task_3", "task_4"}
    assert order.index("task_1") < order.index("task_2")
    assert order.index("task_1") < order.index("task_3")
    assert order.index("task_2") < order.index("task_4")
    assert order.index("task_3") < order.index("task_4")


def test_compute_execution_order_detects_cycle():
    plan = ExecutionPlan(goal="环")
    plan.add_task(make_task("task_1", deps=("task_2",)))
    plan.add_task(make_task("task_2", deps=("task_1",)))
    assert plan.compute_execution_order() is False


def test_execution_batches_layers():
    plan = make_plan()
    plan.compute_execution_order()
    batches = plan.execution_batches()
    assert [t.id for t in batches[0]] == ["task_1"]
    assert {t.id for t in batches[1]} == {"task_2", "task_3"}
    assert [t.id for t in batches[2]] == ["task_4"]


# ---------- 可执行性与状态 ----------


def test_is_executable_requires_all_deps_completed():
    plan = make_plan()
    plan.compute_execution_order()
    t1 = plan.get_task("task_1")
    t2 = plan.get_task("task_2")

    assert t1.is_executable(plan.tasks) is True
    assert t2.is_executable(plan.tasks) is False

    t1.mark_started()
    assert t1.is_executable(plan.tasks) is False  # RUNNING 不可再执行
    t1.mark_completed("ok")
    assert t2.is_executable(plan.tasks) is True


def test_is_executable_with_missing_dep_returns_false():
    plan = ExecutionPlan(goal="g")
    t = make_task("task_1", deps=("task_ghost",))
    plan.add_task(t)
    assert t.is_executable(plan.tasks) is False


def test_progress_and_completion_flags():
    plan = make_plan()
    assert plan.progress() == 0.0
    plan.get_task("task_1").mark_completed("a")
    assert plan.progress() == 0.25
    assert not plan.is_all_completed()
    assert not plan.has_failed()

    for tid in ("task_2", "task_3", "task_4"):
        plan.get_task(tid).mark_completed("x")
    assert plan.progress() == 1.0
    assert plan.is_all_completed()


def test_failed_task_flags_plan_failure():
    plan = make_plan()
    plan.get_task("task_2").mark_started()
    plan.get_task("task_2").mark_failed("boom")
    assert plan.has_failed()
    assert plan.get_task("task_2").status == TaskStatus.FAILED


def test_mark_skipped_sets_status():
    t = make_task("t")
    t.mark_skipped()
    assert t.status == TaskStatus.SKIPPED


def test_empty_plan_progress_is_one():
    plan = ExecutionPlan(goal="空")
    assert plan.progress() == 1.0
    assert plan.is_all_completed() is False  # 空计划不算"全部完成"


# ---------- 可视化 ----------


def test_visualize_lists_ordered_tasks_with_deps():
    plan = make_plan()
    plan.compute_execution_order()
    text = plan.visualize()
    assert "执行计划" in text
    assert "task_4" in text
    assert "task_2,task_3" in text
    assert "依赖: 无" in text  # task_1 无依赖


def test_summarize_reports_batches_and_goal():
    plan = make_plan()
    plan.compute_execution_order()
    text = plan.summarize()
    assert "计划摘要" in text
    assert "任务数: 4" in text
    assert "并行批次: 3" in text
    assert "首批执行: task_1" in text
    assert "最终收敛: task_4" in text
    assert "完成某复杂目标" in text
