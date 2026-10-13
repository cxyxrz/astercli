"""计划数据模型（对齐 Java plan.Task / plan.ExecutionPlan）。

- Task：任务单元（类型 / 状态 / 依赖 / 结果）
- ExecutionPlan：任务集合 + 拓扑排序执行顺序 + 批次划分

可视化不使用右对齐边框（CJK 字宽会导致错位），只做左对齐列表。
"""

from __future__ import annotations

import time
from enum import Enum


class TaskType(str, Enum):
    PLANNING = "PLANNING"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    COMMAND = "COMMAND"
    ANALYSIS = "ANALYSIS"
    VERIFICATION = "VERIFICATION"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PlanStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_STATUS_ICONS = {
    TaskStatus.PENDING: "⏳",
    TaskStatus.RUNNING: "▶️",
    TaskStatus.COMPLETED: "✅",
    TaskStatus.FAILED: "❌",
    TaskStatus.SKIPPED: "⏭️",
}


class Task:
    def __init__(self, task_id: str, description: str, type_: TaskType = TaskType.ANALYSIS) -> None:
        self.id = task_id
        self.description = description or ""
        self.type = type_
        self.status = TaskStatus.PENDING
        self.result: str = ""
        self.error: str = ""
        self.dependencies: list = []
        self.dependents: list = []
        self.start_time = 0.0
        self.end_time = 0.0

    def add_dependency(self, task_id: str) -> None:
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def add_dependent(self, task_id: str) -> None:
        if task_id not in self.dependents:
            self.dependents.append(task_id)

    def mark_started(self) -> None:
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result or ""
        self.end_time = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error or ""
        self.end_time = time.time()

    def mark_skipped(self) -> None:
        self.status = TaskStatus.SKIPPED
        self.end_time = time.time()

    def is_executable(self, all_tasks: dict) -> bool:
        """PENDING 且所有依赖都 COMPLETED。"""
        if self.status != TaskStatus.PENDING:
            return False
        for dep_id in self.dependencies:
            dep = all_tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def __repr__(self) -> str:  # pragma: no cover
        return f"Task[{self.id}: {self.description}] ({self.status})"


class ExecutionPlan:
    def __init__(self, goal: str, summary: str = "", plan_id: str = "") -> None:
        self.id = plan_id or f"plan_{int(time.time() * 1000)}"
        self.goal = goal or ""
        self.summary = summary or ""
        self.tasks: dict = {}  # 保持插入顺序
        self.execution_order: list = []
        self.status = PlanStatus.CREATED
        self.start_time = 0.0
        self.end_time = 0.0

    # ---------- 结构 ----------

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task
        for dep_id in task.dependencies:
            dep = self.tasks.get(dep_id)
            if dep is not None:
                dep.add_dependent(task.id)

    def get_task(self, task_id: str):
        return self.tasks.get(task_id)

    def all_tasks(self) -> list:
        return list(self.tasks.values())

    def executable_tasks(self) -> list:
        return [t for t in self.tasks.values() if t.is_executable(self.tasks)]

    def compute_execution_order(self) -> bool:
        """DFS 拓扑排序；有环返回 False。"""
        self.execution_order = []
        visited: set = set()
        visiting: set = set()

        def visit(task: Task) -> bool:
            if task.id in visiting:
                return False
            if task.id in visited:
                return True
            visiting.add(task.id)
            for dep_id in task.dependencies:
                dep = self.tasks.get(dep_id)
                if dep is not None and not visit(dep):
                    return False
            visiting.discard(task.id)
            visited.add(task.id)
            self.execution_order.append(task.id)
            return True

        return all(visit(t) for t in self.tasks.values())

    def execution_batches(self) -> list:
        """按依赖层级分批（同层任务可并行；P4 顺序执行，仅用于展示）。"""
        if not self.tasks:
            return []
        remaining = dict(self.tasks)
        completed: set = set()
        batches: list = []
        while remaining:
            batch = [t for t in remaining.values() if completed.issuperset(t.dependencies)]
            if not batch:
                break
            batches.append(batch)
            for task in batch:
                remaining.pop(task.id)
                completed.add(task.id)
        return batches

    # ---------- 状态 ----------

    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    def is_all_completed(self) -> bool:
        return bool(self.tasks) and all(
            t.status == TaskStatus.COMPLETED for t in self.tasks.values()
        )

    def has_failed(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())

    def mark_started(self) -> None:
        self.status = PlanStatus.RUNNING
        self.start_time = time.time()

    def mark_completed(self) -> None:
        self.status = PlanStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self) -> None:
        self.status = PlanStatus.FAILED
        self.end_time = time.time()

    # ---------- 可视化 ----------

    def visualize(self) -> str:
        lines = [f"📋 执行计划: {self.goal}", "─" * 40]
        for i, task_id in enumerate(self.execution_order, 1):
            task = self.tasks[task_id]
            deps = ",".join(task.dependencies) if task.dependencies else "无"
            lines.append(f"{i}. {_STATUS_ICONS[task.status]} {task.id} [{task.type.value}] 依赖: {deps}")
            lines.append(f"   {task.description}")
        lines.append("─" * 40)
        lines.append(f"进度: {self.progress() * 100:.0f}% | 状态: {self.status.value}")
        return "\n".join(lines)

    def summarize(self) -> str:
        batches = self.execution_batches()
        ready = self.executable_tasks()
        lines = [
            "📋 计划摘要",
            f"   - 目标: {self._compact_goal(self.goal, 48)}",
            f"   - 任务数: {len(self.tasks)} | 并行批次: {len(batches)}"
            f" | 当前可执行: {len(ready)} | 状态: {self.status.value}",
        ]
        if batches:
            lines.append(f"   - 首批执行: {self._format_task_ids(batches[0])}")
            if len(batches) > 1:
                lines.append(f"   - 最终收敛: {self._format_task_ids(batches[-1])}")
        return "\n".join(lines)

    @staticmethod
    def _compact_goal(raw: str, max_length: int) -> str:
        single_line = " ".join((raw or "").split())
        if len(single_line) <= max_length:
            return single_line
        return single_line[: max_length - 3] + "..."

    @staticmethod
    def _format_task_ids(batch: list, limit: int = 5) -> str:
        ids = [t.id for t in batch]
        if len(ids) <= limit:
            return ", ".join(ids) if ids else "无"
        return ", ".join(ids[:limit]) + f" 等 {len(ids)} 个任务"
