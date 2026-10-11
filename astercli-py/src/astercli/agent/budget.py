"""ReAct 循环退出预算（对齐 Java agent.AgentBudget）。

主退出条件 = LLM 自己决定（不再调用工具就结束）；本类只承担三种保险阀：
1. Token 预算：累计 input + output 超阈值强制收尾（默认不限，显式配置时生效）
2. 停滞检测：连续 N 轮完全相同的工具名 + 参数，判定死循环
3. 硬轮数兜底：超过 hard_max_iterations 强制收尾

环境变量（对齐 Java 系统属性 astercli.react.*）：
ASTERCLI_REACT_TOKEN_BUDGET / ASTERCLI_REACT_STAGNATION_WINDOW / ASTERCLI_REACT_HARD_MAX_ITERATIONS
"""

from __future__ import annotations

import json
from collections import deque
from enum import Enum
from typing import Optional


class ExitReason(Enum):
    WITHIN_BUDGET = "within_budget"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    STAGNATION_DETECTED = "stagnation_detected"
    HARD_ITERATION_LIMIT = "hard_iteration_limit"


class AgentBudget:
    def __init__(
        self,
        token_budget: Optional[int] = None,
        stagnation_window: int = 3,
        hard_max_iterations: int = 50,
    ) -> None:
        if stagnation_window < 2:
            raise ValueError("stagnation_window must be >= 2")
        if hard_max_iterations <= 0:
            raise ValueError("hard_max_iterations must be positive")
        self.token_budget = token_budget  # None = 不限
        self.stagnation_window = stagnation_window
        self.hard_max_iterations = hard_max_iterations
        self._recent_signatures: deque = deque()
        self.iteration = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_input_tokens = 0
        self._stagnant = False

    def begin_iteration(self) -> int:
        self.iteration += 1
        return self.iteration

    def record_tokens(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        self.total_input_tokens += max(0, input_tokens)
        self.total_output_tokens += max(0, output_tokens)
        self.total_cached_input_tokens += max(0, cached_input_tokens)

    def record_tool_calls(self, tool_calls: list) -> None:
        """记录本轮工具签名；连续 N 轮完全相同则进入停滞（状态保持）。"""
        if not tool_calls:
            self._recent_signatures.clear()
            return
        self._recent_signatures.append(_signature_of(tool_calls))
        while len(self._recent_signatures) > self.stagnation_window:
            self._recent_signatures.popleft()
        if len(self._recent_signatures) == self.stagnation_window:
            first = self._recent_signatures[0]
            self._stagnant = all(sig == first for sig in self._recent_signatures)

    def check(self) -> ExitReason:
        if self._stagnant:
            return ExitReason.STAGNATION_DETECTED
        if self.token_budget is not None and (
            self.total_input_tokens + self.total_output_tokens >= self.token_budget
        ):
            return ExitReason.TOKEN_BUDGET_EXCEEDED
        if self.iteration >= self.hard_max_iterations:
            return ExitReason.HARD_ITERATION_LIMIT
        return ExitReason.WITHIN_BUDGET

    def describe_exit(self, reason: ExitReason) -> str:
        if reason is ExitReason.TOKEN_BUDGET_EXCEEDED:
            used = self.total_input_tokens + self.total_output_tokens
            return f"Token 预算已用尽（{used} / {self.token_budget}），任务被强制收尾"
        if reason is ExitReason.STAGNATION_DETECTED:
            return (
                f"检测到连续 {self.stagnation_window} 轮重复的工具调用，疑似死循环，已强制收尾"
            )
        if reason is ExitReason.HARD_ITERATION_LIMIT:
            return f"达到硬轮数上限（{self.hard_max_iterations}），已强制收尾"
        return "未触发兜底条件"


def _signature_of(tool_calls: list) -> str:
    parts = []
    for tc in tool_calls:
        args = tc.get("args")
        try:
            args_text = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_text = str(args)
        parts.append(f"{tc.get('name', '?')}|{args_text};")
    return "".join(parts)
