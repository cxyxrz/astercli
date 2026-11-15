"""计划审阅交互（对齐 Java PlanReviewInputParser + createPlanReviewHandler）。

输入改为文本菜单（rich 没有按键回调，readline 也拿不到单独 ESC）：
- Enter       按当前计划执行
- v           展开完整计划（再次输入 v 折叠回摘要）
- q           已展开时折叠；未展开时取消本次计划
- i           输入补充要求后重新规划

审阅在 asyncio 事件循环内经 ``asyncio.to_thread`` 执行。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlanReviewDecision:
    action: str  # execute / cancel / supplement
    feedback: Optional[str] = None

    @staticmethod
    def execute() -> "PlanReviewDecision":
        return PlanReviewDecision("execute")

    @staticmethod
    def cancel() -> "PlanReviewDecision":
        return PlanReviewDecision("cancel")

    @staticmethod
    def supplement(feedback: str) -> "PlanReviewDecision":
        return PlanReviewDecision("supplement", feedback)


def parse_supplement_line(text: Optional[str]) -> PlanReviewDecision:
    """补充输入行解析（Java PlanReviewInputParser.parse 直译）。

    空 / y / yes / run / /run → 执行；cancel / esc / /cancel / 裸 ESC → 取消；
    其余文本 → 补充要求重新规划。
    """
    if text == "\u001B":
        return PlanReviewDecision.cancel()

    trimmed = (text or "").strip()
    if trimmed in ("", "y", "Y", "yes", "YES", "run", "/run"):
        return PlanReviewDecision.execute()
    if trimmed in ("cancel", "esc", "/cancel"):
        return PlanReviewDecision.cancel()
    return PlanReviewDecision.supplement(trimmed)


def parse_review_command(text: Optional[str]) -> str:
    """审阅菜单解析：execute / expand / cancel / supplement / unknown。"""
    if text == "\u001B":
        return "cancel"
    trimmed = (text or "").strip().lower()
    if trimmed in ("", "enter", "y", "yes", "run"):
        return "execute"
    if trimmed in ("v", "view", "expand"):
        return "expand"
    if trimmed in ("q", "quit", "cancel", "esc"):
        return "cancel"
    if trimmed in ("i", "input", "edit"):
        return "supplement"
    return "unknown"


def _read_line(console, prompt: str) -> str:
    return console.input(prompt) if console is not None else input(prompt)


def review_plan_sync(goal: str, plan, console=None) -> PlanReviewDecision:
    """同步审阅循环（在工作线程内执行）。"""
    _print(console, plan.summarize())
    _print(console, "📝 计划已生成。")
    _print(console, "   - 回车：按当前计划执行")
    _print(console, "   - v：展开 / 折叠完整计划")
    _print(console, "   - q：折叠或取消本次计划")
    _print(console, "   - i：输入补充要求后重新规划")

    expanded = False
    while True:
        try:
            line = _read_line(console, "审阅> ")
        except (EOFError, KeyboardInterrupt, OSError):
            return PlanReviewDecision.cancel()
        action = parse_review_command(line)
        if action == "execute":
            return PlanReviewDecision.execute()
        if action == "expand":
            expanded = not expanded
            _print(console, plan.visualize() if expanded else plan.summarize())
            continue
        if action == "cancel":
            if expanded:
                expanded = False
                _print(console, plan.summarize())
                _print(console, "📁 已折叠回摘要，继续按 Enter / v / q / i。")
                continue
            return PlanReviewDecision.cancel()
        if action == "supplement":
            try:
                line = _read_line(console, "补充> ")
            except (EOFError, KeyboardInterrupt, OSError):
                return PlanReviewDecision.cancel()
            return parse_supplement_line(line)
        _print(console, "未识别输入，请按 Enter / v / q / i。")


async def review_plan(goal: str, plan, console=None) -> PlanReviewDecision:
    """审阅入口（事件循环内调用，阻塞读键放到工作线程）。"""
    return await asyncio.to_thread(review_plan_sync, goal, plan, console)


def _print(console, text: str) -> None:
    if console is not None:
        console.print(text)
    else:
        print(text)
