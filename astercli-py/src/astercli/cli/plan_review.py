"""计划审阅交互（对齐 Java PlanReviewInputParser + createPlanReviewHandler）。

键位（与 Java 一致）：
- Enter      按当前计划执行
- Ctrl+O     展开完整计划（再按 ESC 折叠回摘要）
- ESC        已展开时折叠；未展开时取消本次计划
- I          输入补充要求后重新规划

方向键等未绑定按键由 prompt_toolkit 解析为独立按键事件（up/down），不会
误触发 ESC。审阅在 asyncio 事件循环内经 ``asyncio.to_thread`` 执行。
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


def _read_review_key() -> Optional[str]:
    """读单个审阅键：返回 execute / expand / escape / supplement；EOF → None。"""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result="execute")

    @kb.add("c-o")
    def _expand(event):
        event.app.exit(result="expand")

    @kb.add("escape")
    def _escape(event):
        event.app.exit(result="escape")

    @kb.add("i", eager=True)
    def _supplement_lower(event):
        event.app.exit(result="supplement")

    @kb.add("I", eager=True)
    def _supplement_upper(event):
        event.app.exit(result="supplement")

    session = PromptSession()
    try:
        result = session.prompt("审阅> ", key_bindings=kb)
    except (EOFError, KeyboardInterrupt):
        return None
    return result if isinstance(result, str) else None


def review_plan_sync(goal: str, plan, console=None) -> PlanReviewDecision:
    """同步审阅循环（在工作线程内执行）。"""
    _print(console, plan.summarize())
    _print(console, "📝 计划已生成。")
    _print(console, "   - 回车：按当前计划执行")
    _print(console, "   - Ctrl+O：展开完整计划")
    _print(console, "   - ESC：折叠或取消本次计划")
    _print(console, "   - I：输入补充要求后重新规划")

    expanded = False
    while True:
        action = _read_review_key()
        if action is None:
            return PlanReviewDecision.cancel()
        if action == "execute":
            return PlanReviewDecision.execute()
        if action == "expand":
            _print(console, plan.visualize())
            expanded = True
            _print(console, "👆 已展开完整计划，继续按 Enter / Ctrl+O / ESC / I。")
            continue
        if action == "escape":
            if expanded:
                expanded = False
                _print(console, plan.summarize())
                _print(console, "📁 已退出完整计划视图，继续按 Enter / Ctrl+O / ESC / I。")
                continue
            return PlanReviewDecision.cancel()
        if action == "supplement":
            try:
                line = input("补充> ")
            except (EOFError, OSError):
                return PlanReviewDecision.cancel()
            return parse_supplement_line(line)
        # 其它按键（未绑定的字母被收入缓冲区后经 Enter 以 execute 退出，
        # 不会走到这里）；防御性继续循环
        _print(console, "未识别按键，请按 Enter / Ctrl+O / ESC / I。")


async def review_plan(goal: str, plan, console=None) -> PlanReviewDecision:
    """审阅入口（事件循环内调用，阻塞读键放到工作线程）。"""
    return await asyncio.to_thread(review_plan_sync, goal, plan, console)


def _print(console, text: str) -> None:
    if console is not None:
        console.print(text)
    else:
        print(text)
