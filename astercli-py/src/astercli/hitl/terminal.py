"""终端 HITL 审批交互（对齐 Java TerminalHitlHandler）。

行式交互（与 Java 相同，读整行而非单键）：
  y / Enter - 批准本次操作
  a         - 批准本次 + 本会话后续同类操作自动放行（工具维度）
  n         - 拒绝本次（可补一行原因）
  s         - 跳过本步骤
  m         - 修改参数后执行（输入 JSON）

fail-safe：输入流关闭 / 连续 5 次无效输入 → 保守拒绝。
交互走 ``asyncio.to_thread`` 阻塞读行（图已挂起，无并发输出方）。
"""

from __future__ import annotations

import asyncio
import json

from .approval import ApprovalSession, parse_hitl_choice

_MAX_INVALID_ATTEMPTS = 5
_ARG_PREVIEW_CHARS = 500


class HitlPrompter:
    """把 interrupt 载荷渲染成审批框并收集用户决策。"""

    def __init__(self, console=None) -> None:
        self._console = console

    async def collect(self, interrupts) -> dict:
        """处理图挂起返回的 __interrupt__，返回 ``{call_id: decision}`` 作为 resume 值。"""
        payload = None
        if interrupts:
            first = interrupts[0]
            payload = getattr(first, "value", None)
        if not isinstance(payload, dict) or not payload.get("calls"):
            # 不认识的挂起类型：整体拒绝，避免无审批放行
            return {}
        return await asyncio.to_thread(self._prompt_sync, payload)

    # ---------- 同步交互（工作线程内执行） ----------

    def _prompt_sync(self, payload: dict) -> dict:
        decisions: dict = {}
        for call in payload.get("calls", []):
            self._render_request(call)
            decision = self._prompt_until_decision(call)
            decisions[call.get("id", "")] = decision
        return decisions

    def _render_request(self, call: dict) -> None:
        name = call.get("name", "?")
        args_text = json.dumps(call.get("args", {}), ensure_ascii=False)
        if len(args_text) > _ARG_PREVIEW_CHARS:
            args_text = args_text[:_ARG_PREVIEW_CHARS] + "…"
        if self._console is None:
            return
        self._console.print()
        self._console.print("────────── ⚠️  HITL 审批请求 ──────────", style="bold yellow")
        self._console.print(f"{call.get('danger_level', '🟡')} {name}", style="bold")
        self._console.print(f"   风险: {call.get('risk', '')}", style="grey50")
        self._console.print(f"   参数: {args_text}", style="grey50")

    def _prompt_until_decision(self, call: dict) -> dict:
        args = call.get("args", {})
        for _attempt in range(_MAX_INVALID_ATTEMPTS):
            try:
                self._say("")
                self._say("请选择操作：[y/Enter] 批准  [a] 全部放行  [n] 拒绝  [s] 跳过  [m] 修改参数")
                choice = input("审批> ")
            except (EOFError, OSError):
                self._say("  [HITL] 输入流不可用，保守处理为拒绝")
                return {"action": "reject", "reason": "输入流不可用"}

            parsed = parse_hitl_choice(choice)
            if parsed is None:
                self._say(f"  ❓ 无法识别的选项：'{choice}'，请输入 y/a/n/s/m 之一（Enter 等价于 y）")
                continue

            if parsed == "approve":
                self._say("  已批准")
                return {"action": "approve"}
            if parsed == "approve_all":
                self._say(f"  已批准，后续 {call.get('name')} 操作将自动通过")
                return {"action": "approve_all"}
            if parsed == "reject":
                reason = self._read_line("  拒绝原因（可直接回车跳过）：")
                return {"action": "reject", "reason": reason}
            if parsed == "skip":
                self._say("  已跳过本次操作")
                return {"action": "skip"}
            if parsed == "modify":
                modified = self._prompt_modified_arguments(args)
                if modified is not None:
                    return modified
                # JSON 非法等：回到主菜单重新提示

        self._say("  [HITL] 连续多次无效输入，保守处理为拒绝")
        return {"action": "reject", "reason": "连续多次无效输入"}

    def _prompt_modified_arguments(self, args: dict):
        self._say(f"  当前参数：{json.dumps(args, ensure_ascii=False)}")
        modified = self._read_line("  请输入修改后的参数（JSON 格式，空行则使用原始参数）：")
        if not modified.strip():
            self._say("  输入为空，改为批准原始参数")
            return {"action": "approve"}
        try:
            parsed = json.loads(modified)
        except json.JSONDecodeError as e:
            self._say(f"  ❌ 修改后的参数不是合法 JSON：{e}")
            return None
        if not isinstance(parsed, dict):
            self._say("  ❌ 修改后的参数必须是 JSON 对象")
            return None
        return {"action": "modify", "args": parsed}

    def _read_line(self, prompt: str) -> str:
        try:
            return input(prompt).strip()
        except (EOFError, OSError):
            return ""

    def _say(self, message: str) -> None:
        if self._console is not None:
            self._console.print(message)


__all__ = ["HitlPrompter", "ApprovalSession"]
