"""HITL 审批会话与决策（LangGraph interrupt 驱动）。

决策动作（对齐 Java TerminalHitlHandler 的交互选项）：
- ``approve``     批准本次操作
- ``approve_all`` 批准本次，且本会话后续同工具调用自动放行（工具维度）
- ``reject``      拒绝本次（原因回灌给 LLM）
- ``skip``        跳过本步骤
- ``modify``      修改参数后执行

流程：tools 节点执行前先 ``pending_calls`` 找出需审批且未放行的调用，
``interrupt(payload)`` 挂起图 → CLI 层渲染审批框收集用户决策 →
``Command(resume={call_id: decision})`` 恢复 → ``apply_response`` 归一化
后交给 ToolRegistry 按决策执行。

resume 值与 decision 均为 JSON 友好的 dict（interrupt 载荷要进 checkpoint）。
"""

from __future__ import annotations

from typing import Optional

from .policy import danger_level, requires_approval, risk_description

VALID_ACTIONS = ("approve", "approve_all", "reject", "skip", "modify")


def parse_hitl_choice(text: str) -> Optional[str]:
    """解析用户键入的审批选项（纯函数，供终端层与测试复用）。

    - 空串 / ``y`` → approve；``a`` → approve_all；``n`` → reject；
      ``s`` → skip；``m`` → modify
    - 无法识别返回 None，由调用方重新提示（fail-safe，不默认放行）
    """
    normalized = (text or "").strip().lower()
    if normalized in ("", "y", "yes"):
        return "approve"
    if normalized == "a":
        return "approve_all"
    if normalized == "n":
        return "reject"
    if normalized == "s":
        return "skip"
    if normalized == "m":
        return "modify"
    return None


def normalize_decision(raw: object) -> dict:
    """把 resume 值里的单条决策归一化为标准 dict；非法输入保守拒绝。"""
    if not isinstance(raw, dict):
        return {"action": "reject", "reason": "未收到有效审批"}
    action = raw.get("action")
    if action == "approve" or action == "approve_all":
        return {"action": "approve"}
    if action == "reject":
        reason = raw.get("reason")
        return {"action": "reject", "reason": reason if isinstance(reason, str) else ""}
    if action == "skip":
        return {"action": "skip"}
    if action == "modify":
        args = raw.get("args")
        if not isinstance(args, dict):
            return {"action": "reject", "reason": "修改后的参数不是合法 JSON 对象"}
        return {"action": "modify", "args": args}
    return {"action": "reject", "reason": "未收到有效审批"}


class ApprovalSession:
    """会话级审批状态：持有"全部放行"名单 + pending 计算 + resume 归一化。

    放行名单在 interrupt 返回之后的同一轮执行里才写入（apply_response），
    因此节点挂起重入（resume 重跑 tools 节点）时 pending 计算保持确定性，
    不会因为提前写入而丢掉 resume 值。
    """

    def __init__(self) -> None:
        self._approved_all_by_tool: set = set()

    # ---------- 会话状态 ----------

    def clear_approved_all(self) -> None:
        self._approved_all_by_tool.clear()

    def is_approved_all(self, tool_name: Optional[str]) -> bool:
        return bool(tool_name) and tool_name in self._approved_all_by_tool

    # ---------- pending 计算（tools 节点挂起前） ----------

    def pending_calls(self, tool_calls: list, registry) -> list:
        """筛出需要审批且未在会话放行名单内的调用。"""
        pending: list = []
        for call in tool_calls:
            name = call.get("name", "")
            if self.is_approved_all(name):
                continue
            tool_def = registry.get(name) if registry is not None else None
            if requires_approval(name, tool_def):
                pending.append(call)
        return pending

    def build_payload(self, pending: list) -> dict:
        """interrupt 载荷：列出待审批调用 + 风险信息。"""
        calls = []
        for call in pending:
            name = call.get("name", "")
            calls.append(
                {
                    "id": call.get("id", ""),
                    "name": name,
                    "args": call.get("args", {}),
                    "danger_level": danger_level(name),
                    "risk": risk_description(name),
                }
            )
        return {"type": "hitl_approval", "calls": calls}

    # ---------- resume 归一化（interrupt 返回后） ----------

    def apply_response(self, pending: list, response: object) -> dict:
        """把 resume 值映射为 ``{call_id: decision}``。

        - ``approve_all`` 同时把工具名写入会话放行名单（工具维度）
        - 缺失 / 非法决策保守拒绝（fail-safe）
        """
        decisions: dict = {}
        for call in pending:
            call_id = call.get("id", "")
            raw = response.get(call_id) if isinstance(response, dict) else None
            decision = normalize_decision(raw)
            if raw is not None and isinstance(raw, dict) and raw.get("action") == "approve_all":
                self._approved_all_by_tool.add(call.get("name", ""))
            decisions[call_id] = decision
        return decisions

    @staticmethod
    def reject_all(pending: list, reason: str = "无审批处理器，保守拒绝") -> dict:
        """无交互处理器时的兜底：全部拒绝。"""
        return {call.get("id", ""): {"action": "reject", "reason": reason} for call in pending}
