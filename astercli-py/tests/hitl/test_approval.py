"""HITL 审批策略与会话测试：危险识别 / 决策解析 / pending 筛选 / resume 归一化。

对齐 Java ApprovalPolicy + TerminalHitlHandler 的语义（fail-safe：非法输入
保守拒绝，绝不默认放行）。
"""

from __future__ import annotations

from types import SimpleNamespace

from astercli.hitl import ApprovalSession, HitlPrompter
from astercli.hitl.approval import normalize_decision, parse_hitl_choice
from astercli.hitl.policy import (
    danger_level,
    is_mcp_tool,
    mcp_server_name,
    requires_approval,
    risk_description,
)


# ---------- 危险识别策略 ----------


def test_static_dangerous_tools_require_approval():
    for name in ("write_file", "execute_command", "create_project", "revert_turn"):
        assert requires_approval(name) is True


def test_read_only_tools_do_not_require_approval():
    for name in ("read_file", "list_dir", "glob_files", "grep_code", "search_code"):
        assert requires_approval(name) is False


def test_registered_dangerous_flag_requires_approval():
    class ToolDef:
        dangerous = True

    assert requires_approval("custom_tool", ToolDef()) is True
    assert requires_approval("custom_tool", None) is False


def test_mcp_tools_always_require_approval():
    assert is_mcp_tool("mcp__github__create_issue") is True
    assert is_mcp_tool("read_file") is False
    assert requires_approval("mcp__github__create_issue") is True


def test_empty_tool_name_never_requires_approval():
    assert requires_approval(None) is False
    assert requires_approval("") is False


def test_danger_level_and_risk_description():
    assert "高危" in danger_level("execute_command")
    assert "高危" in danger_level("revert_turn")
    assert "中危" in danger_level("write_file")
    assert "MCP" in danger_level("mcp__s__t")
    assert "安全" in danger_level("read_file")
    assert "Shell" in risk_description("execute_command")
    assert "写入或覆盖" in risk_description("write_file")
    assert "只读" in risk_description("read_file")


def test_mcp_server_name_extraction():
    assert mcp_server_name("mcp__github__create_issue") == "github"
    assert mcp_server_name("read_file") is None


# ---------- 审批选项解析（parse_hitl_choice） ----------


def test_parse_hitl_choice_variants():
    assert parse_hitl_choice("") == "approve"
    assert parse_hitl_choice("y") == "approve"
    assert parse_hitl_choice(" yes ") == "approve"
    assert parse_hitl_choice("Y") == "approve"
    assert parse_hitl_choice("a") == "approve_all"
    assert parse_hitl_choice("A") == "approve_all"
    assert parse_hitl_choice("n") == "reject"
    assert parse_hitl_choice("s") == "skip"
    assert parse_hitl_choice("m") == "modify"


def test_parse_hitl_choice_invalid_returns_none():
    assert parse_hitl_choice("x") is None
    assert parse_hitl_choice("approve") is None  # 不接受完整单词，只要单字母
    # None 与空串同语义（等价 Enter → 批准）；EOF 由终端层兜底拒绝
    assert parse_hitl_choice(None) == "approve"


# ---------- resume 值归一化（normalize_decision） ----------


def test_normalize_decision_valid_actions():
    assert normalize_decision({"action": "approve"}) == {"action": "approve"}
    assert normalize_decision({"action": "approve_all"}) == {"action": "approve"}
    assert normalize_decision({"action": "skip"}) == {"action": "skip"}
    assert normalize_decision({"action": "reject", "reason": "太危险"}) == {
        "action": "reject",
        "reason": "太危险",
    }


def test_normalize_decision_modify_requires_dict_args():
    assert normalize_decision({"action": "modify", "args": {"path": "a"}}) == {
        "action": "modify",
        "args": {"path": "a"},
    }
    # args 非 dict → 保守拒绝
    rejected = normalize_decision({"action": "modify", "args": ["a"]})
    assert rejected["action"] == "reject"


def test_normalize_decision_invalid_input_rejects():
    assert normalize_decision(None)["action"] == "reject"
    assert normalize_decision("yes")["action"] == "reject"
    assert normalize_decision({"action": "destroy"})["action"] == "reject"
    # reject 的 reason 非 str → 空字符串
    assert normalize_decision({"action": "reject", "reason": 42}) == {
        "action": "reject",
        "reason": "",
    }


# ---------- ApprovalSession ----------


def make_registry_stub(dangerous_names=frozenset({"write_file"})):
    class Registry:
        def get(self, name):
            if name in dangerous_names:
                return SimpleNamespace(dangerous=True)
            return SimpleNamespace(dangerous=False)

    return Registry()


def test_pending_calls_filters_dangerous_and_approved_all():
    session = ApprovalSession()
    registry = make_registry_stub()
    calls = [
        {"name": "read_file", "args": {}, "id": "c1"},
        {"name": "write_file", "args": {}, "id": "c2"},
        {"name": "mcp__x__y", "args": {}, "id": "c3"},
    ]
    pending = session.pending_calls(calls, registry)
    assert [c["id"] for c in pending] == ["c2", "c3"]

    # 会话放行名单内的工具不再 pending
    session._approved_all_by_tool.add("write_file")
    pending = session.pending_calls(calls, registry)
    assert [c["id"] for c in pending] == ["c3"]


def test_build_payload_carries_risk_info():
    session = ApprovalSession()
    payload = session.build_payload([{"name": "write_file", "args": {"path": "a"}, "id": "c1"}])
    assert payload["type"] == "hitl_approval"
    call = payload["calls"][0]
    assert call["id"] == "c1"
    assert call["name"] == "write_file"
    assert call["danger_level"]
    assert call["risk"]


def test_apply_response_maps_decisions_and_approve_all():
    session = ApprovalSession()
    pending = [
        {"name": "write_file", "args": {}, "id": "c1"},
        {"name": "execute_command", "args": {}, "id": "c2"},
    ]
    decisions = session.apply_response(
        pending,
        {"c1": {"action": "approve_all"}, "c2": {"action": "reject", "reason": "no"}},
    )
    assert decisions["c1"] == {"action": "approve"}
    assert decisions["c2"] == {"action": "reject", "reason": "no"}
    # approve_all 写入会话放行名单（工具维度）
    assert session.is_approved_all("write_file")
    assert not session.is_approved_all("execute_command")


def test_apply_response_missing_decision_rejects():
    session = ApprovalSession()
    pending = [{"name": "write_file", "args": {}, "id": "c1"}]
    decisions = session.apply_response(pending, {})  # resume 值缺失该调用
    assert decisions["c1"]["action"] == "reject"


def test_reject_all_static_fallback():
    pending = [{"name": "write_file", "args": {}, "id": "c1"}]
    decisions = ApprovalSession.reject_all(pending)
    assert decisions == {"c1": {"action": "reject", "reason": "无审批处理器，保守拒绝"}}


def test_clear_approved_all_resets_session():
    session = ApprovalSession()
    session._approved_all_by_tool.add("write_file")
    session.clear_approved_all()
    assert not session.is_approved_all("write_file")


# ---------- HitlPrompter（monkeypatch input 的交互序列） ----------


def fake_interrupt(calls):
    return [SimpleNamespace(value={"type": "hitl_approval", "calls": calls})]


async def test_prompter_collect_approve(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    prompter = HitlPrompter(console=None)
    decisions = await prompter.collect(
        fake_interrupt([{"id": "c1", "name": "write_file", "args": {"path": "a"}, "danger_level": "🟡", "risk": "r"}])
    )
    assert decisions == {"c1": {"action": "approve"}}


async def test_prompter_collect_reject_reads_reason(monkeypatch):
    inputs = iter(["n", "太危险"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    prompter = HitlPrompter(console=None)
    decisions = await prompter.collect(
        fake_interrupt([{"id": "c1", "name": "write_file", "args": {}, "danger_level": "🟡", "risk": "r"}])
    )
    assert decisions == {"c1": {"action": "reject", "reason": "太危险"}}


async def test_prompter_collect_modify_with_json(monkeypatch):
    inputs = iter(["m", '{"path": "b"}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    prompter = HitlPrompter(console=None)
    decisions = await prompter.collect(
        fake_interrupt([{"id": "c1", "name": "write_file", "args": {"path": "a"}, "danger_level": "🟡", "risk": "r"}])
    )
    assert decisions == {"c1": {"action": "modify", "args": {"path": "b"}}}


async def test_prompter_collect_invalid_then_approve(monkeypatch):
    inputs = iter(["x", "??", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    prompter = HitlPrompter(console=None)
    decisions = await prompter.collect(
        fake_interrupt([{"id": "c1", "name": "write_file", "args": {}, "danger_level": "🟡", "risk": "r"}])
    )
    assert decisions == {"c1": {"action": "approve"}}


async def test_prompter_collect_eof_rejects(monkeypatch):
    def boom(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    prompter = HitlPrompter(console=None)
    decisions = await prompter.collect(
        fake_interrupt([{"id": "c1", "name": "write_file", "args": {}, "danger_level": "🟡", "risk": "r"}])
    )
    assert decisions["c1"]["action"] == "reject"


async def test_prompter_collect_unknown_payload_rejects_all():
    async def noop(_):
        raise AssertionError("不应进入交互")

    prompter = HitlPrompter(console=None)
    # 无法识别的挂起类型：整体空决策（registry 层按缺失决策保守拒绝）
    decisions = await prompter.collect([SimpleNamespace(value={"type": "other"})])
    assert decisions == {}
