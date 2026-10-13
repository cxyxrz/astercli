"""计划审阅输入解析测试（parse_supplement_line，Java PlanReviewInputParser 直译）。"""

from __future__ import annotations

from astercli.cli.plan_review import PlanReviewDecision, parse_supplement_line


def test_empty_or_confirm_words_execute():
    for text in ("", "   ", "y", "Y", "yes", "YES", "run", "/run"):
        assert parse_supplement_line(text).action == "execute"


def test_bare_escape_cancels():
    assert parse_supplement_line("\u001B").action == "cancel"


def test_cancel_words_cancel():
    for text in ("cancel", "esc", "/cancel"):
        assert parse_supplement_line(text).action == "cancel"


def test_other_text_becomes_supplement():
    decision = parse_supplement_line("请把测试覆盖率提到 80%")
    assert decision.action == "supplement"
    assert decision.feedback == "请把测试覆盖率提到 80%"


def test_none_input_treated_as_supplement_empty():
    # None（input() 不可用时理论上不出现）→ 空串语义 → execute
    assert parse_supplement_line(None).action == "execute"


def test_decision_factories():
    assert PlanReviewDecision.execute().action == "execute"
    assert PlanReviewDecision.cancel().action == "cancel"
    d = PlanReviewDecision.supplement("加一步")
    assert d.action == "supplement"
    assert d.feedback == "加一步"
