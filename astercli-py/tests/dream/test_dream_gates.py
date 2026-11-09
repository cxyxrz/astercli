"""AutoDream 五道门测试：开关 / 时间 / 节流 / 新会话数 / 判定顺序。"""

from __future__ import annotations

from astercli.dream import (
    GATE_DISABLED,
    GATE_NOT_ENOUGH_SESSIONS,
    GATE_OK,
    GATE_SCAN_THROTTLED,
    GATE_TOO_SOON,
    DreamState,
    evaluate,
)

NOW = 1_800_000_000.0
DAY = 86_400.0
HOUR = 3_600.0
MINUTE = 60.0


def test_五道门全部通过():
    state = DreamState(last_dream_at=NOW - 25 * HOUR, last_scan_at=NOW - 20 * MINUTE)
    result = evaluate(state, NOW, new_sessions=5)
    assert result.allowed is True
    assert result.reason == GATE_OK


def test_开关关闭优先于其它门():
    state = DreamState(last_dream_at=NOW - 100 * HOUR)
    result = evaluate(state, NOW, enabled=False, new_sessions=99)
    assert result.allowed is False
    assert result.reason == GATE_DISABLED


def test_时间门不足24小时():
    state = DreamState(last_dream_at=NOW - 2 * HOUR)
    result = evaluate(state, NOW, new_sessions=99)
    assert result.reason == GATE_TOO_SOON


def test_从未整合过不受时间门限制():
    result = evaluate(DreamState(), NOW, new_sessions=5)
    assert result.allowed is True


def test_扫描节流():
    state = DreamState(last_dream_at=NOW - 30 * HOUR, last_scan_at=NOW - 3 * MINUTE)
    result = evaluate(state, NOW, new_sessions=99)
    assert result.reason == GATE_SCAN_THROTTLED


def test_扫描节流到期后放行():
    state = DreamState(last_dream_at=NOW - 30 * HOUR, last_scan_at=NOW - 11 * MINUTE)
    result = evaluate(state, NOW, new_sessions=5)
    assert result.allowed is True


def test_新会话不足():
    state = DreamState(last_dream_at=NOW - 30 * HOUR, last_scan_at=NOW - 30 * MINUTE)
    result = evaluate(state, NOW, new_sessions=4)
    assert result.reason == GATE_NOT_ENOUGH_SESSIONS
    assert "4" in result.detail


def test_可配置阈值():
    state = DreamState(last_dream_at=NOW - 1 * HOUR, last_scan_at=NOW - 60 * MINUTE)
    result = evaluate(
        state,
        NOW,
        min_hours=0.5,
        min_sessions=1,
        scan_interval_minutes=5,
        new_sessions=1,
    )
    assert result.allowed is True


def test_门顺序为时间先于节流():
    """同时触发时间门与节流门时，应先报告时间门（与方案顺序一致）。"""
    state = DreamState(last_dream_at=NOW - 1 * HOUR, last_scan_at=NOW - 1 * MINUTE)
    result = evaluate(state, NOW, new_sessions=0)
    assert result.reason == GATE_TOO_SOON
