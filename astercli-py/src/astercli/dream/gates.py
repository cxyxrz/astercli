"""AutoDream 五道门（P10 M1）：启动期廉价前置检查。

顺序（对齐方案）：开关 → 时间门 → 扫描节流 → 新会话门 → PID 锁。

前四道在这里做纯函数判定，只依赖 :class:`DreamState` 与外部传入的计数；
第五道是文件系统 + 进程状态，落在 :class:`astercli.dream.runner.DreamLock`。
门的作用是"不做无意义扫描、不重复烧 LLM token"，任何一道不通过都静默退出。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import DreamState

GATE_OK = "ok"
GATE_DISABLED = "disabled"
GATE_TOO_SOON = "too_soon"
GATE_SCAN_THROTTLED = "scan_throttled"
GATE_NOT_ENOUGH_SESSIONS = "not_enough_sessions"

GATE_LABELS = {
    GATE_OK: "通过",
    GATE_DISABLED: "未开启",
    GATE_TOO_SOON: "距上次整合不足",
    GATE_SCAN_THROTTLED: "扫描节流中",
    GATE_NOT_ENOUGH_SESSIONS: "新增会话不足",
}


@dataclass(frozen=True)
class GateResult:
    """门判定结果：``reason`` 为机器码，``detail`` 给人看。"""

    allowed: bool
    reason: str
    detail: str = ""

    @property
    def label(self) -> str:
        return GATE_LABELS.get(self.reason, self.reason)


def evaluate(
    state: DreamState,
    now: float,
    *,
    enabled: bool = True,
    min_hours: float = 24.0,
    min_sessions: int = 5,
    scan_interval_minutes: float = 10.0,
    new_sessions: int = 0,
) -> GateResult:
    """判定本次启动是否允许开始一次 dream（不含 PID 锁）。"""
    if not enabled:
        return GateResult(False, GATE_DISABLED, "ASTERCLI_DREAM_ENABLED=false")

    # 时间门：上次成功整合距今不足 min_hours → 跳过
    if state.last_dream_at > 0:
        elapsed_hours = max(0.0, (now - state.last_dream_at) / 3600.0)
        if elapsed_hours < min_hours:
            return GateResult(
                False,
                GATE_TOO_SOON,
                f"距上次整合 {elapsed_hours:.1f}h < {min_hours:g}h",
            )

    # 扫描节流：避免每次启动都重新扫会话目录
    if state.last_scan_at > 0:
        elapsed_minutes = max(0.0, (now - state.last_scan_at) / 60.0)
        if elapsed_minutes < scan_interval_minutes:
            return GateResult(
                False,
                GATE_SCAN_THROTTLED,
                f"距上次扫描 {elapsed_minutes:.1f}min < {scan_interval_minutes:g}min",
            )

    # 新会话门：上次整合之后产生的其它会话数不足 → 没有新素材
    if new_sessions < min_sessions:
        return GateResult(
            False,
            GATE_NOT_ENOUGH_SESSIONS,
            f"新增会话 {new_sessions} 个 < {min_sessions} 个",
        )

    return GateResult(True, GATE_OK, f"新增会话 {new_sessions} 个，可以整合")
