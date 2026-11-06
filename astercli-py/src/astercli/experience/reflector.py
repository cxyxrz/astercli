"""反思器（P9 层一）：触发采集 → 复盘提示词 → 结构化产出 → 落盘。

触发采集（ReflectionCollector）由三处埋点写入：
- 工具层（ToolRegistry）：工具失败 / 超时 / 策略拒绝 / HITL 用户拒绝
- LSP 层（LspManager 回调）：write_file 后的 error 级诊断

反思时机：每轮任务结束后由 CLI 调用 :meth:`Reflector.consume`，主回答已经
流式输出完毕，因此不阻塞用户看到结果；单次复盘是一次独立 LLM 调用，
不进入主对话历史。

护栏：
- ``ASTERCLI_REFLECTION_ENABLED``（默认开）总开关，``/experience off`` 可在会话内关闭
- 每会话最多 ``ASTERCLI_REFLECTION_SESSION_LIMIT`` 次反思（默认 3，按尝试计数）
- 同一触发签名在会话内只反思一次；落盘侧按内容签名合并（hits + 1）
- 单次 LLM 调用超时 ``ASTERCLI_REFLECTION_TIMEOUT_SECONDS``（默认 30s）
- 解析失败 / 置信度 low / lesson 为空 → 直接丢弃，不写盘
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..llm.factory import ProviderRuntime
from .models import (
    DEFAULT_TRIGGER_CATEGORY,
    TRIGGER_HITL_REJECT,
    TRIGGER_LSP_ERROR,
    TRIGGER_TOOL_FAILURE,
    VALID_CATEGORIES,
    ExperienceEntry,
    ReflectionTrigger,
    args_skeleton,
    first_line,
)
from .store import ExperienceStore

log = logging.getLogger(__name__)

MAX_LESSON_CHARS = 200
MAX_ACTION_CHARS = 300
MAX_PENDING_TRIGGERS = 20

REFLECTION_PROMPT = """你是 AsterCLI 的复盘助手，负责把一次失败信号提炼成一条可复用的操作经验。

输入包含：触发类型、工具名、错误摘要、可能的用户拒绝原因与参数骨架。

请只输出 JSON：
{"lesson": "...", "correct_action": "...", "category": "error-lesson|user-correction|workflow", "confidence": "high|medium|low"}

规则：

1. lesson 用一句话点出可避免的根因，不要复述整段报错。
2. correct_action 给出下次可直接照做的具体动作（例如"写文件前先用 glob_files 确认父目录存在"）。
3. 触发类型是 hitl_reject 时，用户的拒绝原因就是最重要信号，lesson 应反映用户偏好或风险认知。
4. 触发类型是 lsp_error 时，lesson 指向语法 / 结构问题，不要编造运行期错误。
5. 不要写入绝对路径、API Key、token 等敏感内容；能泛化就泛化。
6. 信息不足以形成经验时输出 {"lesson": ""}。
7. 只输出 JSON，不要代码块标记，不要其他内容。"""


@dataclass
class ReflectionCollector:
    """收集三类触发信号（单会话内，drain 语义取走并清空）。"""

    _triggers: List[ReflectionTrigger] = field(default_factory=list)

    def record(self, trigger: ReflectionTrigger) -> None:
        if len(self._triggers) >= MAX_PENDING_TRIGGERS:
            return
        self._triggers.append(trigger)

    def tool_failure(self, tool_name: str, error: str, args: object = None) -> None:
        self.record(
            ReflectionTrigger(
                kind=TRIGGER_TOOL_FAILURE,
                tool_name=tool_name,
                error=first_line(error, limit=200),
                args_skeleton=args_skeleton(args),
            )
        )

    def hitl_reject(self, tool_name: str, reason: str, args: object = None) -> None:
        self.record(
            ReflectionTrigger(
                kind=TRIGGER_HITL_REJECT,
                tool_name=tool_name,
                reason=first_line(reason, limit=200),
                args_skeleton=args_skeleton(args),
            )
        )

    def lsp_error(self, diagnostic) -> None:
        """LSP error 级诊断（由 LspManager 回调）；非 error 级直接忽略。"""
        severity = getattr(diagnostic, "severity", None)
        value = getattr(severity, "value", severity)
        if str(value).lower() != "error":
            return
        message = str(getattr(diagnostic, "message", "") or "")
        self.record(
            ReflectionTrigger(
                kind=TRIGGER_LSP_ERROR,
                tool_name="write_file",
                file_path=str(getattr(diagnostic, "file_path", "") or ""),
                line=int(getattr(diagnostic, "line", 0) or 0),
                message=first_line(message, limit=160),
                error=first_line(message, limit=160),
            )
        )

    def drain(self) -> List[ReflectionTrigger]:
        triggers = list(self._triggers)
        self._triggers.clear()
        return triggers

    def pending_count(self) -> int:
        return len(self._triggers)


class Reflector:
    """单会话反思器：end-of-turn 消费触发信号并落盘经验。"""

    def __init__(
        self,
        runtime: ProviderRuntime,
        store: ExperienceStore,
        renderer,
        settings: Settings,
        collector: ReflectionCollector,
        *,
        project_key: str = "",
        enabled: Optional[bool] = None,
    ) -> None:
        self._runtime = runtime
        self._store = store
        self._renderer = renderer
        self._settings = settings
        self._collector = collector
        self._project_key = project_key
        self._enabled = settings.reflection_enabled() if enabled is None else bool(enabled)
        self._session_reflections = 0
        self._seen_signatures: set = set()

    # ---------- 开关 / 会话状态 ----------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._collector.drain()

    def set_runtime(self, runtime: ProviderRuntime) -> None:
        """/model 切换后同步新的客户端（反射用同一 provider）。"""
        self._runtime = runtime

    def reset_session(self) -> None:
        """/clear 时重置会话级去重与次数配额（经验库本身保留）。"""
        self._session_reflections = 0
        self._seen_signatures.clear()
        self._collector.drain()

    def status(self) -> str:
        limit = self._settings.reflection_session_limit()
        state = "开启" if self._enabled else "关闭"
        return f"反思: {state} · 本会话已反思 {self._session_reflections}/{limit} 次"

    # ---------- 消费 ----------

    async def consume(self) -> List[ExperienceEntry]:
        """消费本轮收集到的触发信号，返回本次新建 / 合并的经验条目。"""
        triggers = self._collector.drain()
        if not triggers or not self._enabled:
            return []
        limit = self._settings.reflection_session_limit()
        produced: List[ExperienceEntry] = []
        for trigger in triggers:
            if self._session_reflections >= limit:
                log.debug("反思次数达到会话上限 %s，跳过剩余触发", limit)
                break
            signature = trigger.signature()
            if signature in self._seen_signatures:
                continue
            self._seen_signatures.add(signature)
            self._session_reflections += 1
            try:
                entry = await self._reflect(trigger)
            except Exception:  # noqa: BLE001 —— 反思失败绝不影响主任务
                log.debug("反思调用失败", exc_info=True)
                entry = None
            if entry is None:
                continue
            saved, created = self._store.save(entry)
            if saved is None:
                continue
            if created:
                self._renderer.info(f"🧠 已沉淀经验 [{saved.id}]: {saved.lesson}")
            else:
                self._renderer.info(f"🧠 命中已有经验 [{saved.id}]（hits={saved.hits}）: {saved.lesson}")
            produced.append(saved)
        return produced

    # ---------- 单次反思 ----------

    async def _reflect(self, trigger: ReflectionTrigger) -> Optional[ExperienceEntry]:
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=trigger.describe()),
        ]
        timeout = self._settings.reflection_timeout_seconds()
        response = await asyncio.wait_for(
            self._runtime.client.ainvoke(messages), timeout=timeout
        )
        data = parse_reflection_output(_extract_text(response))
        if not data:
            return None
        lesson = first_line(str(data.get("lesson") or ""), limit=MAX_LESSON_CHARS)
        if not lesson:
            return None
        confidence = str(data.get("confidence") or "").strip().lower()
        if confidence == "low":
            return None
        action = first_line(str(data.get("correct_action") or ""), limit=MAX_ACTION_CHARS)
        category = str(data.get("category") or "").strip()
        if category not in VALID_CATEGORIES:
            category = DEFAULT_TRIGGER_CATEGORY.get(trigger.kind, VALID_CATEGORIES[0])
        # 证据只保留首行并做最小脱敏（避免把绝对路径写进可共享的 git 目录）
        evidence = _sanitize(trigger.evidence(), self._project_key)
        return ExperienceEntry(
            id=f"exp-{uuid.uuid4().hex[:8]}",
            lesson=_sanitize(lesson, self._project_key),
            correct_action=_sanitize(action, self._project_key),
            category=category,
            trigger=trigger.kind,
            evidence=evidence,
            scope="project",
            project_key=self._project_key,
            tool=trigger.tool_name,
        )


# ---------- 纯函数（供测试复用） ----------


def parse_reflection_output(text: str) -> Optional[dict]:
    """解析 LLM 返回的 JSON（容忍 ```json 包裹与前后闲聊）。"""
    if not text or not text.strip():
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _extract_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def _sanitize(text: str, project_key: str = "") -> str:
    """最小脱敏：折叠换行、替换项目根路径占位、限长。"""
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = " ".join(line.strip() for line in value.split("\n") if line.strip())
    if project_key:
        value = value.replace(project_key, "<project>")
        value = value.replace(project_key.replace("\\", "/"), "<project>")
    if len(value) > 320:
        value = value[:320] + "…"
    return value
