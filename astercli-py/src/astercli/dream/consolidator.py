"""Dream 整合器（P10 M2）：素材拼装 → 单次 LLM 结构化输出 → 提案落盘。

刻意做成"一次调用、四阶段在提示词内完成"（归纳事实 → 去重冲突 → 提炼经验 →
输出 ops），而不是多次串行调用：dream 在后台跑，token 与失败面都要可控。

护栏：
- 输入按 ``ASTERCLI_DREAM_MAX_INPUT_TOKENS``（默认 30k）预算裁剪
- 输出要求纯 JSON，ops 再由 :mod:`astercli.dream.proposal` 做代码侧校验
- 单次超时 ``ASTERCLI_DREAM_TIMEOUT_SECONDS``（默认 120s）
- LLM 异常 / 解析失败 / 无新会话 → 返回空 ops 提案，绝不抛出到主流程
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..experience import ExperienceStore, sanitize_text
from ..llm.factory import ProviderRuntime
from ..memory import LongTermMemory, estimate_tokens, is_visible_in_project
from .models import DreamPaths, DreamState
from .proposal import Proposal, normalize_ops, parse_proposal_output
from .session_log import SessionInfo, SessionLog

log = logging.getLogger(__name__)

MAX_SESSIONS = 30
MAX_SESSION_BLOCK_CHARS = 4_000
MAX_SESSION_LINE_CHARS = 600
MAX_SUMMARY_CHARS = 400
MAX_MEMORY_CHARS = 12_000
MAX_EXPERIENCE_CHARS = 8_000

DREAM_PROMPT = """你是 AsterCLI 的后台记忆整理助手（AutoDream）。输入是本项目的长期记忆、经验库和最近若干轮会话转写。请在内部按四个阶段处理，最终只输出一个 JSON：

阶段 1 归纳：从会话里找出跨会话仍然成立、值得长期保留的事实（技术栈、项目约定、用户偏好、环境约束）。
阶段 2 去重与冲突：与现有长期记忆比较；同义条目用 memory_merge 合并，过时或错误的用 memory_update，重复/无价值的用 memory_delete；新事实用 memory_add。
阶段 3 提炼经验：会话里反复出现或明显的失败模式/正确做法，沉淀为 experience_add（最多 3 条）；已过时的经验用 experience_delete。
阶段 4 审慎输出：宁少勿多，没有价值就输出空 ops。

只输出：
{"summary": "一句话说明这次整理做了什么", "ops": [ ... ]}

可用 op 结构：
{"type":"memory_add","scope":"project|global","content":"..."}
{"type":"memory_update","id":"fact-xxxxxxxx","content":"..."}
{"type":"memory_delete","id":"fact-xxxxxxxx"}
{"type":"memory_merge","ids":["fact-a","fact-b"],"scope":"project","content":"合并后的事实"}
{"type":"experience_add","category":"error-lesson|user-correction|workflow","lesson":"...","correct_action":"..."}
{"type":"experience_delete","id":"exp-xxxxxxxx","reason":"..."}

硬性规则：
1. 只能引用输入里出现过的 fact-/exp- id，不能编造 id。
2. 事实内容 ≤400 字、lesson ≤200 字、correct_action ≤300 字，全部单行。
3. 禁止写入 API Key、token、密码、绝对路径；路径用相对路径或 <project> 占位。
4. 不要复述会话原文，不要保留临时任务进度、一次性命令输出。
5. 单次最多 20 条 op；不确定就不写。
6. 只输出 JSON，不要代码块标记，不要解释。"""


@dataclass(frozen=True)
class Material:
    """拼装好的 dream 输入。"""

    text: str
    session_count: int
    memory_count: int
    experience_count: int


class Consolidator:
    """把项目记忆 + 会话转写整理成一份可校验的提案。"""

    def __init__(
        self,
        runtime: ProviderRuntime,
        settings: Settings,
        paths: DreamPaths,
        session_log: SessionLog,
        long_term: LongTermMemory,
        experience: ExperienceStore,
        *,
        project_key: str,
        renderer=None,
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._paths = paths
        self._session_log = session_log
        self._long_term = long_term
        self._experience = experience
        self._project_key = project_key
        self._renderer = renderer

    # ---------- 素材 ----------

    def collect(self, state: DreamState, now: Optional[float] = None) -> Material:
        """按预算拼装输入；会话按"上次 dream 之后有更新"筛选。"""
        memory_entries = [
            entry
            for entry in self._long_term.get_all()
            if is_visible_in_project(entry, self._project_key)
        ]
        experience_entries = self._experience.list_entries()

        memory_lines = ["# 当前长期记忆"]
        for entry in memory_entries:
            scope = entry.metadata.get("scope") or "global"
            memory_lines.append(f"[{entry.id}] ({scope}) {entry.content}")
        memory_block = _clip("\n".join(memory_lines), MAX_MEMORY_CHARS)

        experience_lines = ["# 当前经验库"]
        for entry in experience_entries:
            line = f"[{entry.id}] ({entry.scope}/{entry.category}) {entry.lesson}"
            if entry.correct_action:
                line += f" | 正确做法: {entry.correct_action}"
            experience_lines.append(line)
        experience_block = _clip("\n".join(experience_lines), MAX_EXPERIENCE_CHARS)

        budget = self._settings.dream_max_input_tokens()
        base = f"{memory_block}\n\n{experience_block}"
        base = _trim_to_tokens(base, budget)
        used = estimate_tokens(base)

        candidates = self._session_log.sessions_since(state.last_dream_at)[:MAX_SESSIONS]
        chosen: List[str] = []
        session_count = 0
        for session in candidates:
            block = self._format_session(session)
            if not block:
                continue
            cost = estimate_tokens(block)
            if used + cost > budget:
                break
            chosen.append(block)
            used += cost
            session_count += 1
        chosen.reverse()

        sections = [base]
        if chosen:
            sections.append("# 最近会话（自上次整合以来）\n\n" + "\n\n".join(chosen))
        text = "\n\n".join(sections)
        return Material(
            text=text,
            session_count=session_count,
            memory_count=len(memory_entries),
            experience_count=len(experience_entries),
        )

    def _format_session(self, session: SessionInfo) -> str:
        entries = self._session_log.read_session(session.session_id)
        if not entries:
            return ""
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(session.mtime))
        lines = [f"## session {session.session_id}（{when}）"]
        for entry in entries:
            role = str(entry.get("role") or "")
            if role == "tool":
                name = str(entry.get("name") or "").strip()
                if name:
                    lines.append(f"tool: {name}")
            elif role in ("user", "assistant"):
                text = str(entry.get("text") or "").strip()
                if text:
                    lines.append(f"{role}: {_clip(text, MAX_SESSION_LINE_CHARS)}")
            if sum(len(line) for line in lines) > MAX_SESSION_BLOCK_CHARS:
                lines.append("…（会话过长已截断）")
                break
        return _clip("\n".join(lines), MAX_SESSION_BLOCK_CHARS)

    # ---------- LLM ----------

    async def consolidate(
        self, state: DreamState, dream_id: str, now: Optional[float] = None
    ) -> Proposal:
        """执行一次整合；任何失败都折叠进 ``Proposal.errors``，不向上抛。"""
        material = self.collect(state, now)
        proposal = Proposal(dream_id=dream_id, model=self._runtime.model)

        if material.session_count == 0:
            proposal.errors.append("没有可整合的新会话")
            proposal.save(self._paths.proposal_dir(dream_id))
            return proposal

        messages = [
            SystemMessage(content=DREAM_PROMPT),
            HumanMessage(content=material.text),
        ]
        timeout = self._settings.dream_timeout_seconds()
        try:
            response = await asyncio.wait_for(
                self._runtime.client.ainvoke(messages), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001 —— 后台任务失败不影响主流程
            proposal.errors.append(f"LLM 调用失败: {type(exc).__name__}: {_clip(str(exc), 200)}")
            proposal.save(self._paths.proposal_dir(dream_id))
            return proposal

        text = _extract_text(response)
        proposal.raw = _clip(text, 4_000)
        data = parse_proposal_output(text)
        if data is None:
            proposal.errors.append("LLM 输出不是合法 JSON")
            proposal.save(self._paths.proposal_dir(dream_id))
            return proposal

        proposal.summary = sanitize_text(
            str(data.get("summary") or ""), self._project_key
        )[:MAX_SUMMARY_CHARS]
        memory_ids = {entry.id for entry in self._long_term.get_all()}
        experience_ids = {entry.id for entry in self._experience.list_entries()}
        ops, errors = normalize_ops(
            data.get("ops"),
            memory_ids=memory_ids,
            experience_ids=experience_ids,
            project_key=self._project_key,
        )
        proposal.ops = ops
        proposal.errors.extend(errors)
        if not proposal.save(self._paths.proposal_dir(dream_id)):
            proposal.errors.append("提案落盘失败")
        return proposal


# ---------- 纯函数 ----------


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


def _clip(text: str, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit] + "…"


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    """粗裁剪：按 token 估算比例截断字符，避免超预算输入直接发给模型。"""
    value = text
    while value and estimate_tokens(value) > max_tokens:
        keep = max(1, int(len(value) * 0.8))
        if keep >= len(value):
            break
        value = value[:keep]
    return value + "\n…（素材过长已截断）" if value != text else value
