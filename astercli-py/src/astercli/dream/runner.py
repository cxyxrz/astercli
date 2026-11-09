"""AutoDream 运行器（P10 M1 骨架）：状态读写 + PID 锁。

守护线程、LLM 整合与事务 apply 在后续里程碑追加；本文件先把两件事做稳：
1. 谁能开始一次 dream（跨进程 PID 锁，1 小时过期自动抢占）
2. 上次做了什么（state.json 的读取 / 原子写回）

锁文件内容 ``{"pid": ..., "created_at": ...}``。进程崩溃留下的锁在 TTL 后
或持有进程已消失时自动回收，避免永久卡死后台整合。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ..config import Settings
from .applier import Applier
from .consolidator import Consolidator
from .gates import GateResult, evaluate
from .models import DreamPaths, DreamRecord, DreamState
from .session_log import SessionLog
from .snapshot import Snapshot

LOCK_TTL_SECONDS = 3600


def pid_alive(pid: int) -> bool:
    """进程是否存活（无权限视为存活，避免误抢他人锁）。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class DreamLock:
    """跨进程互斥锁：同一项目同一时间只允许一个 dream 线程。"""

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: int = LOCK_TTL_SECONDS,
        pid: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.pid = os.getpid() if pid is None else int(pid)
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self, now: Optional[float] = None) -> bool:
        """尝试获取锁；已有存活持有者且未过期时返回 False。"""
        current = time.time() if now is None else float(now)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False

        if self.path.exists():
            holder = self.read_holder()
            if holder and not self._is_stale(holder, current):
                return False
            try:
                self.path.unlink()
            except OSError:
                return False

        payload = json.dumps({"pid": self.pid, "created_at": current})
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            # 与其它进程竞争失败：让对方执行
            return False
        except OSError:
            return False
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        self._held = True
        return True

    def release(self) -> None:
        """只删除自己持有的锁，避免误删抢占者刚建立的锁。"""
        if not self._held:
            return
        holder = self.read_holder()
        if holder and holder.get("pid") != self.pid:
            self._held = False
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self._held = False

    def read_holder(self) -> Optional[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def _is_stale(self, holder: dict, now: float) -> bool:
        try:
            created = float(holder.get("created_at") or 0)
            holder_pid = int(holder.get("pid") or 0)
        except (TypeError, ValueError):
            return True  # 内容损坏：视为过期
        if created <= 0 or now - created > self.ttl_seconds:
            return True
        return not pid_alive(holder_pid)

    def __enter__(self) -> "DreamLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class DreamRunner:
    """启动期门判定 + 状态读写的轻量门面（M4 在其上挂守护线程）。"""

    def __init__(
        self,
        project_key: str,
        settings: Settings,
        *,
        paths: Optional[DreamPaths] = None,
        session_log: Optional[SessionLog] = None,
    ) -> None:
        self.settings = settings
        if paths is None:
            raw_dream = settings.get("ASTERCLI_DREAM_DIR")
            raw_session = settings.get("ASTERCLI_SESSION_LOG_DIR")
            paths = DreamPaths.for_project(
                project_key,
                dream_root=Path(raw_dream) if raw_dream else None,
                session_root=Path(raw_session) if raw_session else None,
            )
        self.paths = paths
        self.session_log = session_log or SessionLog(project_key, settings)

    # ---------- 状态 ----------

    def load_state(self) -> DreamState:
        return DreamState.load(self.paths.state_file)

    def save_state(self, state: DreamState) -> bool:
        return state.save(self.paths.state_file)

    def record_result(self, record: DreamRecord) -> DreamState:
        """梦境成功应用后落状态（时间门 / 新会话门的基准时间）。"""
        state = self.load_state()
        state.mark_dream(record)
        self.save_state(state)
        return state

    # ---------- 门 ----------

    def check(self, now: Optional[float] = None) -> GateResult:
        """前四道门的合并判定（不含 PID 锁）。"""
        current = time.time() if now is None else float(now)
        state = self.load_state()
        new_sessions = len(self.session_log.sessions_since(state.last_dream_at))
        return evaluate(
            state,
            current,
            enabled=self.settings.dream_enabled(),
            min_hours=self.settings.dream_min_hours(),
            min_sessions=self.settings.dream_min_sessions(),
            scan_interval_minutes=self.settings.dream_scan_interval_minutes(),
            new_sessions=new_sessions,
        )

    def lock(self) -> DreamLock:
        return DreamLock(self.paths.lock_file)

    def mark_scan(self, now: Optional[float] = None) -> DreamState:
        state = self.load_state()
        state.mark_scan(now)
        self.save_state(state)
        return state

    # ---------- 撤销 ----------

    def find_record(self, dream_id: str = "") -> Optional[DreamRecord]:
        """按 id 查找记录；空 id 表示最近一次**成功应用**的 dream。"""
        state = self.load_state()
        dream_id = (dream_id or "").strip()
        if dream_id:
            for record in state.history:
                if record.dream_id == dream_id:
                    return record
            return None
        for record in state.history:
            if record.status == "applied":
                return record
        return None

    def undo(self, dream_id: str = "") -> Tuple[bool, str]:
        """按快照还原上一次 dream；成功后回退状态时间戳并移除该记录。"""
        state = self.load_state()
        record = self.find_record(dream_id)
        if record is None:
            return False, f"未找到可撤销的 dream 记录: {dream_id or '（暂无成功记录）'}"
        snapshot = Snapshot.load(self.paths.backup_dir(record.dream_id), record.dream_id)
        if snapshot is None:
            return False, f"未找到快照: {record.dream_id}（无法安全撤销）"

        restored = snapshot.restore()
        state.history = [item for item in state.history if item.dream_id != record.dream_id]
        state.dream_count = max(0, state.dream_count - 1)
        state.last_dream_id = state.history[0].dream_id if state.history else ""
        state.last_dream_at = state.history[0].finished_at if state.history else 0.0
        self.save_state(state)
        return True, f"已撤销 dream {record.dream_id}（还原 {restored} 个文件）"


class DreamWorker:
    """启动期后台整合：主线程只做廉价门检查，LLM 与写盘都在守护线程。

    线程内自建事件循环（``asyncio.run``），因为 CLI 每轮任务结束后主事件循环
    会被销毁。完成通知不能直接从后台线程打印到 prompt_toolkit 输入区，统一进
    ``queue.Queue``，由 REPL 在下一次 ``session.prompt`` 前 flush。
    """

    def __init__(
        self,
        runner: DreamRunner,
        settings: Settings,
        session_log: SessionLog,
        long_term,
        experience,
        runtime_provider: Callable[[], object],
        *,
        project_key: str,
        memory_lock=None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.runner = runner
        self.settings = settings
        self.session_log = session_log
        self.long_term = long_term
        self.experience = experience
        self.project_key = project_key
        self.memory_lock = memory_lock
        self._runtime_provider = runtime_provider
        self._enabled = settings.dream_enabled() if enabled is None else bool(enabled)
        self._notices: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._dirty = False

    # ---------- 状态 ----------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def dirty(self) -> bool:
        """后台是否改写过记忆库（REPL 据此重建 system prompt 索引段）。"""
        return self._dirty

    def clear_dirty(self) -> None:
        self._dirty = False

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def drain_notices(self) -> List[str]:
        notices: List[str] = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except queue.Empty:
                break
        return notices

    def _notify(self, message: str) -> None:
        self._notices.put(message)

    # ---------- 启动 ----------

    def start(self) -> Optional[str]:
        """门检查 → 抢锁 → 起守护线程；返回跳过原因（None 表示已启动）。"""
        if not self._enabled:
            return "AutoDream 已关闭"
        if self.busy:
            return "AutoDream 正在运行"
        gate = self.runner.check()
        if not gate.allowed:
            return gate.detail
        lock = self.runner.lock()
        if not lock.acquire():
            return "另一个进程正在整合"
        self.runner.mark_scan()
        self._thread = threading.Thread(
            target=self._thread_main, args=(lock,), name="astercli-dream", daemon=True
        )
        self._thread.start()
        return None

    def wait(self, timeout: float = 5.0) -> bool:
        """等待后台线程结束（测试与退出前使用）；返回是否已结束。"""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _thread_main(self, lock: DreamLock) -> None:
        try:
            record = asyncio.run(self._execute())
            if record.status == "applied":
                self._dirty = True
                self._notify(
                    f"🧠 AutoDream 已整理记忆：{record.ops} 条操作 · {record.summary}"
                    f"（/dream log {record.dream_id} 查看，/dream undo 撤销）"
                )
            elif record.status == "failed":
                self._notify(f"⚠️ AutoDream 未完成：{record.summary}")
        except Exception as exc:  # noqa: BLE001 —— 后台失败只通知，不影响主流程
            self._notify(f"⚠️ AutoDream 失败: {type(exc).__name__}: {exc}")
        finally:
            lock.release()

    # ---------- 手动执行 ----------

    async def run_now(self) -> str:
        """`/dream` 立即整合（忽略时间 / 会话门，但仍需独占锁）。"""
        if self.busy:
            return "AutoDream 正在后台运行，请稍后再试。"
        lock = self.runner.lock()
        if not lock.acquire():
            return "另一个进程正在整合，请稍后再试。"
        self.runner.mark_scan()
        try:
            record = await self._execute()
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ AutoDream 执行失败: {type(exc).__name__}: {exc}"
        finally:
            lock.release()
        if record.status == "applied":
            self._dirty = True
            return (
                f"🧠 已整理记忆：{record.ops} 条操作 · {record.summary}"
                f"（/dream log {record.dream_id} 查看，/dream undo 撤销）"
            )
        if record.status == "skipped":
            return f"🧠 本次没有值得整合的内容（{record.summary}）"
        return f"⚠️ AutoDream 未完成：{record.summary}"

    # ---------- 核心 ----------

    async def _execute(self) -> DreamRecord:
        started = time.time()
        dream_id = f"dream-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        state = self.runner.load_state()
        consolidator = Consolidator(
            self._runtime_provider(),
            self.settings,
            self.runner.paths,
            self.session_log,
            self.long_term,
            self.experience,
            project_key=self.project_key,
        )
        proposal = await consolidator.consolidate(state, dream_id)
        record = DreamRecord(
            dream_id=dream_id, started_at=started, status="failed", summary=""
        )

        if proposal.ops:
            result = Applier(
                self.long_term,
                self.experience,
                self.runner.paths,
                project_key=self.project_key,
                lock=self.memory_lock,
            ).apply(proposal)
            if result.rolled_back:
                record.summary = "；".join(result.errors)[:200] or "应用失败已回滚"
            elif result.ok:
                record.status = "applied"
                record.ops = result.applied
                record.summary = proposal.summary or f"{result.applied} 条操作"
            else:
                record.summary = "；".join(result.errors)[:200] or "没有可应用的操作"
        else:
            reasons = proposal.errors or ["没有可整合的内容"]
            record.status = "skipped" if not proposal.errors else "failed"
            record.summary = "；".join(reasons)[:200]

        self.runner.record_result(record)
        return record
