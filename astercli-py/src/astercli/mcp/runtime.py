"""MCP runtime：后台线程 event loop + ClientSessionGroup 会话池。

为什么需要后台 loop（Python 版特有的架构适配）：
- REPL 主循环每轮任务用独立的 ``asyncio.run()`` 新建事件循环，而 MCP stdio
  子进程 / HTTP 会话必须跨轮存活——官方 SDK 的 async context 绑定创建它的
  loop。因此所有 MCP 会话住在一个专属后台线程的常驻 loop 里。
- 主 loop 通过 ``submit()`` 提交协程（run_coroutine_threadsafe），工具
  ``func`` 为 async 时用 ``call_async()``（wrap_future 桥接，不阻塞主 loop）；
  同步上下文（mention 展开）用 ``call_sync()``。

任务边界（anyio 约束，实测踩坑）：
- ``ClientSessionGroup.connect_to_server`` 会进入 stdio/HTTP transport 的
  anyio task group + cancel scope；anyio 强制 scope 进入与退出必须发生在
  **同一个 task**。因此 connect/disconnect 这类生命周期调用不能走
  ``run_coroutine_threadsafe``（那会创建临时 task，退出时在 ``_main`` 里
  aclose 触发 "Attempted to exit cancel scope in a different task"）。
- 生命周期调用统一经 :meth:`submit_lifecycle` 投递到队列，由常驻 ``_main``
  任务顺序执行；session 级调用（call_tool / list_resources / read_resource）
  不触碰 transport scope，走 :meth:`submit` 独立 task，保持工具并发能力。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Awaitable, Callable


class McpRuntimeError(Exception):
    """MCP runtime 未启动 / 已关闭。"""


class McpRuntime:
    """后台线程 + 常驻 event loop；协程工厂统一以 ``group`` 为参数提交。"""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stop: asyncio.Event | None = None
        self._jobs: asyncio.Queue | None = None   # 生命周期任务队列（_main 消费）
        self._group = None  # mcp.ClientSessionGroup，仅后台 loop 内可操作
        self._closed = False

    # ---------- 生命周期 ----------

    def start(self, *, startup_timeout: float = 10.0) -> bool:
        """启动后台线程与 ClientSessionGroup；失败返回 False（调用方降级为无 MCP）。"""
        if self._thread is not None:
            return self.is_running()
        self._thread = threading.Thread(target=self._run, name="astercli-mcp", daemon=True)
        self._thread.start()
        return self._started.wait(startup_timeout)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001 —— 关闭失败不影响主进程退出
                pass

    async def _main(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSessionGroup

        self._stop = asyncio.Event()
        self._jobs = asyncio.Queue()
        async with AsyncExitStack() as stack:
            group = await stack.enter_async_context(ClientSessionGroup())
            self._group = group
            self._started.set()
            while True:
                job = await self._jobs.get()
                if job is None:  # shutdown 哨兵：退出 async with（aclose 同任务）
                    break
                coro_fn, fut = job
                try:
                    fut.set_result(await coro_fn(group))
                except BaseException as exc:  # noqa: BLE001 —— future 携带异常
                    fut.set_exception(exc)
        self._group = None

    def shutdown(self, timeout: float = 5.0) -> None:
        """请求后台 loop 退出并回收线程；幂等。"""
        if self._closed or self._thread is None:
            return
        self._closed = True
        loop, jobs, stop = self._loop, self._jobs, self._stop
        if loop is not None:
            def _signal() -> None:
                if jobs is not None:
                    jobs.put_nowait(None)  # 正常路径：_main 走到 aclose
                elif stop is not None:
                    stop.set()  # 兜底：queue 尚未建立（极早期 shutdown）
            try:
                loop.call_soon_threadsafe(_signal)
            except RuntimeError:  # loop 已停
                pass
        self._thread.join(timeout)
        self._group = None

    def is_running(self) -> bool:
        return self._group is not None and not self._closed

    # ---------- 提交调用 ----------

    def submit(self, coro_fn: Callable[[object], Awaitable]) -> concurrent.futures.Future:
        """session 级调用：新 task 执行（不触碰 transport scope，可并发）。"""
        if not self.is_running():
            raise McpRuntimeError("MCP runtime 未启动或已关闭")
        return asyncio.run_coroutine_threadsafe(coro_fn(self._group), self._loop)

    def submit_lifecycle(self, coro_fn: Callable[[object], Awaitable]) -> concurrent.futures.Future:
        """生命周期调用（connect / disconnect）：投递到常驻 ``_main`` 任务执行。

        anyio 的 cancel scope 要求进出同 task——connect 必须在 _main 里跑，
        才能在 shutdown aclose 时同 task 退出。
        """
        if not self.is_running():
            raise McpRuntimeError("MCP runtime 未启动或已关闭")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try:
            self._loop.call_soon_threadsafe(self._jobs.put_nowait, (coro_fn, fut))
        except RuntimeError as e:  # loop 已停
            raise McpRuntimeError("MCP runtime 未启动或已关闭") from e
        return fut

    async def call_async(self, coro_fn: Callable[[object], Awaitable], timeout: float | None = None):
        """主 loop 侧 await 桥接（MCP 工具 func 用；不阻塞任一 loop）。"""
        future = self.submit(coro_fn)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout)

    def call_sync(self, coro_fn: Callable[[object], Awaitable], timeout: float | None = None):
        """同步阻塞等待（mention 展开等 REPL 非协程上下文用）。"""
        return self.submit(coro_fn).result(timeout)
