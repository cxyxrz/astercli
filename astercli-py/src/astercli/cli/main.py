"""CLI 入口：REPL + 一次性运行模式。

对齐 Java 侧约定：
- 未识别的 /xxx 在 CLI 层直接报"未知命令"，不回退给 Agent
- 主退出条件交给 LLM；Ctrl+C 取消当前任务并回滚本轮历史
- /model 切换 provider（仅当前会话；~/.astercli/config.json 持久化后续分期接入）
- 每轮任务前检索相关长期记忆，前置注入 user message（Java 注入 system prompt，
  Python 版保持 system prompt 稳定以友好 prompt cache，语义等价）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rich.console import Console

from .. import __version__
from ..agent import AgentBudget
from ..agent.graph import ainvoke_with_hitl, build_react_graph, make_checkpointer
from ..agent.orchestrator import AgentOrchestrator
from ..agent.plan_execute import PlanExecuteAgent
from ..config import Settings
from ..dream import DreamRunner, DreamWorker, Proposal, SessionLog, describe_op
from ..experience import (
    CATEGORY_LABELS,
    ExperienceStore,
    ReflectionCollector,
    Reflector,
    TRIGGER_LABELS,
)
from ..hitl import ApprovalSession, HitlPrompter
from ..llm.factory import (
    ProviderRuntime,
    available_providers,
    create_client,
    create_from_settings,
)
from ..lsp import LspManager
from ..mcp import McpServerManager, expand_mentions
from ..memory import (
    ConversationHistoryCompactor,
    LongTermMemory,
    auto_compact_trigger_tokens,
    build_memory_context,
    estimate_messages_tokens,
    memory_context_tokens,
    normalize_project_key,
)
from ..plan import Planner
from ..policy import CommandSandbox, PathGuard
from ..prompt import build_system_prompt
from ..skill import (
    SkillContextBuffer,
    default_skill_registry,
    format_index,
    register_skill_tool,
)
from ..tools import ToolRegistry
from ..tools.builtin import register_file_tools, register_shell_tools
from ..web import register_web_tools
from ..rag import (
    CodeIndex,
    CodeRetriever,
    EmbeddingClient,
    EmbeddingNotConfiguredError,
    IndexNotReadyError,
    format_for_cli,
    register_rag_tool,
)
from .plan_review import review_plan
from .renderer import ConsoleRenderer
from .status_bar import StatusState, toolbar_fragments


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="astercli", description="AsterCLI Python 重写版（LangChain + LangGraph）"
    )
    parser.add_argument("--provider", help="指定 provider（glm/deepseek/step/kimi/freellmapi/xfyun/agnes）")
    parser.add_argument("prompt", nargs="*", help="一次性运行模式：直接给出任务后退出")
    args = parser.parse_args(argv)

    settings = Settings()
    if args.provider:
        runtime = create_client(args.provider, settings)
        if runtime is None:
            print(f"provider {args.provider} 未配置 API Key 或不可用", file=sys.stderr)
            print(f"可用 provider: {', '.join(available_providers(settings)) or '（无）'}", file=sys.stderr)
            return 1
    else:
        runtime = create_from_settings(settings)
        if runtime is None:
            print(
                "没有可用的 API Key。请参考仓库根 .env.example 配置 "
                "GLM_API_KEY / DEEPSEEK_API_KEY / STEP_API_KEY / KIMI_API_KEY / "
                "FREELLMAPI_API_KEY / XFYUN_MAAS_API_KEY / AGNES_API_KEY 之一。",
                file=sys.stderr,
            )
            return 1

    one_shot = " ".join(args.prompt).strip()
    console = Console(highlight=False, markup=False)
    # 底部状态栏：REPL 输入期由 prompt_toolkit 的 bottom_toolbar 渲染；
    # 一次性模式没有输入态，保持 None，usage() 回到轮末打印 token 行。
    status = None
    if not one_shot:
        status = StatusState(
            model=runtime.display_name,
            context_window=runtime.context_window,
            cwd=str(Path.cwd()),
        )
    renderer = ConsoleRenderer(console, status=status)
    app = ReplApp(settings, runtime, renderer, Path.cwd())
    if one_shot:
        try:
            asyncio.run(app.run_turn(one_shot))
        except (KeyboardInterrupt, asyncio.CancelledError):
            renderer.warn("⏹️ 已取消当前任务。")
        finally:
            app.mcp.shutdown()
        return 0

    return app.run()


class ReplApp:
    def __init__(
        self,
        settings: Settings,
        runtime: ProviderRuntime,
        renderer,
        root: Path,
    ) -> None:
        self.settings = settings
        self.renderer = renderer
        self.root = root.resolve()
        self.path_guard = PathGuard(self.root)
        # HITL 审批：interrupt 挂起 → 终端收集决策 → Command(resume) 恢复；
        # 会话级"全部放行"名单挂在 ApprovalSession 上（/clear 清空）
        self.approval = ApprovalSession()
        self.checkpointer = make_checkpointer()
        self.hitl_prompt = HitlPrompter(getattr(renderer, "_console", None))
        # 经验反思（P9）：采集器挂到工具层（工具失败 / HITL 拒绝）与 LSP 层（语法 error），
        # 每轮结束后由 Reflector 统一复盘成 Markdown 经验
        self.experience = ExperienceStore(self.root, self.settings)
        self.reflection_collector = ReflectionCollector()
        self.registry = ToolRegistry(
            settings=settings, approval=self.approval, reflection=self.reflection_collector
        )
        register_file_tools(self.registry, self.path_guard)
        # 行内 diff（P11）：write_file 成功后由 registry 出口回调渲染器，
        # 与 LSP 钩子同级，异常在 registry 内静默
        self.registry.set_file_write_listener(
            lambda path, before, after: self.renderer.diff(path, before, after)
        )
        # 第三层命令沙箱：开启后 execute_command 子进程在写围栏内运行
        # （未开启时 wrap 返回裸 bash，行为与之前完全一致）
        register_shell_tools(
            self.registry,
            self.path_guard,
            sandbox=CommandSandbox.from_settings(self.settings),
        )
        # RAG（P6）：search_code 语义检索（只读不审批）；索引由 /index 惰性建立
        register_rag_tool(self.registry, self.root)
        # Web（P7）：web_search / web_fetch（只读不审批；provider 未配置时返回配置提示）
        register_web_tools(self.registry, self.settings)
        # LSP 诊断（P8）：write_file 出口钩子收集语法诊断，ReAct 图调 LLM 前
        # 作为 user message 注入（ReAct / Plan / team worker 三路径共享此 registry）
        self.lsp_manager = LspManager(
            self.root, self.settings, on_diagnostics=self.reflection_collector.lsp_error
        )
        self.registry.set_lsp_manager(self.lsp_manager)
        # Skill（P7）：三层目录加载 + load_skill 工具 + 注入缓冲区
        self.skill_registry = default_skill_registry(self.root)
        self.skill_buffer = SkillContextBuffer()
        register_skill_tool(self.registry, self.skill_registry, self.skill_buffer)
        # MCP（P5）：动态工具注册进同一 registry（后台续载的工具实时进 schemas()）；
        # startup 同步最多等 8 秒，超时部分后台续载，不阻塞首屏（对齐 Java）
        self.mcp = McpServerManager(self.registry, self.settings, root=self.root, renderer=renderer)
        self.mcp.startup()
        self.runtime = runtime
        self.long_term = LongTermMemory()
        self.project_key = normalize_project_key(self.root)
        # AutoDream（P10）：每轮最小会话转写（用户输入 / 工具名 / 最终回答），
        # 不落工具参数与完整输出；后台整合由 dream runner 在后续里程碑消费
        self.session_log = SessionLog(self.project_key, self.settings)
        # 长期记忆是单文件 JSON：/save、/memory 与 dream 应用共用一把锁，
        # 保证"读 - 改 - 写"不被后台线程打断（临界区很短）
        self.memory_lock = threading.RLock()
        self.dream_runner = DreamRunner(
            self.project_key, self.settings, session_log=self.session_log
        )
        # 经验反思器（P9）：会话级开关 / 次数配额 / 去重都在这里；/model 切换后同步 runtime
        self.reflector = Reflector(
            runtime,
            self.experience,
            self.renderer,
            self.settings,
            self.reflection_collector,
            project_key=self.project_key,
        )
        # Skill 索引段 + 经验索引段进 system prompt（经验只注入一次，避免 prompt 前缀抖动）
        self._skill_index = format_index(self.skill_registry.enabled_skills())
        self._experience_index = self.experience.index_text()
        self.history: list = [
            SystemMessage(
                build_system_prompt(self.root, self._skill_index, self._experience_index)
            )
        ]
        self._next_use_plan = False
        self._next_use_team = False
        self._orchestrator: AgentOrchestrator | None = None
        self._rebuild_graph()
        # AutoDream（P10）：启动期只做廉价门检查，通过后守护线程后台整合；
        # 完成通知入队，下一次输入前由 REPL 统一打印（不抢 prompt_toolkit 输入区）
        self.dream_worker = DreamWorker(
            self.dream_runner,
            self.settings,
            self.session_log,
            self.long_term,
            self.experience,
            self._dream_runtime,
            project_key=self.project_key,
            memory_lock=self.memory_lock,
        )
        self.dream_worker.start()

    def _dream_runtime(self):
        """dream 专用 runtime：ASTERCLI_DREAM_MODEL 覆盖当前 provider 的模型。"""
        override = self.settings.dream_model()
        if not override:
            return self.runtime
        dream_runtime = create_client(self.runtime.provider, self.settings, model=override)
        return dream_runtime or self.runtime

    def _rebuild_graph(self) -> None:
        """按当前 runtime 重建压缩器与 ReAct 图 / Plan Agent（/model 切换后复用）。"""
        self.reflector.set_runtime(self.runtime)
        self.compactor = ConversationHistoryCompactor(self.runtime.client)
        self.trigger_tokens = auto_compact_trigger_tokens(self.runtime.context_window)
        self.graph = build_react_graph(
            self.runtime,
            self.registry,
            self.renderer,
            tool_mode=self.settings.tool_mode(),
            compactor=self.compactor,
            trigger_tokens=self.trigger_tokens,
            approval=self.approval,
            checkpointer=self.checkpointer,
        )
        self.plan_agent = PlanExecuteAgent(
            self.runtime,
            self.registry,
            self.renderer,
            Planner(self.runtime, self.renderer),
            self._review_plan,
            tool_mode=self.settings.tool_mode(),
            compactor=self.compactor,
            trigger_tokens=self.trigger_tokens,
            approval=self.approval,
            checkpointer=self.checkpointer,
            on_interrupt=self._on_hitl_interrupt,
            memory_provider=self._task_memory,
        )
        # 编排器持有 runtime 引用，/model 切换后懒重建
        self._orchestrator = None

    # ---------- HITL 审批 / 计划审阅回调 ----------

    async def _on_hitl_interrupt(self, interrupts) -> dict:
        return await self.hitl_prompt.collect(interrupts)

    async def _review_plan(self, goal, plan):
        return await review_plan(goal, plan, getattr(self.renderer, "_console", None))

    def _task_memory(self, description: str) -> str:
        memory = build_memory_context(
            self.long_term,
            description,
            memory_context_tokens(self.runtime.context_window),
            project_key=self.project_key,
        )
        experience = self.experience.context_for_query(description) if self.reflector.enabled else ""
        return "\n\n".join(block for block in (memory, experience) if block)

    def _delete_thread(self, thread_id: str) -> None:
        """释放单轮 thread 的 checkpoint（MemorySaver 常驻会话内增长）。"""
        try:
            self.checkpointer.delete_thread(thread_id)
        except Exception:  # noqa: BLE001 —— 清理失败不影响任务结果
            pass

    # ---------- 主循环 ----------

    def run(self) -> int:
        self._print_banner()
        import shutil

        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()

        @bindings.add("c-o")
        def _expand_last_block(event) -> None:
            # 折叠块写在 transcript 里，需先把输入区让出来再重绘（对齐 Java Ctrl+O）
            event.app.run_in_terminal(lambda: self.renderer.toggle_last_block())

        def _toolbar():
            status = getattr(self.renderer, "status", None)
            if status is None:
                return []
            # 宽度必须和 prompt_toolkit 当前渲染宽度一致，否则差一列就会折行，
            # 底栏多出一行会盖住上面的模型输出；减 1 留出边界，避免正好占满折行
            try:
                from prompt_toolkit.application import get_app

                cols = max(20, get_app().output.get_size().columns - 1)
            except Exception:  # noqa: BLE001 —— 非 prompt_toolkit 上下文回退
                cols = max(20, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            # 状态栏两行输出后再补一个换行，避免和输入行/后续输出贴在一起
            return toolbar_fragments(status, cols) + [("", "\n")]

        session: PromptSession = PromptSession(key_bindings=bindings, bottom_toolbar=_toolbar)
        while True:
            self._flush_dream_notices()
            try:
                text = session.prompt("* ")
            except (KeyboardInterrupt, EOFError):
                break
            stripped = text.strip()
            if not stripped:
                continue
            # 提交回显：JLine 侧用 `>` 前缀写回 transcript，这里等价补一行
            self.renderer.echo_input(stripped)
            if stripped.startswith("/"):
                if self._handle_command(stripped):
                    break
                continue
            started = time.monotonic()
            try:
                # /plan 或 /team 无参数后下一次输入进入对应模式
                use_plan = self._next_use_plan
                use_team = self._next_use_team
                self._next_use_plan = False
                self._next_use_team = False
                self.renderer.begin_turn()
                if use_team:
                    asyncio.run(self.run_team_turn(stripped))
                elif use_plan:
                    asyncio.run(self.run_plan_turn(stripped))
                else:
                    asyncio.run(self.run_turn(stripped))
                self._refresh_status(int((time.monotonic() - started) * 1000))
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.renderer.warn("⏹️ 已取消当前任务。")
                self._refresh_status(int((time.monotonic() - started) * 1000))
        self.mcp.shutdown()
        return 0

    def _refresh_status(self, elapsed_ms: int) -> None:
        """刷新底部状态栏：phase 回 idle + 当前上下文 token 估算。"""
        status = getattr(self.renderer, "status", None)
        if status is None:
            return
        try:
            context_tokens = estimate_messages_tokens(self.history)
        except Exception:  # noqa: BLE001 —— 估算失败不影响主循环
            context_tokens = 0
        status.update(
            phase="idle",
            model=self.runtime.display_name,
            context_window=self.runtime.context_window,
            context_tokens=context_tokens,
            elapsed_ms=elapsed_ms,
        )

    # ---------- 单轮执行 ----------

    async def run_turn(self, text: str) -> None:
        # 会话转写：先记用户原始输入（后续 enrich 不改变已记录内容）
        self.session_log.record_user(text)
        # MCP resource mention 展开（@server:uri → <resource> 块；失败降级错误块）
        text = expand_mentions(text, self.mcp)
        snapshot = list(self.history)
        # 每轮独立 thread：interrupt 挂起需要 checkpointer，轮后 delete_thread 释放
        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        try:
            # 检索相关长期记忆，前置注入本轮 user message（不进 system prompt，
            # 保持前缀稳定以友好 prompt cache）
            memory_block = build_memory_context(
                self.long_term,
                text,
                memory_context_tokens(self.runtime.context_window),
                project_key=self.project_key,
            )
            # 经验检索注入（P9）：按当前输入召回相关历史复盘，只取少量条目
            experience_block = (
                self.experience.context_for_query(text) if self.reflector.enabled else ""
            )
            blocks = [b for b in (memory_block, experience_block) if b]
            enriched = "\n\n---\n\n".join(blocks + [text]) if blocks else text
            # Skill body 前置注入（load_skill 之后一轮生效；对齐 Java prependSkillBodies）
            skill_block = self.skill_buffer.drain()
            if skill_block:
                enriched = f"{skill_block}\n\n{enriched}"
            self.history.append(HumanMessage(enriched))
            budget = AgentBudget(
                self.settings.react_token_budget(),
                self.settings.react_stagnation_window(),
                self.settings.react_hard_max_iterations(),
            )
            result = await ainvoke_with_hitl(
                self.graph,
                {"messages": list(self.history), "budget": budget, "halt_reason": None},
                config,
                self._on_hitl_interrupt,
            )
            consumed = len(self.history)
            self.history[:] = result["messages"]
            # resume 恢复后 result["budget"] 是反序列化副本，优先取回（token 记账完整）
            budget = result.get("budget") or budget
            new_messages = result["messages"][consumed:]
            self.session_log.record_tools(_tool_names(new_messages))
            self.session_log.record_assistant(_last_ai_text(result["messages"]))
        except (KeyboardInterrupt, asyncio.CancelledError):
            # 取消当前任务：回滚本轮半途历史，不污染下一轮
            self.history[:] = snapshot
            self._delete_thread(thread_id)
            raise
        self._delete_thread(thread_id)
        self.renderer.usage(budget)
        await self._reflect_after_turn()

    async def _reflect_after_turn(self) -> None:
        """轮末复盘：主回答已输出完毕后再消费触发信号，失败绝不影响任务结果。"""
        try:
            await self.reflector.consume()
        except Exception:  # noqa: BLE001
            pass

    async def run_plan_turn(self, text: str) -> None:
        """计划模式单轮：/plan 触发，Planner 出 DAG → 审阅 → 按序执行。"""
        self.session_log.record_user(text)
        text = expand_mentions(text, self.mcp)
        # Skill body 前置注入（计划任务执行者与主 Agent 共用同一语义）
        skill_block = self.skill_buffer.drain()
        if skill_block:
            text = f"{skill_block}\n\n{text}"
        try:
            result = await self.plan_agent.run(text)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.renderer.warn("⏹️ 已取消计划任务。")
            return
        # 计划结果只进历史不重印全文（任务过程已流式输出）
        self.history.append(HumanMessage(text))
        self.history.append(AIMessage(f"[计划结果] {result}"))
        self.session_log.record_assistant(str(result))
        await self._reflect_after_turn()

    async def run_team_turn(self, text: str) -> None:
        """多 Agent 协作单轮：/team 触发，planner → worker → reviewer 流水线。"""
        self.session_log.record_user(text)
        text = expand_mentions(text, self.mcp)
        if self._orchestrator is None:
            self._orchestrator = AgentOrchestrator(
                self.runtime,
                self.registry,
                self.renderer,
                project_root=self.root,
                settings=self.settings,
                skill_registry=self.skill_registry,
                skill_buffer=self.skill_buffer,
                approval=self.approval,
                checkpointer=self.checkpointer,
                on_interrupt=self._on_hitl_interrupt,
                tool_mode=self.settings.tool_mode(),
                experience_index=self._experience_index,
            )
        try:
            result = await self._orchestrator.run(text)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.renderer.warn("⏹️ 已取消多 Agent 任务。")
            return
        # 协作结果进历史（主对话上下文可见，供后续轮次引用）
        self.history.append(HumanMessage(text))
        self.history.append(AIMessage(f"[多 Agent 协作结果] {result}"))
        self.session_log.record_assistant(str(result))
        await self._reflect_after_turn()

    def _flush_dream_notices(self) -> None:
        """输入前统一输出后台 dream 结果；若记忆库被改写则重建经验索引段。"""
        for notice in self.dream_worker.drain_notices():
            self.renderer.info(notice)
        if self.dream_worker.dirty:
            self.dream_worker.clear_dirty()
            self._refresh_experience_index()

    # ---------- 斜杠命令（未知命令在 CLI 层直接报错，不回退给 Agent） ----------

    def _handle_command(self, text: str) -> bool:
        """返回 True 表示退出 REPL。"""
        parts = text.split()
        cmd = parts[0].lower()
        payload = text[len(parts[0]) :].strip()
        if cmd in ("/quit", "/exit"):
            return True
        if cmd == "/clear":
            # 清空待注入 Skill buffer，system prompt 用当前 skill 索引段重建（对齐 Java）
            self.skill_buffer.clear()
            self.history = [
                SystemMessage(
                    build_system_prompt(self.root, self._skill_index, self._experience_index)
                )
            ]
            self.approval.clear_approved_all()
            self.reflector.reset_session()
            self.renderer.info("已清空会话历史（system prompt 已重建，长期记忆与经验库保留）。")
        elif cmd == "/compact":
            self._compact_history()
        elif cmd == "/plan":
            if payload:
                asyncio.run(self.run_plan_turn(payload))
            else:
                self._next_use_plan = True
                self.renderer.info("计划模式已开启，下一条输入将进入任务规划与审阅。")
        elif cmd == "/team":
            if payload:
                asyncio.run(self.run_team_turn(payload))
            else:
                self._next_use_team = True
                self.renderer.info("多 Agent 协作模式已开启，下一条输入将进入团队协作。")
        elif cmd == "/model":
            self._handle_model(payload.split()[0] if payload else "")
        elif cmd == "/save":
            self._handle_save(payload)
        elif cmd == "/memory":
            self._handle_memory(parts)
        elif cmd == "/tools":
            names = [
                f"{name}{' *' if (t := self.registry.get(name)) and t.dangerous else ''}"
                for name in self.registry.names()
            ]
            self.renderer.info(
                f"工具（{self.settings.tool_mode()} 模式，* = 需审批）:\n  " + "\n  ".join(names)
            )
        elif cmd == "/mcp":
            self._handle_mcp(payload)
        elif cmd == "/index":
            self._handle_index(payload)
        elif cmd == "/search":
            self._handle_search(payload)
        elif cmd == "/skill":
            self._handle_skill(payload)
        elif cmd == "/experience":
            self._handle_experience(payload)
        elif cmd == "/dream":
            self._handle_dream(payload)
        elif cmd == "/help":
            self.renderer.info(
                "命令: /quit /exit 退出 · /clear 清空历史 · /compact 压缩历史 · "
                "/model [provider] 切换 · /tools 工具列表 · /plan [任务] 计划模式 · "
                "/team [任务] 多 Agent 协作 · "
                "/skill [list|show|off|on|reload] Skill 管理 · "
                "/experience [list|search|show|delete|clear|on|off] 经验管理 · "
                "/dream [status|log|undo|on|off] 后台记忆整合 · "
                "/mcp [server|resources|restart|disable|enable] MCP · "
                "/index [路径] 建立代码索引 · /search <query> 语义检索 · "
                "/save [--global] <事实> 存长期记忆 · /memory list|search|delete|clear · /help 帮助"
            )
        else:
            self.renderer.warn(f"未知命令: {text}")
        return False

    # ---------- 记忆命令 ----------

    def _compact_history(self) -> None:
        """手动压缩：跳过阈值，保留最近 1 个 user 轮次。"""
        rebuilt = asyncio.run(self.compactor.compact_now(self.history))
        if rebuilt is None:
            self.renderer.info("当前历史较短，没有可压缩的早期轮次。")
            return
        before = estimate_messages_tokens(self.history)
        after = estimate_messages_tokens(rebuilt)
        self.history[:] = rebuilt
        self.renderer.info(
            f"已压缩会话历史: {before} → {after} tokens（{len(rebuilt)} 条消息，保留最近 1 个 user 轮次）"
        )

    def _handle_save(self, payload: str) -> None:
        scope, fact = parse_save_payload(payload)
        if not fact:
            self.renderer.warn(
                "请提供要保存的内容，例如 /save 这个项目使用Java 17，或 /save --global 默认用中文回答"
            )
            return
        with self.memory_lock:
            entry = self.long_term.store_fact(fact, scope=scope, project_key=self.project_key)
        if entry is None:
            self.renderer.warn("长期记忆中已存在相同内容，未重复保存。")
        else:
            self.renderer.info(f"💾 已保存到长期记忆({scope}): {fact}")

    def _handle_memory(self, parts: list) -> None:
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub in ("", "status"):
            self.renderer.info(
                "📋 记忆系统状态：\n"
                + self.long_term.status_summary()
                + f"\n   当前项目作用域: {self.project_key}"
                + "\n   /memory list - 查看长期记忆"
                + "\n   /memory search <关键词> - 搜索当前项目可见长期记忆"
                + "\n   /memory delete <id> - 删除单条长期记忆"
                + "\n   /memory clear - 清空长期记忆"
                + "\n   /save <事实> - 保存项目级长期记忆；/save --global <事实> 保存全局记忆"
            )
        elif sub == "list":
            self.renderer.info(format_memory_entries("📋 长期记忆列表", self.long_term.get_all()))
        elif sub == "search":
            query = " ".join(parts[2:]).strip()
            if not query:
                self.renderer.warn("请提供搜索关键词，例如 /memory search Chrome 登录态")
            else:
                entries = self.long_term.search(query, limit=20, project_key=self.project_key)
                self.renderer.info(format_memory_entries(f"🔎 长期记忆搜索: {query}", entries))
        elif sub == "delete":
            entry_id = parts[2] if len(parts) > 2 else ""
            if not entry_id:
                self.renderer.warn("请提供要删除的记忆 id，例如 /memory delete fact-abcd1234")
            elif self._delete_memory(entry_id):
                self.renderer.info(f"🗑️ 已删除长期记忆: {entry_id}")
            else:
                self.renderer.warn(f"📭 未找到长期记忆: {entry_id}")
        elif sub == "clear":
            with self.memory_lock:
                self.long_term.clear()
            self.renderer.info("🧹 长期记忆已清空")
        else:
            self.renderer.warn(f"未知的 /memory 子命令: {sub}")

    def _delete_memory(self, entry_id: str) -> bool:
        """删除单条长期记忆（与 dream 应用共用锁）。"""
        with self.memory_lock:
            return self.long_term.delete(entry_id)

    # ---------- AutoDream ----------

    def _handle_dream(self, payload: str) -> None:
        """`/dream` 子命令（P10）: 立即整合 / status / log / undo / on / off。"""
        parts = payload.split()
        sub = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""
        if sub in ("", "run"):
            self._run_dream_now()
        elif sub == "status":
            self._dream_status()
        elif sub == "log":
            self._dream_log(arg)
        elif sub == "undo":
            self._dream_undo(arg)
        elif sub in ("on", "off"):
            enabled = sub == "on"
            self.dream_worker.set_enabled(enabled)
            self.renderer.info(
                f"AutoDream 自动整合已{'开启' if enabled else '关闭'}"
                "（本会话生效；/dream 仍可手动触发）"
            )
        else:
            self.renderer.warn(
                f"未知的 /dream 子命令: {sub}"
                "（/dream 立即整合 · status · log [id] · undo [id] · on|off）"
            )

    def _run_dream_now(self) -> None:
        """手动触发一次整合（忽略时间 / 会话门，仍走锁与事务回滚）。"""
        self.renderer.info("🧠 正在整理记忆（后台同款流程，稍候）…")
        message = asyncio.run(self.dream_worker.run_now())
        if self.dream_worker.dirty:
            self.dream_worker.clear_dirty()
            self._refresh_experience_index()
        self.renderer.info(message)

    def _dream_status(self) -> None:
        state = self.dream_runner.load_state()
        gate = self.dream_runner.check()
        enabled = self.dream_worker.enabled
        lines = [
            "🧠 AutoDream 状态",
            f"   自动整合: {'开启' if enabled else '关闭'}（本会话；环境默认 "
            f"{'开' if self.settings.dream_enabled() else '关'}）",
            f"   当前门判定: {gate.label} — {gate.detail}",
            f"   后台任务: {'运行中' if self.dream_worker.busy else '空闲'}",
            f"   累计整合: {state.dream_count} 次 · 上次: {_fmt_ts(state.last_dream_at)}",
            f"   上次扫描: {_fmt_ts(state.last_scan_at)}",
        ]
        if state.last_dream_id:
            lines.append(f"   最近 dream: {state.last_dream_id}")
        lines.extend(
            [
                f"   会话转写: {self.session_log.directory}",
                f"   整合目录: {self.dream_runner.paths.root}",
                "   /dream 立即整合 · log [id] 查看记录 · undo [id] 撤销 · on|off 开关",
            ]
        )
        self.renderer.info("\n".join(lines))

    def _dream_log(self, dream_id: str) -> None:
        state = self.dream_runner.load_state()
        if dream_id:
            record = next((r for r in state.history if r.dream_id == dream_id), None)
            if record is None:
                self.renderer.warn(f"未找到 dream 记录: {dream_id}")
                return
            lines = [
                f"🧠 Dream {record.dream_id}",
                f"   时间: {_fmt_ts(record.finished_at)} · 状态: {record.status or '未知'}"
                f" · {record.ops} ops",
                f"   摘要: {record.summary or '（无）'}",
                f"   提案: {self.dream_runner.paths.proposal_dir(record.dream_id)}",
            ]
            proposal = Proposal.load(self.dream_runner.paths.proposal_dir(record.dream_id))
            if proposal and proposal.ops:
                lines.append("   操作:")
                lines.extend(
                    f"     {index}. {describe_op(op)}"
                    for index, op in enumerate(proposal.ops, 1)
                )
            if proposal and proposal.errors:
                lines.append("   校验问题:")
                lines.extend(f"     - {error}" for error in proposal.errors)
            self.renderer.info("\n".join(lines))
            return

        if not state.history:
            self.renderer.info("还没有 dream 记录。满足条件后会在启动期自动整理。")
            return
        lines = ["🧠 Dream 记录（最新在前）"]
        for item in state.history:
            lines.append(
                f"   {item.dream_id} · {_fmt_ts(item.finished_at)}"
                f" · {item.status or '未知'} · {item.ops} ops · {item.summary or '（无摘要）'}"
            )
        lines.append("   /dream log <id> 查看操作明细 · /dream undo [id] 撤销整合")
        self.renderer.info("\n".join(lines))

    def _dream_undo(self, dream_id: str) -> None:
        ok, message = self.dream_runner.undo(dream_id)
        if not ok:
            self.renderer.warn(message)
            return
        # 撤销直接改写了磁盘文件：重新加载内存记忆并重建 system prompt 索引段
        self.long_term = LongTermMemory()
        self._refresh_experience_index()
        self.renderer.info(f"↩️ {message}")

    def _handle_model(self, arg: str) -> None:
        available = available_providers(self.settings)
        if not arg:
            self.renderer.info(
                f"当前模型: {self.runtime.display_name}；可用 provider: "
                + (", ".join(available) or "（无）")
            )
            return
        runtime = create_client(arg, self.settings)
        if runtime is None:
            self.renderer.warn(
                f"provider {arg} 不可用（未配置 Key 或未知名称）；可用: "
                + (", ".join(available) or "（无）")
            )
            return
        self.runtime = runtime
        self._rebuild_graph()
        self.renderer.info(f"已切换模型: {runtime.display_name}")

    def _handle_index(self, payload: str) -> None:
        """`/index [路径]`：对项目（或子目录）建立/重建 RAG 向量索引。"""
        target = self.path_guard.resolve_safe(payload or ".")
        if target is None:
            self.renderer.warn("路径不在项目根内，已拒绝。")
            return
        try:
            embedding = EmbeddingClient(self.settings)
            embedding.ensure_configured()
            indexer = CodeIndex(target, embedding)
            stats = indexer.index(progress=lambda msg: self.renderer.info(msg))
            self.renderer.info(
                f"✅ 索引完成: {stats.files} 文件 · {stats.chunks} 代码块 · "
                f"{stats.elapsed_seconds}s · 后端 {stats.backend}"
            )
        except EmbeddingNotConfiguredError as e:
            self.renderer.warn(str(e))
        except (RuntimeError, OSError, ValueError) as e:
            self.renderer.warn(f"索引失败: {e}")

    def _handle_search(self, payload: str) -> None:
        """`/search <query>`：不经过 LLM 直接做语义混合检索（对齐 Java /search）。"""
        if not payload:
            self.renderer.warn("用法: /search <自然语言查询>")
            return
        try:
            retriever = CodeRetriever(self.root, EmbeddingClient(self.settings))
            results = retriever.hybrid_search(payload)
            self.renderer.info(format_for_cli(payload, results))
        except IndexNotReadyError:
            self.renderer.warn("代码索引尚未建立，请先执行 /index。")
        except EmbeddingNotConfiguredError as e:
            self.renderer.warn(str(e))
        except (RuntimeError, OSError, ValueError) as e:
            self.renderer.warn(f"检索失败: {e}")

    def _handle_mcp(self, payload: str) -> None:
        """`/mcp` 子命令（对齐 Java：status / resources / restart / disable / enable）。"""
        parts = payload.split()
        if not parts or parts[0].lower() == "status":
            self.renderer.info("\n".join(self.mcp.status_lines()))
            return
        sub = parts[0].lower()
        server = parts[1] if len(parts) > 1 else ""
        if not server:
            self.renderer.warn(f"用法: /mcp {sub} <server>")
        elif sub == "resources":
            self.renderer.info(asyncio.run(self.mcp.list_resources_async(server)))
        elif sub == "restart":
            if self.mcp.restart(server):
                self.renderer.info(f"🔄 MCP server '{server}' 正在重连（/mcp 查看进度）。")
            else:
                self.renderer.warn(f"未找到 MCP server: {server}")
        elif sub == "disable":
            if self.mcp.disable(server):
                self.renderer.info(f"已禁用 MCP server '{server}'（工具已注销）。")
            else:
                self.renderer.warn(f"未找到 MCP server: {server}")
        elif sub == "enable":
            if self.mcp.enable(server):
                self.renderer.info(f"正在启用 MCP server '{server}'（/mcp 查看进度）。")
            else:
                self.renderer.warn(f"未找到 MCP server: {server}")
        else:
            self.renderer.warn(
                f"未知的 /mcp 子命令: {sub}（可用: status / resources / restart / disable / enable）"
            )

    def _handle_skill(self, payload: str) -> None:
        """`/skill` 子命令（对齐 Java：list / show / off / on / reload）。"""
        parts = payload.split()
        sub = parts[0].lower() if parts else "list"
        store = self.skill_registry.state_store()
        disabled = store.disabled()

        if sub in ("", "list"):
            skills = self.skill_registry.all_skills()
            if not skills:
                self.renderer.info(
                    "暂无可用 skill。可创建 ~/.astercli-py/skills/<name>/SKILL.md（用户级）或 "
                    ".astercli/skills/<name>/SKILL.md（项目级）后执行 /skill reload。"
                )
                return
            lines = ["📋 Skill 列表："]
            for s in skills:
                flag = "（已禁用）" if s.name in disabled else ""
                desc = s.description if len(s.description) <= 60 else s.description[:60] + "..."
                lines.append(f"  [{s.display_source()}] {s.name}{flag}: {desc}")
            lines.append("  /skill show <name> 查看详情 · /skill off|on <name> 禁用/启用 · /skill reload 重载")
            self.renderer.info("\n".join(lines))

        elif sub == "show":
            name = parts[1] if len(parts) > 1 else ""
            skill = self.skill_registry.find_any_skill(name)
            if skill is None:
                self.renderer.warn(f"未找到 skill: {name or '（未提供名称）'}")
                return
            state = "已禁用" if skill.name in disabled else "已启用"
            lines = [
                f"📄 Skill: {skill.name}（{skill.display_source()} · {state}）",
                f"   描述: {skill.description or '（无）'}",
            ]
            if skill.version:
                lines.append(f"   版本: {skill.version}")
            if skill.author:
                lines.append(f"   作者: {skill.author}")
            if skill.tags:
                lines.append(f"   标签: {', '.join(skill.tags)}")
            lines.append(f"   路径: {skill.skill_md_path}")
            lines.append("")
            lines.append("--- body 预览（前 500 字符）---")
            lines.append(skill.body[:500])
            self.renderer.info("\n".join(lines))

        elif sub in ("off", "on"):
            name = parts[1] if len(parts) > 1 else ""
            if not name:
                self.renderer.warn(f"用法: /skill {sub} <name>")
                return
            if self.skill_registry.find_any_skill(name) is None:
                self.renderer.warn(f"未找到 skill: {name}")
                return
            names = set(disabled)
            if sub == "off":
                names.add(name)
                self.renderer.info(f"已禁用 skill: {name}（下轮 system prompt 生效）")
            else:
                names.discard(name)
                self.renderer.info(f"已启用 skill: {name}（下轮 system prompt 生效）")
            store.set_disabled(sorted(names))
            self._refresh_skill_index()

        elif sub == "reload":
            self.skill_registry.reload()
            for warning in self.skill_registry.warnings():
                self.renderer.warn(f"skill 加载警告: {warning}")
            enabled = self._refresh_skill_index()
            self.renderer.info(f"已重新加载 skill（{len(enabled)} 个启用），system prompt 索引段已更新。")

        else:
            self.renderer.warn(
                f"未知的 /skill 子命令: {sub}（可用: list / show / off / on / reload）"
            )

    def _handle_experience(self, payload: str) -> None:
        """`/experience` 子命令（P9）: list / search / show / delete / clear / on / off。"""
        parts = payload.split()
        sub = parts[0].lower() if parts else "list"

        if sub in ("", "list"):
            scope = _parse_experience_scope(parts[1:])
            entries = self.experience.list_entries(scope=scope)
            self.renderer.info(
                format_experience_entries("🧠 经验列表", entries)
                + f"\n  {self.reflector.status()}"
                + "\n  /experience search <关键词> · show <id> · delete <id> · clear · on|off"
            )

        elif sub == "search":
            query = " ".join(parts[1:]).strip()
            if not query:
                self.renderer.warn("请提供搜索关键词，例如 /experience search 路径")
                return
            self.renderer.info(
                format_experience_entries(f"🔎 经验搜索: {query}", self.experience.search(query))
            )

        elif sub == "show":
            entry_id = parts[1] if len(parts) > 1 else ""
            entry = self.experience.get(entry_id)
            if entry is None:
                self.renderer.warn(f"未找到经验: {entry_id or '（未提供 id）'}")
                return
            path = self.experience.path_of(entry)
            self.renderer.info(
                format_experience_detail(entry, str(path) if path else "")
            )

        elif sub == "delete":
            entry_id = parts[1] if len(parts) > 1 else ""
            if not entry_id:
                self.renderer.warn("请提供要删除的经验 id，例如 /experience delete exp-abcd1234")
            elif self._delete_experience(entry_id):
                self._refresh_experience_index()
                self.renderer.info(f"🗑️ 已删除经验: {entry_id}")
            else:
                self.renderer.warn(f"📭 未找到经验: {entry_id}")

        elif sub == "clear":
            if len(parts) > 1 and parts[1].lower() == "--all":
                scope = None
            else:
                scope = _parse_experience_scope(parts[1:]) or "project"
            removed = self._clear_experience(scope)
            self._refresh_experience_index()
            label = "全部" if scope is None else ("项目级" if scope == "project" else "全局")
            self.renderer.info(f"🧹 已清空{label}经验（{removed} 条）")

        elif sub in ("on", "off"):
            enabled = sub == "on"
            self.reflector.set_enabled(enabled)
            state = "开启" if enabled else "关闭"
            self.renderer.info(f"经验反思已{state}（自动复盘与经验注入同时{state}）")

        else:
            self.renderer.warn(
                f"未知的 /experience 子命令: {sub}"
                "（可用: list / search / show / delete / clear / on / off）"
            )

    def _delete_experience(self, entry_id: str) -> bool:
        with self.memory_lock:
            return self.experience.delete(entry_id)

    def _clear_experience(self, scope) -> int:
        with self.memory_lock:
            return self.experience.clear(scope)

    def _refresh_experience_index(self) -> None:
        """经验库变更后重建 system prompt 索引段（历史首条消息一起更新）。"""
        self._experience_index = self.experience.index_text()
        if self.history and isinstance(self.history[0], SystemMessage):
            self.history[0] = SystemMessage(
                build_system_prompt(self.root, self._skill_index, self._experience_index)
            )

    def _refresh_skill_index(self) -> list:
        """重建 skill 索引段并更新当前 system prompt（含历史首条消息）。"""
        enabled = self.skill_registry.enabled_skills()
        self._skill_index = format_index(enabled)
        if self.history and isinstance(self.history[0], SystemMessage):
            self.history[0] = SystemMessage(
                build_system_prompt(self.root, self._skill_index, self._experience_index)
            )
        return enabled

    # ---------- Banner ----------

    def _print_banner(self) -> None:
        console = getattr(self.renderer, "_console", None)
        if console is None:
            print(f"AsterCLI Python v{__version__} | {self.runtime.display_name}")
            return
        console.print(f"AsterCLI Python v{__version__}", style="bold cyan")
        enabled_skills = len(self.skill_registry.enabled_skills())
        console.print(
            f"模型: {self.runtime.display_name} · 工具: {len(self.registry.names())} 个"
            f"（{self.settings.tool_mode()} 模式） · Skill: {enabled_skills} 个"
            f" · AutoDream: {'开' if self.dream_worker.enabled else '关'}"
            f" · 项目根: {self.root}",
            style="grey50",
        )
        console.print(
            "命令: /model 切换 · /tools 工具 · /plan 计划模式 · /team 多 Agent 协作 · "
            "/skill 管理 Skill · /experience 经验 · /dream 记忆整合 · /compact 压缩 · "
            "/save 与 /memory 记忆 · /index 与 /search 语义检索 · /clear 清空 · /quit 退出",
            style="grey50",
        )


# ---------- 纯函数（供 CLI 命令与测试复用） ----------


def _fmt_ts(ts: float) -> str:
    """时间戳 → 本地时间文本；0 表示从未发生。"""
    if not ts:
        return "从未"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _last_ai_text(messages: list) -> str:
    """取最后一条 AI 消息的文本（会话转写用；兼容 str / block 列表两种 content）。"""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
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
    return ""


def _tool_names(messages: list) -> list:
    """提取新增 AI 消息里的工具名（只记名字，不记参数，供 AutoDream 使用）。"""
    names = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if name:
                names.append(str(name))
    return names


def parse_save_payload(payload: str) -> tuple:
    """解析 /save 负载：``--global`` 前缀 → (scope, fact)。"""
    fact = (payload or "").strip()
    if fact.startswith("--global"):
        return "global", fact[len("--global") :].strip()
    return "project", fact


def format_memory_entries(title: str, entries: list) -> str:
    if not entries:
        return f"{title}\n  📭 没有长期记忆"
    lines = [title]
    for e in entries:
        scope = (e.metadata.get("scope") or "global").lower()
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.timestamp))
        preview = e.content if len(e.content) <= 80 else e.content[:80] + "..."
        lines.append(f"  [{e.id}] ({scope}) {ts} · {preview}")
    return "\n".join(lines)


def _parse_experience_scope(tokens: list) -> str:
    """解析 /experience 的 ``--project`` / ``--global`` 过滤参数；缺省表示全部。"""
    for token in tokens:
        value = token.lower()
        if value in ("--global", "global"):
            return "global"
        if value in ("--project", "project"):
            return "project"
    return ""


def format_experience_entries(title: str, entries: list) -> str:
    if not entries:
        return f"{title}\n  📭 没有经验记录（失败 / 拒绝 / 语法错误后会自动沉淀）"
    lines = [title]
    for e in entries:
        when = time.strftime("%Y-%m-%d", time.localtime(e.created))
        lesson = e.lesson if len(e.lesson) <= 80 else e.lesson[:80] + "..."
        lines.append(
            f"  [{e.id}] ({e.scope} · {e.category} · {e.trigger} · hits={e.hits} · {when}) {lesson}"
        )
    return "\n".join(lines)


def format_experience_detail(entry, path: str = "") -> str:
    lines = [
        f"🧠 经验 {entry.id}",
        f"   作用域: {entry.scope} · 分类: {CATEGORY_LABELS.get(entry.category, entry.category)}"
        f" · 触发: {TRIGGER_LABELS.get(entry.trigger, entry.trigger)}",
        f"   工具: {entry.tool or '（无）'} · 命中: {entry.hits} 次"
        f" · 创建: {time.strftime('%Y-%m-%d %H:%M', time.localtime(entry.created))}",
        f"   教训: {entry.lesson}",
        f"   正确做法: {entry.correct_action or '（未提炼）'}",
    ]
    if entry.evidence:
        lines.append(f"   证据: {entry.evidence}")
    if path:
        lines.append(f"   文件: {path}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
