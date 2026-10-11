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
import json
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from .. import __version__
from ..agent import AgentBudget
from ..agent.graph import build_react_graph
from ..config import Settings
from ..llm.factory import (
    ProviderRuntime,
    available_providers,
    create_client,
    create_from_settings,
)
from ..memory import (
    ConversationHistoryCompactor,
    LongTermMemory,
    auto_compact_trigger_tokens,
    build_memory_context,
    estimate_messages_tokens,
    memory_context_tokens,
    normalize_project_key,
)
from ..policy import PathGuard
from ..prompt import build_system_prompt
from ..tools import ToolRegistry
from ..tools.builtin import register_file_tools, register_shell_tools
from .renderer import ConsoleRenderer


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

    console = Console(highlight=False, markup=False)
    renderer = ConsoleRenderer(console)
    app = ReplApp(settings, runtime, renderer, Path.cwd())

    one_shot = " ".join(args.prompt).strip()
    if one_shot:
        try:
            asyncio.run(app.run_turn(one_shot))
        except (KeyboardInterrupt, asyncio.CancelledError):
            renderer.warn("⏹️ 已取消当前任务。")
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
        self.registry = ToolRegistry(settings=settings, approval=_make_approval(renderer))
        register_file_tools(self.registry, self.path_guard)
        register_shell_tools(self.registry, self.path_guard)
        self.runtime = runtime
        self.long_term = LongTermMemory()
        self.project_key = normalize_project_key(self.root)
        self.history: list = [SystemMessage(build_system_prompt(self.root))]
        self._rebuild_graph()

    def _rebuild_graph(self) -> None:
        """按当前 runtime 重建压缩器与 ReAct 图（/model 切换后复用）。"""
        self.compactor = ConversationHistoryCompactor(self.runtime.client)
        self.trigger_tokens = auto_compact_trigger_tokens(self.runtime.context_window)
        self.graph = build_react_graph(
            self.runtime,
            self.registry,
            self.renderer,
            tool_mode=self.settings.tool_mode(),
            compactor=self.compactor,
            trigger_tokens=self.trigger_tokens,
        )

    # ---------- 主循环 ----------

    def run(self) -> int:
        self._print_banner()
        from prompt_toolkit import PromptSession

        session: PromptSession = PromptSession()
        while True:
            try:
                text = session.prompt("* ")
            except (KeyboardInterrupt, EOFError):
                break
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.startswith("/"):
                if self._handle_command(stripped):
                    break
                continue
            try:
                asyncio.run(self.run_turn(stripped))
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.renderer.warn("⏹️ 已取消当前任务。")
        return 0

    # ---------- 单轮执行 ----------

    async def run_turn(self, text: str) -> None:
        snapshot = list(self.history)
        try:
            # 检索相关长期记忆，前置注入本轮 user message（不进 system prompt，
            # 保持前缀稳定以友好 prompt cache）
            memory_block = build_memory_context(
                self.long_term,
                text,
                memory_context_tokens(self.runtime.context_window),
                project_key=self.project_key,
            )
            enriched = f"{memory_block}\n\n---\n\n{text}" if memory_block else text
            self.history.append(HumanMessage(enriched))
            budget = AgentBudget(
                self.settings.react_token_budget(),
                self.settings.react_stagnation_window(),
                self.settings.react_hard_max_iterations(),
            )
            result = await self.graph.ainvoke(
                {"messages": list(self.history), "budget": budget, "halt_reason": None}
            )
            self.history[:] = result["messages"]
        except (KeyboardInterrupt, asyncio.CancelledError):
            # 取消当前任务：回滚本轮半途历史，不污染下一轮
            self.history[:] = snapshot
            raise
        self.renderer.usage(budget)

    # ---------- 斜杠命令（未知命令在 CLI 层直接报错，不回退给 Agent） ----------

    def _handle_command(self, text: str) -> bool:
        """返回 True 表示退出 REPL。"""
        parts = text.split()
        cmd = parts[0].lower()
        payload = text[len(parts[0]) :].strip()
        if cmd in ("/quit", "/exit"):
            return True
        if cmd == "/clear":
            self.history = [SystemMessage(build_system_prompt(self.root))]
            self.renderer.info("已清空会话历史（system prompt 已重建，长期记忆保留）。")
        elif cmd == "/compact":
            self._compact_history()
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
        elif cmd == "/help":
            self.renderer.info(
                "命令: /quit /exit 退出 · /clear 清空历史 · /compact 压缩历史 · "
                "/model [provider] 切换 · /tools 工具列表 · "
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
            elif self.long_term.delete(entry_id):
                self.renderer.info(f"🗑️ 已删除长期记忆: {entry_id}")
            else:
                self.renderer.warn(f"📭 未找到长期记忆: {entry_id}")
        elif sub == "clear":
            self.long_term.clear()
            self.renderer.info("🧹 长期记忆已清空")
        else:
            self.renderer.warn(f"未知的 /memory 子命令: {sub}")

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

    # ---------- Banner ----------

    def _print_banner(self) -> None:
        console = getattr(self.renderer, "_console", None)
        if console is None:
            print(f"AsterCLI Python v{__version__} | {self.runtime.display_name}")
            return
        console.print(f"AsterCLI Python v{__version__}", style="bold cyan")
        console.print(
            f"模型: {self.runtime.display_name} · 工具: {len(self.registry.names())} 个"
            f"（{self.settings.tool_mode()} 模式） · 项目根: {self.root}",
            style="grey50",
        )
        console.print(
            "命令: /model 切换 · /tools 工具 · /compact 压缩 · /save 与 /memory 记忆 · "
            "/clear 清空 · /quit 退出",
            style="grey50",
        )


# ---------- 纯函数（供 CLI 命令与测试复用） ----------


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


def _make_approval(renderer):
    async def approve(tool_def, args: dict) -> bool:
        from rich.prompt import Confirm

        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 160:
            preview = preview[:160] + "…"
        return await asyncio.to_thread(
            Confirm.ask, f"允许执行 {tool_def.name} {preview} ?", default=False
        )

    return approve


if __name__ == "__main__":
    raise SystemExit(main())
