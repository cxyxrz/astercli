"""LangGraph ReAct 主循环。

对齐 Java Agent.java 的循环语义：
- 主退出条件 = LLM 自己决定（返回 content 不再调工具）；budget 仅兜底
- 每轮 LLM 调用前检查 budget（停滞 / token / 硬轮数），命中即 halt
- 每轮 LLM 调用前评估历史压缩（maybeCompactHistory）：达到触发阈值时把早期
  轮次摘要成"[已压缩的历史对话摘要]"，通过 RemoveMessage(REMOVE_ALL_MESSAGES)
  全量替换通道重建历史
- 工具结果按 ToolMessage 回灌继续循环

渲染器（renderer）为鸭子类型接口：begin_thinking / stream_chunk / end_thinking /
take_reasoning / tool_calls / tool_results / halt / info；由 cli 层注入实现。
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command, interrupt

from ..config import TOOL_MODE_SLIM
from ..llm.factory import ProviderRuntime
from ..memory.tokens import estimate_messages_tokens
from ..tools.registry import ToolRegistry
from .budget import AgentBudget, ExitReason
from .state import AgentState


def make_checkpointer():
    """MemorySaver + pickle 兜底序列化。

    state 里的 AgentBudget 是自定义类，JsonPlusSerializer 的 msgpack 编码
    不认识（"Type is not msgpack serializable"）→ 打开官方 pickle_fallback
    开关兜底。MemorySaver 纯内存不落盘，pickle.loads 无持久化攻击面。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))


def build_react_graph(
    runtime: ProviderRuntime,
    registry: ToolRegistry,
    renderer,
    *,
    tool_mode: Optional[str] = None,
    compactor=None,
    trigger_tokens: Optional[int] = None,
    approval=None,
    checkpointer=None,
):
    """编译 ReAct 图：agent →(有 tool_calls?)→ tools → agent → ... → END。

    :param compactor: ConversationHistoryCompactor；None 时关闭自动压缩
    :param trigger_tokens: 自动压缩触发阈值（context window - 摘要/安全预留）
    :param approval: ApprovalSession（HITL 审批）；None 时禁用审批（对齐 Java
        "HITL 未启用行为与父类完全相同"）。审批经 interrupt() 挂起，需要
        checkpointer 提供 resume 能力；未提供 checkpointer 时审批自动禁用。
    :param checkpointer: LangGraph checkpointer（如 MemorySaver）
    """
    if tool_mode is None:
        tool_mode = TOOL_MODE_SLIM
    if approval is not None and checkpointer is None:
        # 审批需要 checkpointer 提供 resume 能力；缺失时整体禁用（graph 层
        # 不拦截 + registry 层直接执行），对齐 Java "HITL 未启用行为与父类相同"
        approval = None
        registry.set_hitl_active(False)
    graph = StateGraph(AgentState)
    graph.add_node(
        "agent",
        make_agent_node(
            runtime, registry, renderer, tool_mode, compactor=compactor, trigger_tokens=trigger_tokens
        ),
    )
    graph.add_node("tools", make_tools_node(registry, renderer, approval=approval))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


async def ainvoke_with_hitl(graph, state: dict, config: dict, on_interrupt=None) -> dict:
    """ainvoke + HITL 挂起恢复循环。

    图因审批 interrupt() 挂起时，把 ``__interrupt__`` 交给 on_interrupt 收集
    用户决策，再以 ``Command(resume=...)`` 恢复，直到拿到最终状态。
    on_interrupt 为 None 时对挂起调用保守拒绝（fail-safe）。
    """
    result = await graph.ainvoke(state, config)
    while isinstance(result, dict) and result.get("__interrupt__"):
        interrupts = result["__interrupt__"]
        if on_interrupt is not None:
            response = await on_interrupt(interrupts)
        else:
            payload = getattr(interrupts[0], "value", None) if interrupts else {}
            calls = (payload or {}).get("calls", [])
            response = {c.get("id", ""): {"action": "reject", "reason": "无审批处理器，保守拒绝"} for c in calls}
        result = await graph.ainvoke(Command(resume=response), config)
    return result


def route_after_agent(state: AgentState):
    if state.get("halt_reason"):
        return END
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, AIMessage) and extract_tool_calls(last):
        return "tools"
    return END


def make_agent_node(
    runtime: ProviderRuntime,
    registry: ToolRegistry,
    renderer,
    tool_mode: str,
    *,
    compactor=None,
    trigger_tokens: Optional[int] = None,
):
    async def agent_node(state: AgentState):
        budget: AgentBudget = state["budget"]
        exit_reason = budget.check()
        if exit_reason is not ExitReason.WITHIN_BUDGET:
            message = "❌ " + budget.describe_exit(exit_reason)
            renderer.halt(message)
            return {"halt_reason": message}

        budget.begin_iteration()

        # 调 LLM 前评估历史压缩（对齐 Java maybeCompactHistory）。
        # add_messages 对同 id 消息是"替换原位置 + 抵消删除"语义，逐 id
        # RemoveMessage 会让含同对象的重建尾段错乱；走 REMOVE_ALL_MESSAGES
        # 全量替换通道，reducer 直接返回 right[idx+1:] = rebuilt + 本轮新消息。
        messages = state["messages"]
        prefix: list = []
        if compactor is not None and trigger_tokens:
            rebuilt = await compactor.compact_if_needed(messages, trigger_tokens)
            if rebuilt is not None:
                prefix = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + rebuilt
                messages = rebuilt
                renderer.info(
                    "已自动压缩历史对话: "
                    f"{estimate_messages_tokens(state['messages'])} → "
                    f"{estimate_messages_tokens(rebuilt)} tokens"
                )

        # LSP 诊断注入（P8，对齐 Java injectPendingLspDiagnostics：每次调 LLM
        # 前把 write_file 触发的 pending 诊断作为 user message 注入 + renderer
        # 显示；registry 未挂 LspManager / 无 pending 时为空报告 no-op）。
        # 注入点在历史压缩之后：注入消息永远属尾部轮次不会被压进摘要，
        # 也不会随 rebuilt 全量替换通道重复入列（与 Java 行为等价）
        injected: list = []
        report = registry.flush_pending_lsp_diagnostics()
        if not report.is_empty():
            injected = [HumanMessage(content=report.prompt_text)]
            messages = messages + injected
            renderer.info(report.display_text)

        schemas = registry.schemas(tool_mode) if runtime.supports_tools else None
        llm = runtime.client.bind_tools(schemas) if schemas else runtime.client

        final = None
        renderer.begin_thinking()
        try:
            async for chunk in llm.astream(messages):
                if not isinstance(chunk, AIMessage):
                    continue
                renderer.stream_chunk(chunk)
                if final is None:
                    final = chunk
                elif isinstance(final, AIMessageChunk) and isinstance(chunk, AIMessageChunk):
                    final = final + chunk
                else:
                    final = chunk
        finally:
            renderer.end_thinking()

        if final is None:
            message = "LLM 未返回内容"
            renderer.halt(message)
            return {"halt_reason": message}

        # reasoning 完整性兜底：以渲染端累积为准回写 additional_kwargs，
        # 保证 DeepSeek/Kimi 的 reasoning_content 随历史回传不缺斤短两
        full_reasoning = renderer.take_reasoning()
        if full_reasoning and not final.additional_kwargs.get("reasoning_content"):
            final.additional_kwargs["reasoning_content"] = full_reasoning

        usage = getattr(final, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        budget.record_tokens(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            details.get("cached_tokens", 0),
        )

        tool_calls = extract_tool_calls(final)
        if tool_calls:
            renderer.tool_calls(tool_calls)
            budget.record_tool_calls(tool_calls)
        # injected 是本轮 LLM 输入的一部分，同样要进 state 历史（对齐 Java
        # conversationHistory.add(user)：诊断消息跨轮次保留）
        return {"messages": prefix + injected + [final]}

    return agent_node


def make_tools_node(registry: ToolRegistry, renderer, approval=None):
    async def tools_node(state: AgentState):
        last = state["messages"][-1]
        tool_calls = extract_tool_calls(last)
        invalid = getattr(last, "invalid_tool_calls", None) or []
        # HITL 审批（对齐 Java HitlToolRegistry：执行前拦截，拦截顺序
        # Hitl → Registry → Guard）：挂起重入时节点从头重跑，interrupt()
        # 返回 resume 值，之前的执行路径无副作用
        decisions: dict = {}
        if approval is not None and tool_calls:
            pending = approval.pending_calls(tool_calls, registry)
            if pending:
                response = interrupt(approval.build_payload(pending))
                decisions = approval.apply_response(pending, response)
            # 会话放行名单（approve_all）内的调用无需再挂起，但 registry 层
            # 仍要求危险调用携带决策，这里补齐放行决策
            for call in tool_calls:
                call_id = call.get("id", "")
                if call_id not in decisions and approval.is_approved_all(call.get("name")):
                    decisions[call_id] = {"action": "approve"}
        results = await registry.execute_all(tool_calls, decisions=decisions)
        # 参数解析失败的调用也要回灌，否则下一轮 LLM 会困惑为什么没有工具结果
        for bad in invalid:
            try:
                id_ = bad.get("id") or ""
            except AttributeError:
                id_ = ""
            results.append(
                ToolMessage(
                    content=f"工具调用参数解析失败: {bad}",
                    tool_call_id=id_,
                    name=str(bad.get("name", "?")) if isinstance(bad, dict) else "?",
                )
            )
        renderer.tool_results(tool_calls, results)
        return {"messages": results}

    return tools_node


def extract_tool_calls(message: AIMessage) -> list:
    """AIMessage / AIMessageChunk 通用 tool_calls 提取（args 统一为 dict）。"""
    calls = list(getattr(message, "tool_calls", None) or [])
    if calls:
        return calls
    # 流式聚合的 AIMessageChunk：从 tool_call_chunks 兜底解析
    chunks = getattr(message, "tool_call_chunks", None) or []
    parsed: list = []
    for i, chunk in enumerate(chunks):
        args = chunk.get("args") or ""
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": str(args)}
        parsed.append(
            {
                "name": chunk.get("name") or "",
                "args": args,
                "id": chunk.get("id") or f"call_{i}",
                "type": "tool_call",
            }
        )
    return parsed
