"""工具注册表（对齐 Java tool.ToolRegistry 的三块核心语义）：

1. 精简 Schema 双模式：默认 slim——description 追加参数文字说明、
   parameters 只留 {"type":"object"} + required；ASTERCLI_TOOL_MODE=schema 回退完整 Schema。
2. 并发执行：最多 4 并发、结果保持原始顺序、单工具超时、异常转失败结果占位。
3. 统一输出预算：任何工具结果超过 max_chars 在出口截断并追加缩小范围提示
   （各工具自身细粒度预算优先生效，这里只做最后兜底）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Tuple, Union

from langchain_core.messages import ToolMessage

from ..config import TOOL_MODE_SCHEMA, TOOL_MODE_SLIM, Settings
from ..policy import PolicyException, record_audit


# ---------- 参数与 Schema（对齐 Java createParameters / Param record） ----------


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    required: bool = False
    enum_values: Tuple[str, ...] = ()
    item_type: Optional[str] = None
    fields: Tuple[str, ...] = ()


def create_parameters(*params: Param) -> dict:
    """构建完整 JSON Schema（schema 模式用；slim 模式只提取 required）。"""
    properties: dict = {}
    required: list = []
    for p in params:
        prop: dict = {"type": p.type, "description": p.description}
        if p.enum_values:
            prop["enum"] = list(p.enum_values)
        if p.type == "array" and p.item_type:
            prop["items"] = {"type": p.item_type}
        if p.type == "object" and p.fields:
            prop["properties"] = {f: {} for f in p.fields}
        properties[p.name] = prop
        if p.required:
            required.append(p.name)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------- 工具定义 ----------


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    func: Callable[[dict], Union[str, Awaitable[str]]]
    dangerous: bool = False


ApprovalHook = Callable[[ToolDef, dict], Awaitable[bool]]


class ToolRegistry:
    def __init__(
        self,
        *,
        settings: Settings,
        approval: Optional[ApprovalHook] = None,
        max_concurrency: int = 4,
    ) -> None:
        self._tools: dict = {}
        self._settings = settings
        self._approval = approval
        self._max_concurrency = max_concurrency
        self._max_output_chars = settings.tool_output_max_chars()
        self._default_timeout = settings.tool_timeout_seconds()

    # ---------- 注册 ----------

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable[[dict], Union[str, Awaitable[str]]],
        *,
        dangerous: bool = False,
    ) -> None:
        self._tools[name] = ToolDef(name, description, parameters, func, dangerous)

    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def names(self) -> list:
        return list(self._tools)

    # ---------- Schema 出口 ----------

    def schemas(self, mode: Optional[str] = None) -> list:
        """OpenAI function 格式工具定义，按 mode 选择 slim/schema。"""
        if mode is None:
            mode = self._settings.tool_mode()
        out: list = []
        for tool in self._tools.values():
            if mode == TOOL_MODE_SCHEMA:
                function = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            else:
                function = compact_tool(tool)
            out.append({"type": "function", "function": function})
        return out

    # ---------- 执行（对齐 Java executeTools：4 并发 / 保序 / 超时） ----------

    async def execute_all(self, tool_calls: list) -> list:
        """并发执行 tool_calls（[{"name","args","id"}]），返回顺序一致的 ToolMessage 列表。"""
        if not tool_calls:
            return []
        semaphore = asyncio.Semaphore(self._max_concurrency)
        tasks = [self._run_one(c, semaphore) for c in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final: list = []
        for call, res in zip(tool_calls, results):
            name = call.get("name", "?")
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, BaseException):
                final.append(
                    ToolMessage(
                        content=f"工具执行失败: {res}",
                        tool_call_id=call.get("id") or "",
                        name=name,
                    )
                )
            else:
                final.append(res)
        return final

    async def _run_one(self, call: dict, semaphore: asyncio.Semaphore) -> ToolMessage:
        name = call.get("name", "?")
        args = call.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        tool = self._tools.get(name)
        if tool is None:
            return ToolMessage(
                content=f"未知工具: {name}", tool_call_id=call.get("id") or "", name=name
            )

        try:
            content = await asyncio.wait_for(
                self._invoke(tool, args, semaphore), timeout=self._default_timeout
            )
        except asyncio.TimeoutError:
            content = f"工具执行超时（>{self._default_timeout}s）: {name}"
        content = truncate_output(content, self._max_output_chars)
        return ToolMessage(content=content, tool_call_id=call.get("id") or "", name=name)

    async def _invoke(self, tool: ToolDef, args: dict, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            if tool.dangerous:
                if self._approval is not None and not await self._approval(tool, args):
                    record_audit(tool.name, json.dumps(args, ensure_ascii=False), allowed=False)
                    return "[HITL] 用户拒绝了本次工具调用，可询问用户或调整方案。"
                record_audit(tool.name, json.dumps(args, ensure_ascii=False), allowed=True)
            try:
                result = tool.func(args)
                if inspect.isawaitable(result):
                    result = await result
                return str(result)
            except PolicyException as e:
                return f"🛡️ {e}"
            except Exception as e:  # noqa: BLE001 —— 对齐 Java：异常转失败文本回给 LLM
                return f"工具执行失败: {e}"


def truncate_output(text: str, max_chars: int) -> str:
    """统一工具输出预算：超过 max_chars 截断并追加缩小范围提示。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    hint = (
        f"\n\n⚠️ 输出已截断（{len(text)} > {max_chars} 字符），"
        "请缩小读取范围或用 offset/limit、max_chars 分批获取。"
    )
    return text[:max_chars] + hint


# ---------- 精简 Schema（对齐 Java compactTool / buildParameterSummary） ----------


def compact_tool(tool: ToolDef) -> dict:
    """description 追加参数文字说明，parameters 只留 {"type":"object"} + required。"""
    slim_parameters: dict = {"type": "object"}
    required = tool.parameters.get("required")
    if isinstance(required, list) and required:
        slim_parameters["required"] = list(required)
    description = tool.description or ""
    summary = build_parameter_summary(tool.parameters)
    if summary:
        description = description + "\n\n参数:\n" + summary.strip()
    return {"name": tool.name, "description": description, "parameters": slim_parameters}


def build_parameter_summary(schema: Optional[dict]) -> str:
    if not isinstance(schema, dict):
        return ""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return ""
    required_names = set(schema.get("required") or [])
    return "".join(
        format_parameter_line(name, prop, required_names) for name, prop in properties.items()
    )


def format_parameter_line(name: str, prop: dict, required_names: set) -> str:
    line = f"- {name} ({property_type(prop)}"
    enum_node = prop.get("enum")
    if isinstance(enum_node, list) and enum_node:
        line += ", enum: " + " | ".join(str(v) for v in enum_node)
    if prop.get("type") == "object":
        sub = prop.get("properties")
        if isinstance(sub, dict) and sub:
            line += ", fields: " + ", ".join(sub)
    line += ", " + ("required" if name in required_names else "optional") + ")"
    description = prop.get("description") or ""
    if description.strip():
        line += ": " + description.strip()
    return line + "\n"


def property_type(prop: dict) -> str:
    ptype = prop.get("type") or ""
    if not ptype:
        return "any"
    if ptype == "array":
        items = prop.get("items") or {}
        item_type = items.get("type") or ""
        return f"array<{item_type}>" if item_type else "array"
    return ptype
