"""LangGraph 状态定义。"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # 消息列表：add_messages 累积（agent → tools → agent 循环追加；
    # 同时支持 RemoveMessage 删除——历史压缩需要替换早期消息）
    messages: Annotated[list[AnyMessage], add_messages]
    # AgentBudget 对象：每轮新建、原地更新，无 reducer（整值覆盖语义）
    budget: object
    # 兜底退出原因；非 None 时路由直接结束
    halt_reason: Optional[str]
