"""OpenAI-compatible 客户端基类（所有 provider 共用）。

背景：langchain-openai 1.x 明确不提取第三方 provider 的非标准字段
（reasoning_content / reasoning），历史消息回传时也会丢弃。AsterCLI 需要：

1. 【输出】流式 SSE 的 reasoning_content/reasoning delta 落到
   AIMessageChunk.additional_kwargs["reasoning_content"]，供渲染器灰字展示；
2. 【输入】DeepSeek / Kimi thinking 模式下，assistant tool-call 消息的
   reasoning_content 必须随下一轮请求历史带回（对齐 Java 侧行为约束）；
3. 【输入】DeepSeek 不支持图片输入，历史图片 content 统一降级为文本提示。

钩子（langchain-openai 1.6 实测）：
- BaseChatOpenAI._convert_chunk_to_generation_chunk：拿到的是原始 chunk dict，
  可从 choices[0].delta 提取 reasoning 字段 —— 输出方向；
- BaseChatOpenAI._get_request_payload：请求组装出口，payload["messages"]
  与输入消息一一对应 —— 输入方向。
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI

REASONING_KEYS = ("reasoning_content", "reasoning")


class OpenAiCompatClient(ChatOpenAI):
    """所有 OpenAI-compatible provider 的统一基类。

    ClassVar 开关由具体 provider 子类/工厂设置：
    - reasoning_roundtrip: 历史 assistant 消息的 reasoning_content 是否随请求回传
    - degrade_image_history: 历史图片 content 是否降级为文本提示
    """

    reasoning_roundtrip: ClassVar[bool] = False
    degrade_image_history: ClassVar[bool] = False

    # ---------- 输出方向：流式 reasoning 提取 ----------

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Any:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return generation
        try:
            choices = chunk.get("choices") or []
            if not choices:
                return generation
            delta = choices[0].get("delta") or {}
            reasoning_delta = None
            for key in REASONING_KEYS:
                value = delta.get(key)
                if isinstance(value, str) and value:
                    reasoning_delta = value
                    break
            if reasoning_delta:
                msg = generation.message
                merged = (msg.additional_kwargs.get("reasoning_content") or "") + reasoning_delta
                msg.additional_kwargs["reasoning_content"] = merged
        except Exception:
            # 定制提取失败不影响标准 chunk 解析
            pass
        return generation

    # ---------- 输入方向：reasoning 回传 + 图片降级 ----------

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list] = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            self._postprocess_request_payload(payload, input_)
        except Exception:
            # 定制失败退化为标准 OpenAI 请求，不阻断主路径
            pass
        return payload

    def _postprocess_request_payload(self, payload: dict, input_: Any) -> None:
        reqs = payload.get("messages")
        if not isinstance(reqs, list):
            return
        if self.reasoning_roundtrip or self.degrade_image_history:
            try:
                original = self._convert_input(input_).to_messages()
            except Exception:
                return
            if len(original) != len(reqs):
                return
            for req, msg in zip(reqs, original):
                if not isinstance(req, dict):
                    continue
                if (
                    self.reasoning_roundtrip
                    and req.get("role") == "assistant"
                    and isinstance(msg, AIMessage)
                ):
                    reasoning = msg.additional_kwargs.get("reasoning_content")
                    if reasoning and "reasoning_content" not in req:
                        req["reasoning_content"] = reasoning
                if self.degrade_image_history and isinstance(content := req.get("content"), list):
                    req["content"] = _degrade_image_blocks(content)


def _degrade_image_blocks(blocks: list) -> list:
    """把 content blocks 里的图片替换为省略提示（对齐 Java Message.withoutImageContent）。"""
    has_image = any(isinstance(b, dict) and b.get("type") == "image_url" for b in blocks)
    if not has_image:
        return blocks
    kept = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "image_url")]
    omitted = len(blocks) - len(kept)
    kept.append(
        {
            "type": "text",
            "text": f"[历史图片附件已省略 {omitted} 张；如需重新查看，请使用相关工具结果。]",
        }
    )
    return kept
