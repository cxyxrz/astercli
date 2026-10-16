"""客户端工厂（对齐 Java LlmClientFactory：别名归一化 + Key 回退链）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ..config import Settings
from .deepseek import DeepSeekClient
from .openai_compat import OpenAiCompatClient
from .providers import (
    FALLBACK_ORDER,
    PROVIDER_SPECS,
    ProviderSpec,
    default_base_url,
    normalize_provider,
)


class KimiClient(OpenAiCompatClient):
    """Kimi thinking 模式同样要求 reasoning_content 随历史回传（对齐 Java 侧约束）。"""

    reasoning_roundtrip = True


@dataclass
class ProviderRuntime:
    client: Any
    provider: str
    model: str
    spec: ProviderSpec

    @property
    def supports_tools(self) -> bool:
        return self.spec.supports_tools

    @property
    def supports_image_input(self) -> bool:
        return self.spec.supports_image_input

    @property
    def context_window(self) -> int:
        return self.spec.context_window

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"


# 对齐 Java 侧 LLM HTTP 超时：connect=60s / read=300s（SSE 两次 read 间隔）/ call=600s
_LLM_TIMEOUT = httpx.Timeout(600.0, connect=60.0, read=300.0, write=60.0)

# langchain-openai 的相邻 chunk 间隔计时器默认 100s；thinking 模型长推理期间
# 可能长时间无 content chunk 被误杀（/team reviewer 实测触发过），放宽到 300s
# 与 HTTP read timeout 对齐
_STREAM_CHUNK_TIMEOUT = 300.0


def create_client(provider: Optional[str], settings: Settings) -> Optional[ProviderRuntime]:
    """按 provider 名创建客户端；未配置 Key 或未知 provider 返回 None（对齐 Java 语义）。"""
    key = normalize_provider(provider)
    if key is None or key not in PROVIDER_SPECS:
        return None
    spec = PROVIDER_SPECS[key]
    api_key = settings.api_key(spec)
    if not api_key:
        return None
    model = settings.model(spec)
    base_url = settings.base_url(spec) or default_base_url(key, model)
    common = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=_LLM_TIMEOUT,
        stream_chunk_timeout=_STREAM_CHUNK_TIMEOUT,
    )

    if key == "deepseek":
        client: Any = DeepSeekClient(**common)
    elif key == "kimi":
        client = KimiClient(**common)
    else:
        extra: dict = {}
        if key == "xfyun":
            lora_id = settings.lora_id(spec)
            if lora_id:
                extra["default_headers"] = {"lora_id": lora_id}
        client = OpenAiCompatClient(**common, **extra)
    return ProviderRuntime(client=client, provider=key, model=model, spec=spec)


def create_from_settings(settings: Settings) -> Optional[ProviderRuntime]:
    """默认 provider 优先，其次按 glm→deepseek→step→kimi→freellmapi→xfyun→agnes 回退。"""
    runtime = create_client(settings.default_provider(), settings)
    if runtime is not None:
        return runtime
    for provider in FALLBACK_ORDER:
        runtime = create_client(provider, settings)
        if runtime is not None:
            return runtime
    return None


def available_providers(settings: Settings) -> list:
    """已配置 API Key 的 provider 列表（按 FALLBACK_ORDER）。"""
    return [p for p in FALLBACK_ORDER if settings.api_key(PROVIDER_SPECS[p])]
