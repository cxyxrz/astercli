"""Provider 元数据表。

对齐 Java 侧两处语义：
- LlmClientFactory.normalizeProvider 的别名归一化（moonshot→kimi、iflytek→xfyun 等）
- 各 *Client 类的默认 model / base URL / contextWindow / 能力标记
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    api_key_envs: Tuple[str, ...]
    model_envs: Tuple[str, ...]
    base_url_envs: Tuple[str, ...]
    default_model: str
    context_window: int
    supports_tools: bool = True
    supports_image_input: bool = True
    lora_id_envs: Tuple[str, ...] = ()


PROVIDER_SPECS: dict = {
    "glm": ProviderSpec(
        key="glm",
        api_key_envs=("GLM_API_KEY",),
        model_envs=("GLM_MODEL",),
        base_url_envs=("GLM_BASE_URL",),
        default_model="glm-5.1",
        context_window=200_000,
    ),
    "deepseek": ProviderSpec(
        key="deepseek",
        api_key_envs=("DEEPSEEK_API_KEY",),
        model_envs=("DEEPSEEK_MODEL",),
        base_url_envs=("DEEPSEEK_BASE_URL",),
        default_model="deepseek-v4-flash",
        context_window=1_000_000,
        # DeepSeek 按文本 provider 处理：不支持图片输入，历史图片需降级为文本提示
        supports_image_input=False,
    ),
    "step": ProviderSpec(
        key="step",
        api_key_envs=("STEP_API_KEY",),
        model_envs=("STEP_MODEL",),
        base_url_envs=("STEP_BASE_URL",),
        default_model="step-3.5-flash",
        context_window=256_000,
    ),
    "kimi": ProviderSpec(
        key="kimi",
        api_key_envs=("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        model_envs=("KIMI_MODEL", "MOONSHOT_MODEL"),
        base_url_envs=("KIMI_BASE_URL", "MOONSHOT_BASE_URL"),
        default_model="kimi-k2.6",
        context_window=256_000,
    ),
    "freellmapi": ProviderSpec(
        key="freellmapi",
        api_key_envs=("FREELLMAPI_API_KEY",),
        model_envs=("FREELLMAPI_MODEL",),
        base_url_envs=("FREELLMAPI_BASE_URL",),
        default_model="auto",
        context_window=128_000,
    ),
    "xfyun": ProviderSpec(
        key="xfyun",
        api_key_envs=("XFYUN_MAAS_API_KEY", "XFYUN_API_KEY"),
        model_envs=("XFYUN_MAAS_MODEL", "XFYUN_MODEL"),
        base_url_envs=("XFYUN_MAAS_BASE_URL", "XFYUN_BASE_URL"),
        default_model="Qwen3.6-35B-A3B",
        context_window=128_000,
        # 讯飞 MaaS 按纯对话请求处理，不向上游发送内置工具列表
        supports_tools=False,
        lora_id_envs=("XFYUN_MAAS_LORA_ID", "XFYUN_LORA_ID"),
    ),
    "agnes": ProviderSpec(
        key="agnes",
        api_key_envs=("AGNES_API_KEY",),
        model_envs=("AGNES_MODEL",),
        base_url_envs=("AGNES_BASE_URL",),
        default_model="agnes-2.0-flash",
        context_window=1_000_000,
    ),
}

FALLBACK_ORDER = ("glm", "deepseek", "step", "kimi", "freellmapi", "xfyun", "agnes")

# 对齐 Java normalizeProvider 的别名表（未知名称原样返回，由工厂校验是否在表中）
_PROVIDER_ALIASES = {
    "stepfun": "step",
    "step-fun": "step",
    "moonshot": "kimi",
    "moonshotai": "kimi",
    "moonshot-ai": "kimi",
    "free-llm-api": "freellmapi",
    "free_llm_api": "freellmapi",
    "freellm": "freellmapi",
    "free-llm": "freellmapi",
    "xfyun-maas": "xfyun",
    "xfyun_maas": "xfyun",
    "iflytek": "xfyun",
    "iflytek-maas": "xfyun",
    "iflytek_maas": "xfyun",
    "maas": "xfyun",
    "agnes-ai": "agnes",
    "agnes_ai": "agnes",
    "sapiens": "agnes",
    "sapiens-ai": "agnes",
    "sapiens_ai": "agnes",
}


def normalize_provider(raw: Optional[str]) -> Optional[str]:
    """别名归一化；未知名称原样返回，由调用方校验。"""
    if raw is None:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    return _PROVIDER_ALIASES.get(key, key)


def default_base_url(provider_key: str, model: str) -> str:
    """默认 Base URL（可用 *_BASE_URL 环境变量覆盖）。

    GLM 按 Java GLMClient.selectApiUrl 的语义分流：
    glm-5v* 走多模态端点，其余走 coding 端点。
    """
    if provider_key == "glm":
        if model.strip().lower().startswith("glm-5v"):
            return "https://open.bigmodel.cn/api/paas/v4"
        return "https://open.bigmodel.cn/api/coding/paas/v4"
    return {
        "deepseek": "https://api.deepseek.com/v1",
        "step": "https://api.stepfun.com/v1",
        "kimi": "https://api.moonshot.ai/v1",
        "freellmapi": "http://localhost:5173/v1",
        "xfyun": "https://maas-api.cn-huabei-1.xf-yun.com/v2",
        "agnes": "https://apihub.agnes-ai.com/v1",
    }[provider_key]
