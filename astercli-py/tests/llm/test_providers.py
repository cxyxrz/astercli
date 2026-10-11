"""Provider 元数据 + 工厂测试：别名归一化 / 默认 URL / Key 回退链 / 能力标记。"""

from __future__ import annotations

import httpx
import pytest

from astercli.config import Settings
from astercli.llm.deepseek import DeepSeekClient
from astercli.llm.factory import (
    KimiClient,
    ProviderRuntime,
    available_providers,
    create_client,
    create_from_settings,
)
from astercli.llm.openai_compat import OpenAiCompatClient
from astercli.llm.providers import (
    PROVIDER_SPECS,
    default_base_url,
    normalize_provider,
)


def settings(env=None):
    return Settings(env=env, load_files=False)


# ---------- 别名归一化 ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stepfun", "step"),
        ("step-fun", "step"),
        ("moonshot", "kimi"),
        ("moonshotai", "kimi"),
        ("moonshot-ai", "kimi"),
        ("Moonshot", "kimi"),
        ("free-llm-api", "freellmapi"),
        ("freellm", "freellmapi"),
        ("xfyun-maas", "xfyun"),
        ("iflytek", "xfyun"),
        ("iflytek_maas", "xfyun"),
        ("maas", "xfyun"),
        ("agnes-ai", "agnes"),
        ("sapiens", "agnes"),
        ("GLM", "glm"),
        ("deepseek", "deepseek"),
        ("unknown-provider", "unknown-provider"),
    ],
)
def test_normalize_provider_aliases(raw, expected):
    assert normalize_provider(raw) == expected


def test_normalize_provider_blank():
    assert normalize_provider(None) is None
    assert normalize_provider("") is None
    assert normalize_provider("   ") is None


# ---------- 默认 Base URL ----------


def test_glm_coding_url_by_default():
    assert default_base_url("glm", "glm-5.1") == "https://open.bigmodel.cn/api/coding/paas/v4"


def test_glm_multimodal_url_for_5v_models():
    assert default_base_url("glm", "glm-5v-plus") == "https://open.bigmodel.cn/api/paas/v4"
    assert default_base_url("glm", "GLM-5V") == "https://open.bigmodel.cn/api/paas/v4"


@pytest.mark.parametrize(
    ("provider", "url"),
    [
        ("deepseek", "https://api.deepseek.com/v1"),
        ("step", "https://api.stepfun.com/v1"),
        ("kimi", "https://api.moonshot.ai/v1"),
        ("freellmapi", "http://localhost:5173/v1"),
        ("xfyun", "https://maas-api.cn-huabei-1.xf-yun.com/v2"),
        ("agnes", "https://apihub.agnes-ai.com/v1"),
    ],
)
def test_provider_default_base_urls(provider, url):
    assert default_base_url(provider, "any-model") == url


# ---------- 工厂 ----------


def test_factory_deepseek_uses_custom_client():
    s = settings({"DEEPSEEK_API_KEY": "sk-x"})
    runtime = create_client("deepseek", s)
    assert isinstance(runtime.client, DeepSeekClient)
    assert runtime.provider == "deepseek"
    assert runtime.model == "deepseek-v4-flash"
    assert runtime.context_window == 1_000_000
    assert runtime.supports_image_input is False
    assert runtime.display_name == "deepseek/deepseek-v4-flash"


def test_factory_kimi_uses_reasoning_roundtrip_client():
    s = settings({"KIMI_API_KEY": "sk-k"})
    runtime = create_client("kimi", s)
    assert isinstance(runtime.client, KimiClient)
    assert KimiClient.reasoning_roundtrip is True
    assert KimiClient.degrade_image_history is False


def test_factory_moonshot_alias_and_key_fallback():
    s = settings({"MOONSHOT_API_KEY": "sk-m"})
    runtime = create_client("moonshot", s)
    assert isinstance(runtime.client, KimiClient)
    assert runtime.provider == "kimi"
    assert runtime.model == "kimi-k2.6"


def test_factory_glm_default_client_and_coding_url():
    s = settings({"GLM_API_KEY": "sk-g"})
    runtime = create_client("glm", s)
    assert type(runtime.client) is OpenAiCompatClient
    assert OpenAiCompatClient.reasoning_roundtrip is False
    assert OpenAiCompatClient.degrade_image_history is False
    assert runtime.client.openai_api_base == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert runtime.supports_tools is True
    assert runtime.supports_image_input is True


def test_factory_base_url_env_override():
    s = settings({"GLM_API_KEY": "sk-g", "GLM_BASE_URL": "http://localhost:9000/v1"})
    runtime = create_client("glm", s)
    assert runtime.client.openai_api_base == "http://localhost:9000/v1"


def test_factory_unknown_provider_returns_none():
    assert create_client("unknown", settings()) is None
    assert create_client(None, settings()) is None


def test_factory_missing_key_returns_none():
    assert create_client("deepseek", settings()) is None


def test_factory_xfyun_plain_chat_no_tools():
    s = settings({"XFYUN_MAAS_API_KEY": "sk-f"})
    runtime = create_client("xfyun", s)
    assert runtime.supports_tools is False
    assert runtime.model == "Qwen3.6-35B-A3B"
    assert runtime.client.default_headers is None


def test_factory_xfyun_lora_id_header():
    s = settings({"XFYUN_MAAS_API_KEY": "sk-f", "XFYUN_MAAS_LORA_ID": "res-123"})
    runtime = create_client("xfyun", s)
    assert runtime.client.default_headers == {"lora_id": "res-123"}


def test_factory_agnes_defaults():
    s = settings({"AGNES_API_KEY": "sk-a"})
    runtime = create_client("agnes", s)
    assert runtime.model == "agnes-2.0-flash"
    assert runtime.context_window == 1_000_000
    assert runtime.client.openai_api_base == "https://apihub.agnes-ai.com/v1"


def test_deepseek_client_type_flags():
    assert DeepSeekClient.reasoning_roundtrip is True
    assert DeepSeekClient.degrade_image_history is True


# ---------- 回退链 ----------


def test_default_provider_from_settings():
    assert settings().default_provider() == "glm"
    assert settings({"ASTERCLI_PROVIDER": "kimi"}).default_provider() == "kimi"


def test_create_from_settings_prefers_default_provider():
    s = settings({"ASTERCLI_PROVIDER": "agnes", "AGNES_API_KEY": "a", "GLM_API_KEY": "g"})
    runtime = create_from_settings(s)
    assert runtime.provider == "agnes"


def test_create_from_settings_fallback_order():
    # 默认 glm 无 Key → 按 FALLBACK_ORDER 找到第一个有 Key 的 provider
    s = settings({"STEP_API_KEY": "sk-s", "KIMI_API_KEY": "sk-k"})
    runtime = create_from_settings(s)
    assert runtime.provider == "step"


def test_create_from_settings_no_key_returns_none():
    assert create_from_settings(settings()) is None


def test_available_providers_follow_fallback_order():
    s = settings({"AGNES_API_KEY": "a", "GLM_API_KEY": "g", "KIMI_API_KEY": "k"})
    assert available_providers(s) == ["glm", "kimi", "agnes"]


# ---------- 模型 env 覆盖 ----------


def test_model_env_override():
    s = settings({"DEEPSEEK_API_KEY": "k", "DEEPSEEK_MODEL": "deepseek-reasoner"})
    runtime = create_client("deepseek", s)
    assert runtime.model == "deepseek-reasoner"


def test_provider_runtime_is_dataclass():
    assert hasattr(ProviderRuntime, "__dataclass_fields__")


def test_llm_timeout_shape():
    # 超时配置应作为 httpx.Timeout 传给客户端
    from astercli.llm.factory import _LLM_TIMEOUT

    assert isinstance(_LLM_TIMEOUT, httpx.Timeout)
