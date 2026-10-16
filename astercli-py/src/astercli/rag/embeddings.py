"""Embedding 客户端：包装 LangChain OpenAIEmbeddings，对齐 Java EmbeddingClient。

- provider 二分：ollama（本机原生服务，OpenAI 兼容端点自动补 /v1）与
  OpenAI 兼容云服务（openai / zhipu / glm / 阿里云 dashscope 等，配
  EMBEDDING_BASE_URL + EMBEDDING_API_KEY 即可）
- 单条输入超 2000 字符截断（对齐 Java），避免长块把 embedding 端点打爆
- 维度由服务端模型决定，首次索引时探测一次（detect_dimension）

环境变量（沿用仓库根 .env，与 Java 同名）：
- EMBEDDING_PROVIDER：ollama | openai | zhipu | glm | 其他任意 OpenAI 兼容名
- EMBEDDING_MODEL / EMBEDDING_BASE_URL / EMBEDDING_API_KEY
"""

from __future__ import annotations

from typing import Optional

from ..config import Settings

MAX_EMBED_CHARS = 2000
# 每次请求批量条数（保守值，兼容各类服务的批量上限）
EMBED_BATCH_SIZE = 10

# 各 provider 的默认模型（显式配置 EMBEDDING_MODEL 时优先生效）
_DEFAULT_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
    "zhipu": "embedding-3",
    "glm": "embedding-3",
}


class EmbeddingNotConfiguredError(RuntimeError):
    """未配置任何 embedding 服务（/index 前置检查点）。"""


class EmbeddingClient:
    """同步 embedding 封装（索引用同步批处理即可，无需 async）。"""

    def __init__(self, settings: Settings, http_client=None) -> None:
        self._settings = settings
        self._http_client = http_client
        provider = (settings.get("EMBEDDING_PROVIDER") or "").strip().lower()
        model = (settings.get("EMBEDDING_MODEL") or "").strip()
        base_url = (settings.get("EMBEDDING_BASE_URL") or "").strip()
        api_key = (settings.get("EMBEDDING_API_KEY") or "").strip()

        if provider == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "ollama"
            model = model or _DEFAULT_MODELS["ollama"]
        else:
            model = model or _DEFAULT_MODELS.get(provider, "")

        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self._embedder = None
        self._dim_probe: Optional[list] = None

    def is_configured(self) -> bool:
        """是否具备可用配置：provider=ollama 有默认端点即可用；其余需要 base_url + api_key + model。"""
        if self.provider == "ollama":
            return True
        return bool(self.base_url and self.api_key and self.model)

    def ensure_configured(self) -> None:
        if not self.is_configured():
            raise EmbeddingNotConfiguredError(
                "未配置 embedding 服务：请在 .env 设置 EMBEDDING_PROVIDER（ollama 或 openai 兼容）"
                "及对应的 EMBEDDING_MODEL / EMBEDDING_BASE_URL / EMBEDDING_API_KEY"
            )

    def _get_embedder(self):
        if self._embedder is None:
            from langchain_openai import OpenAIEmbeddings

            kwargs = {
                "model": self.model,
                "check_embedding_ctx_length": False,   # 非 OpenAI 模型跳过 tiktoken 预检
                "tiktoken_enabled": False,
                "chunk_size": EMBED_BATCH_SIZE,
            }
            if self.base_url:
                kwargs["openai_api_base"] = self.base_url
            if self.api_key:
                kwargs["openai_api_key"] = self.api_key
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            self._embedder = OpenAIEmbeddings(**kwargs)
        return self._embedder

    @staticmethod
    def _clip(text: str) -> str:
        """单条输入 2000 字符截断（对齐 Java）。"""
        return text if len(text) <= MAX_EMBED_CHARS else text[:MAX_EMBED_CHARS]

    def embed_documents(self, texts: list) -> list:
        """批量嵌入；texts 为空直接返回空列表。"""
        if not texts:
            return []
        self.ensure_configured()
        clipped = [self._clip(t) for t in texts]
        return self._get_embedder().embed_documents(clipped)

    def embed_query(self, text: str) -> list:
        self.ensure_configured()
        return self._get_embedder().embed_query(self._clip(text))

    def detect_dimension(self) -> int:
        """探测 embedding 维度（建向量表用）；一次真实调用，缓存结果。"""
        if self._dim_probe is None:
            self._dim_probe = self.embed_query("dimension probe")
        return len(self._dim_probe)
