"""索引器与 embedding 客户端单测（不发真实网络请求）。"""

from __future__ import annotations

import math

import httpx
import pytest

from astercli.config import Settings
from astercli.rag import CodeIndex, EmbeddingClient, EmbeddingNotConfiguredError
from astercli.rag.index import CodeIndex as _CodeIndex  # noqa: F401 同一类的直接引用
from astercli.rag.store import VectorStore


class FakeEmbedding:
    """确定性伪 embedding，模拟与内容相关的向量。"""

    model = "fake-embed"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list:
        v = [0.0] * self.dim
        for i, ch in enumerate(text):
            v[(ord(ch) + i) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _make_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pay.py").write_text(
        "class PaymentService:\n"
        "    def pay(self):\n"
        "        return 'payment 支付流程'\n", encoding="utf-8")
    (tmp_path / "src" / "auth.py").write_text(
        "def login():\n    return 'user session'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\nsemantic search\n", encoding="utf-8")
    return tmp_path


def test_full_index_flow(tmp_path, monkeypatch):
    """端到端：索引临时项目 → 语义/混合检索命中支付相关代码块。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    root = _make_project(tmp_path)
    emb = FakeEmbedding()
    index = CodeIndex(root, emb, store=VectorStore(root, base_dir=tmp_path / "rag"))
    progress: list = []
    stats = index.index(progress=progress.append)
    assert stats.files == 3
    assert stats.chunks >= 3
    assert stats.backend == "json-fallback"
    assert any("索引完成" in p for p in progress)

    from astercli.rag import CodeRetriever
    retriever = CodeRetriever(root, emb, store=index.store)
    results = retriever.hybrid_search("payment 支付流程", top_k=3)
    assert results, "语义检索应命中"
    assert results[0].file_path in ("src/pay.py", "README.md")
    assert any(r.file_path == "src/pay.py" for r in results)


def test_index_empty_dir_raises(tmp_path, monkeypatch):
    """无可索引文件时报错而非静默成功。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    (tmp_path / "only.bin").write_bytes(b"\x00")
    index = CodeIndex(tmp_path, FakeEmbedding(),
                      store=VectorStore(tmp_path, base_dir=tmp_path / "rag"))
    with pytest.raises(RuntimeError):
        index.index()


# ---------- EmbeddingClient（MockTransport 模拟 OpenAI 兼容 /embeddings） ----------

_DIM = 4


def _ok_response(request: httpx.Request) -> httpx.Response:
    import json as _json

    body = _json.loads(request.content.decode())
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [{"object": "embedding", "index": i,
             "embedding": [0.1, 0.2, 0.3, 0.4]} for i in range(len(inputs))]
    return httpx.Response(200, json={"object": "list", "data": data, "usage": {}})


def _settings(**env):
    return Settings(env, load_files=False)


def test_ollama_defaults():
    """provider=ollama：端点补 /v1、key 默认占位、模型有默认。"""
    client = EmbeddingClient(_settings(EMBEDDING_PROVIDER="ollama"))
    assert client.is_configured()
    assert client.base_url == "http://localhost:11434/v1"
    assert client.api_key == "ollama"
    assert client.model == "nomic-embed-text"


def test_not_configured_raises():
    """云 provider 缺 base_url/key 时明确报错（/index 前置检查）。"""
    client = EmbeddingClient(_settings())
    assert not client.is_configured()
    with pytest.raises(EmbeddingNotConfiguredError):
        client.embed_query("x")


def test_openai_compatible_configured():
    client = EmbeddingClient(_settings(
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="text-embedding-v4",
        EMBEDDING_BASE_URL="https://dashscope.example.com/compatible-mode/v1/",
        EMBEDDING_API_KEY="sk-test",
    ))
    assert client.is_configured()
    assert client.base_url.endswith("/v1") and not client.base_url.endswith("/")
    assert client.model == "text-embedding-v4"


def test_embed_query_via_mock_transport():
    """embed_query/embed_documents 走 MockTransport：请求打到配置端点且返回解析正确。"""
    client = EmbeddingClient(
        _settings(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="m",
            EMBEDDING_BASE_URL="https://emb.example.com/v1",
            EMBEDDING_API_KEY="sk-test",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(_ok_response)),
    )
    vec = client.embed_query("hello")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    vecs = client.embed_documents(["a", "b"])
    assert len(vecs) == 2
    assert client.detect_dimension() == _DIM


def test_embed_documents_batching_and_clip():
    """批量分批 + 单条 2000 字符截断（对齐 Java EmbeddingClient）。"""
    seen_batches: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content.decode())
        inputs = body["input"]
        seen_batches.append([len(x) if isinstance(x, str) else -1 for x in inputs])
        data = [{"object": "embedding", "index": i,
                 "embedding": [0.5, 0.5, 0.5, 0.5]} for i in range(len(inputs))]
        return httpx.Response(200, json={"object": "list", "data": data})

    client = EmbeddingClient(
        _settings(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="m",
            EMBEDDING_BASE_URL="https://emb.example.com/v1",
            EMBEDDING_API_KEY="sk-test",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    long_text = "x" * 5000
    out = client.embed_documents([long_text, "short"])
    assert len(out) == 2
    assert max(seen_batches[0]) == 2000   # 长文本被截到 2000
