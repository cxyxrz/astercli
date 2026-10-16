"""search_code 工具注册与调用单测（不依赖真实 embedding 服务）。"""

from __future__ import annotations

import math

from astercli.config import Settings
from astercli.rag import NO_INDEX_HINT, register_rag_tool
from astercli.rag.store import VectorStore
from astercli.tools import ToolRegistry


class FakeEmbedding:
    model = "fake-embed"

    def __init__(self, settings=None, http_client=None, dim: int = 8) -> None:
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


def _registry(tmp_path):
    registry = ToolRegistry(settings=Settings(env={}, load_files=False))
    register_rag_tool(registry, tmp_path)
    return registry


def test_tool_registered_not_dangerous(tmp_path):
    """search_code 已注册、只读不审批（对齐 Java 危险级别）。"""
    registry = _registry(tmp_path)
    tool = registry.get("search_code")
    assert tool is not None
    assert tool.dangerous is False
    assert "search_code" in registry.names()


def test_call_without_index_returns_hint(tmp_path):
    """未建索引时返回 /index 提示，不抛异常（对齐 Java 行为）。"""
    registry = _registry(tmp_path)
    tool = registry.get("search_code")
    result = tool.func({"query": "支付流程"})
    assert NO_INDEX_HINT in result


def test_call_after_index_hits(tmp_path, monkeypatch):
    """索引后调用 search_code：返回带摘要与相似度的结构化文本。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    monkeypatch.setenv("ASTERCLI_RAG_DIR", str(tmp_path / "rag"))
    (tmp_path / "pay.py").write_text(
        "class PaymentService:\n"
        "    def pay(self):\n"
        "        return 'payment 支付'\n", encoding="utf-8")
    # monkeypatch rag 命名空间的 EmbeddingClient，避免真实网络调用
    import astercli.rag as rag_pkg
    monkeypatch.setattr(rag_pkg, "EmbeddingClient", FakeEmbedding)

    from astercli.rag import CodeIndex
    CodeIndex(tmp_path, FakeEmbedding()).index()   # 走 ASTERCLI_RAG_DIR 同一库

    registry = _registry(tmp_path)
    tool = registry.get("search_code")
    result = tool.func({"query": "payment 支付", "top_k": 3})
    assert "检索摘要:" in result
    assert "PaymentService" in result
    assert "相似度" in result


def test_call_bad_args(tmp_path, monkeypatch):
    """非法参数安全降级：缺 query / 非法 top_k 不炸。"""
    registry = _registry(tmp_path)
    tool = registry.get("search_code")
    assert "缺少 query" in tool.func({})
    # 未索引场景下非法 top_k 也只会返回索引提示
    result = tool.func({"query": "x", "top_k": "abc"})
    assert isinstance(result, str)
