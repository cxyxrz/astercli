"""检索器单测：混合融合策略 + 查询分词 + 结果格式化。"""

from __future__ import annotations

import math

import pytest

from astercli.rag.chunker import CodeChunk
from astercli.rag.retriever import (
    CodeRetriever,
    IndexNotReadyError,
    build_summary,
    format_for_cli,
    format_for_tool,
    tokenize_query,
)
from astercli.rag.store import SearchResult, VectorStore


class FakeEmbedding:
    """确定性伪 embedding：相同文本 → 相同向量（模拟语义命中）。"""

    model = "fake"

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


def _chunk(name, content, chunk_type="method", file_path="a.py"):
    return CodeChunk(file_path, chunk_type, name, content, 10, 20)


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    """建好索引的 retriever（JSON 回退模式，避免依赖 sqlite-vec 构建）。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    store = VectorStore(tmp_path, base_dir=tmp_path / "rag")
    chunks = [
        _chunk("PaymentService", "class PaymentService 处理 payment 支付",
               chunk_type="class", file_path="pay/PaymentService.java"),
        _chunk("pay", "def pay(self): 处理 payment 支付流程", file_path="pay/api.py"),
        _chunk("login", "def login(self): user login session 逻辑", file_path="auth/api.py"),
        _chunk("cache", "def cache(self): cache eviction policy", file_path="core/cache.py"),
    ]
    emb = FakeEmbedding()
    store.ensure_schema(emb.dim, emb.model)
    store.insert(chunks, emb.embed_documents([c.to_embed_text() for c in chunks]))
    return CodeRetriever(tmp_path, emb, store=store), emb


def test_tokenize_query():
    """ASCII 整词小写化 + 中文 2-gram + 停用词过滤 + 去重保序。"""
    tokens = tokenize_query("How to handle Payment 支付流程 flow")
    assert "payment" in tokens
    assert "支付" in tokens and "付流" in tokens and "流程" in tokens
    assert "how" not in tokens          # 停用词
    assert tokens.count("flow") == 1    # 去重
    assert tokenize_query("") == []
    assert tokenize_query("的") == []   # 单字停用词


def test_hybrid_returns_ranked_results(indexed):
    retriever, emb = indexed
    results = retriever.hybrid_search("处理 payment 支付", top_k=3)
    assert 0 < len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)
    # 支付相关的块应排进前二
    top_names = [r.name for r in results[:2]]
    assert "PaymentService" in top_names or "pay" in top_names


def test_double_hit_bonus(indexed):
    """语义与关键词双命中的块应排在只有单路命中的块之前。"""
    retriever, emb = indexed
    # "cache eviction policy"：与 cache 块文本完全一致（语义必命中），
    # 同时关键词 policy/eviction/cache LIKE 命中 → 双路加分
    results = retriever.hybrid_search("cache eviction policy", top_k=4)
    assert results[0].name == "cache"


def test_per_file_limit(tmp_path, monkeypatch):
    """单文件最多 2 条（对齐 Java maxPerFile=2）。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    store = VectorStore(tmp_path, base_dir=tmp_path / "rag")
    chunks = [
        _chunk(f"m{i}", f"same file method {i} common-topic xyz", file_path="big.py")
        for i in range(5)
    ]
    emb = FakeEmbedding()
    store.ensure_schema(emb.dim, emb.model)
    store.insert(chunks, emb.embed_documents([c.to_embed_text() for c in chunks]))
    retriever = CodeRetriever(tmp_path, emb, store=store)
    results = retriever.hybrid_search("common-topic xyz", top_k=5)
    assert len(results) == 2
    assert all(r.file_path == "big.py" for r in results)


def test_top_k_clamp(indexed):
    retriever, _ = indexed
    assert retriever.hybrid_search("", top_k=5) == []    # 空 query 直接返回
    results = retriever.hybrid_search("cache eviction policy", top_k=999)
    assert len(results) <= 30                              # MAX_TOP_K 上限


def test_not_ready_raises(tmp_path, monkeypatch):
    """未建立索引时抛 IndexNotReadyError（提示先 /index）。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    retriever = CodeRetriever(tmp_path, FakeEmbedding(),
                              store=VectorStore(tmp_path, base_dir=tmp_path / "rag"))
    with pytest.raises(IndexNotReadyError):
        retriever.hybrid_search("anything")


def test_format_for_tool(indexed):
    retriever, _ = indexed
    results = retriever.hybrid_search("cache eviction policy", top_k=2)
    text = format_for_tool("cache eviction policy", results)
    assert text.startswith("检索摘要:")
    assert "检索结果:" in text
    assert "1. [method:cache]" in text
    assert "相似度: 0." in text
    assert "L10-20" in text       # 行号信息


def test_semantic_beats_keyword_noise(tmp_path, monkeypatch):
    """融合权重回归：语义命中的块必须排在仅关键词 LIKE 命中的块之前，
    且结果 similarity 反映融合分（写回语义），而非关键词底分 0.15。

    语义块的向量直接取查询向量（余弦精确 1.0），绕开 Fake 向量对
    to_embed_text 前缀的敏感性，保证断言只验证融合权重本身。
    """
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    store = VectorStore(tmp_path, base_dir=tmp_path / "rag")
    emb = FakeEmbedding()
    query = "exact query target text"
    chunks = [
        CodeChunk("sem.py", "method", "semantic_hit", "exact query target text", 1, 2),
        CodeChunk("kw.py", "method", "keyword_noise", "noise noise target noise", 1, 2),
    ]
    store.ensure_schema(emb.dim, emb.model)
    store.insert(chunks, [emb.embed_query(query), emb.embed_query("unrelated stuff")])
    retriever = CodeRetriever(tmp_path, emb, store=store)
    results = retriever.hybrid_search(query, top_k=2)
    assert results[0].name == "semantic_hit"
    # 语义 1.0 + 双命中 0.1 + method 0.15 = 1.25 融合分写回
    assert results[0].similarity == pytest.approx(1.25, abs=1e-3)
    assert results[1].name == "keyword_noise"
    assert results[1].similarity < results[0].similarity


def test_format_for_cli_and_empty():
    assert "没有命中可用代码块" in format_for_cli("q", [])
    r = SearchResult("a/b/c.py", "class", "Foo", "content here", 3, 9, 0.87)
    text = format_for_cli("q", [r])
    assert "📋 找到 1 个相关代码块" in text
    assert "a/b/c.py" in text
    assert build_summary("q", []) .startswith("搜索摘要:")
