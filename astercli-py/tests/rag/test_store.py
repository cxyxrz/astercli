"""向量存储单测：sqlite-vec KNN 与 JSON 余弦回退双路径。"""

from __future__ import annotations

import math

import pytest

from astercli.rag.chunker import CodeChunk
from astercli.rag.store import VectorStore, _cosine


def _vec(text: str, dim: int = 8) -> list:
    """确定性伪向量：按字符位置累积再归一化。"""
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[(ord(ch) + i) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _chunk(name: str, content: str, chunk_type: str = "method",
           file_path: str = "a.py") -> CodeChunk:
    return CodeChunk(file_path, chunk_type, name, content, 1, 3)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """JSON 回退路径的 store（强制禁用 sqlite-vec）。"""
    monkeypatch.setattr(VectorStore, "_try_load_vec", lambda self: False)
    return VectorStore(tmp_path, base_dir=tmp_path / "rag")


@pytest.fixture
def vec_store(tmp_path):
    """真实 sqlite-vec 路径的 store（构建不支持时自动回退，测试自适应）。"""
    return VectorStore(tmp_path, base_dir=tmp_path / "rag")


def test_json_fallback_search(store):
    """JSON 回退：全量余弦排序正确。"""
    chunks = [
        _chunk("payment", "payment flow logic here"),
        _chunk("login", "user login session logic"),
        _chunk("cache", "cache eviction policy"),
    ]
    vectors = [_vec(c.content) for c in chunks]
    store.ensure_schema(8, "fake")
    store.insert(chunks, vectors)
    assert store.count() == 3
    results = store.search(_vec("payment flow logic here"), 2)
    assert len(results) == 2
    assert results[0].name == "payment"
    assert results[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_vec_knn_search(vec_store):
    """sqlite-vec 路径：KNN 结果与余弦口径一致（不可用时该用例验证回退等价）。"""
    chunks = [_chunk("alpha", "alpha content"), _chunk("beta", "beta content")]
    vectors = [_vec(c.content) for c in chunks]
    vec_store.ensure_schema(8, "fake")
    vec_store.insert(chunks, vectors)
    results = vec_store.search(_vec("alpha content"), 1)
    assert len(results) == 1
    assert results[0].name == "alpha"
    assert results[0].similarity > 0.99


def test_keyword_search(store):
    """关键词 LIKE 命中 + 固定基础相似度（0.15，低于 Java 0.3 的原因见实现注释）。"""
    store.ensure_schema(8, "fake")
    store.insert(
        [_chunk("pay", "处理 payment 支付流程", file_path="pay.py"),
         _chunk("log", "user login session", file_path="log.py")],
        [_vec("a"), _vec("b")],
    )
    results = store.search_by_keyword(["payment"], 5)
    assert len(results) == 1
    assert results[0].name == "pay"
    assert results[0].similarity == pytest.approx(0.15)


def test_dim_change_rebuild(store):
    """换模型导致维度变化时整库重建，旧数据不残留。"""
    store.ensure_schema(8, "model-a")
    store.insert([_chunk("x", "x")], [_vec("x")])
    assert store.count() == 1
    store.ensure_schema(16, "model-b")   # 维度+模型都变 → 重建
    assert store.count() == 0
    store.insert([_chunk("y", "y")], [_vec("y", dim=16)])
    assert store.count() == 1


def test_clear_keeps_schema(store):
    """clear 清数据但保留 meta/schema，可重复索引。"""
    store.ensure_schema(8, "fake")
    store.insert([_chunk("x", "x")], [_vec("x")])
    store.clear()
    assert store.count() == 0
    store.insert([_chunk("y", "y")], [_vec("y")])
    assert store.count() == 1


def test_search_before_index(store):
    """未索引时语义/关键词检索安全返回空，不抛异常。"""
    assert store.search(_vec("q"), 5) == []
    assert store.search_by_keyword(["q"], 5) == []


def test_insert_dim_mismatch(store):
    """向量维度与索引维度不一致时拒绝写入（防静默错配）。"""
    store.ensure_schema(8, "fake")
    with pytest.raises(ValueError):
        store.insert([_chunk("x", "x")], [_vec("x", dim=16)])


def test_cosine_basics():
    """内存余弦：同向=1，正交=0。"""
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine([0, 0], [1, 1]) == 0.0
