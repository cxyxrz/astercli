"""长期记忆测试（对齐 Java LongTermMemory / MemoryRetriever 语义）。

覆盖：存储去重、scope 可见性、检索排序（时间衰减）、持久化 roundtrip、
build_memory_context 预算截断。
"""

from __future__ import annotations

import time

from astercli.memory.long_term import (
    LongTermMemory,
    MemoryEntry,
    build_memory_context,
    is_visible_in_project,
    normalize_project_key,
    tokenize,
)


def make_memory(tmp_path) -> LongTermMemory:
    return LongTermMemory(storage_dir=tmp_path / "memory")


# ---------- 存储 ----------


def test_store_fact_project_scope(tmp_path):
    mem = make_memory(tmp_path)
    entry = mem.store_fact("这个项目使用 Java 17", project_key="/proj/a")
    assert entry is not None
    assert entry.id.startswith("fact-")
    assert entry.metadata["scope"] == "project"
    assert entry.metadata["project"] == "/proj/a"
    assert entry.token_count > 0


def test_store_fact_global_scope(tmp_path):
    mem = make_memory(tmp_path)
    entry = mem.store_fact("默认用中文回答", scope="global")
    assert entry.metadata["scope"] == "global"
    assert "project" not in entry.metadata


def test_store_fact_duplicate_skipped(tmp_path):
    mem = make_memory(tmp_path)
    assert mem.store_fact("相同事实", project_key="/p") is not None
    assert mem.store_fact("相同事实", scope="global") is None  # 内容相同全局去重
    assert len(mem.get_all()) == 1


def test_store_fact_blank_returns_none(tmp_path):
    mem = make_memory(tmp_path)
    assert mem.store_fact("   ") is None


def test_delete_and_clear(tmp_path):
    mem = make_memory(tmp_path)
    e1 = mem.store_fact("事实一", project_key="/p")
    mem.store_fact("事实二", project_key="/p")
    assert mem.delete(e1.id) is True
    assert mem.delete(e1.id) is False
    assert len(mem.get_all()) == 1
    mem.clear()
    assert mem.get_all() == []


# ---------- 可见性与检索 ----------


def test_visibility_scope_rules(tmp_path):
    proj_entry = MemoryEntry(
        id="fact-x", content="项目事实", metadata={"scope": "project", "project": "/proj/a"}, token_count=5
    )
    global_entry = MemoryEntry(
        id="fact-y", content="全局事实", metadata={"scope": "global"}, token_count=5
    )
    assert is_visible_in_project(global_entry, None) is True
    assert is_visible_in_project(proj_entry, "/proj/a") is True
    assert is_visible_in_project(proj_entry, "/proj/b") is False
    assert is_visible_in_project(proj_entry, None) is False


def test_search_filters_by_project(tmp_path):
    mem = make_memory(tmp_path)
    mem.store_fact("Java 17 项目规则", project_key="/proj/a")
    mem.store_fact("Java 17 其他项目", project_key="/proj/b")
    mem.store_fact("全局 Java 偏好", scope="global")

    hits = mem.search("Java 17", project_key="/proj/a")
    contents = [e.content for e in hits]
    assert "Java 17 项目规则" in contents
    assert "全局 Java 偏好" in contents
    assert "Java 17 其他项目" not in contents


def test_search_empty_query_returns_empty(tmp_path):
    mem = make_memory(tmp_path)
    mem.store_fact("事实", scope="global")
    assert mem.search("") == []
    assert mem.search("!!") == []  # 全标点 → 无 token


def test_search_ranks_fresh_above_stale(tmp_path):
    mem = make_memory(tmp_path)
    fresh = mem.store_fact("部署流程使用 docker", scope="global")
    stale = mem.store_fact("部署流程使用 k8s", scope="global")
    stale.timestamp = time.time() - 48 * 3600  # 2 天前：衰减到底 0.5
    mem._save()
    hits = mem.search("部署流程", project_key=None)
    assert hits[0].id == fresh.id


def test_search_limit(tmp_path):
    mem = make_memory(tmp_path)
    for i in range(10):
        mem.store_fact(f"docker 配置条目 {i}", scope="global")
    assert len(mem.search("docker", limit=3)) == 3


# ---------- 持久化 ----------


def test_persistence_roundtrip(tmp_path):
    mem = make_memory(tmp_path)
    entry = mem.store_fact("持久化事实", scope="global")
    assert (tmp_path / "memory" / "long_term_memory.json").exists()

    mem2 = LongTermMemory(storage_dir=tmp_path / "memory")
    loaded = mem2.get_all()
    assert len(loaded) == 1
    assert loaded[0].id == entry.id
    assert loaded[0].content == "持久化事实"
    assert loaded[0].metadata == {"source": "fact", "scope": "global"}


def test_load_tolerates_corrupt_file(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    (d / "long_term_memory.json").write_text("{broken json", encoding="utf-8")
    mem = LongTermMemory(storage_dir=d)
    assert mem.get_all() == []


def test_load_tolerates_bad_entries(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    (d / "long_term_memory.json").write_text('[{"no_content": 1}, {"id": "a", "content": "ok"}]', encoding="utf-8")
    mem = LongTermMemory(storage_dir=d)
    assert len(mem.get_all()) == 1


# ---------- build_memory_context ----------


def test_build_memory_context_format_and_budget(tmp_path):
    mem = make_memory(tmp_path)
    mem.store_fact("短事实", scope="global")
    context = build_memory_context(mem, "短事实", max_tokens=10_000)
    assert context.startswith("## 相关长期记忆")
    assert "- [FACT] 短事实" in context


def test_build_memory_context_token_budget_cutoff(tmp_path):
    mem = make_memory(tmp_path)
    big = "很长的" * 200  # 约 600 tokens
    mem.store_fact(big, scope="global")
    mem.store_fact("小事实", scope="global")
    context = build_memory_context(mem, big, max_tokens=100)
    assert "小事实" not in context  # 预算内放不下新条目时停止追加
    assert context == ""  # 第一条就超预算 → 整段为空


def test_build_memory_context_no_hits(tmp_path):
    mem = make_memory(tmp_path)
    mem.store_fact("无关事实", scope="global")
    assert build_memory_context(mem, "完全不同的关键词", max_tokens=1000) == ""


# ---------- 分词与项目 key ----------


def test_tokenize_english_and_chinese():
    assert "docker" in tokenize("docker compose 配置")
    # 中文 2-gram：相邻两字成词
    grams = tokenize("部署流程")
    assert "部署" in grams
    assert "流程" in grams
    # 单字与单字符英文被过滤
    assert tokenize("a！") == set()


def test_normalize_project_key(tmp_path):
    key = normalize_project_key(tmp_path)
    assert key == str(tmp_path.resolve())
    # 不存在的路径不抛异常
    assert normalize_project_key(tmp_path / "nope") == str((tmp_path / "nope").resolve())
