"""混合检索器：语义 + 关键词 + 重叠加分 + 类型加分 + 单文件限流。

对齐 Java CodeRetriever.hybridSearch 的融合策略：
1. 语义检索 top_k*2（余弦相似度）
2. 关键词检索 top_k*2（LIKE 命中，基础相似度 0.15——低于 Java 的 0.3，
   原因见 store.search_by_keyword：防止关键词噪音块压过低绝对值的语义命中）
3. 融合：两路都命中 +0.1；method 块 +0.15、class 块 +0.10（file 不加分）；
   融合分写回结果的 similarity 字段，展示值与排序依据一致
4. 按融合分排序，单文件最多 2 条（避免一个文件刷屏），取 top_k

查询分词与 Java 的 jieba 不同：沿用本项目长期记忆的零依赖方案
（ASCII 整词 + 中文 2-gram + 停用词），不引入 jieba。
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

from .embeddings import EmbeddingClient
from .store import SearchResult, VectorStore

DEFAULT_TOP_K = 5
MAX_TOP_K = 30
# 单文件在最终结果中的最大条数（对齐 Java maxPerFile=2）
MAX_PER_FILE = 2

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for",
    "on", "with", "how", "what", "的", "了", "是", "在", "和", "与", "有",
}


def tokenize_query(query: str) -> list:
    """查询分词：ASCII 整词小写化 + 中文 2-gram，过滤停用词与单字符噪音。"""
    tokens: list = []
    ascii_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", query)
    for w in ascii_words:
        low = w.lower()
        if low not in _STOPWORDS and len(low) >= 2:
            tokens.append(low)
    han = re.findall(r"[\u4e00-\u9fff]+", query)
    for seg in han:
        if len(seg) == 1:
            if seg not in _STOPWORDS:
                tokens.append(seg)
            continue
        for i in range(len(seg) - 1):
            gram = seg[i:i + 2]
            if gram not in _STOPWORDS:
                tokens.append(gram)
        if len(seg) == 2 and seg not in _STOPWORDS and seg not in tokens:
            tokens.append(seg)
    # 去重保序
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


class IndexNotReadyError(RuntimeError):
    """索引尚未建立（提示先 /index）。"""


class CodeRetriever:
    """RAG 检索入口：search_code 工具与 /search 命令共用。"""

    def __init__(self, project_path: Path, embedding_client: EmbeddingClient,
                 store: Optional[VectorStore] = None) -> None:
        self.project_path = Path(project_path).resolve()
        self.embedding_client = embedding_client
        self.store = store or VectorStore(self.project_path)

    def has_index(self) -> bool:
        return self.store.count() > 0

    def hybrid_search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """语义 + 关键词混合检索（对齐 Java hybridSearch）。"""
        if not query or not query.strip():
            return []
        top_k = max(1, min(top_k, MAX_TOP_K))
        if not self.has_index():
            raise IndexNotReadyError("代码索引尚未建立，请先执行 /index")

        # 语义路
        semantic: dict = {}
        try:
            for r in self.store.search(self.embedding_client.embed_query(query), top_k * 2):
                semantic[(r.file_path, r.name)] = r
        except Exception:
            semantic = {}  # 语义路失败不致命，退化为纯关键词

        # 关键词路
        keyword: dict = {}
        terms = tokenize_query(query)
        if terms:
            for r in self.store.search_by_keyword(terms[:8], top_k * 2):
                keyword[(r.file_path, r.name)] = r

        # 融合打分（分数写回 similarity，展示值与排序依据一致）
        merged: dict = {}
        for key, r in semantic.items():
            merged[key] = (r, r.similarity)
        for key, r in keyword.items():
            if key in merged:
                _, score = merged[key]
                merged[key] = (r, score + 0.1)   # 双路命中加分
            else:
                merged[key] = (r, r.similarity)

        scored: list = []
        for r, score in merged.values():
            bonus = 0.0
            if r.chunk_type == "method":
                bonus = 0.15
            elif r.chunk_type == "class":
                bonus = 0.10
            final = score + bonus
            scored.append((final, dataclasses.replace(r, similarity=round(final, 4))))

        # 排序 + 单文件限流 + 截断
        scored.sort(key=lambda t: t[0], reverse=True)
        results: list = []
        per_file: dict = {}
        for score, r in scored:
            if per_file.get(r.file_path, 0) >= MAX_PER_FILE:
                continue
            per_file[r.file_path] = per_file.get(r.file_path, 0) + 1
            results.append(r)
            if len(results) >= top_k:
                break
        return results


# ---------- 结果格式化（对齐 Java SearchResultFormatter 文案） ----------

def _snippet(content: str, max_chars: int) -> str:
    normalized = (content or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return "(无内容片段)"
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def _shorten_path(file_path: str) -> str:
    parts = Path(file_path).parts
    if len(parts) <= 3:
        return file_path
    return str(Path(*parts[-3:]))


def build_summary(query: str, results: list) -> str:
    """三行式检索摘要（对齐 Java buildSummary：入口/文件分布/检索依据）。"""
    if not results:
        return "搜索摘要:\n- 没有命中可用代码块。"
    top = results[0]
    file_names: list = []
    for r in results:
        name = Path(r.file_path).name
        if name not in file_names:
            file_names.append(name)
    query_tokens = tokenize_query(query)[:3]
    related = "、".join(file_names[:3])
    token_text = "、".join(query_tokens) if query_tokens else "自然语言语义"
    suffix = " 等文件" if len(file_names) > 3 else " 这些文件"
    lines = [
        "搜索摘要:",
        f"- 最相关的入口是 [{top.chunk_type}:{top.name}]，位于 {_shorten_path(top.file_path)}。",
        f"- 当前结果主要集中在 {related}{suffix}。",
        f"- 这次排序综合参考了 {token_text} 等关键词与语义相似度；先看第 1 条，再按文件继续展开最稳妥。",
    ]
    return "\n".join(lines)


def format_for_tool(query: str, results: list) -> str:
    """search_code 工具输出格式（对齐 Java formatForTool）。"""
    if not results:
        return "检索摘要:\n- 没有命中可用代码块。"
    parts = ["检索摘要:", build_summary(query, results), "", "检索结果:"]
    for i, r in enumerate(results):
        head = (f"{i + 1}. [{r.chunk_type}:{r.name}] (相似度: {r.similarity:.3f}) {r.file_path}"
                f" L{r.start_line}-{r.end_line}")
        parts.append(head)
        parts.append("   " + _snippet(r.content, 180).replace("\n", "\n   "))
        parts.append("")
    return "\n".join(parts).rstrip()


def format_for_cli(query: str, results: list) -> str:
    """/search 命令输出格式（对齐 Java formatForCli）。"""
    parts = [f"📋 找到 {len(results)} 个相关代码块:", "", build_summary(query, results), ""]
    for i, r in enumerate(results):
        head = (f"{i + 1}. [{r.chunk_type}:{r.name}] (相似度: {r.similarity:.3f}) {r.file_path}"
                f" L{r.start_line}-{r.end_line}")
        parts.append(head)
        parts.append("   " + _snippet(r.content, 120).replace("\n", "\n   "))
        parts.append("")
    return "\n".join(parts).rstrip()
