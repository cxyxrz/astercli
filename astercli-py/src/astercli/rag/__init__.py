"""RAG 模块（P6）：tree-sitter 分块 + sqlite-vec 向量存储 + 混合检索。

对齐 Java rag/ 包职责，实现重新设计：
- 分块：tree-sitter AST 符号级（Java 版 JavaParser），文本分段兜底
- 向量：sqlite-vec vec0 KNN（Java 版手写余弦），JSON 回退保跨环境可用
- 检索：语义 + 关键词 + 重叠加分 + 类型加分 + 单文件限流，策略对齐 Java
- 分词：零依赖 ASCII 整词 + 中文 2-gram（Java 版 jieba，Python 侧不引依赖）

search_code 是 RAG 语义辅助工具：适合模糊自然语言、关键词不明确、
常规 glob/grep 无果的场景；精确符号/字符串定位仍应优先 grep_code。
"""

from __future__ import annotations

from pathlib import Path

from .chunker import CodeChunk, chunk_file, collect_files
from .embeddings import EmbeddingClient, EmbeddingNotConfiguredError
from .index import CodeIndex, IndexStats
from .retriever import (
    CodeRetriever,
    IndexNotReadyError,
    build_summary,
    format_for_cli,
    format_for_tool,
    tokenize_query,
)
from .store import SearchResult, VectorStore

NO_INDEX_HINT = "代码索引尚未建立，请先执行 /index 建立向量索引后再使用语义检索"


def register_rag_tool(registry, project_path: Path) -> None:
    """注册 search_code 语义检索工具（只读、无需审批，对齐 Java 危险级别）。"""
    from ..tools.registry import Param, create_parameters

    project_path = Path(project_path).resolve()

    def _search_code(args) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "缺少 query 参数"
        top_k = args.get("top_k") or 5
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 30))
        retriever = CodeRetriever(project_path, EmbeddingClient(_load_settings()))
        try:
            results = retriever.hybrid_search(query, top_k)
        except IndexNotReadyError:
            return NO_INDEX_HINT
        except EmbeddingNotConfiguredError as e:
            return f"语义检索不可用：{e}"
        return format_for_tool(query, results)

    registry.register(
        "search_code",
        "语义代码检索（RAG 辅助）。基于向量索引做自然语言/模糊描述检索，"
        "适合关键词不明确、常规搜索无果、跨文件主题汇总的场景；"
        "需要先执行 /index 建立索引；精确符号或字符串定位请优先使用 grep_code",
        create_parameters(
            Param("query", "string", "自然语言查询或关键词", True),
            Param("top_k", "integer", "最多返回结果数，默认 5，上限 30", False),
        ),
        _search_code,
    )


def _load_settings():
    """延迟导入避免循环依赖（cli 启动时 Settings 已就绪，工具调用时再取）。"""
    from ..config import Settings

    return Settings()


__all__ = [
    "CodeChunk", "CodeIndex", "IndexStats", "CodeRetriever", "VectorStore",
    "SearchResult", "EmbeddingClient", "EmbeddingNotConfiguredError",
    "IndexNotReadyError", "chunk_file", "collect_files", "tokenize_query",
    "build_summary", "format_for_tool", "format_for_cli",
    "register_rag_tool", "NO_INDEX_HINT",
]
