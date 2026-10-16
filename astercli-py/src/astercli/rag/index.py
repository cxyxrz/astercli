"""代码索引器：收集 → 分块 → 嵌入 → 入库，带进度回调。

对齐 Java CodeIndex 的编排职责（ProgressListener 进度 + 全量重建式索引）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .chunker import chunk_file, collect_files
from .embeddings import EmbeddingClient
from .store import VectorStore

# 嵌入批大小（批内进度回调一次，避免逐条刷屏）
EMBED_PROGRESS_BATCH = 16


@dataclass
class IndexStats:
    files: int
    chunks: int
    elapsed_seconds: float
    backend: str          # sqlite-vec | json-fallback


class CodeIndex:
    """全量重建式索引器（对齐 Java：每次 /index 重建而非增量更新）。"""

    def __init__(self, project_path: Path, embedding_client: EmbeddingClient,
                 store: Optional[VectorStore] = None) -> None:
        self.project_path = Path(project_path).resolve()
        self.embedding_client = embedding_client
        self.store = store or VectorStore(self.project_path)

    def index(self, progress: Optional[Callable[[str], None]] = None) -> IndexStats:
        """建立/重建索引。progress 接收人类可读的进度行。"""
        started = time.time()

        def emit(msg: str) -> None:
            if progress is not None:
                progress(msg)

        emit(f"扫描 {self.project_path} ...")
        files = collect_files(self.project_path)
        if not files:
            raise RuntimeError("未找到可索引文件（检查目录与扩展名）")

        emit(f"解析 {len(files)} 个文件 ...")
        chunks = []
        for path in files:
            rel = path.relative_to(self.project_path).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks.extend(chunk_file(path, rel, text))
        if not chunks:
            raise RuntimeError("解析后无有效代码块")

        emit(f"共 {len(chunks)} 个代码块，开始嵌入 ...")
        texts = [c.to_embed_text() for c in chunks]
        embeddings: list = []
        for i in range(0, len(texts), EMBED_PROGRESS_BATCH):
            batch = texts[i:i + EMBED_PROGRESS_BATCH]
            embeddings.extend(self.embedding_client.embed_documents(batch))
            emit(f"嵌入进度 {min(i + EMBED_PROGRESS_BATCH, len(texts))}/{len(texts)}")

        dim = len(embeddings[0])
        model = self.embedding_client.model or "unknown"
        self.store.ensure_schema(dim, model)
        self.store.clear()
        self.store.insert(chunks, embeddings)

        stats = IndexStats(
            files=len(files),
            chunks=len(chunks),
            elapsed_seconds=round(time.time() - started, 2),
            backend="sqlite-vec" if self.store.vec_enabled else "json-fallback",
        )
        emit(f"索引完成：{stats.files} 文件 / {stats.chunks} 块 / {stats.elapsed_seconds}s / {stats.backend}")
        return stats
