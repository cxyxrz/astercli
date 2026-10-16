"""向量存储：sqlite-vec KNN 优先，JSON 全量余弦回退。

对齐 Java VectorStore 的接口语义（clear / insert / search / searchByKeyword），
存储结构重新设计而非移植：
- 每个项目一个独立库 `~/.astercli-py/rag/{key}.db`（key = 项目路径 sha1 前 12 位），
  替代 Java 单库多项目混合行
- sqlite-vec 可用（Python 构建支持 load_extension 且包可导入）时用 vec0 虚拟表 KNN；
  不可用时回退 JSON 列 + 内存余弦（等价 Java 手写余弦方案），保证跨环境可用
- 维度/模型变化（换 embedding 模型）时自动整库重建，避免维度错配静默查空
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .chunker import CodeChunk

DEFAULT_RAG_DIR = Path.home() / ".astercli-py" / "rag"


@dataclass
class SearchResult:
    file_path: str
    chunk_type: str
    name: str
    content: str
    start_line: int
    end_line: int
    similarity: float


def _cosine(a: list, b: list) -> float:
    """内存余弦相似度（回退路径与测试用）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _pack_f32(vector: list) -> bytes:
    """float32 小端序列化（sqlite-vec BLOB 入库格式）。"""
    import struct

    return struct.pack(f"<{len(vector)}f", *vector)


class VectorStore:
    """单项目向量库。"""

    def __init__(self, project_path: Path, base_dir: Optional[Path] = None) -> None:
        self.project_path = Path(project_path).resolve()
        base = Path(base_dir) if base_dir is not None else None
        if base is None:
            env_dir = os.environ.get("ASTERCLI_RAG_DIR")
            base = Path(env_dir) if env_dir else DEFAULT_RAG_DIR
        base.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(str(self.project_path).encode()).hexdigest()[:12]
        self.db_path = base / f"{key}.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._vec_enabled: Optional[bool] = None
        self._dim: Optional[int] = None
        self._model: Optional[str] = None

    # ---------- 连接与 schema ----------

    @property
    def vec_enabled(self) -> bool:
        """sqlite-vec 是否可用（首次访问时探测并缓存）。"""
        if self._vec_enabled is None:
            self._vec_enabled = self._try_load_vec()
        return self._vec_enabled

    def _try_load_vec(self) -> bool:
        try:
            import sqlite_vec  # noqa: F401

            conn = self._get_raw()
            conn.enable_load_extension(True)
            conn.load_extension(sqlite_vec.loadable_path())
            conn.enable_load_extension(False)
            # 用一次真实建表验证扩展可用（部分构建 load_extension 是空操作）
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS rag_vec_probe USING vec0(embedding float[2])"
            )
            conn.execute("DROP TABLE IF EXISTS rag_vec_probe")
            return True
        except Exception:
            return False

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_base_schema()
            self._load_meta()
        return self._conn

    def _create_base_schema(self) -> None:
        conn = self._get_raw()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON rag_chunks(file_path)")
        # 向量表（vec0 或 JSON）由 ensure_schema 按探测维度统一创建

    def _get_raw(self) -> sqlite3.Connection:
        """裸连接（_get_conn 初始化过程中使用，避免递归）。"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _load_meta(self) -> None:
        rows = dict(self._get_raw().execute("SELECT key, value FROM rag_meta").fetchall())
        self._dim = int(rows["dim"]) if "dim" in rows else None
        self._model = rows.get("model")

    def ensure_schema(self, dim: int, model: str) -> None:
        """按维度/模型准备向量表；换模型导致维度/模型变化时整库重建。"""
        conn = self._get_conn()
        rebuild = self._dim is not None and (self._dim != dim or self._model != model)
        if rebuild:
            self._drop_all_data()
        if self.vec_enabled:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS rag_vectors "
                f"USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
        else:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_json_vectors "
                "(chunk_id INTEGER PRIMARY KEY, embedding TEXT NOT NULL)"
            )
        conn.execute(
            "INSERT INTO rag_meta(key, value) VALUES ('dim', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(dim),),
        )
        conn.execute(
            "INSERT INTO rag_meta(key, value) VALUES ('model', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (model,),
        )
        conn.commit()
        self._dim, self._model = dim, model

    def _drop_all_data(self) -> None:
        conn = self._get_raw()
        conn.execute("DELETE FROM rag_chunks")
        for table in ("rag_vectors", "rag_json_vectors"):
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            except sqlite3.OperationalError:
                pass

    # ---------- 写入与查询 ----------

    def clear(self) -> None:
        """清空数据（保留 meta，下次 ensure_schema 判断维度）。"""
        conn = self._get_conn()
        conn.execute("DELETE FROM rag_chunks")
        try:
            conn.execute("DELETE FROM rag_vectors")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("DELETE FROM rag_json_vectors")
        except sqlite3.OperationalError:
            pass
        conn.commit()

    def insert(self, chunks: list, embeddings: list) -> None:
        """按序写入块与向量（长度必须一致）。"""
        if len(chunks) != len(embeddings):
            raise ValueError(f"块数 {len(chunks)} 与向量数 {len(embeddings)} 不一致")
        if not chunks:
            return
        conn = self._get_conn()
        vec_table = "rag_vectors" if self.vec_enabled else "rag_json_vectors"
        for chunk, vector in zip(chunks, embeddings):
            if self._dim is not None and len(vector) != self._dim:
                raise ValueError(f"向量维度 {len(vector)} 与索引维度 {self._dim} 不一致")
            cur = conn.execute(
                "INSERT INTO rag_chunks(file_path, chunk_type, name, content, start_line, end_line) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk.file_path, chunk.chunk_type, chunk.name, chunk.content,
                 chunk.start_line, chunk.end_line),
            )
            chunk_id = cur.lastrowid
            if self.vec_enabled:
                conn.execute(
                    f"INSERT INTO {vec_table}(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, _pack_f32(vector)),
                )
            else:
                conn.execute(
                    f"INSERT INTO {vec_table}(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, json.dumps(vector)),
                )
        conn.commit()

    def count(self) -> int:
        return self._get_conn().execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]

    def _row_to_result(self, row, similarity: float) -> SearchResult:
        return SearchResult(row[0], row[1], row[2], row[3], row[4], row[5], similarity)

    def search(self, embedding: list, top_k: int) -> list:
        """语义近邻检索：vec0 KNN 优先，回退全量 JSON 余弦。"""
        if top_k <= 0:
            return []
        conn = self._get_conn()
        if self.vec_enabled:
            try:
                rows = conn.execute(
                    "SELECT c.file_path, c.chunk_type, c.name, c.content, c.start_line, c.end_line, v.distance "
                    "FROM rag_vectors v JOIN rag_chunks c ON c.chunk_id = v.chunk_id "
                    "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                    (_pack_f32(embedding), top_k),
                ).fetchall()
                # vec0 distance=1-cosine，换回相似度口径
                return [self._row_to_result(r[:6], 1.0 - r[6]) for r in rows]
            except sqlite3.OperationalError:
                pass  # 表不存在/维度不匹配 → 走回退返回空
        # JSON 回退：全量取回内存余弦
        try:
            rows = conn.execute(
                "SELECT c.chunk_id, c.file_path, c.chunk_type, c.name, c.content, c.start_line, c.end_line, j.embedding "
                "FROM rag_json_vectors j JOIN rag_chunks c ON c.chunk_id = j.chunk_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # 向量表尚未建立（未索引）
        scored = []
        for r in rows:
            try:
                vector = json.loads(r[7])
            except (json.JSONDecodeError, TypeError):
                continue
            scored.append((_cosine(embedding, vector), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._row_to_result(r[1:8], s) for s, r in scored[:top_k]]

    def search_by_keyword(self, terms: list, top_k: int, base_similarity: float = 0.15) -> list:
        """关键词检索（LIKE 命中 + 固定基础相似度）。

        base_similarity 取 0.15 而非 Java 的 0.3：现代 embedding 模型的余弦
        绝对值普遍偏低（语义最相关块常在 0.2 上下），0.3 的底分会让纯关键词
        命中的噪音块压过真实语义命中，融合排序失真。
        """
        if not terms or top_k <= 0:
            return []
        conn = self._get_conn()
        conditions = []
        params: list = []
        for term in terms:
            conditions.append("(content LIKE ? OR name LIKE ? OR file_path LIKE ?)")
            like = f"%{term}%"
            params.extend([like, like, like])
        sql = (
            "SELECT file_path, chunk_type, name, content, start_line, end_line "
            "FROM rag_chunks WHERE " + " OR ".join(conditions) + " LIMIT ?"
        )
        params.append(top_k)
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_result(r, base_similarity) for r in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
