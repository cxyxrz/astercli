"""第 1 层压缩：大结果落盘测试。"""

from __future__ import annotations

import asyncio

from astercli.config import Settings
from astercli.tools.registry import OFFLOAD_THRESHOLD_CHARS, ToolRegistry


def make_registry(tmp_path, result: str) -> ToolRegistry:
    registry = ToolRegistry(settings=Settings(load_files=False))
    registry.register("read_file", "读文件", {"type": "object"}, lambda args: result)
    registry.set_offload_root(tmp_path)
    return registry


def run(registry: ToolRegistry, path: str = "a.txt"):
    return asyncio.run(
        registry.execute_all([{"name": "read_file", "args": {"path": path}, "id": "c1"}])
    )


def test_大结果落盘并返回指针(tmp_path):
    big = "x" * (OFFLOAD_THRESHOLD_CHARS + 500)
    content = run(make_registry(tmp_path, big))[0].content
    assert "[offload]" in content
    assert len(content) < OFFLOAD_THRESHOLD_CHARS
    files = list((tmp_path / ".astercli" / "offload").glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == big


def test_小结果不落盘(tmp_path):
    content = run(make_registry(tmp_path, "ok"))[0].content
    assert content == "ok"
    assert not (tmp_path / ".astercli" / "offload").exists()


def test_读取落盘文件不二次落盘(tmp_path):
    big = "y" * (OFFLOAD_THRESHOLD_CHARS + 500)
    registry = make_registry(tmp_path, big)
    content = run(registry, path=".astercli/offload/old.txt")[0].content
    assert content == big


def test_未挂载根目录时关闭该层(tmp_path):
    registry = ToolRegistry(settings=Settings(load_files=False))
    registry.register("read_file", "读文件", {"type": "object"}, lambda args: "x" * 20_000)
    assert len(run(registry)[0].content) == 20_000
