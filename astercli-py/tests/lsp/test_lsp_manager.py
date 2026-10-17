"""LspManager 测试：post-edit 钩子收集/清除、flush 排序清空、开关与 tree-sitter 路径。

对齐 Java LspManagerTest 的两个核心用例（坏文件收集诊断 / 修复后清除），
并覆盖 Python 版扩展：.py 走 ast、其余语言复用 P6 tree-sitter、
ASTERCLI_LSP_ENABLED 关闭、severity 排序。
"""

from __future__ import annotations

from pathlib import Path

from astercli.config import Settings
from astercli.lsp import LspDiagnostic, LspManager, LspSeverity


def _manager(project: Path, env: dict | None = None) -> LspManager:
    return LspManager(project, Settings(env=env or {}, load_files=False))


def test_写入坏_python_文件后收集语法诊断(tmp_path):
    file = tmp_path / "broken.py"
    file.write_text("def broken(:\n    pass\n", encoding="utf-8")

    manager = _manager(tmp_path)
    manager.run_post_edit_lsp_hook("broken.py", file)

    report = manager.flush_pending_diagnostics()
    assert not report.is_empty()
    assert "[LSP 诊断注入]" in report.prompt_text
    assert "broken.py" in report.prompt_text
    assert "[error]" in report.prompt_text
    assert "invalid syntax" in report.prompt_text
    assert "(ast)" in report.prompt_text


def test_文件修复后诊断清除(tmp_path):
    file = tmp_path / "fixed.py"
    manager = _manager(tmp_path)

    file.write_text("def fixed(:\n", encoding="utf-8")
    manager.run_post_edit_lsp_hook("fixed.py", file)
    assert manager.pending_diagnostics_snapshot()

    file.write_text("def fixed():\n    pass\n", encoding="utf-8")
    manager.run_post_edit_lsp_hook("fixed.py", file)
    assert manager.flush_pending_diagnostics().is_empty()


def test_不受支持后缀清除该文件旧诊断(tmp_path):
    manager = _manager(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("纯文本不是代码", encoding="utf-8")
    # 手工注入旧诊断后触发 hook：.txt 不受支持 → 清除（对齐 Java isJavaFile 分支）
    manager._pending[txt.resolve()] = [
        LspDiagnostic(LspSeverity.ERROR, "notes.txt", 1, 1, "旧诊断", "ast")
    ]

    manager.run_post_edit_lsp_hook("notes.txt", txt)

    assert not manager.pending_diagnostics_snapshot()


def test_文件不存在时清除(tmp_path):
    manager = _manager(tmp_path)
    file = tmp_path / "gone.py"
    manager._pending[file.resolve()] = [
        LspDiagnostic(LspSeverity.ERROR, "gone.py", 1, 1, "旧诊断", "ast")
    ]
    manager.run_post_edit_lsp_hook("gone.py", file)
    assert not manager.pending_diagnostics_snapshot()


def test_开关关闭时不收集(tmp_path):
    file = tmp_path / "broken.py"
    file.write_text("def broken(:\n", encoding="utf-8")

    manager = _manager(tmp_path, {"ASTERCLI_LSP_ENABLED": "false"})
    manager.run_post_edit_lsp_hook("broken.py", file)

    assert manager.flush_pending_diagnostics().is_empty()


def test_开关仅显式_false_关闭(tmp_path):
    assert Settings(env={"ASTERCLI_LSP_ENABLED": "TRUE"}, load_files=False).lsp_enabled()
    assert Settings(env={"ASTERCLI_LSP_ENABLED": "0"}, load_files=False).lsp_enabled()
    assert not Settings(env={"ASTERCLI_LSP_ENABLED": "FALSE"}, load_files=False).lsp_enabled()
    assert Settings(load_files=False).lsp_enabled()


def test_坏_java_文件_tree_sitter_诊断(tmp_path):
    file = tmp_path / "Broken.java"
    file.write_text("class Broken {", encoding="utf-8")

    manager = _manager(tmp_path)
    manager.run_post_edit_lsp_hook("Broken.java", file)

    report = manager.flush_pending_diagnostics()
    assert not report.is_empty()
    assert "Broken.java" in report.prompt_text
    assert "[error]" in report.prompt_text
    assert "(tree-sitter)" in report.prompt_text


def test_好_java_文件无诊断(tmp_path):
    file = tmp_path / "Good.java"
    file.write_text("class Good {}\n", encoding="utf-8")

    manager = _manager(tmp_path)
    manager.run_post_edit_lsp_hook("Good.java", file)
    assert manager.flush_pending_diagnostics().is_empty()


def test_flush_排序并清空(tmp_path):
    manager = _manager(tmp_path)
    file = tmp_path / "multi.py"

    manager._pending[file.resolve()] = [
        LspDiagnostic(LspSeverity.WARNING, "b.py", 5, 1, "w2", "ast"),
        LspDiagnostic(LspSeverity.ERROR, "b.py", 9, 1, "e2", "ast"),
        LspDiagnostic(LspSeverity.ERROR, "a.py", 1, 1, "e1", "ast"),
        LspDiagnostic(LspSeverity.INFO, "a.py", 3, 1, "i1", "ast"),
    ]
    report = manager.flush_pending_diagnostics()

    lines = [ln for ln in report.prompt_text.splitlines() if ln.startswith("- [")]
    assert lines == [
        "- [error] a.py:1:1 e1 (ast)",
        "- [error] b.py:9:1 e2 (ast)",
        "- [warning] b.py:5:1 w2 (ast)",
        "- [info] a.py:3:1 i1 (ast)",
    ]
    # flush 取走语义：再次 flush 为空
    assert manager.flush_pending_diagnostics().is_empty()
    assert not manager.pending_diagnostics_snapshot()


def test_display_path_缺省回退项目根(tmp_path):
    file = tmp_path / "broken.py"
    file.write_text("def broken(:\n", encoding="utf-8")

    manager = _manager(tmp_path)
    manager.run_post_edit_lsp_hook("", file)

    report = manager.flush_pending_diagnostics()
    assert str(tmp_path) in report.prompt_text


def test_钩子异常静默不影响主路径(tmp_path):
    file = tmp_path / "broken.py"
    file.write_text("def broken(:\n", encoding="utf-8")
    manager = _manager(tmp_path)
    # display_path 非法（无 strip 方法）触发内部异常 → 钩子吞掉，不向调用方抛出
    manager.run_post_edit_lsp_hook(object(), file)
