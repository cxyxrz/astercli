"""PathGuard 语义测试：项目根限定 / .. 穿越 / 符号链接逃逸 / 不存在路径。"""

from __future__ import annotations

import os

import pytest

from astercli.policy import PathGuard, PolicyException


class TestBasic:
    def test_empty_root_rejected(self):
        with pytest.raises(ValueError):
            PathGuard("  ")

    def test_relative_path_resolved_inside_root(self, project):
        guard = PathGuard(project)
        safe = guard.resolve_safe("README.md")
        assert safe == project / "README.md"
        assert safe.is_file()

    def test_absolute_path_inside_root_ok(self, project):
        guard = PathGuard(project)
        safe = guard.resolve_safe(str(project / "src" / "Main.java"))
        assert safe.is_file()
        assert guard.root_path in safe.parents

    def test_root_itself_allowed(self, project):
        guard = PathGuard(project)
        assert guard.resolve_safe(".") == guard.root_path


class TestEscape:
    def test_absolute_escape_rejected(self, project):
        guard = PathGuard(project)
        with pytest.raises(PolicyException, match="路径越界"):
            guard.resolve_safe("/etc/passwd")

    def test_dotdot_escape_rejected(self, project):
        guard = PathGuard(project)
        with pytest.raises(PolicyException, match="路径越界"):
            guard.resolve_safe("../outside.txt")

    def test_nested_dotdot_escape_rejected(self, project):
        guard = PathGuard(project)
        with pytest.raises(PolicyException):
            guard.resolve_safe("src/../../escape.txt")

    def test_empty_path_rejected(self, project):
        guard = PathGuard(project)
        with pytest.raises(PolicyException):
            guard.resolve_safe("   ")


class TestNonExistent:
    def test_new_file_inside_root_ok(self, project):
        """write_file 新建文件场景：路径不存在也要能校验。"""
        guard = PathGuard(project)
        safe = guard.resolve_safe("src/new/Deep.java")
        assert guard.root_path in safe.parents
        assert not safe.exists()

    def test_new_file_dotdot_escape_rejected(self, project):
        guard = PathGuard(project)
        with pytest.raises(PolicyException):
            guard.resolve_safe("new/../../escape.txt")


class TestSymlink:
    def test_symlink_escape_rejected(self, project, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.txt").write_text("s3cret", encoding="utf-8")
        link = project / "link"
        os.symlink(outside, link)
        guard = PathGuard(project)
        with pytest.raises(PolicyException, match="路径越界"):
            guard.resolve_safe("link/secret.txt")

    def test_symlink_inside_root_ok(self, project):
        os.symlink(project / "README.md", project / "alias.md")
        guard = PathGuard(project)
        safe = guard.resolve_safe("alias.md")
        assert safe.is_file()

    def test_root_on_tmpdir_symlink_normalized(self, project, monkeypatch):
        """macOS /var → /private/var：根本身经符号链接时目标仍应可校验。"""
        base = project
        # 通过 /var 前缀构造符号链接根（macOS 上 /var -> /private/var）
        if not os.path.exists("/var/tmp"):
            pytest.skip("非 macOS 布局")
        linked_root = "/var/tmp/astercli-guard-root"
        if os.path.islink(linked_root) or os.path.exists(linked_root):
            import shutil

            shutil.rmtree(linked_root, ignore_errors=True)
        os.symlink(base, linked_root, target_is_directory=True)
        try:
            guard = PathGuard(linked_root)
            safe = guard.resolve_safe("README.md")
            assert safe.is_file()
            with pytest.raises(PolicyException):
                guard.resolve_safe("../escape.txt")
        finally:
            os.unlink(linked_root)
