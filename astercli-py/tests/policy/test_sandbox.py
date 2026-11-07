"""CommandSandbox 命令沙箱测试：纯函数生成 profile / argv，不真正执行沙箱。"""

from __future__ import annotations

import os

import pytest

from astercli.config import Settings
from astercli.policy import CommandSandbox, PathGuard, PolicyException
from astercli.policy.sandbox import (
    build_bwrap_argv,
    build_seatbelt_profile,
    canonicalize,
    find_executable,
    parse_writable_paths,
)
from astercli.tools.builtin.shell import register_shell_tools
from astercli.tools.registry import ToolRegistry


def make_mac(**overrides) -> CommandSandbox:
    params = dict(
        enabled=True, os_name="Darwin", sandbox_exec_path="/usr/bin/sandbox-exec", allow_net=True
    )
    params.update(overrides)
    return CommandSandbox(**params)


# ---------- 未开启 ----------


def test_disabled_returns_plain_bash():
    sandbox = CommandSandbox.disabled()
    assert sandbox.enabled is False
    assert sandbox.wrap("ls -la", "/tmp") == ["bash", "-c", "ls -la"]


def test_from_settings_disabled_by_default():
    sandbox = CommandSandbox.from_settings(Settings(load_files=False, env={}))
    assert sandbox.enabled is False
    assert sandbox.allow_net is True
    assert sandbox.extra_writable_paths == []


# ---------- 配置解析 ----------


@pytest.mark.parametrize("raw", ["true", "TRUE", "on", " On "])
def test_switch_values_enable(raw):
    sandbox = CommandSandbox.from_settings(
        Settings(load_files=False, env={"ASTERCLI_COMMAND_SANDBOX": raw})
    )
    assert sandbox.enabled is True


@pytest.mark.parametrize("raw", ["yes", "1", "enabled", "false"])
def test_other_switch_values_stay_disabled(raw):
    sandbox = CommandSandbox.from_settings(
        Settings(load_files=False, env={"ASTERCLI_COMMAND_SANDBOX": raw})
    )
    assert sandbox.enabled is False


def test_allow_net_only_explicit_false_disables():
    files = {"ASTERCLI_SANDBOX_ALLOW_NET": "FALSE"}
    settings = Settings(load_files=False, env=files)
    assert CommandSandbox.from_settings(settings).allow_net is False
    settings = Settings(load_files=False, env={"ASTERCLI_SANDBOX_ALLOW_NET": "no"})
    assert CommandSandbox.from_settings(settings).allow_net is True


def test_parse_writable_paths_expands_tilde_and_drops_blank():
    paths = parse_writable_paths(" ~/.m2 , ,/tmp/build, ")
    assert paths == [os.path.expanduser("~/.m2"), "/tmp/build"]
    assert parse_writable_paths(None) == []
    assert parse_writable_paths("") == []


# ---------- macOS Seatbelt ----------


def test_seatbelt_profile_allows_only_whitelisted_writes(tmp_path):
    profile = build_seatbelt_profile(tmp_path, [str(tmp_path / "cache")], allow_net=True)
    assert profile.startswith("(version 1)")
    assert "(allow default)" in profile
    assert "(deny file-write*)" in profile
    assert '(allow file-write* (subpath "/private/tmp"))' in profile
    assert f'(allow file-write* (subpath "{tmp_path.resolve()}"))' in profile
    assert f'(allow file-write* (subpath "{(tmp_path / "cache").resolve()}"))' in profile
    assert "network" not in profile


def test_seatbelt_profile_denies_network_when_disabled(tmp_path):
    profile = build_seatbelt_profile(tmp_path, [], allow_net=False)
    assert "(deny network*)" in profile


def test_seatbelt_profile_skips_duplicate_root(tmp_path):
    profile = build_seatbelt_profile(tmp_path, [str(tmp_path)], allow_net=True)
    root_rule = f'(allow file-write* (subpath "{tmp_path.resolve()}"))'
    assert profile.count(root_rule) == 1


def test_mac_wrap_builds_sandbox_exec_argv(tmp_path):
    argv = make_mac().wrap("echo hi", tmp_path)
    assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert argv[-3:] == ["bash", "-c", "echo hi"]
    assert f'(subpath "{tmp_path.resolve()}")' in argv[2]


# ---------- Linux bubblewrap ----------


def test_bwrap_argv_layout(tmp_path):
    build = tmp_path / "build"
    argv = build_bwrap_argv("/usr/bin/bwrap", tmp_path, [str(build)], True, "echo hi")
    assert argv[:4] == ["/usr/bin/bwrap", "--ro-bind", "/", "/"]
    assert ["--bind", str(tmp_path.resolve()), str(tmp_path.resolve())] == argv[4:7]
    assert ["--bind", "/tmp", "/tmp"] == argv[7:10]
    assert ["--bind", str(build.resolve()), str(build.resolve())] == argv[10:13]
    assert argv[13:] == ["--dev", "/dev", "--proc", "/proc", "bash", "-c", "echo hi"]


def test_bwrap_argv_unshare_net_when_disabled(tmp_path):
    argv = build_bwrap_argv("/usr/bin/bwrap", tmp_path, [], False, "ls")
    assert "--unshare-net" in argv
    assert argv[-3:] == ["bash", "-c", "ls"]


def test_linux_wrap_uses_bwrap_path(tmp_path):
    sandbox = CommandSandbox(
        enabled=True, os_name="Linux", bwrap_path="/usr/bin/bwrap", allow_net=True
    )
    argv = sandbox.wrap("ls", tmp_path)
    assert argv[0] == "/usr/bin/bwrap"


# ---------- fail-closed ----------


def test_enabled_on_unsupported_platform_rejected():
    sandbox = CommandSandbox(enabled=True, os_name="Windows")
    with pytest.raises(PolicyException, match="沙箱已开启但当前平台不可用"):
        sandbox.wrap("ls", "/tmp")


def test_mac_without_sandbox_exec_rejected():
    sandbox = CommandSandbox(enabled=True, os_name="Darwin", sandbox_exec_path=None)
    with pytest.raises(PolicyException):
        sandbox.wrap("ls", "/tmp")


def test_linux_without_bwrap_rejected():
    sandbox = CommandSandbox(enabled=True, os_name="Linux", bwrap_path=None)
    with pytest.raises(PolicyException):
        sandbox.wrap("ls", "/tmp")


# ---------- 辅助函数 ----------


def test_canonicalize_missing_path_returns_absolute(tmp_path):
    missing = tmp_path / "not-created"
    assert canonicalize(missing) == str(missing.resolve())
    assert canonicalize("") is None
    assert canonicalize(None) is None


def test_find_executable_hit_and_miss(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "bwrap"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    assert find_executable("bwrap", path_env=str(bin_dir)) == str(tool)
    assert find_executable("bwrap", path_env=str(tmp_path / "empty")) is None
    assert find_executable("", path_env=str(bin_dir)) is None


# ---------- execute_command 接线 ----------


class _FakeCompleted:
    stdout = "ok"
    stderr = ""
    returncode = 0


def test_shell_tool_uses_argv_and_no_shell_when_enabled(monkeypatch, tmp_path):
    from astercli.tools.builtin import shell as shell_module

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["shell"] = kwargs.get("shell")
        return _FakeCompleted()

    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)
    registry = ToolRegistry(settings=Settings(load_files=False), approval=None)
    register_shell_tools(registry, PathGuard(tmp_path), sandbox=make_mac())
    out = registry.get("execute_command").func({"command": "echo hi"})

    assert "命令输出 (exit 0)" in out
    assert captured["shell"] is False
    assert captured["cmd"][0] == "/usr/bin/sandbox-exec"
    assert captured["cmd"][-3:] == ["bash", "-c", "echo hi"]


async def test_shell_tool_unavailable_sandbox_is_policy_rejection(tmp_path):
    registry = ToolRegistry(settings=Settings(load_files=False), approval=None)
    register_shell_tools(
        registry, PathGuard(tmp_path), sandbox=CommandSandbox(enabled=True, os_name="Windows")
    )
    results = await registry.execute_all(
        [{"name": "execute_command", "args": {"command": "echo hi"}, "id": "c1"}]
    )
    assert results[0].content.startswith("🛡️")
    assert "沙箱已开启但当前平台不可用" in results[0].content


def test_shell_tool_blacklist_still_wins(tmp_path):
    registry = ToolRegistry(settings=Settings(load_files=False), approval=None)
    register_shell_tools(registry, PathGuard(tmp_path), sandbox=make_mac())
    with pytest.raises(PolicyException, match="禁止 sudo"):
        registry.get("execute_command").func({"command": "sudo rm file"})
