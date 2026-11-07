"""execute_command 可选 OS 级沙箱包装（对齐 Java policy.CommandSandbox）。

ASTERCLI_COMMAND_SANDBOX=true 开启后，子进程在写文件系统围栏内运行——
仅项目根 / /tmp / ASTERCLI_SANDBOX_WRITABLE_PATHS 可写，其余全盘禁写（HOME 只读）。
读、执行、网络（默认）放行，落盘行为由内核拦截，用来补黑名单（命令文本）
与白名单（放行集合）共同盲区：子进程启动后的任意写路径。

平台实现：
- macOS：系统自带 /usr/bin/sandbox-exec + Seatbelt profile（-p 内联传参）
- Linux：bubblewrap（bwrap），全盘只读挂载 + 项目根 / /tmp / 追加路径可写 bind

fail-closed：显式开启但平台不支持（Windows）或工具缺失（Linux 无 bwrap）时拒绝执行，
不静默降级成裸 bash——与白名单“配置缺失全拒”同一设计语言。

配置（与 Java 版同名，可写在 .env / 环境变量）：
- ASTERCLI_COMMAND_SANDBOX：true / on 开启，默认关
- ASTERCLI_SANDBOX_WRITABLE_PATHS：逗号分隔追加可写路径，支持 ~ 展开
- ASTERCLI_SANDBOX_ALLOW_NET：默认 true；false 时断网
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .exceptions import PolicyException

if TYPE_CHECKING:  # 避免 policy -> config 的运行时耦合
    from ..config import Settings

MAC_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


class CommandSandbox:
    """命令包装器：未开启时原样返回 bash -c，开启时返回 sandbox-exec / bwrap 的完整 argv。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        os_name: str = "",
        sandbox_exec_path: Optional[str] = None,
        bwrap_path: Optional[str] = None,
        extra_writable_paths: Optional[List[str]] = None,
        allow_net: bool = True,
    ) -> None:
        self.enabled = enabled
        self.os_name = os_name or ""
        self.sandbox_exec_path = sandbox_exec_path
        self.bwrap_path = bwrap_path
        self.extra_writable_paths = list(extra_writable_paths or [])
        self.allow_net = allow_net

    @classmethod
    def disabled(cls) -> "CommandSandbox":
        return cls()

    @classmethod
    def from_settings(cls, settings: "Settings") -> "CommandSandbox":
        """生产入口：读配置 + 探测真实沙箱工具（对齐 Java fromEnvironment）。"""
        return cls(
            enabled=_parse_enabled(settings.get("ASTERCLI_COMMAND_SANDBOX")),
            os_name=platform.system(),
            sandbox_exec_path=probe_sandbox_exec(),
            bwrap_path=find_executable("bwrap"),
            extra_writable_paths=parse_writable_paths(
                settings.get("ASTERCLI_SANDBOX_WRITABLE_PATHS")
            ),
            allow_net=_parse_allow_net(settings.get("ASTERCLI_SANDBOX_ALLOW_NET")),
        )

    def wrap(self, command: str, project_root) -> List[str]:
        """把命令包装成完整 argv；沙箱未开启时返回 ["bash", "-c", command]。

        开启但平台 / 工具不可用时抛 PolicyException（fail-closed）。
        """
        if not self.enabled:
            return ["bash", "-c", command]
        if self._is_mac() and self.sandbox_exec_path:
            profile = build_seatbelt_profile(
                project_root, self.extra_writable_paths, self.allow_net
            )
            return [self.sandbox_exec_path, "-p", profile, "bash", "-c", command]
        if self._is_linux() and self.bwrap_path:
            return build_bwrap_argv(
                self.bwrap_path, project_root, self.extra_writable_paths, self.allow_net, command
            )
        raise PolicyException(
            "命令沙箱已开启但当前平台不可用（macOS 需要 /usr/bin/sandbox-exec，"
            "Linux 需要安装 bubblewrap）；如需继续执行请清除 ASTERCLI_COMMAND_SANDBOX"
        )

    # ---------- 平台判定 ----------

    def _is_mac(self) -> bool:
        # platform.system() 在 macOS 返回 "Darwin"，Java 的 os.name 是 "Mac OS X"，两者都要认
        name = self.os_name.lower()
        return "mac" in name or "darwin" in name

    def _is_linux(self) -> bool:
        return "linux" in self.os_name.lower()


# ---------- Seatbelt（macOS） ----------


def build_seatbelt_profile(project_root, extra_writable_paths, allow_net: bool) -> str:
    """生成 Seatbelt profile：读操作全部放行，写操作默认拒绝，仅白名单 subpath 可写。"""
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        # macOS 的 /tmp 是 /private/tmp 的符号链接，Seatbelt 按真实路径匹配，必须写后者
        '(allow file-write* (subpath "/private/tmp"))',
    ]
    root = canonicalize(project_root)
    if root:
        lines.append(f'(allow file-write* (subpath "{root}"))')
    for extra in extra_writable_paths:
        canonical = canonicalize(extra)
        if canonical and canonical != root:
            lines.append(f'(allow file-write* (subpath "{canonical}"))')
    if not allow_net:
        lines.append("(deny network*)")
    return "\n".join(lines) + "\n"


# ---------- bubblewrap（Linux） ----------


def build_bwrap_argv(
    bwrap_path: str, project_root, extra_writable_paths, allow_net: bool, command: str
) -> List[str]:
    """生成 bwrap argv：全盘只读挂载 + 可写路径 bind + dev/proc 补挂 + 可选断网。"""
    argv = [bwrap_path, "--ro-bind", "/", "/"]
    # 后挂载覆盖先挂载：ro-bind / / 之后再普通 --bind 同一路径即恢复可写
    root = canonicalize(project_root)
    if root:
        argv += ["--bind", root, root]
    argv += ["--bind", "/tmp", "/tmp"]
    for extra in extra_writable_paths:
        canonical = canonicalize(extra)
        if canonical:
            argv += ["--bind", canonical, canonical]
    # ro-bind 下的 /dev /proc 只读，bash 写 /dev/null 需要 --dev，ps 等需要 --proc
    argv += ["--dev", "/dev", "--proc", "/proc"]
    if not allow_net:
        argv.append("--unshare-net")
    argv += ["bash", "-c", command]
    return argv


# ---------- 辅助 ----------


def _parse_enabled(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("true", "on")


def _parse_allow_net(raw: Optional[str]) -> bool:
    """默认放行网络；显式 false（忽略大小写）断网，其他任意值维持默认。"""
    return (raw or "").strip().lower() != "false"


def parse_writable_paths(raw: Optional[str]) -> List[str]:
    """逗号分隔路径，去掉空白项并展开 ~；空配置返回空列表。"""
    if not raw:
        return []
    result = []
    for part in raw.split(","):
        expanded = os.path.expanduser(part.strip())
        if expanded:
            result.append(expanded)
    return result


def canonicalize(raw) -> Optional[str]:
    """尽量解析成真实绝对路径；不存在时退化为规范化绝对路径（fail-safe 不抛错）。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve())
    except (OSError, RuntimeError):
        try:
            return os.path.abspath(os.path.expanduser(text))
        except OSError:
            return None


def probe_sandbox_exec() -> Optional[str]:
    path = Path(MAC_SANDBOX_EXEC)
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def find_executable(name: str, path_env: Optional[str] = None) -> Optional[str]:
    """沿 PATH 逐目录查找可执行文件；找不到返回 None。"""
    if not name:
        return None
    raw_path = path_env if path_env is not None else os.environ.get("PATH", "")
    for directory in raw_path.split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
