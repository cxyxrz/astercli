"""命令快速拒绝黑名单（对齐 Java policy.CommandGuard 的 9 条规则与正则）。

定位：辅助防线而非主防线——黑名单永远列不全，但能拦住明显破坏性命令。
不做完整 shell 解析，只做正则模式匹配；命令替换段 $(...) / 反引号以原文一并扫描。
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_RULES: Tuple[Tuple[str, str], ...] = (
    ("禁止 sudo 提权", r"(?i)\bsudo\b"),
    # rm 路径黑名单：/、~、$HOME 及其子路径前缀全拦
    (
        "禁止 rm -rf 删除全盘或用户目录",
        r"(?i)\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(/|~|\$home)"
        r"|\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+(/|~|\$home)",
    ),
    ("禁止 mkfs 格式化磁盘", r"(?i)\bmkfs(\.|\b)"),
    ("禁止 dd 写入裸设备", r"(?i)\bdd\b[^\n]*\bof=/dev/"),
    (
        "识别为 fork bomb",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    ),
    (
        "禁止 curl / wget 管道直接执行远端脚本",
        r"(?i)\b(curl|wget)\b[^|\n]*\|\s*(sh|bash|zsh|fish|ksh)\b",
    ),
    ("不允许扫描 /、~ 或整个文件系统", r"(?i)\bfind\s+(/|~|\$home)"),
    ("禁止 chmod 777 全盘", r"(?i)\bchmod\s+-R\s+777\s+(/|~)"),
    ("禁止 shutdown / reboot / halt", r"(?i)\b(shutdown|reboot|halt|poweroff)\b"),
)

_COMPILED = tuple((reason, re.compile(pattern)) for reason, pattern in _RULES)


def check(command: Optional[str]) -> Optional[str]:
    """返回 None 表示放行；非 None 为拒绝原因。"""
    if command is None or not command.strip():
        return None
    normalized = " ".join(command.split())
    for reason, pattern in _COMPILED:
        if pattern.search(normalized):
            return reason
    return None
