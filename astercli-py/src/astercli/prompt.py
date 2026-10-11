"""系统提示组装（P2 精简版：身份 + 环境 + 工具准则 + ASTER.md 项目记忆注入）。

完整 PromptAssembler 分层（memory 检索注入、Skill 索引等）在后续分期对齐。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

ASTER_MD_CHAR_BUDGET = 8_000


def build_system_prompt(project_root: Path) -> str:
    now = datetime.now().astimezone()
    tz_name = getattr(now.tzinfo, "tzname", lambda *_: None)(now) or "UTC"
    lines = [
        "你是 AsterCLI，一个面向商业使用的 Agent CLI 编码助手，运行在用户终端中。",
        "",
        "环境信息:",
        f"- 当前日期: {now.strftime('%Y-%m-%d %A')}",
        f"- 时区: {tz_name}",
        f"- 项目根目录: {project_root}（所有文件读写都被限制在该目录内）",
        "",
        "工具使用准则:",
        "- 定位代码的顺序：glob_files 找候选文件 → grep_code 精确定位 → read_file 按行读取上下文",
        "- 修改文件用 write_file；运行短命令用 execute_command（60 秒超时）",
        "- 工具结果带 partial: true 或 suggested_reads 时，按提示缩小范围或读取建议行段",
        "- 路径越界和危险命令会被策略层拒绝，请调整方案而不是重复尝试",
        "",
        "回答使用与用户提问相同的语言。",
    ]
    aster = _load_aster_md(project_root)
    if aster:
        lines.append("")
        lines.append("项目记忆（ASTER.md）:")
        lines.append(aster)
    return "\n".join(lines)


def _load_aster_md(project_root: Path) -> str:
    for candidate in (project_root / "ASTER.md", project_root / ".astercli" / "ASTER.md"):
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    if len(text) > ASTER_MD_CHAR_BUDGET:
                        text = text[:ASTER_MD_CHAR_BUDGET] + "\n…（已按预算截断）"
                    return text
        except OSError:
            continue
    return ""
