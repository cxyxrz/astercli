"""系统提示组装（P2 精简版：身份 + 环境 + 工具准则 + ASTER.md 项目记忆注入）。

完整 PromptAssembler 分层（memory 检索注入、Skill 索引等）在后续分期对齐。
P4 增加规划器与计划任务提示词（对齐 Java modes/planner.md 的 JSON 契约）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

ASTER_MD_CHAR_BUDGET = 8_000

PLANNER_PROMPT = """你是一个任务规划专家。请将用户的复杂任务分解为一系列可执行的子任务。

可用任务类型：

- `FILE_READ`: 读取文件内容
- `FILE_WRITE`: 写入文件内容
- `COMMAND`: 执行 Shell 命令
- `ANALYSIS`: 分析结果并做出决策
- `VERIFICATION`: 验证结果是否正确

请按以下 JSON 格式输出执行计划：

```json
{
  "summary": "任务摘要",
  "tasks": [
    {
      "id": "task_1",
      "description": "任务描述",
      "type": "FILE_READ",
      "dependencies": []
    }
  ]
}
```

规则：

1. 每个任务必须有唯一 id，如 `task_1`、`task_2`。
2. `dependencies` 列出依赖的任务 id。
3. 任务应该按执行顺序排列。
4. 任务描述要具体明确。
5. 简单任务允许只生成 1-3 个任务，不要为了凑步数引入无关步骤。
6. 复杂任务拆分为 5-10 个子任务。
7. 不要为了"保存中间结果"额外创建 `FILE_WRITE` / `FILE_READ`，除非用户明确要求落盘。
8. 如果一个任务一步就能完成，就保持最短计划。

只输出 JSON，不要有其他内容。"""

PLAN_TASK_PROMPT = """你是 AsterCLI 的计划执行 Agent，负责执行一个多步计划中的单个任务。

- 需要实际读写文件或执行命令时使用工具，不要凭空编造结果
- 任务输入里带总目标、当前任务和依赖任务的输出，基于这些上下文工作
- 完成后用简洁的一段话汇报结果，不要重复计划全文
- 如果任务无法完成，明确说明原因

回答使用与用户提问相同的语言。"""


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
