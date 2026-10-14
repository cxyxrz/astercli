# AsterCLI Python 重写版（P1–P5）

用 **LangChain + LangGraph** 重写 Java 版 AsterCLI 的核心路径，与仓库根的 Java 版并行演进：

- Java 版 = 生产版本，Python 侧改动不回写 Java 代码
- 分期路线图：仓库 `.trae/documents/python-langchain-rewrite.md`
- 当前已交付：P1（配置 + 7 provider + 流式 REPL）、P2（6 内置工具 + LangGraph ReAct + PathGuard/CommandGuard + 预算兜底）、P3（conversationHistory 自动压缩 + `/compact` + 长期记忆 `/save` `/memory`）、P4（HITL interrupt 审批 + `/plan` Plan-Execute + DAG 审阅）、P5（MCP：官方 SDK + 动态工具 + resources mention）

## 运行

```bash
cd astercli-py
uv sync                                  # 需要 uv；首次会创建 .venv
uv run astercli                          # 交互 REPL，提示符 * ；沿用仓库根 ../.env 的 API Key
uv run astercli "总结一下 README.md"      # 一次性运行
uv run astercli --provider deepseek      # 显式指定 provider
uv run pytest                            # 全量单测
```

## REPL 命令

- `/quit` `/exit` 退出；`/clear` 清空会话历史（长期记忆保留，审批"全部放行"名单清空）；`/model` 查看/切换 provider；`/tools` 查看工具列表（`*` 标记需审批）
- `/compact` 手动压缩当前会话历史（保留最近 1 个轮次，早期轮次摘要为"[已压缩的历史对话摘要]"）
- `/save [内容|--global 内容]` 保存长期记忆（默认项目级作用域；`--global` 为跨项目全局）
- `/memory status|list|search <关键词>|delete <id>|clear` 长期记忆管理
- `/plan [任务]` 计划模式：无参数时下一条输入进入规划；带参数直接规划该任务
- `/mcp [status|resources <server>|restart <server>|disable <server>|enable <server>]` MCP server 管理

## HITL 审批（P4）

危险工具（`write_file` / `execute_command` 等，见 `hitl/policy.py`）执行前经 LangGraph `interrupt()` 挂起图，终端弹出审批框：

```
────────── ⚠️  HITL 审批请求 ──────────
🟡 中危 write_file
   风险: 将写入或覆盖文件内容，原有内容将丢失
   参数: {"path": "hello.txt", "content": "..."}

请选择操作：[y/Enter] 批准  [a] 全部放行  [n] 拒绝  [s] 跳过  [m] 修改参数
审批>
```

- `y`/Enter 批准本次；`a` 批准且本会话同工具后续自动放行（`/clear` 重置）；`n` 拒绝（可填原因，回灌给 LLM 调整方案）；`s` 跳过；`m` 输入修改后的 JSON 参数再执行
- 拦截顺序对齐 Java：HITL → ToolRegistry → PathGuard/CommandGuard
- fail-safe：非法输入连续 5 次、决策缺失（interrupt 丢失）、输入流关闭（非交互管道）一律保守拒绝，绝不默认放行
- 审批挂起依赖 checkpointer（`make_checkpointer()`：MemorySaver + pickle 兜底序列化）；未提供时审批整体禁用，危险工具直接执行（对齐 Java"HITL 未启用行为与父类相同"）

## Plan-Execute（P4）

`/plan` 触发：Planner 把复杂任务分解为 DAG（两遍 JSON 解析，容忍 LLM 自造 id 与前向引用；环检测失败即报错），展示计划摘要后进入审阅：

- **Enter** 按当前计划执行
- **Ctrl+O** 展开完整计划（再按 ESC 折叠回摘要）
- **ESC** 已展开时折叠；未展开时取消本次计划
- **I** 输入补充要求后重新规划

执行语义：按拓扑序逐任务执行（同层多任务提示"⚡ 本轮 N 个任务就绪"，但顺序执行保证审批与渲染不交错，任务级并行留后续）；单任务失败且计划进度 < 0.5 时自动 replan 并重新审阅；每任务独立 ReAct 图 + 5 轮硬上限 + 同一条 HITL 审批通道；最终结果汇总叶子任务输出。

## Memory（P3）

对齐 Java `ConversationHistoryCompactor` / `LongTermMemory` 语义：

- **自动压缩**：每轮 LLM 调用前评估历史 token，达到 `window - 20k - 13k` 阈值（200k 窗口 ≈ 167k）时把早期轮次摘要成一对 `[已压缩的历史对话摘要]` 消息，通过 LangGraph `RemoveMessage(REMOVE_ALL_MESSAGES)` 全量替换通道重建历史；摘要失败不破坏原历史
- **手动压缩**：`/compact` 立即压缩，保留最近 1 个轮次
- **长期记忆**：`/save` 存储到 `~/.astercli-py/memory/long_term_memory.json`（`ASTERCLI_MEMORY_DIR` 可覆盖），scope=project 按项目根过滤、global 全局可见；每轮任务前按关键词命中率 + 24h 时间衰减检索，以"## 相关长期记忆"段落前置注入 user message（Java 版注入 system prompt，Python 版保持 system prompt 稳定以友好 prompt cache）

## MCP（P5）

基于官方 `mcp` SDK 2.x（`ClientSessionGroup` 会话池），对齐 Java 版 MCP 行为：

- **配置**：用户级 `~/.astercli-py/mcp.json` 先读，项目级 `<项目根>/.astercli/mcp.json` 覆盖同名 server；顶层 key `mcpServers`。每条 `command`+`args`+`env`（stdio 子进程）或 `url`+`headers`+`type=sse|http`（远端），支持 `${VAR}` 展开（查找顺序：注入 env → 系统环境变量 → 仓库根 .env → 运行目录 .env → ~/.env），未定义变量该 server 置错不影响其余
- **动态工具**：连接后以 `mcp__{server}__{tool}` 注册进 ToolRegistry，schema 透传 SDK 原生 inputSchema；所有 MCP 工具 `dangerous=True`，经 P4 的 HITL 审批通道执行
- **resources**：server 支持时追加 `mcp__{server}__list_resources` / `mcp__{server}__read_resource` 虚拟工具
- **mention**：输入中 `@server:uri` 自动展开为 `<resource ...>` 块（20 万字符截断；引号内不展开；失败降级为 `<resource_error>` 不中断输入）
- **生命周期**：启动同步最多等 `ASTERCLI_MCP_STARTUP_WAIT_SECONDS`（默认 8 秒），未完成的保持 STARTING 后台续载，`/mcp` 查看最新状态；状态机 STARTING/READY/DISABLED/ERROR；后台线程常驻 event loop（会话跨 REPL 每轮 asyncio.run 存活）
- **审计**：mention 读取按 `allowByMention` 记审计；工具调用走统一审批审计链路

## 与 Java 版的已知差异

- `grep_code` 暂为纯 Python 扫描引擎（Java 版优先 ripgrep，后续对齐）
- Java 的 shortTermMemory/ContextCompressor 未迁移（Java javadoc 自认度量错位属补丁前旧设计，Python 版只保留 conversationHistory 压缩一道）
- 长期记忆检索分词：Java 用 jieba，Python 用英文整词 + 中文 2-gram（不引 jieba 依赖）
- 记忆注入位置：user message 前置（见上）而非 system prompt
- Plan-Execute 任务批次顺序执行（Java 4 线程并行）；审阅键位一致（Enter/Ctrl+O/ESC/I）
- MCP 用户级配置目录为 `~/.astercli-py`（与 Java `~/.astercli` 隔离）；step_search 自动内置未迁移；HTTP/SSE transport 由官方 SDK 保证（单测仅覆盖 stdio 集成）
- LSP 诊断、RAG、Multi-Agent、Skill 在后续分期
