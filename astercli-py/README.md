# AsterCLI Python 重写版（P1/P2/P3）

用 **LangChain + LangGraph** 重写 Java 版 AsterCLI 的核心路径，与仓库根的 Java 版并行演进：

- Java 版 = 生产版本，Python 侧改动不回写 Java 代码
- 分期路线图：仓库 `.trae/documents/python-langchain-rewrite.md`
- 当前已交付：P1（配置 + 7 provider + 流式 REPL）、P2（6 内置工具 + LangGraph ReAct + PathGuard/CommandGuard + 预算兜底）、P3（conversationHistory 自动压缩 + `/compact` + 长期记忆 `/save` `/memory`）

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

- `/quit` `/exit` 退出；`/clear` 清空会话历史（长期记忆保留）；`/model` 查看/切换 provider；`/tools` 查看工具列表
- `/compact` 手动压缩当前会话历史（保留最近 1 个轮次，早期轮次摘要为"[已压缩的历史对话摘要]"）
- `/save [内容|--global 内容]` 保存长期记忆（默认项目级作用域；`--global` 为跨项目全局）
- `/memory status|list|search <关键词>|delete <id>|clear` 长期记忆管理

## Memory（P3）

对齐 Java `ConversationHistoryCompactor` / `LongTermMemory` 语义：

- **自动压缩**：每轮 LLM 调用前评估历史 token，达到 `window - 20k - 13k` 阈值（200k 窗口 ≈ 167k）时把早期轮次摘要成一对 `[已压缩的历史对话摘要]` 消息，通过 LangGraph `RemoveMessage(REMOVE_ALL_MESSAGES)` 全量替换通道重建历史；摘要失败不破坏原历史
- **手动压缩**：`/compact` 立即压缩，保留最近 1 个轮次
- **长期记忆**：`/save` 存储到 `~/.astercli-py/memory/long_term_memory.json`（`ASTERCLI_MEMORY_DIR` 可覆盖），scope=project 按项目根过滤、global 全局可见；每轮任务前按关键词命中率 + 24h 时间衰减检索，以"## 相关长期记忆"段落前置注入 user message（Java 版注入 system prompt，Python 版保持 system prompt 稳定以友好 prompt cache）

## 与 Java 版的已知差异

- `grep_code` 暂为纯 Python 扫描引擎（Java 版优先 ripgrep，后续对齐）
- Java 的 shortTermMemory/ContextCompressor 未迁移（Java javadoc 自认度量错位属补丁前旧设计，Python 版只保留 conversationHistory 压缩一道）
- 长期记忆检索分词：Java 用 jieba，Python 用英文整词 + 中文 2-gram（不引 jieba 依赖）
- 记忆注入位置：user message 前置（见上）而非 system prompt
- LSP 诊断、HITL interrupt 审批、MCP、RAG、Plan/Multi-Agent 在后续分期
- `write_file` / `execute_command` 有最简化行内 y/n 审批（P4 换 LangGraph interrupt）
