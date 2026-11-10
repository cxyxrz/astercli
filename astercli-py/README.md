# AsterCLI Python 重写版（P1–P11）

用 **LangChain + LangGraph** 重写 Java 版 AsterCLI 的核心路径，与仓库根的 Java 版并行演进：

- Java 版 = 生产版本，Python 侧改动不回写 Java 代码
- 分期路线图：仓库 `.trae/documents/python-langchain-rewrite.md`
- 当前已交付：P1（配置 + 7 provider + 流式 REPL）、P2（6 内置工具 + LangGraph ReAct + PathGuard/CommandGuard + 预算兜底）、P3（conversationHistory 自动压缩 + `/compact` + 长期记忆 `/save` `/memory`）、P4（HITL interrupt 审批 + `/plan` Plan-Execute + DAG 审阅）、P5（MCP：官方 SDK + 动态工具 + resources mention）、P6（RAG：tree-sitter 分块 + sqlite-vec 向量 + 混合检索）、P7（Multi-Agent `/team` + Skill `/skill` + web 工具）、P8（LSP 诊断：write_file 出口钩子 + 下一轮 user message 注入）、P9（反思经验：三类触发自动复盘 + Markdown 归档 + `/experience`）、P10（AutoDream：会话转写 + 五道门 + 后台事务化记忆整合 + `/dream`）、P11（终端渲染对齐 Java inline 形态：思考活动区 + 工具折叠块 + 行内 diff + 代码围栏折叠 + 底部状态栏）

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

- `/quit` `/exit` 退出；`/clear` 清空会话历史（长期记忆与经验库保留，审批"全部放行"名单与会话反思配额重置）；`/model` 查看/切换 provider；`/tools` 查看工具列表（`*` 标记需审批）
- `/compact` 手动压缩当前会话历史（保留最近 1 个轮次，早期轮次摘要为"[已压缩的历史对话摘要]"）
- `/save [内容|--global 内容]` 保存长期记忆（默认项目级作用域；`--global` 为跨项目全局）
- `/memory status|list|search <关键词>|delete <id>|clear` 长期记忆管理
- `/plan [任务]` 计划模式：无参数时下一条输入进入规划；带参数直接规划该任务
- `/team [任务]` 多 Agent 协作：无参数时下一条输入进入团队协作；带参数直接协作该任务
- `/skill [list|show <name>|off <name>|on <name>|reload]` Skill 管理（三层目录加载，banner 显示启用数）
- `/experience [list|search <关键词>|show <id>|delete <id>|clear [--global|--all]|on|off]` 经验管理（P9；`list` 支持 `--project` / `--global` 过滤）
- `/dream [status|log <id>|undo [id]|on|off]` 后台记忆整合（P10；无参数立即整合一次，`status` 看门判定与统计）
- `/mcp [status|resources <server>|restart <server>|disable <server>|enable <server>]` MCP server 管理
- `/index [路径]` 对项目（或项目内子目录）建立/重建 RAG 向量索引
- `/search <query>` 不经过 LLM 直接做语义混合检索

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
- 工具参数在执行前按 Schema 做浅层校验（required 齐全 / 基本类型 / enum），失败直接以「参数校验失败: …」文本回给 LLM 修正，不进入审批与执行；`m` 修改后的参数走同一校验（Java 仅在 modify 提交时校验 required）
- 可选 OS 级命令沙箱（与 Java 同名同语义）：`ASTERCLI_COMMAND_SANDBOX=true` 后 `execute_command` 子进程在写围栏内运行——仅项目根 / `/tmp` / `ASTERCLI_SANDBOX_WRITABLE_PATHS` 可写，HOME 只读；macOS 用 `sandbox-exec`，Linux 需 `bubblewrap`，`ASTERCLI_SANDBOX_ALLOW_NET=false` 断网。开启但平台/工具不可用时 fail-closed 拒绝执行，不静默降级
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

## RAG（P6）

对齐 Java `rag/` 包职责（`search_code` 语义辅助定位：模糊自然语言、常规搜索无果时用；精确符号/字符串定位仍优先 `grep_code`）：

- **分块**：tree-sitter AST 符号级（py/java/js/ts/tsx/go/rs/c/cpp 八语言）——class 块取声明头 5 行、method 块取完整函数体；无符号文件回退整文件块，超 2000 字符按行边界分段
- **向量**：`sqlite-vec` vec0 虚拟表 KNN（float32 BLOB），扩展不可用的环境自动回退 JSON 列 + 余弦扫描；每个项目独立库 `~/.astercli-py/rag/{hash}.db`，换 embedding 模型自动整库重建
- **embedding**：`EMBEDDING_*` 配置（与 Java/.env 同名）；`ollama` 自动补 `http://localhost:11434/v1` 端点，或任意 OpenAI 兼容云服务（base_url + api_key + model）；单条 2000 字符截断
- **检索**：语义 + 关键词（LIKE）双路混合——双路命中加分、method/class 类型加分、单文件最多 2 条；查询分词零依赖（ASCII 整词 + 中文 2-gram）
- **工具与命令**：`search_code` 工具（只读不审批，LLM 可调用）；`/index [路径]` 建立索引（全量重建式，进度实时显示）；`/search <query>` 直接检索不经 LLM

## Multi-Agent / Skill / Web（P7）

### `/team` 多 Agent 协作

对齐 Java `AgentOrchestrator` 主从架构（planner → worker → reviewer）：

- **规划**：planner 单轮流式输出步骤计划 JSON（FILE_READ/FILE_WRITE/COMMAND/ANALYSIS/VERIFICATION 五类），id 统一重编号 `step_N` 并映射依赖
- **执行**：按依赖批次推进；worker 复用 P2 的 ReAct 图（工具 / budget 兜底 / reasoning 流式 / HITL 审批全量继承），输入含总任务上下文与已完成依赖步骤的结果摘要（500 字符预览）
- **审查**：reviewer 输出 `{"approved", "issues", "suggestions", "summary"}`；不通过带反馈重试（每步最多 2 次），超限保留结果并警示；审查 LLM 调用失败兜底放行（保留执行结果）
- 前置步骤失败的后续步骤按"跳过"处理；最终汇总各步骤状态与结果预览

### Skill 系统

对齐 Java `skill/` 包（沉淀决策与经验的复用单元）：

- **三层目录**：`~/.astercli-py/skills/<name>/SKILL.md`（用户级）+ `<项目根>/.astercli/skills/<name>/SKILL.md`（项目级，同名整体覆盖）
- **SKILL.md**：frontmatter（name/description/version/author/tags，零 PyYAML 解析器，嵌套对象降级警告）+ body 指引正文
- **索引段**：启用 skill 列表注入 system prompt 末尾（≤20 个 / ≤4KB，`/skill off|on|reload` 即时刷新）
- **load_skill 工具**：LLM 按索引段触发场景调用，body 经 `SkillContextBuffer`（LRU≤3）前置注入下一轮 user message（`## 已加载 Skill：<name>` 段）；`/clear` 清空 buffer
- **启用状态**：`~/.astercli-py/skills.json` 持久化 disabled 列表

### Web 工具

- **web_search(query, top_k=5)**：三 provider——zhipu（与 GLM 推理共用 `GLM_API_KEY`，`ZHIPU_SEARCH_ENGINE` 可选引擎）/ serpapi（`SERPAPI_KEY`）/ searxng（`SEARXNG_URL` 自托管）；`SEARCH_PROVIDER` 显式指定，否则按 Key 存在性自动选择
- **web_fetch(url, max_chars=8000)**：`NetworkPolicy` SSRF 围栏（仅 http/https + DNS 解析后拒绝环回/内网/链路本地地址 + 60s/30 次限流）→ 流式抓取（5MB 上限）→ BeautifulSoup 极简 readability 转 Markdown（标题/正文/表格/代码块/相对链接补全）；SPA 空正文提示改用 web_search
- 两者均只读不审批；失败转文本回给 LLM 自行调整

## 反思经验（P9）

三层设计交付前两层 + 最小注入（完整动态检索策略留 P10）：

- **层一 触发与复盘**：三类信号自动进入反思队列——工具失败（异常 / 超时 / 策略拒绝）、HITL 用户拒绝（含拒绝原因）、LSP error 级诊断；每轮任务结束后用一次**独立 LLM 调用**提炼 `lesson / correct_action / category / confidence` 四项 JSON，不进入主对话历史
- **层二 持久归档**：一经验一 Markdown——项目级 `<项目根>/.astercli/experience/exp-<id>.md`（可随 git 共享）+ 全局 `~/.astercli-py/experience/`（`ASTERCLI_EXPERIENCE_DIR` 可覆盖）；frontmatter 存 category/trigger/scope/created/hits/signature，正文四段（教训 / 正确做法 / 触发信号 / 证据）；目录内维护 `index.md`；单目录超 200 条按 `hits × 时效` 淘汰；同内容经验合并为 hits +1
- **层三 最小注入**：会话启动时把经验索引段注入 system prompt（一次，避免前缀抖动）；每轮按当前输入检索 top-K 相关经验前置注入 user message（普通轮 + Plan 任务级均覆盖）
- **护栏**：`ASTERCLI_REFLECTION_ENABLED`（默认开，仅显式 false 关闭）/ `ASTERCLI_REFLECTION_SESSION_LIMIT`（默认 3，按尝试计数）/ `ASTERCLI_REFLECTION_TIMEOUT_SECONDS`（默认 30）/ `ASTERCLI_EXPERIENCE_MAX_ENTRIES`（默认 200）；会话内同触发签名只反思一次；`confidence=low`、空 lesson、JSON 解析失败、LLM 异常一律丢弃，绝不影响主任务；`/experience off` 同时关闭复盘与注入，`/clear` 重置会话配额但保留经验库
- **长期记忆 vs 经验**：`/save` 保存的是**事实**（手动、稳定、跨会话）；经验是**自动复盘**的操作教训（按失败信号沉淀、可删除、可命中累加），两条通道互不替代

## AutoDream 后台记忆整合（P10）

Claude Code 式 auto-dream 的 Python 版实现：把"每次都要手动 `/save`"变成"启动期后台离线整理"。

- **会话转写**：每轮只记最小信号（用户原文 / 最终回答 / 工具名，单条截断 2000 字，不落工具参数与完整输出），按项目哈希写到 `~/.astercli-py/sessions/<hash>/<session-id>.jsonl`；`ASTERCLI_SESSION_LOG=false` 关闭
- **五道门**：开关 → 距上次整合 ≥24h → 距上次扫描 ≥10min → 上次整合后新增 ≥5 个会话 → 跨进程 PID 锁（1h 过期自动抢占）；主线程只做廉价门检查，通过后才起守护线程，完成通知在下一次输入前打印
- **单次结构化整合**：长期记忆 + 经验库 + 最近会话拼成 ≤30k tokens 输入，一次 LLM 调用完成"归纳 → 去重/冲突 → 提炼经验 → 输出 ops"；单次 ≤20 条 op，代码侧校验 id 必须存在、字段长度上限、项目路径脱敏
- **事务应用**：apply 前对长期记忆 JSON 与全部经验文件做快照；删除一律软删除（归档到 `dream/<hash>/archive/<dream-id>/`）；任一落盘失败整体回滚；支持 memory_add / update / delete / merge 与 experience_add / delete 六类 op
- **可审计可回滚**：一次整合 = 一个 dream-id；`/dream log [id]` 查看摘要与 op 明细，`/dream undo [id]` 按快照还原上一笔（连带回退 state 时间戳）
- **配置**：`ASTERCLI_DREAM_ENABLED`（默认开）/ `ASTERCLI_DREAM_MIN_HOURS`（24）/ `ASTERCLI_DREAM_MIN_SESSIONS`（5）/ `ASTERCLI_DREAM_SCAN_INTERVAL_MINUTES`（10）/ `ASTERCLI_DREAM_MODEL`（默认当前模型）/ `ASTERCLI_DREAM_MAX_INPUT_TOKENS`（30000）/ `ASTERCLI_DREAM_TIMEOUT_SECONDS`（120）/ `ASTERCLI_SESSION_LOG`（默认开）/ `ASTERCLI_SESSION_LOG_DIR` / `ASTERCLI_DREAM_DIR`
- **与经验/长期记忆的关系**：AutoDream 是唯一允许自动写长期记忆的通道（见 `AGENTS.md` Memory 约束），写入条目带 `metadata.source=dream` 与 `dream_id`；经验仍按 P9 自动复盘，dream 只做跨会话去重与补充
- **后台线程模型**：CLI 每轮 `asyncio.run` 结束后主事件循环销毁，dream 用守护线程 + 线程内独立事件循环；不碰 prompt_toolkit 输入区，结果只进通知队列

## 终端渲染（P11）

参照 Java 版 `render/inline` 包（`InlineRenderer` / `FoldableBlock` / `ToolCallRenderer` / `InlineDiffRenderer` / `InlineActivityDisplay` / `BottomStatusBar`）对齐内联形态，实现落在 `cli/render_blocks.py`（纯逻辑）、`cli/activity.py`（活动区）、`cli/status_bar.py`（底栏）、`cli/renderer.py`（编排）：

- **思考活动区**：`Thinking... (esc to cancel, 12s)` + 最近 4 行 reasoning 灰字预览，原地重绘（只清理自己打印的行，不碰 transcript / 输入区）；正文或工具调用开始前清空，并把完整 reasoning 以 `│ ` 引用块落到正文区
- **工具调用折叠块**：按工具名分组，折叠态一行（`⏵ ReadFile(src/a.py) (ctrl+o to expand)` / `⏵ 2 组工具调用 / 3 次`），展开态逐条列出关键参数（path / pattern / command / query / url / name，未知工具回退截断 JSON，MCP 工具显示 `server.tool`）；结果行 `✓` / `✗`，失败红色
- **Ctrl+O 展开**：绑定在 prompt_toolkit `PromptSession` 上，经 `run_in_terminal` 让出输入区后重绘最近一个折叠块；块高超过可视区域时拒绝展开并提示，避免破坏滚动区域
- **行内 diff**：`write_file` 成功后由 `ToolRegistry` 出口回调（与 LSP 钩子同级、异常静默），LCS 生成 hunk（上下文 2 行），新增绿 / 删除红 / hunk 青；旧文件超过 512KB 跳过
- **代码围栏折叠**：流式正文中的 ``` 代码块折叠为 `⏵ code: python (42 行, ctrl+o to expand)`；围栏未闭合（流中断）时按原文输出，避免丢内容
- **底部状态栏**：prompt_toolkit `bottom_toolbar` 两行——模式 + MCP/Skill 摘要 / `Auto Model · 模型  phase  ctx ▰▰▱▱ 42%  in/out/cache  elapsed  cwd`；token 统计从轮末正文行移入底栏（一次性运行模式仍保留 usage 行）
- **提交回显**：每轮输入以 `> ` 暗色行写回 transcript（对齐 JLine 提交回显），斜杠命令同样回显

## 与 Java 版的已知差异

- `grep_code` 暂为纯 Python 扫描引擎（Java 版优先 ripgrep，后续对齐）
- Java 的 shortTermMemory/ContextCompressor 未迁移（Java javadoc 自认度量错位属补丁前旧设计，Python 版只保留 conversationHistory 压缩一道）
- 长期记忆检索分词：Java 用 jieba，Python 用英文整词 + 中文 2-gram（不引 jieba 依赖）
- 记忆注入位置：user message 前置（见上）而非 system prompt
- Plan-Execute 任务批次顺序执行（Java 4 线程并行）；审阅键位一致（Enter/Ctrl+O/ESC/I）
- MCP 用户级配置目录为 `~/.astercli-py`（与 Java `~/.astercli` 隔离）；step_search 自动内置未迁移；HTTP/SSE transport 由官方 SDK 保证（单测仅覆盖 stdio 集成）
- RAG 分块/分词/存储与 Java 的差异：tree-sitter 八语言符号分块（Java 仅 JavaParser 单语言）、查询分词零依赖（Java jieba）、per-project 独立库（Java 单库混多项目）、关键词底分 0.15（Java 0.3，实测修正防关键词噪音压过语义命中）
- Multi-Agent 依赖批次内顺序执行（Java Worker 池并行）；Java builtin（jar 内置）skill 层暂无 Python 内置资产；web_search 不迁移 step-3.7-flash 下转调 StepSearch MCP 的偏好
- LSP 诊断引擎差异：`.py` 内置 ast + 其余 tree-sitter 多语言（Java 仅 JavaParser 单 .java）；诊断同时进 renderer 显示（Java 只进 conversationHistory）；开关仅环境变量口径（Java 另支持系统属性）
- 经验子系统为 Python 版独立设计（Java 暂无对应模块，不回写 Java）；存储选 Markdown 文件而非 SQLite / JSON；长期记忆与经验互不替代（事实手动存、经验自动沉淀）
- AutoDream 为 Python 版独立设计（Java 无对应模块）：会话转写落 jsonl、睡眠式启动门控、事务化 apply + 快照撤销均无 Java 对应实现
- 渲染层差异：Java 用 JLine Status 托管执行期也可见的 dock，Python 的 `bottom_toolbar` 只在输入态可见；折叠块重绘需主动让出输入区，块高超过可视区域时不就地展开（Java 截断重绘）；`Ctrl+O` 只作用于最近一个块
