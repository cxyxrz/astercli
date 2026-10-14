# AsterCLI Python + LangChain 重写计划

## 背景与目标

AsterCLI 是已交付 23 期的 Java 17 Agent CLI（ReAct / Plan / Multi-Agent / Memory / RAG / MCP / HITL），LLM 调用层、SSE 流解析、ReAct 循环全部手写，维护成本高。决定用 **Python + LangChain 生态重写**（非 LangChain4j），动机是减少手写维护成本；节奏为**增量渐进**：在仓库内新建 `astercli-py/` 子目录，按模块分期实现，每期独立可运行、可验收；Java 版继续作为生产版本，Python 侧改动不回写 Java 代码。

## 技术栈选型

| 项 | 选择 | 理由 |
|---|---|---|
| 包管理 | `uv` + Python 3.12+，测试 `pytest` + `pytest-asyncio` | 快、锁文件可靠 |
| LLM | `langchain-core` + `langchain-openai`，7 个 provider 统一 `ChatOpenAI(base_url=...)` 接入 | 全部 OpenAI-compatible；不引入 partner 包（langchain-deepseek 等），因为本项目需要自定义 reasoning 回传、HTTP/1.1 强制、图片降级 |
| ReAct 编排 | **LangGraph 自定义 `StateGraph`**，不用 `create_react_agent` | 需要 HITL interrupt、reasoning 独立流式展示、并行工具保序、64k 输出预算、budget 兜底——prebuilt 封死了这些钩子 |
| TUI | `prompt_toolkit`（JLine 等价：补全/高亮/print_above）+ `rich`（Markdown/流式） | textual 是全屏框架，与 Claude Code 式 inline 形态不符 |
| MCP（P5） | 官方 `mcp` Python SDK | stdio/SSE/HTTP 全覆盖，替代手写 JSON-RPC，Python 重写最大收益点 |
| RAG（P6） | `sqlite-vec` + LangChain Embeddings；JavaParser → tree-sitter | 向量不用手写；符号提取换 tree-sitter，索引结构重新设计而非移植 |

## 项目结构（`astercli-py/`，src 布局对齐 Java 包）

```
astercli-py/
├── pyproject.toml          # uv 管理；console script: astercli
├── README.md               # 交付后补：与 Java 版的关系、运行方式
└── src/astercli/
    ├── config.py           # pydantic-settings，env_file=("../.env", ".env")，沿用根 .env 变量名
    ├── llm/
    │   ├── providers.py    # PROVIDERS 表：7 provider 的默认 model/base_url/别名归一化（照抄 LlmClientFactory 语义）
    │   ├── factory.py      # create_client(provider, settings) -> ChatOpenAI；fallback 顺序 glm→deepseek→step→kimi→freellmapi→xfyun→agnes
    │   └── deepseek.py     # DeepSeekClient(ChatOpenAI)：reasoning_content 历史回传 + http1 强制 + 图片消息降级文本
    ├── agent/
    │   ├── state.py        # AgentState(TypedDict): messages / reasoning / budget
    │   └── graph.py        # build_react_graph(llm, tools, guards) -> CompiledGraph
    ├── tools/
    │   ├── registry.py     # ToolRegistry: register / to_schemas(mode) / async execute_all（4 并发保序、超时、64k 截断）
    │   └── builtin/
    │       ├── files.py    # read_file / write_file / list_dir / glob_files / grep_files
    │       └── shell.py    # execute_command（白名单 + 超时）
    ├── policy/
    │   ├── path_guard.py   # 路径限制项目根内（pathlib 直译 PathGuard）
    │   ├── command_guard.py
    │   └── audit.py
    ├── memory/             # P3 已交付：tokens.py / compactor.py / long_term.py
    ├── mcp/                # P5
    ├── rag/                # P6
    └── cli/
        ├── main.py         # prompt_toolkit REPL + astream_events 流式渲染
        └── renderer.py     # rich 流式 + reasoning 灰字展示
tests/                      # 与 src 镜像：tests/llm/、tests/tools/、tests/policy/、tests/agent/
```

## Java → Python 概念映射（核心）

| Java | Python 落点 |
|---|---|
| `LlmClient.Message/ChatResponse` | `HumanMessage/AIMessage/ToolMessage`；reasoning_content 落 `AIMessage.additional_kwargs["reasoning_content"]`；usage（含 cached_tokens）在 `response_metadata` |
| DeepSeek reasoning 历史回传（AGENTS.md 硬约束） | `ChatOpenAI` 子类重写请求组装：从历史 AIMessage 的 additional_kwargs 取回写入 `reasoning_content` 字段。**langchain 默认序列化会丢弃该键，必须自验证** |
| `Agent.java` 手写 ReAct 循环（L151-225） | StateGraph：agent 节点（bind_tools）→ 条件边（has tool_calls?）→ tools 节点 → 回 agent；主退出条件 = LLM 不再调工具，budget 仅兜底（保留 AgentBudget 语义） |
| `executeTools`（4 并发/超时/保序，ToolRegistry.java L1324-1399） | `asyncio.gather`（保序）+ `Semaphore(4)` + per-task `wait_for`；超时任务转失败占位，与 Java invokeAll 语义对齐；取消用 `asyncio.Event` 对应 CancellationContext |
| 精简工具定义（`ASTERCLI_TOOL_MODE` 双模式） | 工具内部 pydantic 定义，`bind_tools` 前经 `to_slim_schema()` 转精简 dict（bind_tools 接受原始 dict schema），保留双模式 |
| HITL 拦截链（HitlToolRegistry→ToolRegistry→Guard） | LangGraph `interrupt()` 挂起 → 审批 → `Command(resume=...)`；拦截顺序不变 |
| PathGuard / CommandGuard | 纯逻辑 pathlib 直译，单测先行 |
| 64k 统一工具输出预算 | registry.execute_all 出口统一截断装饰器（`ASTERCLI_TOOL_OUTPUT_MAX_CHARS` 可调） |
| PromptAssembler / ASTER.md 注入 | 直译；组装为 LangGraph 入口 system message，注入当前日期/时区 |

## 分期路线图（每期独立验收）

| 期 | 范围 | 验收 |
|---|---|---|
| **P1 骨架+Provider+流式 REPL** | config / providers / factory / DeepSeek 定制 / REPL + reasoning 灰字 | `uv run astercli` 流式问答成功；`uv run pytest tests/llm/`（含 DeepSeek reasoning 回传 mock 单测） |
| **P2 工具+ReAct+Guard** | 6 文件工具 + execute_command + LangGraph ReAct + Path/CommandGuard + 64k 预算 | REPL 内让 Agent 读改本项目文件并 grep；路径逃逸被拒；`uv run pytest tests/tools/ tests/policy/` |
| **P3 Memory（已交付）** | conversationHistory 压缩（window-20k-13k 阈值，REMOVE_ALL_MESSAGES 全量替换通道）+ `/compact` + 长期记忆（`/save` + `/memory`，user message 前置注入） | 长对话自动压缩；`/save` 重启后可检索；`uv run pytest` 232 全绿 |
| **P4 HITL + Plan（已交付）** | 路线 B 定稿：自定义 StateGraph + `interrupt()` 审批（y/a/n/s/m 决策制，替换行内 y/n）+ Plan-Execute（`/plan` 出 DAG、审阅 Enter/Ctrl+O/ESC/I、失败<0.5 重规划） | `write_file` 触发审批框（approve/reject/approve_all 均验证）；`/plan` 出 DAG 摘要并可审阅；`uv run pytest` 308 全绿 |
| **P5 MCP（已交付）** | 官方 `mcp` SDK 2.x（ClientSessionGroup）+ stdio/sse/streamable-http 三 transport + `mcp__{server}__{tool}` 动态注册 + resources 虚拟工具 + `@server:uri` mention + HITL 审批/审计 | 项目级 mcp.json 配真实 stdio server，REPL 内 LLM 经审批调用成功（`mcp__demo__add: 8`）；`uv run pytest` 374 全绿 |
| P6 RAG | sqlite-vec + embedding + tree-sitter 索引 + 混合检索 | `/index` 后语义查询命中 |
| P7 Multi-Agent / Skill / web 工具 | orchestrator / skill buffer / web_search+web_fetch | `/team`、`/skill` 可用 |

**明确暂不迁移**（后续再议）：微信 iLink 通道、LSP 诊断、Chrome DevTools/CDP、完整 TUI 视觉（底部状态栏/dock/live thinking 区）、Side-Git 快照、Runtime API、图片输入、Lanterna 降级。

## 本次执行范围：P1 + P2 文件级步骤

1. **`astercli-py/pyproject.toml`**：deps = langchain-core, langchain-openai, langgraph, prompt_toolkit, rich, pydantic-settings, httpx, python-dotenv；dev deps = pytest, pytest-asyncio, respx（mock HTTP）
2. **`src/astercli/config.py`**：`Settings(BaseSettings)`，`env_file=("../.env", ".env")`；沿用根 .env 变量名：`GLM_API_KEY` / `DEEPSEEK_API_KEY` / `STEP_API_KEY` / `KIMI_API_KEY`（兼容 `MOONSHOT_API_KEY`）/ `FREELLMAPI_API_KEY` / `XFYUN_MAAS_API_KEY` / `AGNES_API_KEY` + `*_MODEL` / `*_BASE_URL` + `ASTERCLI_TOOL_MODE` / `ASTERCLI_TOOL_OUTPUT_MAX_CHARS` + `EMBEDDING_*`
3. **`llm/providers.py` + `factory.py`**：默认 model/base_url 照抄 .env.example 注释（glm-5.1 / deepseek-v4-flash / step-3.5-flash / kimi-k2.6 / freellmapi-auto / agnes-2.0-flash；xfyun `https://maas-api.cn-huabei-1.xf-yun.com/v2` + lora_id header）；别名归一化照抄 `LlmClientFactory.normalizeProvider`（stepfun→step、moonshot→kimi、iflytek→xfyun、sapiens→agnes 等）
4. **`llm/deepseek.py`**：`DeepSeekClient(ChatOpenAI)`——请求组装时把历史 AIMessage 的 `additional_kwargs["reasoning_content"]` 写回请求体；`http_client=httpx.AsyncClient(http1=True)`；`supports_image_input=False` 时历史图片 content 降级为文本提示（参照 Java `Message.withoutImageContent` 语义）
5. **`tools/registry.py`**：`register(tool)` / `to_schemas(mode)`（slim/schema 双模式）/ `async execute_all(invocations) -> list[ToolResult]`（信号量 4、per-task 超时、异常转失败结果、出口 64k 截断）
6. **`tools/builtin/files.py` / `shell.py`**：参照 Java 版参数语义（read_file 行段、grep_code 优先 `rg` 回退 Python 扫描、max_chars/head_limit 预算）；execute_command 8k 输出预算 + 白名单
7. **`policy/path_guard.py`**：`check_path(root, p) -> str | None`（拒绝信息）；命令黑名单 `command_guard.py`；纯逻辑直译 + 单测先行
8. **`agent/state.py` + `graph.py`**：AgentState(messages, reasoning, budget)；`agent` 节点 = `llm.bind_tools(to_schemas(mode))`，条件边 `has_tool_calls` → `tools` 节点 → 回 agent；保留 budget 兜底（token 上限/死循环检测/硬轮数）
9. **`cli/main.py` + `renderer.py`**：prompt_toolkit REPL，`astream_events` 驱动 rich 流式渲染（reasoning 灰字、content 正文、tool call 块）；Ctrl+C 取消当前任务（Event）
10. **tests**：`tests/llm/test_deepseek_reasoning.py`（mock 断言请求体含 `reasoning_content`，最高风险点）；`tests/tools/`（registry 并发保序/超时/截断）；`tests/policy/`（PathGuard 逃逸用例）

## 风险与对策

1. **DeepSeek reasoning 回传**是最高风险：langchain 序列化默认丢 additional_kwargs 非标准键 → P1 就写 mock 单测锁死该行为
2. 流式 reasoning 依赖 chunk 合并行为 → 锁定 langchain-openai 版本范围，pyproject 用 `>=x,<y`
3. `create_react_agent` 看似省事但 HITL/预算/取消都要 hack → 坚持自定义 StateGraph
4. 超时任务在 `gather` 中要转失败占位，保持与 Java invokeAll 结果保序语义一致
5. Java 版保持生产版本不动；Python 子目录不干扰 Maven 构建（.gitignore 加 `astercli-py/.venv`）；完成后按 AGENTS.md 硬规则 1 在 AGENTS.md 补一行说明双版本关系

## 端到端验证

```bash
cd astercli-py && uv sync
uv run pytest                              # 全部单测
uv run astercli                            # REPL：选 deepseek 或 glm（用根 .env 的 key）
# P2 验收任务示例：在 REPL 内输入
#   "读一下 ../pom.xml 的前 30 行，并 grep_code 搜 LlmClientFactory"
#   预期：Agent 走 LangGraph ReAct，多轮工具调用，路径逃逸（如 ../../etc）被 PathGuard 拒绝
uv run astercli <<< "读 /etc/hosts 并写入 /tmp/x"   # 验证写入审批边界与 Guard 拦截
```

## P4 开工前选型：create_agent + 中间件栈 vs 保持自定义 StateGraph

P3 收尾时实测（2026-10）：venv 内 langchain 1.4.3（transitively 安装），`langchain.agents.middleware` 全套可用：`AgentMiddleware`（`before_model` / `after_model` / `wrap_model_call` / `wrap_tool_call` / `modify_state`）+ 内置 `SummarizationMiddleware` / `HumanInTheLoopMiddleware` / `ModelCallLimitMiddleware` / `ToolRetryMiddleware`。P2 "prebuilt 封死钩子"的选型理由在 langchain 1.x 已大幅失效，P4 开工第一件事是评估两条路线：

- **路线 A：迁 `langchain.agents.create_agent` + 中间件栈**。一次拿到三个收益：`HumanInTheLoopMiddleware`（P4 审批）、`SummarizationMiddleware`（替代 P3 compactor）、LangGraph Store + 自定义注入 middleware（替代 run_turn 记忆注入）。`SummarizationMiddleware` 构造支持 `token_counter` / `trigger=('tokens', N)` / `summary_prompt` 自定义；但 `keep` 只支持 fraction/tokens/messages，没有"user 轮次"边界，tool 对不被切断需验证。代价：reasoning 流式聚合、停滞检测、xfyun 纯对话、slim schema 出口全部重新对接
- **路线 B：保持自定义 StateGraph**。P4 只实现 `interrupt()` 审批 + Plan-Execute；memory 留 P3 形态；P6 时只把 LongTermMemory 存储层换 LangGraph Store（Store 不要求 create_agent）
- **估算器结论（两路线通用）**：`count_tokens_approximately` 不能直接替换自研估算——中文低估 ~2×（全局 chars_per_token=4 无分脚本能力，实测中文句 20 vs 11），且完全不计 `reasoning_content`（实测 153 vs 7）。无论哪条路继续用 `estimate_messages_tokens`；迁中间件时通过 `token_counter` 口子传入
- **安全网**：232 个测试是迁移前后行为等价的验收线；若走路线 A，迁移前后跑全量必须全绿

## P4 选型定稿（路线 B）与交付记录

**结论：保持自定义 StateGraph（路线 B）**。spike 实测（2026-10，venv 内 langchain=1.4.3 / langgraph=1.2.12）：

- `interrupt()` + `Command(resume)` 在自定义 StateGraph 上完整可用（挂起 → checkpoint → resume 值回传），HITL 审批不需要 create_agent
- `create_agent` 对 slim dict 工具原样透传 `bind_tools`（调用期而非编译期，与直接 bind 行为一致），不构成路线 A 的加分项
- reasoning_content 聚合在两条路线下行为等价（简单场景验证）
- `MemorySaver.delete_thread` 存在且可用（per-turn thread 释放）
- 路线 A 的 `SummarizationMiddleware` `keep` 只有 fraction/tokens/messages，表达不了 P3 的"保留最近 3 个 user 轮次 + HumanMessage 对齐"语义 → P3 compactor 不可替换，路线 A 收益进一步缩水
- P6 接 LangGraph Store 不要求 create_agent，路线 B 无阻塞

**P4 交付内容**：

- `hitl/`（policy + approval + terminal）：审批决策制 `approve / approve_all / reject / skip / modify`；`ApprovalSession` 持会话放行名单（工具维度，`/clear` 清空）；fail-safe：非法输入 / 决策缺失 / 输入流关闭一律保守拒绝
- `agent/graph.py`：tools 节点执行前 `interrupt()` 挂起；`ainvoke_with_hitl` 恢复循环；`make_checkpointer()`（MemorySaver + `JsonPlusSerializer(pickle_fallback=True)`——AgentBudget 等自定义类 msgpack 不识别，官方开关兜底；纯内存无持久化攻击面）；无 checkpointer 时审批整体禁用（graph 层 + `registry.set_hitl_active(False)` 双层同步）
- `plan/`（models + planner）：两遍 JSON 解析（id 归一化 + 前向引用回填）、DFS 拓扑排序环检测、批次分层展示；`is_simple_goal` 快路径不调 LLM
- `cli/plan_review.py`：审阅键位 Enter / Ctrl+O / ESC / I（KeyBindings eager 绑定，方向键不会误触 ESC；已展开时 ESC 折叠回摘要）
- `agent/plan_execute.py`：拓扑序**顺序**执行（Java 4 线程并行，P4 顺序保证审批与渲染不交错，任务级并行留后续）；失败且 progress<0.5 → replan + 重新审阅；每任务独立 thread + `MAX_TASK_ITERATIONS=5` 预算
- `cli/main.py`：`/plan [任务]` 命令 + 无参数下一轮计划模式；run_turn per-turn thread + resume 后从 `result["budget"]` 取回记账（反序列化副本）
- 测试 232 → 308（新增 hitl/plan/graph-hitl/plan-execute/plan-review + registry 审批测试改决策制）

**已知边界（后续再议）**：任务批次并行执行、`m`（modify）修改参数后 UI 回显、审批框在非 TTY stdin 下退化为 EOF 拒绝（fail-safe 符合预期）。

## P5 交付记录

**选型**：官方 `mcp` SDK 2.3.0（`uv sync` 实装）。相比计划的"三 transport 手写适配"，SDK 2.x 直接提供 `ClientSessionGroup` 多 server 会话池（连接聚合 + 退出清理 + 三 transport 参数类），对应 Java 手写的 McpServerManager transport 池——Python 版最大的"不用重写"收益点。

**交付内容**：

- `mcp/config.py`：`mcpServers` 顶层 key（与 Java/生态通用约定一致）；用户级 `~/.astercli-py/mcp.json` 先读、项目级 `.astercli/mcp.json` 覆盖同名（对齐 Java）；`command`/`url` 互斥必设其一，`type=sse` 走 SSE、默认 http；`${VAR}` 展开 command/args/env/url/headers，未定义变量按单 server 配置错误处理（其余 server 不受影响）
- `mcp/runtime.py`：后台 daemon 线程常驻 event loop——MCP 会话跨 REPL 每轮 `asyncio.run()` 存活的架构基础。**anyio 任务边界坑（实测）**：`connect_to_server` 进入 stdio transport 的 anyio cancel scope，anyio 强制进出同 task；connect 走 `run_coroutine_threadsafe` 会在 shutdown aclose 时炸 `Attempted to exit cancel scope in a different task` → 生命周期调用（connect）经队列投递回常驻 `_main` 任务执行，session 级调用（call_tool/list/read）仍走独立 task 保并发
- `mcp/manager.py`：状态机 `STARTING/READY/DISABLED/ERROR`（对齐 Java McpServerStatus）；启动同步等 `ASTERCLI_MCP_STARTUP_WAIT_SECONDS`（默认 8 秒）超时后台续载不阻塞首屏；`mcp__{server}__{tool}` 注册（dangerous=True，schema 透传 SDK 原生 inputSchema）；resources 虚拟工具 `list_resources`/`read_resource`（server 支持时）；replace 语义 restart；mention 读取 20 万字符截断 + `allowByMention` 审计；`/mcp` 子命令数据接口（status/resources/restart/disable/enable）
- `mcp/mention.py`：`@server:uri` 正则解析（server 字母开头、uri 必须 scheme://、引号内跳过）；展开为 `<resource server uri mimeType>` 块 / 失败降级 `<resource_error>` 不中断输入（对齐 Java AtMentionExpander）
- `cli/main.py`：`run_turn`/`run_plan_turn` 前置 mention 展开；`/mcp` 命令五子命令；退出/一次性模式 finally `mcp.shutdown()`
- 测试 308 → 374（config 24 / mention 17 / manager 25 + **真实 stdio 子进程端到端集成 2**：HITL 审批链路 + 无审批直连）。集成测试用 SDK 自带 `MCPServer`（2.x 的 FastMCP）起真实子进程

**与 Java 版的差异**：用户级配置目录 `~/.astercli-py`（与 Java `~/.astercli` 隔离，与长期记忆/审计同原则）；SDK 原生 schema 直接透传（Java 需自组装 JSON）；暂未做 step_search 自动内置（依赖 STEP_API_KEY 探测，留后续）与 server 级白名单环境变量（非交互通道 P5 未涉及，微信通道在暂不迁移清单）。

**已知边界**：HTTP/SSE transport 单测未覆盖（无公网测试 server，行为由 SDK 保证，仅集成冒烟验证 stdio）；`/mcp` 无 logs 子命令（Java 有，依赖启动日志文件）。
