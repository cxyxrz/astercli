# execute_command 第三层防护：OS 级沙箱（sandbox-exec / bubblewrap）

## Context

黑名单 + 白名单都属于**执行前静态判定**，共同盲区是：子进程一旦启动，其任意写路径不受任何围栏约束（`python -c` 内联代码写 `~/.zshrc` 持久化后门、写 HOME 外任意位置——PathGuard 只管工具参数路径，不管子进程行为）。

本次补**执行时 OS 级围栏**：macOS 用系统自带 `sandbox-exec`（Seatbelt profile），Linux 用 `bubblewrap`（bwrap）。写路径白名单 = 项目根 + /tmp + 可配置追加，其余全盘禁写——即使命令本体绕过了黑/白名单，落盘行为也被内核级拦截。相比 ROADMAP 里"容器/VM 沙箱"，OS 级方案零基础设施、零 daemon，是一条 `ProcessBuilder argv` 包装就能完成的事。

**三层关系**：黑名单（挡语义即破坏的命令文本）→ 白名单（opt-in 收紧放行集合）→ **沙箱（opt-in 限制已放行命令的运行时写能力）**。三层独立开关、串联生效，HITL 审批在最外层不变。

## 配置面（3 个新变量，全部默认关）

| 变量 | 默认 | 语义 |
|------|------|------|
| `ASTERCLI_COMMAND_SANDBOX`（或 `-Dastercli.command.sandbox`） | 未设置=关 | `true/on` 开启沙箱包装 |
| `ASTERCLI_SANDBOX_WRITABLE_PATHS` | 空 | 逗号分隔的额外可写路径，支持 `~` 展开（如 `~/.m2,~/.gradle`——开启沙箱后构建命令需要） |
| `ASTERCLI_SANDBOX_ALLOW_NET` | `true` | `false` 时 macOS profile 加 `(deny network*)`、Linux 加 `--unshare-net` |

**fail-closed 决策**：显式开启沙箱但平台不支持（Windows）或工具缺失（Linux 无 bwrap）→ **拒绝执行**（抛 PolicyException 统一出口），不静默降级——与白名单"文件缺失全拒"同一设计语言，逼用户正视配置错误而非无声失去保护。

## 改动清单

### 1. 新类 `policy/CommandSandbox.java`

实例类，构造注入依赖（可测性关键）：

```java
CommandSandbox(boolean enabled, String osName, boolean sandboxExecAvailable, boolean bwrapAvailable)
static CommandSandbox fromEnvironment(Path projectRoot)   // 读环境变量/系统属性 + 真实工具检测
List<String> wrap(String command, String projectRoot)      // 返回完整 ProcessBuilder argv；不支持时抛 PolicyException
```

- `wrap` 分支：未开启 → `["bash","-c",command]`；macOS → `["/usr/bin/sandbox-exec","-p",profile,"bash","-c",command]`；Linux → `["bwrap",...flags...,"bash","-c",command]`
- 工具检测：sandbox-exec 固定路径 `/usr/bin/sandbox-exec`；bwrap 沿 `PATH` 找（`findExecutable`，惰性一次、缓存）
- profile 与 argv 由**纯函数**生成（单测断言对象）：
  - `buildSeatbeltProfile(root, extraPaths, allowNet) -> String`
  - `buildBwrapArgv(root, extraPaths, allowNet, command) -> List<String>`

**Seatbelt profile**（`-p` 内联传参，ProcessBuilder 是 argv 数组无 shell 转义问题）：

```
(version 1)
(allow default)
(deny file-write*)
(allow file-write* (subpath "/private/tmp"))     ← macOS 的 /tmp 是 /private/tmp 符号链接，必须写真实路径
(allow file-write* (subpath "<root>"))            ← root 用 toRealPath() canonical 化
(allow file-write* (subpath "<追加路径>"))        ← 每条一个
(deny network*)                                   ← 仅 allowNet=false 时
```

读操作全部放行（`ls`/`cat`/`grep` 等正常工作），只围写。

**bwrap argv**：

```
bwrap --ro-bind / / --bind <root> <root> --bind /tmp /tmp
      [--bind <追加路径> <追加路径>]...
      --dev /dev --proc /proc            ← ro-bind 下的 /dev 只读，bash 写 /dev/null 需要 --dev
      [--unshare-net]
      bash -c <command>
```

HOME 不 bind → 继承只读，写 HOME 被拒（持久化后门路径被堵死）。

### 2. [ToolRegistry.java](file:///Users/hrs/IdeaProjects/myCLI/astercli/src/main/java/com/astercli/tool/ToolRegistry.java#L1449) — 两处

- 字段 + `setProjectPath` 重建（完全复制 `commandAllowlist` 的既有模式）：`private CommandSandbox commandSandbox = CommandSandbox.fromEnvironment(...)`
- L1449 唯一改动：`new ProcessBuilder("bash", "-c", normalized)` → `new ProcessBuilder(commandSandbox.wrap(normalized, projectPath))`。`wrap` 调用放 try 块之前（与黑/白名单同区），`PolicyException` 直接传播到 `doExecuteTool` 统一出口（审计 + "🛡️ 策略拒绝"）。超时/输出/销毁逻辑零改动。

### 3. 测试（全中文零网络）

- **CommandSandboxTest**（新）：
  - 纯函数：`buildSeatbeltProfile` 断言含 `deny file-write*`、`/private/tmp`、canonical root、追加路径条目、`deny network*` 分支；`buildBwrapArgv` 断言 `--ro-bind / /`、`--dev /dev`、`--unshare-net` 分支、追加 `--bind`、命令在尾部
  - wrap 矩阵（构造注入 osName/工具存在性）：未开启→裸 bash；mac+有 sandbox-exec→包装 argv；linux+有 bwrap→包装 argv；开启+平台不支持→抛 PolicyException；开启+linux 无 bwrap→抛 PolicyException
  - `~` 展开：`ASTERCLI_SANDBOX_WRITABLE_PATHS` 的路径展开（@TempDir）
  - 开关测试后 `System.clearProperty` 清理（复制 CommandAllowlistTest 的 afterEach 模式）
- **ToolRegistryTest 集成**：`@EnabledOnOs(MAC)` 真实跑 sandbox-exec 两个用例——`echo ok` 正常出结果；`echo x > ~/.astercli-sandbox-deny-test` exit code 非零（文件未产生）。**这两个用例是 profile DSL 语法正确性的唯一真实验证，字符串单测查不出 DSL 写错**。bwrap 用例写 `@EnabledOnOs(LINUX)`（本机不跑，留给 Linux 环境）。

### 4. 文档联动（硬规则 5.4）

- `.env.example`：新"命令沙箱"段（3 变量 + fail-closed 行为说明 + 构建命令需追加 `~/.m2` 的提示）
- `AGENTS.md`「HITL + 策略层」节：补沙箱一行；「当前已知边界」"容器/VM 沙箱未交付"改措辞——OS 级沙箱已交付，容器/VM 继续不做
- `README.md`：策略层描述三处补沙箱
- `ROADMAP.md`：对应"沙箱"条目状态更新（硬规则 1）

## 验证

```bash
mvn test -Dtest='CommandSandboxTest,CommandGuardTest,CommandAllowlistTest,ToolRegistryTest' -DskipTests=false   # 针对性（macOS 上含真实沙箱集成）
mvn test -Pquick    # 回归（39 个存量失败为 TUI/Embedding 环境问题，与本改动无关，已 stash 验证过）
```

手工验收（jar + `ASTERCLI_COMMAND_SANDBOX=true` 启动 REPL）：

1. `执行命令 echo hi > /tmp/ok` → 成功（/tmp 可写）
2. `执行命令 echo hi >> ~/.zshrc` → 命令执行但 exit 非零/被拒（沙箱围栏，HOME 只读）
3. `执行命令 python3 -c "open('/tmp/pwned','w')"` → /tmp 内成功；改写 `~/.zshrc` → 拒（黑名单挡不住的解释器内联，沙箱挡住）
4. `执行命令 mvn -v` → 成功；`mvn test -Pquick` 会因 `~/.m2` 不可写失败 → 设 `ASTERCLI_SANDBOX_WRITABLE_PATHS=~/.m2` 后恢复（验证追加路径生效）
5. 不设开关重启 → 行为与现状一致（默认关，零破坏）
