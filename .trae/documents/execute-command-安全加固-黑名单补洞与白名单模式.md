# execute_command 安全加固：黑名单补洞 + 可选白名单模式

## Context

`execute_command` 目前只有 `CommandGuard` 正则黑名单做静态防线。黑名单存在已知绕过面：`git clean -fdx`（静删未跟踪文件）、`git reset --hard`（丢弃工作区修改）、`base64 -d | sh` / `echo ... | sh`（现有 curl|sh 规则只覆盖下载来源）、重定向写 `~/.ssh/authorized_keys` 或 `/etc/`。CommandGuard 注释自认"黑名单永远列不全，真正的安全责任在 HITL 审批"。

本次修复采取两层：
1. **补黑名单已知洞**（挡命令本身语义即破坏的模式）
2. **新增可选白名单执行模式**（opt-in，`ASTERCLI_COMMAND_MODE=whitelist` 时 execute_command 必须精确命中白名单，语义复用微信通道 `WechatPolicyDecider.commandAllowed` 已验证的精确 equals 设计）

默认行为完全不变（blacklist 现状），零破坏性。

## 改动清单

### 1. [CommandGuard.java](file:///Users/hrs/IdeaProjects/myCLI/astercli/src/main/java/com/astercli/policy/CommandGuard.java) — 补 5 条规则

新增 DenyRule（每条过"命令本身语义即破坏"标准）：

| 模式 | reason |
|------|--------|
| `(?i)\bgit\s+clean\s+-[a-z]*f` | 禁止 git clean -f 删除未跟踪文件（不带 -f 是 dry-run，放行） |
| `(?i)\bgit\s+reset\s+--hard\b` | 禁止 git reset --hard 丢弃工作区修改 |
| `(?i)\|\s*(sh\|bash\|zsh\|fish\|ksh)\b` | 禁止管道进 shell 解释器（通用化，替代并覆盖原 curl/wget 规则；base64 -d \| sh、echo \| sh 全被覆盖） |
| `(?i)(>>?)\s*[^|]*\bauthorized_keys\b` | 禁止重定向写 authorized_keys |
| `(?i)(>>?)\s*/etc/` | 禁止重定向写 /etc/ 系统配置 |

同步更新类注释 L15（"curl / git 默认放行"的表述已过时）。

**明确不拦**（决策记录）：
- `python -c` / `node -e`：`bash -c` 本身就是执行载体，黑名单挡不住解释器内联，这是白名单模式的职责；拦了会误伤正常快速检查
- `git push --force`：有正当用途，HITL 审批兜底

### 2. 新类 `policy/CommandAllowlist.java`

- **实例类**（非静态，便于测试注入）：`check(String command) -> String | null`，签名语义对齐 `CommandGuard.check`
- 匹配语义对齐 [WechatPolicyDecider.commandAllowed](file:///Users/hrs/IdeaProjects/myCLI/astercli/src/main/java/com/astercli/wechat/WechatPolicyDecider.java#L42)：trim 后精确 `equals`，大小写敏感，空白名单全拒
- 模式判定：`ASTERCLI_COMMAND_MODE` 环境变量，未设置或非 `whitelist` → disabled 实例（check 恒返回 null）
- 白名单来源：项目级 `.astercli/commands.json`，格式 `{"allowlist": ["git status", "mvn test -Pquick"]}`（对齐 `.astercli/mcp.json` 惯例），Jackson 读取，文件缺失/解析失败按空名单处理（fail-safe 全拒）+ 明确拒绝文案引导配置
- 拒绝文案示例：`命令不在白名单内（白名单模式已开启，请检查 .astercli/commands.json）: git push`

### 3. [ToolRegistry.java](file:///Users/hrs/IdeaProjects/myCLI/astercli/src/main/java/com/astercli/tool/ToolRegistry.java#L1420) — executeCommand 插入白名单检查

在 L1425-1430 黑名单检查之后、ProcessBuilder 之前插入：

```java
String allowlistDeny = commandAllowlist.check(normalized);
if (allowlistDeny != null) {
    throw new PolicyException(allowlistDeny);  // 复用统一出口：denyByPolicy 审计 + "🛡️ 策略拒绝" 文案
}
```

构造函数新增 `commandAllowlist` 字段初始化（blacklist 模式下为 disabled 实例）。

**顺序语义**：白名单模式是叠加而非替代——黑名单照先生效，白名单再收紧。HITL 审批不豁免（白名单命中仍需人工批准，双保险；ApprovalPolicy 零改动）。

### 4. 微信通道：零改动

微信已有独立的 `WechatPolicyDecider` 白名单机制，本次不碰。

### 5. 测试

- **CommandGuardTest**（现有风格 assertNull=放行 / assertNotNull=拒绝）补负例：`git clean -fdx` 拒 / `git clean -n` 放行、`git reset --hard` 拒、`base64 -d | sh` 拒、`echo xxx | bash` 拒、`cat x.sh | bash` 拒、`>> ~/.ssh/authorized_keys` 拒、`> /etc/hosts` 拒、`bash script.sh` 放行（防误伤）、`git status` 放行（防误伤）
- **新 CommandAllowlistTest**：精确命中放行 / 未命中拒 / 空名单全拒 / disabled 恒放行 / 大小写敏感 / 文件缺失视为空名单 / trim 语义
- **ToolRegistryTest 补集成用例**：whitelist 模式下未命中白名单抛 PolicyException → 走统一策略拒绝出口

### 6. 文档联动（硬规则 5.4）

- `.env.example`：新增 `ASTERCLI_COMMAND_MODE` 条目（含 whitelist 模式 + `.astercli/commands.json` 说明）；**顺手修 bug**：`PAICLI_LSP_ENABLED` → `ASTERCLI_LSP_ENABLED`（grep 证实代码零处读 `PAICLI_`，该行写进 .env 实际不生效）
- `AGENTS.md`「HITL + 策略层」节：CommandGuard 描述补新规则 + 白名单模式一段
- `README.md`：execute_command / CommandGuard 相关小节补白名单模式说明

## 验证

```bash
mvn test -Dtest=CommandGuardTest,CommandAllowlistTest,ToolRegistryTest,ApprovalPolicyTest   # 针对性
mvn test -Pquick                                                                            # 常规回归
```

手工验收：
1. 造 `.astercli/commands.json`（`{"allowlist": ["ls"]}`）+ `ASTERCLI_COMMAND_MODE=whitelist` 启动 REPL
2. 让 Agent 跑 `pwd` → 期待"🛡️ 策略拒绝：命令不在白名单内"；跑 `ls` → HITL 审批批准后正常执行
3. 不设模式变量重启 → 行为与现状完全一致（向后兼容）
