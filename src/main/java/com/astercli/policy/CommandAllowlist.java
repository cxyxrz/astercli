package com.astercli.policy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * execute_command 可选白名单模式：ASTERCLI_COMMAND_MODE=whitelist 时，
 * 命令必须精确命中 .astercli/commands.json 的 allowlist 才能执行。
 *
 * 定位：CommandGuard 黑名单是"挡明显破坏"，白名单模式是"只放行已知安全"——
 * 后者是真正的强约束（黑名单永远列不全），语义对齐微信通道 WechatPolicyDecider
 * 已验证的精确 equals 设计：trim 后全串比较，大小写敏感，不做前缀/模糊匹配。
 *
 * 叠加语义：白名单模式开启时 CommandGuard 黑名单仍然先生效（命中黑名单的命令
 * 即使写进白名单也会被拒）；HITL 审批不因白名单命中而豁免。
 *
 * fail-safe：模式开启但白名单文件缺失/解析失败时按空名单处理（全部拒绝），
 * 拒绝文案明确引导用户配置，避免静默放行。
 */
public class CommandAllowlist {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String CONFIG_FILE = ".astercli/commands.json";

    private final boolean enabled;
    private final List<String> allowlist;
    private final String sourceDescription;

    private CommandAllowlist(boolean enabled, List<String> allowlist, String sourceDescription) {
        this.enabled = enabled;
        this.allowlist = List.copyOf(allowlist);
        this.sourceDescription = sourceDescription;
    }

    /**
     * 按环境与配置构建白名单实例。
     *
     * 模式读取顺序对齐 ToolRegistry 既有惯例：系统属性 astercli.command.mode 优先，
     * 环境变量 ASTERCLI_COMMAND_MODE 次之；值不是 whitelist（忽略大小写）时返回
     * 恒放行的 disabled 实例，默认行为与历史版本完全一致。
     *
     * @param projectRoot 项目根目录，白名单文件位于 projectRoot/.astercli/commands.json
     */
    public static CommandAllowlist fromEnvironment(Path projectRoot) {
        String raw = System.getProperty("astercli.command.mode");
        if (raw == null || raw.isBlank()) {
            raw = System.getenv("ASTERCLI_COMMAND_MODE");
        }
        if (raw == null || !"whitelist".equalsIgnoreCase(raw.trim())) {
            return disabled();
        }
        return of(loadAllowlist(projectRoot));
    }

    /**
     * 校验命令是否在白名单内。
     *
     * @return null 表示放行（模式未开启或精确命中）；非 null 为拒绝原因，
     *         调用方（ToolRegistry.executeCommand）包装成 PolicyException 走统一策略拒绝出口
     */
    public String check(String command) {
        if (!enabled) {
            return null;
        }
        if (command == null || command.trim().isEmpty()) {
            return "白名单模式下拒绝空命令（请检查 " + sourceDescription + "）";
        }
        String normalized = command.trim();
        for (String allowed : allowlist) {
            if (allowed != null && normalized.equals(allowed.trim())) {
                return null;
            }
        }
        return "命令不在白名单内（白名单模式已开启，请检查 " + sourceDescription + "）: " + normalized;
    }

    public boolean isEnabled() {
        return enabled;
    }

    /** 测试与显式关闭场景使用的恒放行实例。 */
    static CommandAllowlist disabled() {
        return new CommandAllowlist(false, List.of(), "");
    }

    /** 测试注入用构造入口：直接给定白名单内容。 */
    static CommandAllowlist of(List<String> allowlist) {
        return new CommandAllowlist(true, allowlist, CONFIG_FILE);
    }

    private static List<String> loadAllowlist(Path projectRoot) {
        if (projectRoot == null) {
            return List.of();
        }
        Path config = projectRoot.resolve(CONFIG_FILE);
        if (!Files.isRegularFile(config)) {
            return List.of();
        }
        try {
            JsonNode node = MAPPER.readTree(config.toFile());
            JsonNode allowlistNode = node.path("allowlist");
            if (!allowlistNode.isArray()) {
                return List.of();
            }
            List<String> result = new ArrayList<>();
            for (JsonNode item : allowlistNode) {
                String entry = item.asText().trim();
                if (!entry.isEmpty()) {
                    result.add(entry);
                }
            }
            return result;
        } catch (Exception e) {
            // 解析失败按空名单处理：fail-safe 全拒，由拒绝文案引导用户修复配置
            return List.of();
        }
    }
}
