package com.astercli.policy;

import java.util.List;
import java.util.regex.Pattern;

/**
 * 命令快速拒绝：在 execute_command 进入 HITL 审批 / 真正调用 ProcessBuilder 之前的黑名单 fast-fail。
 *
 * 定位：辅助 HITL 而非主防线。黑名单是出名的反模式（永远列不全），但能拦住 LLM 容易踩的明显破坏性命令，
 * 减少 HITL 弹窗骚扰。真正的安全责任在 HITL 审批和用户判断。
 *
 * 设计取舍：
 * - 不做完整 shell 解析，只做正则模式匹配，够覆盖明显破坏性命令即可
 * - 命令替换段 $(...) 和反引号内的内容仍以原文存在，正则会一并扫描，不需要单独展开
 * - 管道进 shell 解释器（curl|sh、base64 -d|sh、echo|sh）一律拦截，执行脚本请走 bash script.sh 形式
 * - git 只拦本身语义即破坏的（clean -f 删文件、reset --hard 丢工作区修改）；
 *   push --force 等有正当用途的交由 HITL 审批判断
 * - 解释器内联（python -c / node -e）不在黑名单拦截范围：bash -c 本身就是执行载体，
 *   黑名单挡不住内联代码，强约束请开启白名单模式（ASTERCLI_COMMAND_MODE=whitelist）
 */
public final class CommandGuard {

    private static final List<DenyRule> RULES = List.of(
            new DenyRule("禁止 sudo 提权",
                    Pattern.compile("(?i)\\bsudo\\b")),
            // rm 路径黑名单：匹配开头即可拦截，不强求路径结束边界。
            // /、~、/*、$HOME 是常见的"灾难性删除起点"，包括其作为前缀的所有子路径都拦掉，避免 LLM 误删根目录或用户目录。
            new DenyRule("禁止 rm -rf 删除全盘或用户目录",
                    Pattern.compile("(?i)\\brm\\s+-[a-z]*r[a-z]*f[a-z]*\\s+(/|~|\\$home)|" +
                            "\\brm\\s+-[a-z]*f[a-z]*r[a-z]*\\s+(/|~|\\$home)")),
            new DenyRule("禁止 mkfs 格式化磁盘",
                    Pattern.compile("(?i)\\bmkfs(\\.|\\b)")),
            new DenyRule("禁止 dd 写入裸设备",
                    Pattern.compile("(?i)\\bdd\\b[^\\n]*\\bof=/dev/")),
            new DenyRule("识别为 fork bomb",
                    Pattern.compile(":\\(\\)\\s*\\{\\s*:\\s*\\|\\s*:\\s*&\\s*\\}\\s*;\\s*:")),
            new DenyRule("禁止管道进 shell 解释器执行",
                    Pattern.compile("(?i)\\|\\s*(sh|bash|zsh|fish|ksh|dash)\\b")),
            new DenyRule("不允许扫描 /、~ 或整个文件系统",
                    Pattern.compile("(?i)\\bfind\\s+(/|~|\\$home)")),
            new DenyRule("禁止 chmod 777 全盘",
                    Pattern.compile("(?i)\\bchmod\\s+-R\\s+777\\s+(/|~)")),
            new DenyRule("禁止 shutdown / reboot / halt",
                    Pattern.compile("(?i)\\b(shutdown|reboot|halt|poweroff)\\b")),
            new DenyRule("禁止 git clean -f 删除未跟踪文件",
                    // 不加 \b：-fdx / -xdf 等 f 后跟其他 flag 字母时仍需命中；-n（dry-run）无 f 自然放行
                    Pattern.compile("(?i)\\bgit\\s+clean\\s+(-[a-z]*f|--force)")),
            new DenyRule("禁止 git reset --hard 丢弃工作区修改",
                    Pattern.compile("(?i)\\bgit\\s+reset\\s+--hard\\b")),
            new DenyRule("禁止重定向写 authorized_keys",
                    Pattern.compile("(?i)(>>?)\\s*[^|]*\\bauthorized_keys\\b")),
            new DenyRule("禁止重定向写 /etc/ 系统配置",
                    Pattern.compile("(?i)(>>?)\\s*/etc/"))
    );

    private CommandGuard() {
    }

    /**
     * 校验命令是否安全。
     *
     * @return null 表示放行；非 null 字符串是拒绝原因，调用方包装成用户/LLM 可见的提示
     */
    public static String check(String command) {
        if (command == null || command.isBlank()) {
            return null;
        }
        String normalized = command.replaceAll("\\s+", " ").trim();

        for (DenyRule rule : RULES) {
            if (rule.pattern().matcher(normalized).find()) {
                return rule.reason();
            }
        }
        return null;
    }

    private record DenyRule(String reason, Pattern pattern) {
    }
}
