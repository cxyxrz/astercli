package com.astercli.policy;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * execute_command 可选 OS 级沙箱包装：ASTERCLI_COMMAND_SANDBOX=true 开启后，
 * 子进程在写文件系统围栏内运行——仅项目根 / /tmp / 追加路径可写，其余全盘禁写。
 *
 * 定位：黑名单（挡语义即破坏的命令文本）与白名单（收紧放行集合）都是执行前静态判定，
 * 共同盲区是子进程启动后的任意写路径（如解释器内联代码写 HOME 持久化后门——
 * PathGuard 只管工具参数路径，不管子进程行为）。本类是执行时围栏：
 * 即使命令本体绕过静态判定，落盘行为也被内核级拦截。
 *
 * 平台实现：macOS 用系统自带 sandbox-exec（Seatbelt profile，-p 内联传参，
 * ProcessBuilder 是 argv 数组无 shell 转义问题）；Linux 用 bubblewrap（bwrap）。
 *
 * fail-closed：显式开启但平台不支持（Windows）或工具缺失（Linux 无 bwrap）时
 * 拒绝执行而非静默降级——与白名单"配置缺失全拒"同一设计语言。
 *
 * 配置：
 * - ASTERCLI_COMMAND_SANDBOX（或 -Dastercli.command.sandbox）= true/on 开启，默认关
 * - ASTERCLI_SANDBOX_WRITABLE_PATHS：逗号分隔追加可写路径，支持 ~ 展开（如 ~/.m2,~/.gradle）
 * - ASTERCLI_SANDBOX_ALLOW_NET：默认 true；false 时禁网（macOS 加 (deny network*)，Linux 加 --unshare-net）
 */
public class CommandSandbox {

    private final boolean enabled;
    private final String osName;
    private final String sandboxExecPath;   // null 表示不可用
    private final String bwrapPath;          // null 表示不可用
    private final List<String> extraWritablePaths;
    private final boolean allowNet;

    /**
     * 全依赖注入构造器（包可见，测试用）；fromEnvironment 是生产入口。
     */
    CommandSandbox(boolean enabled, String osName, String sandboxExecPath, String bwrapPath,
                   List<String> extraWritablePaths, boolean allowNet) {
        this.enabled = enabled;
        this.osName = osName == null ? "" : osName;
        this.sandboxExecPath = sandboxExecPath;
        this.bwrapPath = bwrapPath;
        this.extraWritablePaths = List.copyOf(extraWritablePaths);
        this.allowNet = allowNet;
    }

    static CommandSandbox disabled() {
        return new CommandSandbox(false, "", null, null, List.of(), true);
    }

    /**
     * 生产入口：读系统属性 / 环境变量 + 真实沙箱工具检测。
     */
    public static CommandSandbox fromEnvironment(Path projectRoot) {
        boolean enabled = parseEnabled(
                firstNonNull(System.getProperty("astercli.command.sandbox"),
                        System.getenv("ASTERCLI_COMMAND_SANDBOX")));
        String osName = System.getProperty("os.name", "");
        String sandboxExecPath = probeSandboxExec();
        String bwrapPath = findExecutable("bwrap");
        List<String> extraPaths = parseWritablePaths(
                firstNonNull(System.getenv("ASTERCLI_SANDBOX_WRITABLE_PATHS"), ""));
        // 默认放行网络；显式 false（忽略大小写）断网，其他任意值维持默认
        String rawNet = firstNonNull(System.getenv("ASTERCLI_SANDBOX_ALLOW_NET"), "true").trim();
        boolean allowNet = !"false".equalsIgnoreCase(rawNet);
        return new CommandSandbox(enabled, osName, sandboxExecPath, bwrapPath, extraPaths, allowNet);
    }

    /**
     * 把命令包装成完整 ProcessBuilder argv。
     *
     * @return argv 数组；沙箱未开启时返回裸 ["bash","-c",command]
     * @throws PolicyException 开启但平台/工具不可用时拒绝执行（fail-closed）
     */
    public List<String> wrap(String command, String projectRoot) throws PolicyException {
        if (!enabled) {
            return Arrays.asList("bash", "-c", command);
        }
        if (isMac() && sandboxExecPath != null) {
            String profile = buildSeatbeltProfile(projectRoot, extraWritablePaths, allowNet);
            return List.of(sandboxExecPath, "-p", profile, "bash", "-c", command);
        }
        if (isLinux() && bwrapPath != null) {
            return buildBwrapArgv(bwrapPath, projectRoot, extraWritablePaths, allowNet, command);
        }
        throw new PolicyException(
                "命令沙箱已开启但当前平台不可用（macOS 需要 /usr/bin/sandbox-exec，Linux 需要安装 bubblewrap）；"
                        + "如需继续执行请清除 ASTERCLI_COMMAND_SANDBOX");
    }

    public boolean isEnabled() {
        return enabled;
    }

    // ---------- Seatbelt（macOS） ----------

    /**
     * 生成 Seatbelt profile：读操作全部放行，写操作默认拒绝，仅白名单 subpath 可写。
     * 纯函数，单测断言对象。
     */
    static String buildSeatbeltProfile(String projectRoot, List<String> extraWritablePaths, boolean allowNet) {
        StringBuilder sb = new StringBuilder();
        sb.append("(version 1)\n");
        sb.append("(allow default)\n");
        sb.append("(deny file-write*)\n");
        // macOS 的 /tmp 是 /private/tmp 的符号链接，Seatbelt 按真实路径解析，必须写 /private/tmp
        sb.append("(allow file-write* (subpath \"/private/tmp\"))\n");
        String root = canonicalize(projectRoot);
        if (root != null) {
            sb.append("(allow file-write* (subpath \"").append(root).append("\"))\n");
        }
        for (String extra : extraWritablePaths) {
            String canonical = canonicalize(extra);
            if (canonical != null && !canonical.equals(root)) {
                sb.append("(allow file-write* (subpath \"").append(canonical).append("\"))\n");
            }
        }
        if (!allowNet) {
            sb.append("(deny network*)\n");
        }
        return sb.toString();
    }

    // ---------- bubblewrap（Linux） ----------

    /**
     * 生成 bwrap 完整 argv：全盘只读挂载 + 可写路径 bind + dev/proc 补挂 + 可选断网。
     * HOME 不 bind → 继承只读，写 HOME 被拒。
     */
    static List<String> buildBwrapArgv(String bwrapPath, String projectRoot,
                                       List<String> extraWritablePaths, boolean allowNet, String command) {
        List<String> argv = new ArrayList<>();
        argv.add(bwrapPath);
        argv.add("--ro-bind");
        argv.add("/");
        argv.add("/");
        String root = canonicalize(projectRoot);
        if (root != null) {
            argv.addAll(List.of("--bind", root, root));
        }
        argv.addAll(List.of("--bind", "/tmp", "/tmp"));
        for (String extra : extraWritablePaths) {
            String canonical = canonicalize(extra);
            if (canonical != null) {
                argv.addAll(List.of("--bind", canonical, canonical));
            }
        }
        // ro-bind 下的 /dev /proc 只读，bash 写 /dev/null 需要 --dev，ps 等需要 --proc
        argv.addAll(List.of("--dev", "/dev"));
        argv.addAll(List.of("--proc", "/proc"));
        if (!allowNet) {
            argv.add("--unshare-net");
        }
        argv.addAll(List.of("bash", "-c", command));
        return argv;
    }

    // ---------- 辅助 ----------

    private boolean isMac() {
        return osName.toLowerCase(Locale.ROOT).contains("mac");
    }

    private boolean isLinux() {
        return osName.toLowerCase(Locale.ROOT).contains("linux");
    }

    private static boolean parseEnabled(String raw) {
        if (raw == null) {
            return false;
        }
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        return "true".equals(normalized) || "on".equals(normalized);
    }

    static List<String> parseWritablePaths(String raw) {
        List<String> result = new ArrayList<>();
        for (String part : raw.split(",")) {
            String expanded = expandTilde(part.trim());
            if (!expanded.isEmpty()) {
                result.add(expanded);
            }
        }
        return result;
    }

    private static String expandTilde(String path) {
        if (path.startsWith("~/") || path.equals("~")) {
            String home = System.getProperty("user.home", "");
            if (!home.isEmpty()) {
                return path.equals("~") ? home : home + path.substring(1);
            }
        }
        return path;
    }

    private static String canonicalize(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        try {
            Path path = Path.of(raw).toRealPath();
            return path.toString();
        } catch (Exception e) {
            // toRealPath 要求路径存在；不存在时退化为绝对路径规范化（fail-safe 而非抛错）
            try {
                return Path.of(raw).toAbsolutePath().normalize().toString();
            } catch (Exception e2) {
                return null;
            }
        }
    }

    private static String probeSandboxExec() {
        Path path = Path.of("/usr/bin/sandbox-exec");
        return Files.isExecutable(path) ? path.toString() : null;
    }

    /**
     * 沿 PATH 逐目录查找可执行文件；找不到返回 null。
     */
    static String findExecutable(String name) {
        String pathEnv = System.getenv("PATH");
        if (pathEnv == null || name == null || name.isBlank()) {
            return null;
        }
        for (String dir : pathEnv.split(File.pathSeparator)) {
            if (dir.isBlank()) {
                continue;
            }
            Path candidate = Path.of(dir, name);
            if (Files.isExecutable(candidate)) {
                return candidate.toString();
            }
        }
        return null;
    }

    private static String firstNonNull(String a, String b) {
        return a != null ? a : b;
    }
}
