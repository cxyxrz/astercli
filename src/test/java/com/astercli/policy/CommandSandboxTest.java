package com.astercli.policy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * OS 级命令沙箱测试：Seatbelt profile / bwrap argv 纯函数断言 + wrap 平台矩阵 + fail-closed。
 */
class CommandSandboxTest {

    @AfterEach
    void cleanSandboxProperty() {
        System.clearProperty("astercli.command.sandbox");
    }

    // ---------- wrap 矩阵 ----------

    @Test
    void disabledSandboxReturnsBareBashArgv() {
        CommandSandbox sandbox = CommandSandbox.disabled();
        List<String> argv = sandbox.wrap("echo hi", "/tmp/proj");
        assertEquals(List.of("bash", "-c", "echo hi"), argv);
    }

    @Test
    void enabledOnMacWrapsWithSandboxExec() {
        CommandSandbox sandbox = new CommandSandbox(
                true, "Mac OS X", "/usr/bin/sandbox-exec", null, List.of(), true);

        List<String> argv = sandbox.wrap("echo hi", "/tmp/proj");

        assertEquals(6, argv.size());
        assertEquals("/usr/bin/sandbox-exec", argv.get(0));
        assertEquals("-p", argv.get(1));
        // profile 内联在 -p 参数位，其后是原命令链
        assertTrue(argv.get(2).contains("(deny file-write*)"));
        assertEquals("bash", argv.get(3));
        assertEquals("-c", argv.get(4));
        assertEquals("echo hi", argv.get(5));
    }

    @Test
    void enabledOnLinuxWrapsWithBwrap() {
        CommandSandbox sandbox = new CommandSandbox(
                true, "Linux", null, "/usr/bin/bwrap", List.of(), true);

        List<String> argv = sandbox.wrap("echo hi", "/tmp/proj");

        assertEquals("/usr/bin/bwrap", argv.get(0));
        assertTrue(argv.contains("--ro-bind"));
        assertTrue(argv.contains("--dev"));
        assertTrue(argv.contains("--proc"));
        assertFalse(argv.contains("--unshare-net"));
        // 命令链在 argv 尾部
        assertEquals("bash", argv.get(argv.size() - 3));
        assertEquals("-c", argv.get(argv.size() - 2));
        assertEquals("echo hi", argv.get(argv.size() - 1));
    }

    @Test
    void enabledOnLinuxWithoutNetAddsUnshareNet() {
        CommandSandbox sandbox = new CommandSandbox(
                true, "Linux", null, "/usr/bin/bwrap", List.of(), false);
        List<String> argv = sandbox.wrap("echo hi", "/tmp/proj");
        assertTrue(argv.contains("--unshare-net"));
    }

    @Test
    void enabledWithoutBwrapFailsClosed() {
        CommandSandbox sandbox = new CommandSandbox(true, "Linux", null, null, List.of(), true);
        PolicyException e = assertThrows(PolicyException.class, () -> sandbox.wrap("echo hi", "/tmp/proj"));
        // 拒绝文案要引导用户处理（清除开关或安装工具）
        assertTrue(e.getMessage().contains("bubblewrap"));
        assertTrue(e.getMessage().contains("ASTERCLI_COMMAND_SANDBOX"));
    }

    @Test
    void enabledWithoutSandboxExecOnMacFailsClosed() {
        CommandSandbox sandbox = new CommandSandbox(true, "Mac OS X", null, null, List.of(), true);
        assertThrows(PolicyException.class, () -> sandbox.wrap("echo hi", "/tmp/proj"));
    }

    @Test
    void enabledOnUnsupportedPlatformFailsClosed() {
        CommandSandbox sandbox = new CommandSandbox(true, "Windows 11", null, null, List.of(), true);
        PolicyException e = assertThrows(PolicyException.class, () -> sandbox.wrap("echo hi", "/tmp/proj"));
        assertTrue(e.getMessage().contains("当前平台不可用"));
    }

    @Test
    void enabledOnMacDoesNotUseBwrapAndViceVersa() {
        // mac 上即使 bwrap 可用也只用 sandbox-exec；linux 上即使 sandbox-exec 存在也只用 bwrap
        CommandSandbox mac = new CommandSandbox(true, "Mac OS X", "/usr/bin/sandbox-exec", "/usr/bin/bwrap", List.of(), true);
        assertEquals("/usr/bin/sandbox-exec", mac.wrap("echo", "/tmp").get(0));
        CommandSandbox linux = new CommandSandbox(true, "Linux", "/usr/bin/sandbox-exec", "/usr/bin/bwrap", List.of(), true);
        assertEquals("/usr/bin/bwrap", linux.wrap("echo", "/tmp").get(0));
    }

    // ---------- Seatbelt profile 纯函数 ----------

    @Test
    void seatbeltProfileDeniesWritesAndAllowsKeyPaths(@TempDir Path tempDir) throws Exception {
        String profile = CommandSandbox.buildSeatbeltProfile(tempDir.toString(), List.of(), true);

        assertTrue(profile.contains("(version 1)"));
        assertTrue(profile.contains("(allow default)"));
        assertTrue(profile.contains("(deny file-write*)"));
        // /tmp 是 /private/tmp 符号链接，Seatbelt 必须写真实路径
        assertTrue(profile.contains("(allow file-write* (subpath \"/private/tmp\"))"));
        assertTrue(profile.contains(tempDir.toRealPath().toString()));
        // 默认放行网络：不应出现 deny network
        assertFalse(profile.contains("deny network"));
    }

    @Test
    void seatbeltProfileAppendsExtraWritablePaths(@TempDir Path tempDir) throws Exception {
        Path extra = Files.createDirectories(tempDir.resolve("extra"));

        String profile = CommandSandbox.buildSeatbeltProfile(tempDir.toString(), List.of(extra.toString()), true);

        assertTrue(profile.contains(extra.toRealPath().toString()));
    }

    @Test
    void seatbeltProfileDeniesNetworkWhenDisabled(@TempDir Path tempDir) {
        String profile = CommandSandbox.buildSeatbeltProfile(tempDir.toString(), List.of(), false);
        assertTrue(profile.contains("(deny network*)"));
    }

    @Test
    void seatbeltProfileKeepsNonexistentExtraPathAsAbsoluteFallback(@TempDir Path tempDir) {
        // 不存在的追加路径不跳过，而是以绝对路径形态进 profile（前缀匹配容忍未来 mkdir，
        // 对齐 ~/.m2 首次构建时自动创建的场景）
        String profile = CommandSandbox.buildSeatbeltProfile(tempDir.toString(), List.of("relative-path"), true);
        assertTrue(profile.contains(Path.of("relative-path").toAbsolutePath().normalize().toString()));
    }

    // ---------- bwrap argv 纯函数 ----------

    @Test
    void bwrapArgvBindsRootAndTmp(@TempDir Path tempDir) throws Exception {
        List<String> argv = CommandSandbox.buildBwrapArgv(
                "bwrap", tempDir.toString(), List.of(), true, "echo hi");

        assertEquals("bwrap", argv.get(0));
        // 全盘只读 + root /tmp 可写 bind + dev/proc 补挂
        int roBind = argv.indexOf("--ro-bind");
        assertEquals("/", argv.get(roBind + 1));
        int rootBind = argv.indexOf("--bind");
        assertEquals(tempDir.toRealPath().toString(), argv.get(rootBind + 1));
        assertTrue(argv.contains("/tmp"));
        assertTrue(argv.contains("--dev"));
        assertTrue(argv.contains("--proc"));
        assertFalse(argv.contains("--unshare-net"));
    }

    @Test
    void bwrapArgvAppendsExtraWritablePaths(@TempDir Path tempDir) throws Exception {
        Path extra = Files.createDirectories(tempDir.resolve("m2"));

        List<String> argv = CommandSandbox.buildBwrapArgv(
                "bwrap", tempDir.toString(), List.of(extra.toString()), true, "echo hi");

        assertTrue(argv.contains(extra.toRealPath().toString()));
    }

    // ---------- 可写路径解析 ----------

    @Test
    void parseWritablePathsExpandsTildeAndSkipsBlanks() {
        String home = System.getProperty("user.home");

        List<String> paths = CommandSandbox.parseWritablePaths("~/.m2, , ~/.gradle,/var/cache,");

        assertEquals(List.of(home + "/.m2", home + "/.gradle", "/var/cache"), paths);
    }

    // ---------- fromEnvironment 开关 ----------

    @Test
    void fromEnvironmentDefaultsToDisabled() {
        CommandSandbox sandbox = CommandSandbox.fromEnvironment(Path.of("."));
        assertFalse(sandbox.isEnabled());
    }

    @Test
    void fromEnvironmentEnablesViaSystemProperty() {
        System.setProperty("astercli.command.sandbox", "true");
        assertTrue(CommandSandbox.fromEnvironment(Path.of(".")).isEnabled());

        System.setProperty("astercli.command.sandbox", "on");
        assertTrue(CommandSandbox.fromEnvironment(Path.of(".")).isEnabled());

        System.setProperty("astercli.command.sandbox", "false");
        assertFalse(CommandSandbox.fromEnvironment(Path.of(".")).isEnabled());
    }
}
