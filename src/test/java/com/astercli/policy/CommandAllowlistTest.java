package com.astercli.policy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * execute_command 白名单模式测试：精确 equals 语义、fail-safe 空名单、模式开关。
 */
class CommandAllowlistTest {

    @AfterEach
    void cleanModeProperty() {
        // fromEnvironment 以系统属性优先，测试结束后必须清除，避免污染其他测试
        System.clearProperty("astercli.command.mode");
    }

    @Test
    void disabledInstanceAllowsEverything() {
        CommandAllowlist allowlist = CommandAllowlist.disabled();
        assertFalse(allowlist.isEnabled());
        assertNull(allowlist.check("git push --force"));
        assertNull(allowlist.check("rm -rf /"));
        assertNull(allowlist.check(null));
        assertNull(allowlist.check(""));
    }

    @Test
    void allowsExactMatchOnly() {
        CommandAllowlist allowlist = CommandAllowlist.of(List.of("ls", "mvn test -Pquick"));
        assertNull(allowlist.check("ls"));
        assertNull(allowlist.check("mvn test -Pquick"));
        // 精确匹配语义：多一个参数、大小写不同、前缀相似都不放行
        assertNotNull(allowlist.check("ls -la"));
        assertNotNull(allowlist.check("LS"));
        assertNotNull(allowlist.check("mvn"));
    }

    @Test
    void trimsInputBeforeComparing() {
        CommandAllowlist allowlist = CommandAllowlist.of(List.of("pwd"));
        assertNull(allowlist.check("  pwd  "));
        assertNull(allowlist.check("pwd"));
    }

    @Test
    void emptyAllowlistRejectsEverything() {
        CommandAllowlist allowlist = CommandAllowlist.of(List.of());
        assertTrue(allowlist.isEnabled());
        String denial = allowlist.check("ls");
        assertNotNull(denial);
        // 拒绝文案要引导用户去配置文件
        assertTrue(denial.contains("commands.json"));
        assertNotNull(allowlist.check(null));
        assertNotNull(allowlist.check(""));
    }

    @Test
    void fromEnvironmentDefaultsToDisabled() {
        // 未设置模式时保持历史行为：恒放行
        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(Path.of("."));
        assertFalse(allowlist.isEnabled());
        assertNull(allowlist.check("anything"));
    }

    @Test
    void fromEnvironmentLoadsWhitelistFromProjectConfig(@TempDir Path tempDir) throws Exception {
        writeCommandsJson(tempDir, "{\"allowlist\": [\"git status\", \"echo ok\"]}");
        System.setProperty("astercli.command.mode", "whitelist");

        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);

        assertTrue(allowlist.isEnabled());
        assertNull(allowlist.check("git status"));
        assertNotNull(allowlist.check("git log"));
    }

    @Test
    void fromEnvironmentTreatsMissingConfigAsEmptyList(@TempDir Path tempDir) {
        // fail-safe：开了白名单模式但没配文件 → 全拒且文案可引导
        System.setProperty("astercli.command.mode", "whitelist");

        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);

        assertTrue(allowlist.isEnabled());
        assertNotNull(allowlist.check("ls"));
        assertTrue(allowlist.check("ls").contains("commands.json"));
    }

    @Test
    void fromEnvironmentTreatsBrokenConfigAsEmptyList(@TempDir Path tempDir) throws Exception {
        writeCommandsJson(tempDir, "{\"allowlist\": [\"ls\", ");  // 非法 JSON
        System.setProperty("astercli.command.mode", "whitelist");

        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);

        assertTrue(allowlist.isEnabled());
        assertNotNull(allowlist.check("ls"));
    }

    @Test
    void fromEnvironmentTreatsNonArrayAllowlistAsEmptyList(@TempDir Path tempDir) throws Exception {
        writeCommandsJson(tempDir, "{\"allowlist\": \"git status\"}");
        System.setProperty("astercli.command.mode", "whitelist");

        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);

        assertTrue(allowlist.isEnabled());
        assertNotNull(allowlist.check("git status"));
    }

    @Test
    void fromEnvironmentIgnoresBlankEntries(@TempDir Path tempDir) throws Exception {
        writeCommandsJson(tempDir, "{\"allowlist\": [\"  \", \"git status\"]}");
        System.setProperty("astercli.command.mode", "whitelist");

        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);

        assertNull(allowlist.check("git status"));
        assertNotNull(allowlist.check(""));
    }

    @Test
    void fromEnvironmentMatchesWhitelistModeCaseInsensitively(@TempDir Path tempDir) {
        System.setProperty("astercli.command.mode", "WHITELIST ");
        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);
        assertTrue(allowlist.isEnabled());
    }

    @Test
    void fromEnvironmentTreatsUnknownModeAsDisabled(@TempDir Path tempDir) throws Exception {
        writeCommandsJson(tempDir, "{\"allowlist\": [\"ls\"]}");
        System.setProperty("astercli.command.mode", "blacklist");

        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(tempDir);

        assertFalse(allowlist.isEnabled());
        assertNull(allowlist.check("rm -rf /"));
    }

    @Test
    void nullProjectRootFallsBackToEmptyAllowlist() {
        System.setProperty("astercli.command.mode", "whitelist");
        CommandAllowlist allowlist = CommandAllowlist.fromEnvironment(null);
        assertTrue(allowlist.isEnabled());
        assertNotNull(allowlist.check("ls"));
    }

    private static void writeCommandsJson(Path projectRoot, String content) throws Exception {
        Path dir = projectRoot.resolve(".astercli");
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("commands.json"), content);
    }
}
