package com.astercli.tool;

import com.astercli.llm.LlmClient;
import com.astercli.mcp.protocol.McpToolDescriptor;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 统一工具输出预算：超过上限的工具结果被截断并追加缩小范围提示，imageParts 保留。
 */
class ToolOutputBudgetTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @AfterEach
    void clearConfig() {
        System.clearProperty("astercli.tool.output.max.chars");
    }

    @Test
    void oversizedMcpOutputIsTruncatedWithHint(@TempDir Path tempDir) throws Exception {
        withAuditDir(tempDir, () -> {
            ToolRegistry registry = new ToolRegistry();
            registry.registerMcpTool(sampleDescriptor(), args -> "x".repeat(100_000));

            String result = registry.executeTool("mcp__demo__echo", "{}");

            assertTrue(result.startsWith("xxxx"));
            assertTrue(result.contains("输出超长已截断"), "应包含截断提示: " + result.substring(result.length() - 300));
            assertTrue(result.contains("原文 100000 字符"));
            assertTrue(result.length() < 70_000, "截断后应显著变短: " + result.length());
        });
    }

    @Test
    void shortOutputPassesThroughUnchanged(@TempDir Path tempDir) throws Exception {
        withAuditDir(tempDir, () -> {
            ToolRegistry registry = new ToolRegistry();
            registry.registerMcpTool(sampleDescriptor(), args -> "echo:" + args);

            assertEquals("echo:{}", registry.executeTool("mcp__demo__echo", "{}"));
        });
    }

    @Test
    void truncationKeepsImageParts(@TempDir Path tempDir) throws Exception {
        withAuditDir(tempDir, () -> {
            ToolRegistry registry = new ToolRegistry();
            registry.registerMcpToolOutput(sampleDescriptor(), args -> new ToolOutput(
                    "x".repeat(100_000),
                    List.of(LlmClient.ContentPart.imageBase64("aGVsbG8=", "image/png"))));

            ToolOutput output = registry.executeToolOutput("mcp__demo__echo", "{}");

            assertTrue(output.text().contains("输出超长已截断"));
            assertEquals(1, output.imageParts().size());
            assertEquals("image/png", output.imageParts().get(0).mimeType());
        });
    }

    @Test
    void maxCharsIsConfigurableViaSystemProperty(@TempDir Path tempDir) throws Exception {
        withAuditDir(tempDir, () -> {
            System.setProperty("astercli.tool.output.max.chars", "1000");
            ToolRegistry registry = new ToolRegistry();
            registry.registerMcpTool(sampleDescriptor(), args -> "y".repeat(5_000));

            String result = registry.executeTool("mcp__demo__echo", "{}");

            assertEquals(1_000, countLeading(result, 'y'), "应只保留前 1000 个字符");
            assertTrue(result.endsWith("请缩小范围后重试：read_file 用 offset/limit 分段读取，grep_code 调小 max_chars/head_limit，web_fetch 调小 max_chars，MCP 工具请增加过滤/分页参数)"));
        });
    }

    @Test
    void errorMessagesAreNotTruncatedAway(@TempDir Path tempDir) throws Exception {
        withAuditDir(tempDir, () -> {
            System.setProperty("astercli.tool.output.max.chars", "100");
            ToolRegistry registry = new ToolRegistry();
            registry.registerMcpTool(sampleDescriptor(), args -> {
                throw new RuntimeException("upstream broke");
            });

            String result = registry.executeTool("mcp__demo__echo", "{}");

            assertFalse(result.contains("输出超长已截断"));
            assertTrue(result.contains("upstream broke"));
        });
    }

    private static int countLeading(String text, char c) {
        int count = 0;
        while (count < text.length() && text.charAt(count) == c) {
            count++;
        }
        return count;
    }

    private static McpToolDescriptor sampleDescriptor() throws Exception {
        return new McpToolDescriptor(
                "demo",
                "echo",
                "mcp__demo__echo",
                "Echo input",
                MAPPER.readTree("{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}")
        );
    }

    private static void withAuditDir(Path tempDir, ThrowingRunnable body) throws Exception {
        String previous = System.getProperty("astercli.audit.dir");
        System.setProperty("astercli.audit.dir", tempDir.resolve("audit").toString());
        try {
            body.run();
        } finally {
            if (previous == null) {
                System.clearProperty("astercli.audit.dir");
            } else {
                System.setProperty("astercli.audit.dir", previous);
            }
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
