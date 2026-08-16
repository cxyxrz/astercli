package com.astercli.tool;

import com.astercli.llm.LlmClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 精简工具定义模式：参数说明并入 description，parameters 只留 {"type":"object"} + required。
 */
class CompactToolDefinitionTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @AfterEach
    void clearMode() {
        System.clearProperty("astercli.tool.mode");
    }

    @Test
    void compactToolMovesParametersIntoDescription() throws Exception {
        JsonNode schema = MAPPER.readTree("""
                {"type":"object","properties":{
                  "path":{"type":"string","description":"文件路径"},
                  "offset":{"type":"integer","description":"起始行号"}
                },"required":["path"]}
                """);
        LlmClient.Tool compact = ToolRegistry.compactTool(new LlmClient.Tool("read_file", "读取文件", schema));

        assertEquals("read_file", compact.name());
        assertTrue(compact.description().startsWith("读取文件"));
        assertTrue(compact.description().contains("- path (string, required): 文件路径"));
        assertTrue(compact.description().contains("- offset (integer, optional): 起始行号"));

        assertEquals("object", compact.parameters().path("type").asText());
        assertEquals("path", compact.parameters().path("required").get(0).asText());
        assertFalse(compact.parameters().has("properties"));
    }

    @Test
    void compactToolKeepsEnumArrayAndNestedFields() throws Exception {
        JsonNode schema = MAPPER.readTree("""
                {"type":"object","properties":{
                  "mode":{"type":"string","enum":["fast","full"]},
                  "targets":{"type":"array","items":{"type":"string"}},
                  "options":{"type":"object","properties":{"depth":{"type":"integer"},"follow":{"type":"boolean"}}}
                }}
                """);
        LlmClient.Tool compact = ToolRegistry.compactTool(new LlmClient.Tool("t", "d", schema));

        assertTrue(compact.description().contains("enum: fast | full"));
        assertTrue(compact.description().contains("array<string>"));
        assertTrue(compact.description().contains("fields: depth, follow"));
        assertFalse(compact.parameters().has("required"));
    }

    @Test
    void compactToolHandlesMissingSchema() {
        LlmClient.Tool compact = ToolRegistry.compactTool(new LlmClient.Tool("t", "d", null));

        assertEquals("d", compact.description());
        assertEquals("object", compact.parameters().path("type").asText());
        assertFalse(compact.parameters().has("required"));
    }

    @Test
    void registryDefaultsToCompactAndSchemaModeRestoresFullSchema() {
        ToolRegistry registry = new ToolRegistry();

        LlmClient.Tool compact = findTool(registry, "read_file");
        assertTrue(compact.description().contains("- path (string, required)"));
        assertFalse(compact.parameters().has("properties"));

        System.setProperty("astercli.tool.mode", "schema");
        LlmClient.Tool full = findTool(registry, "read_file");
        assertTrue(full.parameters().has("properties"));
        assertFalse(full.description().contains("- path (string, required)"));
    }

    private static LlmClient.Tool findTool(ToolRegistry registry, String name) {
        return registry.getToolDefinitions().stream()
                .filter(t -> t.name().equals(name))
                .findFirst()
                .orElseThrow();
    }
}
