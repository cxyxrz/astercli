package com.astercli.render.inline;

import com.astercli.hitl.ApprovalRequest;
import com.astercli.hitl.ApprovalResult;
import org.jline.terminal.Terminal;
import org.jline.utils.NonBlockingReader;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InlineApprovalPrompterTest {

    @Test
    void singleCharYReturnsApprove() throws Exception {
        Terminal terminal = mockTerminalReturningChars("y");
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("write_file", "{\"path\":\"a\"}", "test"));
        assertEquals(ApprovalResult.Decision.APPROVED, result.decision());
    }

    @Test
    void singleCharSReturnsSkip() throws Exception {
        Terminal terminal = mockTerminalReturningChars("s");
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("write_file", "{}", "test"));
        assertEquals(ApprovalResult.Decision.SKIPPED, result.decision());
    }

    @Test
    void singleCharNFollowedByReasonReturnsRejected() throws Exception {
        Terminal terminal = mockTerminalReturningChars("ntoo risky\n");
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("write_file", "{}", "test"));
        assertEquals(ApprovalResult.Decision.REJECTED, result.decision());
        assertEquals("too risky", result.reason());
    }

    @Test
    void singleCharAOnBuiltinToolApproveAll() throws Exception {
        Terminal terminal = mockTerminalReturningChars("a");
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("write_file", "{}", "test"));
        assertEquals(ApprovalResult.Decision.APPROVED_ALL, result.decision());
    }

    @Test
    void singleCharAOnMcpToolApproveAllByServer() throws Exception {
        Terminal terminal = mockTerminalReturningChars("aserver\n");
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("mcp__chrome-devtools__click", "{}", "test"));
        assertEquals(ApprovalResult.Decision.APPROVED_ALL_BY_SERVER, result.decision());
    }

    @Test
    void singleCharMWithValidJsonReturnsModified() throws Exception {
        Terminal terminal = mockTerminalReturningChars("m{\"path\":\"safe.txt\"}\n");
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("write_file", "{\"path\":\"a\"}", "test"));
        assertEquals(ApprovalResult.Decision.MODIFIED, result.decision());
        assertTrue(result.modifiedArguments().contains("safe.txt"));
    }

    @Test
    void rawModeFailureFallsThroughToReject() throws Exception {
        Terminal terminal = Mockito.mock(Terminal.class);
        Mockito.when(terminal.enterRawMode()).thenThrow(new RuntimeException("no tty"));
        ByteArrayOutputStream sink = new ByteArrayOutputStream();
        InlineApprovalPrompter p = new InlineApprovalPrompter(
                new PrintStream(sink, true, StandardCharsets.UTF_8),
                terminal);
        ApprovalResult result = p.prompt(ApprovalRequest.of("write_file", "{}", "test"));
        assertNotNull(result);
        assertEquals(ApprovalResult.Decision.REJECTED, result.decision());
    }

    private static Terminal mockTerminalReturningChars(String input) throws Exception {
        Terminal terminal = Mockito.mock(Terminal.class);
        Mockito.when(terminal.enterRawMode()).thenReturn(null);
        NonBlockingReader reader = Mockito.mock(NonBlockingReader.class);
        int[] chars = input.chars().toArray();
        int[] index = {0};
        Mockito.when(reader.read()).thenAnswer(invocation -> {
            if (index[0] < chars.length) {
                return chars[index[0]++];
            }
            return -1;
        });
        Mockito.when(terminal.reader()).thenReturn(reader);
        return terminal;
    }
}