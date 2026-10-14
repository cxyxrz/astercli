"""MCP resource mention：``@server:uri`` 解析与展开（对齐 Java AtMentionParser / AtMentionExpander）。

行为契约：
- 解析规则：server 名字母开头（``[A-Za-z][A-Za-z0-9_-]*``），uri 必须是
  ``scheme://...`` 形态；引号（单/双）内的内容跳过，不误展开。
- 展开成功 → ``<resource server=... uri=... mimeType=...>`` 块包裹内容
  （截断预算在 manager 层，对齐 Java 的 20 万字符）。
- 展开失败（server 不存在 / 未就绪 / 读取异常）→ ``<resource_error ...>``
  块，不中断用户输入（对齐 Java：失败降级为错误块）。
"""

from __future__ import annotations

import re
from typing import Optional

MENTION_PATTERN = re.compile(
    r"@([A-Za-z][A-Za-z0-9_-]*):([A-Za-z][A-Za-z0-9+.-]*://[^\s\"']+)"
)


def parse_at_mentions(text: str) -> list:
    """解析非引号区间的全部 mention，返回 ``[(server, uri, raw)]``。"""
    mentions = []
    for segment, in_quote in _split_quoted(text):
        if in_quote:
            continue
        for match in MENTION_PATTERN.finditer(segment):
            server, uri = match.group(1), match.group(2)
            mentions.append((server, uri, match.group(0)))
    return mentions


def expand_mentions(text: str, manager) -> str:
    """把文本中的 ``@server:uri`` 展开为 resource 块；失败降级为错误块。"""
    if "@" not in text:
        return text

    def expand_segment(segment: str) -> str:
        def repl(match) -> str:
            server, uri = match.group(1), match.group(2)
            ok, content, mime, error = manager.read_resource_for_mention(server, uri)
            if ok:
                mime_attr = f' mimeType="{mime}"' if mime else ""
                return (
                    f'<resource server="{server}" uri="{uri}"{mime_attr}>\n'
                    f"{content}\n</resource>"
                )
            return f'<resource_error server="{server}" uri="{uri}">{error}</resource_error>'

        return MENTION_PATTERN.sub(repl, segment)

    return "".join(
        segment if in_quote else expand_segment(segment)
        for segment, in_quote in _split_quoted(text)
    )


def _split_quoted(text: str) -> list:
    """按引号切段：返回 ``[(segment, in_quote)]``，引号本身保留在段内。"""
    segments: list = []
    buf: list = []
    quote: Optional[str] = None
    for ch in text:
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                segments.append(("".join(buf), True))
                buf = []
                quote = None
            continue
        if ch in ('"', "'"):
            if buf:
                segments.append(("".join(buf), False))
                buf = []
            quote = ch
            buf = [ch]
            continue
        buf.append(ch)
    if buf:
        segments.append(("".join(buf), quote is not None))
    return segments
