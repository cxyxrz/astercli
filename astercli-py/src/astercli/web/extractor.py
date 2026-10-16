"""极简 readability：HTML → 主正文 Markdown（对齐 Java web.HtmlExtractor）。

思路（按优先级）：
1. 清理噪声标签：script、style、nav、aside、footer、header、form、iframe、广告 class
2. 找主语义容器：article、main、role=main
3. 都没有则给所有 block 元素打分（文本长度 - 链接占比惩罚），选最高分
4. 把选中容器递归转成 Markdown

Java 用 Jsoup，Python 用 BeautifulSoup（html.parser 后端，零额外解析器依赖）。
不追求与 Mozilla Readability 完全对齐，目标覆盖博客 / 文档 / 官网这类 SSR 页面；
SPA 渲染后的空 HTML 会得到空字符串，由调用方提示边界。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

NOISE_TAGS = (
    "script", "style", "noscript", "iframe", "nav", "aside",
    "header", "footer", "form", "svg", "canvas", "button",
)
NOISE_CLASS_KEYWORDS = (
    "ads", "advert", "banner", "popup", "modal", "subscribe", "newsletter",
    "related", "recommend", "comment", "share", "social", "breadcrumb",
    "sidebar", "promo", "cookie", "footer", "navigation",
)
MIN_MAIN_TEXT = 80


@dataclass(frozen=True)
class Extracted:
    title: str
    markdown: str


class HtmlExtractor:
    def extract(self, html: str, base_url: Optional[str] = None) -> Extracted:
        self._base_url = base_url
        soup = BeautifulSoup(html if html else "", "html.parser")
        title = self._pick_title(soup)
        self._clean_noise(soup)
        main = self._pick_main_element(soup)
        if main is None:
            return Extracted(title=title, markdown="")
        out: list = []
        self._render_children(main, out, False)
        markdown = _collapse_blank_lines("".join(out)).strip()
        return Extracted(title=title, markdown=markdown)

    # ---------- 定位主容器 ----------

    def _pick_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(strip=True)
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else ""

    def _clean_noise(self, soup: BeautifulSoup) -> None:
        for tag in NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()
        # 按 class/id 关键词清理常见广告 / 导航壳（对齐 Java）
        for el in soup.find_all(attrs={"class": True}):
            self._remove_if_noise_marker(el)
        for el in soup.find_all(attrs={"id": True}):
            self._remove_if_noise_marker(el)

    @staticmethod
    def _remove_if_noise_marker(el: Tag) -> None:
        classes = el.get("class") or []
        marker = " ".join(classes) + " " + str(el.get("id") or "")
        marker = marker.lower()
        for kw in NOISE_CLASS_KEYWORDS:
            if kw in marker:
                el.decompose()
                return

    def _pick_main_element(self, soup: BeautifulSoup) -> Optional[Tag]:
        semantic = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
        if semantic is not None and len(semantic.get_text()) > MIN_MAIN_TEXT:
            return semantic
        candidates = soup.find_all(["div", "section", "article", "main"])
        best = soup.body
        best_score = self._score(best) if best is not None else 0.0
        for el in candidates:
            score = self._score(el)
            if score > best_score:
                best = el
                best_score = score
        return best

    def _score(self, el: Optional[Tag]) -> float:
        if el is None:
            return 0.0
        text = el.get_text()
        text_len = len(text)
        if text_len < MIN_MAIN_TEXT:
            return 0.0
        link_len = sum(len(a.get_text()) for a in el.find_all("a"))
        link_ratio = link_len / text_len
        # 链接密度高 → 大概率是导航 / 列表页
        penalty = min(link_ratio * 2.0, 1.0)
        return text_len * (1.0 - penalty)

    # ---------- Markdown 渲染 ----------

    def _render_children(self, parent: Tag, out: list, in_list_context: bool) -> None:
        for child in parent.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                text = str(child)
                if text.strip():
                    out.append(text)
            elif isinstance(child, Tag):
                self._render_element(child, out, in_list_context)

    def _render_element(self, el: Tag, out: list, in_list_context: bool) -> None:
        tag = el.name.lower()
        heading_prefix = {
            "h1": "# ", "h2": "## ", "h3": "### ",
            "h4": "#### ", "h5": "##### ", "h6": "###### ",
        }
        if tag in heading_prefix:
            text = el.get_text(strip=True)
            if text:
                out.append(f"\n\n{heading_prefix[tag]}{text}\n\n")
        elif tag == "p":
            out.append("\n\n")
            self._render_children(el, out, False)
            out.append("\n\n")
        elif tag == "br":
            out.append("\n")
        elif tag == "hr":
            out.append("\n\n---\n\n")
        elif tag in ("strong", "b"):
            out.append("**")
            self._render_children(el, out, in_list_context)
            out.append("**")
        elif tag in ("em", "i"):
            out.append("*")
            self._render_children(el, out, in_list_context)
            out.append("*")
        elif tag == "code":
            if el.parent is not None and el.parent.name == "pre":
                self._render_children(el, out, in_list_context)
            else:
                out.append(f"`{el.get_text()}`")
        elif tag == "pre":
            out.append("\n\n```\n")
            out.append(el.get_text().rstrip())
            out.append("\n```\n\n")
        elif tag == "blockquote":
            inner: list = []
            self._render_children(el, inner, False)
            out.append("\n\n")
            for line in "".join(inner).strip().split("\n"):
                out.append(f"> {line}\n")
            out.append("\n")
        elif tag in ("ul", "ol"):
            self._render_list(el, out, ordered=(tag == "ol"))
        elif tag == "li":
            # 没有外层 ul/ol（罕见）就当无序列表项处理
            out.append("\n- ")
            self._render_children(el, out, True)
        elif tag == "a":
            self._render_link(el, out)
        elif tag == "img":
            # 默认不渲染图片：体积爆涨且 LLM 处理不了图片字节
            alt = el.get("alt") or ""
            if alt.strip():
                out.append(alt)
        elif tag == "table":
            self._render_table(el, out)
        else:
            self._render_children(el, out, in_list_context)

    def _render_link(self, el: Tag, out: list) -> None:
        text = el.get_text()
        if not text.strip():
            return
        href = (el.get("href") or "").strip()
        if not href:
            out.append(text)
            return
        if self._base_url:
            href = urljoin(self._base_url, href)
        out.append(f"[{text}]({href})")

    def _render_list(self, list_el: Tag, out: list, *, ordered: bool) -> None:
        out.append("\n")
        idx = 1
        for li in list_el.find_all("li", recursive=False):
            inner: list = []
            self._render_children(li, inner, True)
            prefix = f"{idx}. " if ordered else "- "
            idx += 1
            out.append(prefix + "".join(inner).strip().replace("\n", " ") + "\n")
        out.append("\n")

    def _render_table(self, table: Tag, out: list) -> None:
        rows = table.find_all("tr")
        if not rows:
            return
        out.append("\n\n")
        header_written = False
        for row in rows:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            texts = [c.get_text().replace("|", "\\|").strip() for c in cells]
            out.append("| " + " | ".join(texts) + " |\n")
            if not header_written:
                out.append("|" + " --- |" * len(texts) + "\n")
                header_written = True
        out.append("\n")

    # base_url 通过实例传递（render_link 需要）
    _base_url: Optional[str] = None


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)
