"""分块器单测：tree-sitter AST 符号分块 + 文本分段兜底 + 文件收集。"""

from __future__ import annotations

from pathlib import Path

from astercli.rag.chunker import (
    MAX_CHUNK_CHARS,
    chunk_file,
    collect_files,
)


def _chunk_text(tmp_path, filename, text):
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return chunk_file(path, path.relative_to(tmp_path).as_posix(), text)


def test_python_ast_chunk(tmp_path):
    """Python 文件：类块（声明头 5 行）+ 方法块（完整函数体）。"""
    text = (
        "class Foo:\n"
        "    '''文档'''\n"
        "    def bar(self):\n"
        "        return 1\n"
        "    def baz(self):\n"
        "        return 2\n"
        "\n"
        "def top():\n"
        "    pass\n"
    )
    chunks = _chunk_text(tmp_path, "src/foo.py", text)
    classes = [c for c in chunks if c.chunk_type == "class"]
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert [c.name for c in classes] == ["Foo"]
    assert {c.name for c in methods} == {"bar", "baz", "top"}
    # 类块只含声明头前 5 行
    assert classes[0].content.count("\n") <= 4
    # 方法块含完整函数体
    bar = next(c for c in methods if c.name == "bar")
    assert "return 1" in bar.content
    assert bar.file_path == "src/foo.py"
    assert bar.start_line == 3 and bar.end_line == 4


def test_java_ast_chunk(tmp_path):
    """Java 文件：类块 + 方法块。"""
    text = (
        "public class Main {\n"
        "    private int x;\n"
        "    public void run() {\n"
        "        System.out.println(x);\n"
        "    }\n"
        "}\n"
    )
    chunks = _chunk_text(tmp_path, "Main.java", text)
    assert ("class", "Main") == (chunks[0].chunk_type, chunks[0].name)
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert [c.name for c in methods] == ["run"]
    assert "System.out.println" in methods[0].content


def test_c_function_name(tmp_path):
    """C 函数名藏在 declarator 子树里，专用提取逻辑应生效。"""
    text = "int add(int a, int b) {\n    return a + b;\n}\n"
    chunks = _chunk_text(tmp_path, "util.c", text)
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert [c.name for c in methods] == ["add"]
    assert "return a + b" in methods[0].content


def test_go_and_rust(tmp_path):
    """Go type/func 与 Rust struct/fn 声明均可识别。"""
    go_chunks = _chunk_text(tmp_path, "main.go",
                             "package main\n\ntype User struct {\n    Name string\n}\n\n"
                             "func Hello() string {\n    return \"hi\"\n}\n")
    assert ("class", "User") in [(c.chunk_type, c.name) for c in go_chunks]
    assert ("method", "Hello") in [(c.chunk_type, c.name) for c in go_chunks]

    rs_chunks = _chunk_text(tmp_path, "lib.rs",
                             "struct P { x: i32 }\nimpl P {\n    fn new() -> Self { P { x: 0 } }\n}\n")
    assert ("class", "P") in [(c.chunk_type, c.name) for c in rs_chunks]
    assert ("method", "new") in [(c.chunk_type, c.name) for c in rs_chunks]


def test_text_fallback_for_plain_file(tmp_path):
    """无符号声明文件（md/配置）回退整文件 file 块。"""
    chunks = _chunk_text(tmp_path, "README.md", "# demo\n\nhello world\n")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"
    assert chunks[0].name == "README.md"
    assert "hello world" in chunks[0].content


def test_large_text_split(tmp_path):
    """超长无符号文本按行边界分段，每段不超预算。"""
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(200))
    chunks = _chunk_text(tmp_path, "notes.txt", text)
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHUNK_CHARS + 60 for c in chunks)
    assert all(c.chunk_type == "file" for c in chunks)
    # 段间行号衔接
    assert chunks[0].start_line == 1


def test_embed_text_format(tmp_path):
    """嵌入文本携带 "[类型:名字]" 前缀（对齐 Java toEmbeddingText）。"""
    chunks = _chunk_text(tmp_path, "a.py", "def run():\n    pass\n")
    assert chunks[0].to_embed_text().startswith("[method:run]")


def test_collect_files(tmp_path):
    """收集遵循扩展名白名单与排除目录。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.java").write_text("class B {}\n", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")   # 不在扩展名白名单
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "d.js").write_text("var a;\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "e.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".git").mkdir()
    (tmp_path / "sub" / ".git" / "f.py").write_text("z = 3\n", encoding="utf-8")
    files = {p.name for p in collect_files(tmp_path)}
    assert files == {"a.py", "b.java"}
