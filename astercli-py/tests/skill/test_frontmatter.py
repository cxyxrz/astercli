"""frontmatter 解析器测试：单行 / 多行 / 数组 / 嵌套警告 / 缺标记。"""

from __future__ import annotations

from astercli.skill.frontmatter import parse


class TestBasic:
    def test_标准单行(self):
        text = """---
name: my-skill
description: 一个测试 skill
version: 1.0
author: 张三
---

正文内容第一行。
正文内容第二行。"""
        result = parse(text)
        assert result.frontmatter["name"] == "my-skill"
        assert result.frontmatter["description"] == "一个测试 skill"
        assert result.frontmatter["version"] == "1.0"
        assert result.frontmatter["author"] == "张三"
        assert result.warnings == []
        assert result.body.startswith("正文内容第一行。")
        assert result.body.endswith("第二行。")

    def test_引号自动剥离(self):
        result = parse('---\nname: "quoted"\ndescription: \'单引号\'\n---\nbody')
        assert result.frontmatter["name"] == "quoted"
        assert result.frontmatter["description"] == "单引号"

    def test_行内数组(self):
        result = parse("---\ntags: [java, python, rust]\n---\nbody")
        assert result.frontmatter["tags"] == ["java", "python", "rust"]

    def test_空数组(self):
        result = parse("---\ntags: []\n---\nbody")
        assert result.frontmatter["tags"] == []


class TestMultiline:
    def test_竖线多行_折叠为单行(self):
        text = """---
name: s
description: |
  第一行描述
  第二行描述
body_start
---
正文"""
        result = parse(text)
        assert result.frontmatter["description"] == "第一行描述 第二行描述"
        assert result.body == "正文"

    def test_缺结束标记_整段当_body_带警告(self):
        result = parse("---\nname: s\nbody without end")
        assert result.frontmatter == {}
        assert "缺少 frontmatter 结束标记" in result.warnings[0]
        assert "body without end" in result.body


class TestWarnings:
    def test_缺起始标记(self):
        result = parse("name: s\n---\nbody")
        assert "缺少 frontmatter 起始标记" in result.warnings[0]
        assert result.body == "name: s\n---\nbody"

    def test_无冒号行_警告并跳过(self):
        result = parse("---\n这是一行没有冒号的内容\nname: ok\n---\nbody")
        assert "无法解析的 frontmatter 行" in result.warnings[0]
        assert result.frontmatter["name"] == "ok"

    def test_嵌套对象_不支持警告(self):
        result = parse("---\nmeta: { a: 1 }\nname: ok\n---\nbody")
        assert any("嵌套对象" in w for w in result.warnings)
        assert "meta" not in result.frontmatter

    def test_注释行与空行_跳过(self):
        result = parse("---\n# 注释\n\nname: s\n---\nbody")
        assert result.warnings == []
        assert result.frontmatter["name"] == "s"

    def test_null_输入(self):
        result = parse(None)
        assert result.frontmatter == {}
        assert "内容为 null" in result.warnings[0]


class TestEdgeCases:
    def test_空_value_警告(self):
        result = parse("---\nname:\ndescription: 有值\n---\nbody")
        assert any("缺少值" in w for w in result.warnings)
        assert result.frontmatter["description"] == "有值"

    def test_引号内冒号_不算分隔符(self):
        result = parse("---\ndescription: 时间: 12:30 出发\n---\nbody")
        assert result.frontmatter["description"] == "时间: 12:30 出发"

    def test_crlf_换行归一(self):
        result = parse("---\r\nname: s\r\n---\r\nbody")
        assert result.frontmatter["name"] == "s"
        assert result.body == "body"
