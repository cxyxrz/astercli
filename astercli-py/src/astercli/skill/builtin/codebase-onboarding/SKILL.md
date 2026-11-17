---
name: codebase-onboarding
description: |
  陌生代码库快速摸底：先看什么、用什么顺序读、如何用 glob_files / grep_code / read_file 定位入口与关键调用链，最后输出模块地图与风险点。
  触发场景：用户要求理解某个仓库或模块、接手新项目、搞清楚某个功能的实现在哪、评估改动影响面。
version: "1.0.0"
author: astercli
tags: [codebase, exploration, architecture]
---

# codebase-onboarding Skill

目标：在不读完整仓库的前提下，形成一张**够用**的心智模型——知道入口在哪、核心模块怎么协作、改动会波及哪里。

## 探查顺序

1. **看文档与门面**：`README` / `AGENTS.md` / `pyproject.toml` / `package.json` / `pom.xml`，一次性拿到定位、技术栈、常用命令。
2. **画目录骨架**：`list_dir` 或 `glob_files` 取两到三层，先识别分层（cli / core / adapters / tests），不要逐文件打开。
3. **找入口**：`grep_code` 搜 `def main`、`if __name__`、`@app.route`、`public static void main`、`console_scripts` 等特征。
4. **顺一条主链路**：从入口到一个典型功能，读 3~6 个关键文件，确认模块边界与数据流。
5. **按需深入**：只在用户关心的功能上展开，其余保持"知道在哪"即可。

## 工具分工

| 目标 | 工具 | 要点 |
|---|---|---|
| 找文件名 / 后缀分布 | `glob_files` | 先宽后窄，如 `**/*.py` → `**/agent/*.py` |
| 找符号定义与调用 | `grep_code` | 搜标识符比搜自然语言可靠；注意 `head_limit` 截断 |
| 读具体实现 | `read_file` | 先看函数签名与 docstring，再决定是否读全文 |
| 模糊概念 / 跨语言检索 | `search_code` | 语义补充，不作为精确定位首选 |

`grep_code` 返回 `partial: true` 或 `suggested_reads` 时，说明结果被预算截断：缩小关键词或按建议直接读行段。

## 输出模板

回答"这个项目怎么组织的"时，用固定结构，避免罗列文件名：

```
定位：一句话说明项目做什么、给谁用
技术栈：语言 / 框架 / 关键依赖（只列影响架构的）
模块地图：模块 → 职责 → 关键文件（每行一个）
主链路：入口 → 关键调用 → 输出（用箭头串起来）
风险点：耦合最重的地方 / 缺测试的地方 / 需要小心改的约定
```

## 常见误区

- 不要用 `read_file` 整篇读大文件：先 `grep_code` 定位行号再带 offset 读段。
- 不要把目录名当架构：`utils/` 里可能有核心逻辑，`core/` 里可能全是 DTO。
- 不要跳过测试目录：测试往往是最准确的"用法文档"。
- 不要在没读代码前给重构建议。

常用搜索式样见 `references/search-recipes.md`，需要时用 `read_skill_file` 读取。
