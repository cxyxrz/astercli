# 常见站点的 URL 形态与抓取建议

本文件是 `web-research` 的附属细则，只在 `web_fetch` 失败或需要选 URL 形态时读取。

## GitHub

- 仓库首页 / README：优先 `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/README.md`。
- 单个源码文件：`raw.githubusercontent.com` 比 `github.com/blob/...` 干净，没有页面导航。
- Issue / PR：`github.com/<owner>/<repo>/issues/<n>` 通常可 fetch；只取首屏内容，后续评论可能不在正文里。
- 抓不到时退回 `web_search`，用仓库名 + 关键词找镜像或文档站。

## 官方文档

- 优先找 `/docs`、`/documentation`、`/reference` 路径，而不是营销首页。
- 部分文档站提供 `llms.txt` 或 `llms-full.txt`，fetch 它比逐页抓更省预算。
- 版本选择：URL 里带版本号时（`/v2/`、`/3.x/`）确认与用户使用的版本一致。

## 微信公众号

- `mp.weixin.qq.com` 对非浏览器请求经常返回验证页或空正文。
- 处理顺序：`web_fetch` 一次 → 失败则用 `web_search` 找同主题的转载 / 摘要 → 告知用户原文不可直读。
- 不要伪装 UA 或尝试绕验证，这既不稳定也不合规。

## 知乎 / 小红书 / X（Twitter）

- 这类站点普遍要求登录或动态渲染，`web_fetch` 命中率低。
- 一次失败后直接退回搜索摘要，并说明"未读到原文，仅基于搜索摘要"。
- 用户需要精确内容时，请用户自行复制正文。

## 通用技巧

- 去掉 `utm_*`、`?spm=`、`#fragment` 等追踪参数再 fetch。
- 有 `print` / `amp` / `?plain=1` 形态时优先用，正文更完整。
- 站点返回 403 / 429 时不要连续重试；换来源或告知用户。
