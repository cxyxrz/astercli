# 常用搜索式样

本文件是 `codebase-onboarding` 的附属细则，按需读取。示例以 ripgrep 语法为准（`grep_code` 优先使用本机 `rg`）。

## 找入口

```
def main\(
if __name__ == .__main__.
console_scripts
public static void main
@(app|router)\.(route|get|post)
@click\.command|argparse\.ArgumentParser
```

## 找分层边界

```
^class .*(Service|Repository|Controller|Handler|Adapter)
^def (handle|dispatch|route|execute)_
^(from|import) \.
```

## 找配置与开关

```
os\.(environ|getenv)\(|getenv\(
settings\.|config\.|Settings\(
feature_flag|ENABLED|_ENABLED
```

## 找数据流

```
\.(query|filter|execute|commit)\(
requests\.(get|post)|httpx\.|aiohttp
AsyncClient\(|Session\(
```

## 找风险点

```
TODO|FIXME|HACK|XXX
except Exception|catch \(Exception
# type: ignore|@ts-ignore|noqa
time\.sleep|setTimeout\(
```

## 使用建议

- 一次只搜一个语义，多个关键词用 `|` 组合会拉低信噪比。
- 先搜定义（`^(class|def|function|const)`），再搜调用（裸标识符）。
- 结果超过 `head_limit` 时加目录前缀限定，例如 `path=src/astercli/agent`。
- 排除产物目录：`--glob '!**/node_modules/**'`、`--glob '!**/dist/**'`（`grep_code` 支持 glob 参数时使用）。
