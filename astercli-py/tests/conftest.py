"""测试环境隔离。

- 清空所有 provider / ASTERCLI_* 环境变量，避免本机 .env 或 shell 注入污染断言
- 审计目录重定向到 tmp_path，避免测试写用户目录
- 提供 project / guard 两个常用 fixture
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_PROVIDER_ENV_PREFIXES = (
    "GLM_",
    "DEEPSEEK_",
    "STEP_",
    "KIMI_",
    "MOONSHOT_",
    "FREELLMAPI_",
    "XFYUN_",
    "AGNES_",
    "ASTERCLI_",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(_PROVIDER_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ASTERCLI_AUDIT_DIR", str(tmp_path / "audit"))


@pytest.fixture
def project(tmp_path):
    """一个最小项目根：src/Main.java + README.md + 子目录。"""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "Main.java").write_text(
        "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"hello\");\n    }\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# demo\n\nhello world\n", encoding="utf-8")
    (tmp_path / "src" / "Util.java").write_text(
        "class Util {\n    static String greet() { return \"hi\"; }\n}\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def guard(project):
    from astercli.policy import PathGuard

    return PathGuard(project)
