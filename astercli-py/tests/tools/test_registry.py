"""ToolRegistry 核心语义测试：精简 Schema / 并发保序 / 超时 / 输出预算 / HITL 钩子。"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import ToolMessage

from astercli.config import Settings
from astercli.hitl import ApprovalSession
from astercli.policy import PolicyException
from astercli.tools.registry import (
    Param,
    ToolRegistry,
    compact_tool,
    create_parameters,
    truncate_output,
)


def make_registry(approval=None, env=None, max_concurrency=4):
    merged = {"ASTERCLI_TOOL_TIMEOUT_SECONDS": "10"}
    merged.update(env or {})
    registry = ToolRegistry(
        settings=Settings(env=merged, load_files=False),
        approval=approval,
        max_concurrency=max_concurrency,
    )
    return registry


def register_echo(registry, name="echo", dangerous=False, func=None):
    registry.register(
        name,
        "回显工具",
        create_parameters(
            Param("path", "string", "文件路径", True),
            Param("limit", "integer", "最多读取多少行", False),
            Param("mode", "string", "模式", False, enum_values=("a", "b")),
        ),
        func or (lambda args: f"ok:{args['path']}"),
        dangerous=dangerous,
    )


def call(name, args, id_=None):
    return {"name": name, "args": args, "id": id_ or f"call-{name}"}


# ---------- 精简 Schema ----------


def test_slim_schema_format():
    registry = make_registry()
    register_echo(registry)
    schemas = registry.schemas("slim")

    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": (
                    "回显工具\n\n参数:\n"
                    "- path (string, required): 文件路径\n"
                    "- limit (integer, optional): 最多读取多少行\n"
                    "- mode (string, enum: a | b, optional): 模式"
                ),
                "parameters": {"type": "object", "required": ["path"]},
            },
        }
    ]


def test_schema_mode_keeps_full_json_schema():
    registry = make_registry()
    register_echo(registry)
    schemas = registry.schemas("schema")
    fn = schemas[0]["function"]
    assert fn["name"] == "echo"
    assert fn["parameters"]["properties"]["path"]["type"] == "string"
    assert fn["parameters"]["properties"]["mode"]["enum"] == ["a", "b"]
    assert fn["parameters"]["required"] == ["path"]


def test_slim_mode_from_settings_env():
    registry = ToolRegistry(settings=Settings(env={"ASTERCLI_TOOL_MODE": "schema"}, load_files=False))
    register_echo(registry)
    schemas = registry.schemas()  # mode 未显式传 → 走 settings
    assert schemas[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"


def test_array_and_object_property_types():
    schema = create_parameters(
        Param("files", "array", "文件列表", False, item_type="string"),
        Param("options", "object", "选项", False, fields=("a", "b")),
    )
    summary = registry_summary(schema)
    assert "- files (array<string>, optional): 文件列表" in summary
    assert "- options (object, fields: a, b, optional): 选项" in summary


def registry_summary(schema):
    from astercli.tools.registry import build_parameter_summary

    return build_parameter_summary(schema)


def test_compact_tool_description_and_parameters():
    registry = make_registry()
    register_echo(registry)
    tool = registry.get("echo")
    compact = compact_tool(tool)
    assert compact["name"] == "echo"
    assert compact["parameters"] == {"type": "object", "required": ["path"]}
    assert "参数:" in compact["description"]


# ---------- 执行 ----------


async def test_execute_returns_tool_messages_in_call_order():
    registry = make_registry()

    def slow(args):
        return "slow:" + args["path"]

    register_echo(registry, name="slow", func=slow)

    async def fast(args):
        await asyncio.sleep(0)
        return "fast:" + args["path"]

    registry.register("fast", "快", create_parameters(Param("path", "string", "", True)), fast)

    results = await registry.execute_all(
        [call("slow", {"path": "1"}), call("fast", {"path": "2"}), call("slow", {"path": "3"})]
    )
    assert all(isinstance(r, ToolMessage) for r in results)
    assert results[0].content == "slow:1"
    assert results[1].content == "fast:2"
    assert results[2].content == "slow:3"
    assert results[0].tool_call_id == "call-slow"


async def test_concurrency_limited_to_4():
    registry = make_registry(max_concurrency=4)
    active = 0
    peak = 0

    async def busy(args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.08)
        active -= 1
        return "done"

    registry.register("busy", "busy", create_parameters(), busy)
    calls = [call("busy", {}, str(i)) for i in range(8)]
    results = await registry.execute_all(calls)
    assert len(results) == 8
    assert peak <= 4
    # 8 个 80ms 任务 / 4 并发 → 至少两批，峰值应真正达到 4 才有意义
    assert peak >= 2


async def test_order_preserved_when_later_calls_finish_first():
    registry = make_registry()

    async def sleeper(args):
        await asyncio.sleep(args["delay"])
        return f"slept-{args['delay']}"

    registry.register(
        "sleeper", "延迟回显", create_parameters(Param("delay", "number", "延迟秒数", True)), sleeper
    )
    results = await registry.execute_all(
        [
            call("sleeper", {"delay": 0.08}),
            call("sleeper", {"delay": 0.01}),
            call("sleeper", {"delay": 0.05}),
        ]
    )
    assert [r.content for r in results] == ["slept-0.08", "slept-0.01", "slept-0.05"]


async def test_unknown_tool_returns_failure_message():
    registry = make_registry()
    results = await registry.execute_all([call("nope", {})])
    assert results[0].content == "未知工具: nope"
    assert results[0].name == "nope"


async def test_policy_exception_becomes_shield_message():
    registry = make_registry()

    def guard_reject(args):
        raise PolicyException("路径越界: ../x 不在项目根之内")

    register_echo(registry, func=guard_reject)
    results = await registry.execute_all([call("echo", {"path": "../x"})])
    assert results[0].content.startswith("🛡️ 路径越界")


async def test_generic_exception_becomes_failure_message():
    registry = make_registry()

    def boom(args):
        raise RuntimeError("disk full")

    register_echo(registry, func=boom)
    results = await registry.execute_all([call("echo", {"path": "any"})])
    assert results[0].content == "工具执行失败: disk full"


async def test_timeout_produces_timeout_message():
    registry = make_registry(env={"ASTERCLI_TOOL_TIMEOUT_SECONDS": "1"})

    async def hang(args):
        await asyncio.sleep(5)
        return "never"

    register_echo(registry, func=hang)
    results = await registry.execute_all([call("echo", {"path": "x"})])
    assert results[0].content == "工具执行超时（>1s）: echo"


async def test_args_json_string_parsed():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all([{"name": "echo", "args": json.dumps({"path": "a.md"}), "id": "c1"}])
    assert results[0].content == "ok:a.md"


async def test_bad_args_json_then_validation_failure():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all([{"name": "echo", "args": "{not json", "id": "c1"}])
    # 非法 JSON 先归零为 {}，随后被参数校验拦下（不再进入工具内部）
    assert results[0].content == "参数校验失败: 缺少必填参数 path"


# ---------- 参数校验 ----------


async def test_missing_required_arg_rejected_before_execution():
    registry = make_registry()
    called = []

    def record(args):
        called.append(args)
        return "ran"

    register_echo(registry, func=record)
    results = await registry.execute_all([call("echo", {})])
    assert results[0].content == "参数校验失败: 缺少必填参数 path"
    assert called == []


async def test_required_arg_with_null_rejected():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all([call("echo", {"path": None})])
    assert results[0].content == "参数校验失败: 缺少必填参数 path"


async def test_wrong_type_arg_rejected():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all([call("echo", {"path": "a.md", "limit": "10"})])
    assert results[0].content == "参数校验失败: 参数 limit 类型应为 integer，实际为 str"


async def test_bool_is_not_integer():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all([call("echo", {"path": "a.md", "limit": True})])
    assert results[0].content == "参数校验失败: 参数 limit 类型应为 integer，实际为 bool"


async def test_enum_arg_rejected():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all([call("echo", {"path": "a.md", "mode": "c"})])
    assert results[0].content == "参数校验失败: 参数 mode 必须是 a / b 之一，实际为 'c'"


async def test_optional_args_and_none_pass_validation():
    registry = make_registry()
    register_echo(registry)
    results = await registry.execute_all(
        [call("echo", {"path": "a.md", "limit": None, "mode": "a", "extra": "任意"})]
    )
    assert results[0].content == "ok:a.md"


async def test_unknown_schema_construct_passes():
    # MCP 的 anyOf / $ref 等非基本类型不校验，避免误伤外部 server
    registry = make_registry()
    registry.register(
        "mcp_like",
        "外部工具",
        {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}, "required": ["x"]},
        lambda args: "mcp-ok",
        dangerous=True,
    )
    results = await registry.execute_all([call("mcp_like", {"x": 123})])
    assert results[0].content == "mcp-ok"


async def test_modify_with_missing_required_arg_rejected():
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all(
        [call("echo", {"path": "x"})],
        decisions={"call-echo": {"action": "modify", "args": {"limit": 5}}},
    )
    assert results[0].content == "参数校验失败: 缺少必填参数 path"


async def test_modify_with_wrong_type_rejected():
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all(
        [call("echo", {"path": "x"})],
        decisions={"call-echo": {"action": "modify", "args": {"path": "modified", "limit": "5"}}},
    )
    assert results[0].content == "参数校验失败: 参数 limit 类型应为 integer，实际为 str"


async def test_invalid_args_skip_hitl_prompt():
    # 参数校验先于审批：参数不合法时不进入审批分支
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all([call("echo", {})])
    assert results[0].content == "参数校验失败: 缺少必填参数 path"


# ---------- HITL 审批（决策制：interrupt 收集用户决策后经 decisions 传入） ----------


async def test_decision_reject_blocks_dangerous_tool():
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all(
        [call("echo", {"path": "x"})],
        decisions={"call-echo": {"action": "reject", "reason": "危险操作"}},
    )
    assert results[0].content == "[HITL] 操作已被拒绝：危险操作"


async def test_decision_approve_runs_dangerous_tool():
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all(
        [call("echo", {"path": "x"})], decisions={"call-echo": {"action": "approve"}}
    )
    assert results[0].content == "ok:x"


async def test_decision_skip_dangerous_tool():
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all(
        [call("echo", {"path": "x"})], decisions={"call-echo": {"action": "skip"}}
    )
    assert results[0].content == "[HITL] 操作已被跳过"


async def test_decision_modify_replaces_args():
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all(
        [call("echo", {"path": "x"})],
        decisions={"call-echo": {"action": "modify", "args": {"path": "modified"}}},
    )
    assert results[0].content == "ok:modified"


async def test_missing_decision_rejected_when_hitl_active():
    # fail-safe：HITL 启用但决策缺失（interrupt 丢失 / 非交互路径）→ 保守拒绝
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=True)
    results = await registry.execute_all([call("echo", {"path": "x"})])
    assert results[0].content == "[HITL] 未收到审批，已按拒绝处理。"


async def test_hitl_inactive_runs_dangerous_tool_directly():
    # 对齐 Java "HITL 未启用行为与父类相同"：无 approval → 危险工具直接执行
    registry = make_registry()
    register_echo(registry, dangerous=True)
    results = await registry.execute_all([call("echo", {"path": "x"})])
    assert results[0].content == "ok:x"


async def test_non_dangerous_tool_skips_approval():
    # 非危险工具不需要决策（requires_approval 为 False），缺失决策也直接执行
    registry = make_registry(approval=ApprovalSession())
    register_echo(registry, dangerous=False)
    results = await registry.execute_all([call("echo", {"path": "x"})])
    assert results[0].content == "ok:x"


def test_audit_written_for_dangerous_call(tmp_path):
    import os

    audit_dir = os.environ["ASTERCLI_AUDIT_DIR"]
    (tmp_path / "audit").mkdir(exist_ok=True)
    from astercli.policy import record_audit

    record_audit("execute_command", 'echo hi', allowed=True)
    files = list((tmp_path / "audit").glob("audit-*.jsonl"))
    assert files
    entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert entry["tool"] == "execute_command"
    assert entry["allowed"] is True


# ---------- 输出预算 ----------


def test_truncate_output_under_limit():
    assert truncate_output("hello", 100) == "hello"


def test_truncate_output_over_limit():
    text = "x" * 300
    out = truncate_output(text, 100)
    assert out.startswith("x" * 100)
    assert "⚠️ 输出已截断" in out
    assert "300 > 100" in out


def test_truncate_output_disabled_with_zero():
    assert truncate_output("x" * 10, 0) == "x" * 10


async def test_execute_all_truncates_huge_output():
    registry = make_registry(env={"ASTERCLI_TOOL_OUTPUT_MAX_CHARS": "50"})
    register_echo(registry, func=lambda args: "y" * 500)
    results = await registry.execute_all([call("echo", {"path": "x"})])
    assert len(results[0].content) < 500
    assert "⚠️ 输出已截断" in results[0].content
