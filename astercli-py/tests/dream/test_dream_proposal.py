"""Dream 提案测试：ops 校验 / 脱敏 / 长度上限 / 提案落盘。"""

from __future__ import annotations

from astercli.dream import Proposal, normalize_ops, parse_proposal_output
from astercli.dream.proposal import MAX_FACT_CHARS, MAX_OPS


MEMORY_IDS = {"fact-aaaa1111", "fact-bbbb2222"}
EXPERIENCE_IDS = {"exp-cccc3333"}


def _normalize(ops, project_key=""):
    return normalize_ops(
        ops,
        memory_ids=MEMORY_IDS,
        experience_ids=EXPERIENCE_IDS,
        project_key=project_key,
    )


def test_六类合法op全部保留():
    ops, errors = _normalize(
        [
            {"type": "memory_add", "scope": "project", "content": "项目使用 Maven"},
            {"type": "memory_update", "id": "fact-aaaa1111", "content": "更新后的事实"},
            {"type": "memory_delete", "id": "fact-bbbb2222"},
            {
                "type": "memory_merge",
                "ids": ["fact-aaaa1111", "fact-bbbb2222"],
                "content": "合并后的事实",
            },
            {
                "type": "experience_add",
                "category": "workflow",
                "lesson": "写文件前先检查目录",
                "correct_action": "先 glob_files",
            },
            {"type": "experience_delete", "id": "exp-cccc3333", "reason": "已过时"},
        ]
    )
    assert errors == []
    assert len(ops) == 6
    assert ops[0]["scope"] == "project"
    assert ops[3]["ids"] == ["fact-aaaa1111", "fact-bbbb2222"]


def test_未知类型与非法id被丢弃():
    ops, errors = _normalize(
        [
            {"type": "drop_database"},
            {"type": "memory_update", "id": "fact-nope", "content": "x"},
            {"type": "experience_delete", "id": "exp-nope"},
        ]
    )
    assert ops == []
    assert len(errors) == 3


def test_空内容与重复新增被丢弃():
    ops, errors = _normalize(
        [
            {"type": "memory_add", "content": "  "},
            {"type": "memory_add", "content": "同一条"},
            {"type": "memory_add", "content": "同一条"},
        ]
    )
    assert len(ops) == 1
    assert len(errors) == 2


def test_超长内容被截断():
    ops, _ = _normalize([{"type": "memory_add", "content": "字" * 900}])
    assert len(ops[0]["content"]) <= MAX_FACT_CHARS


def test_合并需要至少两个有效id():
    ops, errors = _normalize(
        [{"type": "memory_merge", "ids": ["fact-aaaa1111", "fact-missing"], "content": "x"}]
    )
    assert ops == []
    assert "至少 2 个" in errors[0]


def test_单次op上限():
    raw = [{"type": "memory_add", "content": f"事实 {i}"} for i in range(MAX_OPS + 5)]
    ops, errors = _normalize(raw)
    assert len(ops) == MAX_OPS
    assert any("上限" in e for e in errors)


def test_项目路径被脱敏():
    ops, _ = _normalize(
        [{"type": "memory_add", "content": "根目录 /Users/hrs/demo 使用 uv"}],
        project_key="/Users/hrs/demo",
    )
    assert "<project>" in ops[0]["content"]
    assert "/Users/hrs/demo" not in ops[0]["content"]


def test_解析带围栏的JSON():
    text = '```json\n{"summary": "整理", "ops": []}\n```'
    data = parse_proposal_output(text)
    assert data is not None and data["summary"] == "整理"
    assert parse_proposal_output("不是 JSON") is None
    assert parse_proposal_output("") is None


def test_提案落盘与回读(tmp_path):
    proposal = Proposal(
        dream_id="dream-1",
        summary="合并两条重复记忆",
        ops=[{"type": "memory_add", "scope": "project", "content": "项目使用 Maven"}],
        model="glm-test",
    )
    assert proposal.save(tmp_path / "dream-1") is True
    assert (tmp_path / "dream-1" / "proposal.md").is_file()

    loaded = Proposal.load(tmp_path / "dream-1")
    assert loaded is not None
    assert loaded.summary == "合并两条重复记忆"
    assert loaded.ops[0]["content"] == "项目使用 Maven"
    assert "合并两条重复记忆" in (tmp_path / "dream-1" / "proposal.md").read_text(encoding="utf-8")
