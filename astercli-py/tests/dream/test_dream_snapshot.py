"""Dream 快照测试：备份 / 还原 / 删除新文件 / manifest 读取。"""

from __future__ import annotations

from astercli.dream import Snapshot, take_snapshot


def test_快照备份并还原已存在文件(tmp_path):
    memory = tmp_path / "memory" / "long_term_memory.json"
    memory.parent.mkdir()
    memory.write_text("原始内容", encoding="utf-8")

    snapshot = take_snapshot("dream-1", [memory], tmp_path / "backups")
    assert (snapshot.root / "manifest.json").is_file()
    assert snapshot.entries[0].existed is True

    memory.write_text("被 dream 改写", encoding="utf-8")
    assert snapshot.restore() == 1
    assert memory.read_text(encoding="utf-8") == "原始内容"


def test_快照记录不存在文件并还原时删除(tmp_path):
    new_file = tmp_path / "project" / "exp-new.md"
    snapshot = take_snapshot("dream-2", [new_file], tmp_path / "backups")
    assert snapshot.entries[0].existed is False

    new_file.parent.mkdir()
    new_file.write_text("dream 新建的经验", encoding="utf-8")
    assert snapshot.restore() == 1
    assert not new_file.exists()


def test_重复目标只备份一次(tmp_path):
    target = tmp_path / "a.json"
    target.write_text("x", encoding="utf-8")
    snapshot = take_snapshot("dream-3", [target, target], tmp_path / "backups")
    assert len(snapshot.entries) == 1


def test_从manifest重新加载快照(tmp_path):
    target = tmp_path / "a.json"
    target.write_text("v1", encoding="utf-8")
    take_snapshot("dream-4", [target], tmp_path / "backups")

    loaded = Snapshot.load(tmp_path / "backups" / "dream-4", "dream-4")
    assert loaded is not None
    assert loaded.dream_id == "dream-4"
    assert len(loaded.entries) == 1

    target.write_text("v2", encoding="utf-8")
    assert loaded.restore() == 1
    assert target.read_text(encoding="utf-8") == "v1"


def test_缺失manifest返回None(tmp_path):
    assert Snapshot.load(tmp_path / "nope") is None
