"""经验存储测试：落盘 / 去重 / 检索 / 淘汰 / 索引 / 作用域隔离。"""

from __future__ import annotations

import time
import uuid

from astercli.config import Settings
from astercli.experience import ExperienceEntry, ExperienceStore


def _store(tmp_path, **env) -> ExperienceStore:
    settings = Settings(env=env, load_files=False)
    return ExperienceStore(
        tmp_path,
        settings,
        global_dir=tmp_path / "global",
        project_dir=tmp_path / "project",
    )


def _entry(lesson: str = "写文件前先确认目录存在", **kwargs) -> ExperienceEntry:
    defaults = dict(
        id=f"exp-{uuid.uuid4().hex[:8]}",
        lesson=lesson,
        correct_action="先用 glob_files 检查目录",
        tool="write_file",
    )
    defaults.update(kwargs)
    return ExperienceEntry(**defaults)


def test_保存经验生成md与索引(tmp_path):
    store = _store(tmp_path)
    saved, created = store.save(_entry())
    assert created is True
    assert saved is not None
    path = store.path_of(saved)
    assert path is not None and path.is_file()
    assert (tmp_path / "project" / "index.md").is_file()
    text = path.read_text(encoding="utf-8")
    assert "写文件前先确认目录存在" in text


def test_同内容经验合并为hits加一(tmp_path):
    store = _store(tmp_path)
    first, created = store.save(_entry())
    second, created2 = store.save(_entry())
    assert created is True and created2 is False
    assert second is not None and first is not None
    assert second.id == first.id
    assert second.hits == 2
    files = list((tmp_path / "project").glob("exp-*.md"))
    assert len(files) == 1


def test_不同内容经验各自成文件(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("教训A"))
    store.save(_entry("教训B"))
    assert len(list((tmp_path / "project").glob("exp-*.md"))) == 2


def test_项目级与全局级作用域隔离(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("项目经验"))
    store.save(_entry("全局经验", scope="global"))
    project = store.list_entries(scope="project")
    glob = store.list_entries(scope="global")
    assert len(project) == 1 and len(glob) == 1
    assert project[0].lesson == "项目经验"
    assert glob[0].lesson == "全局经验"
    assert len(store.list_entries()) == 2


def test_关键词检索命中经验(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("修改文件前必须检查路径是否存在", correct_action="先 list_dir"))
    store.save(_entry("执行命令前确认工作目录", correct_action="先 pwd"))
    hits = store.search("路径")
    assert hits
    assert "路径" in hits[0].lesson


def test_检索无关键词返回空(tmp_path):
    store = _store(tmp_path)
    store.save(_entry())
    assert store.search("") == []
    assert store.search("毫不相干的内容") == []


def test_删除单条经验(tmp_path):
    store = _store(tmp_path)
    saved, _ = store.save(_entry())
    assert saved is not None
    assert store.delete(saved.id) is True
    assert store.get(saved.id) is None
    assert not list((tmp_path / "project").glob("exp-*.md"))


def test_清空指定作用域经验(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("项目经验"))
    store.save(_entry("全局经验", scope="global"))
    removed = store.clear("project")
    assert removed == 1
    assert store.list_entries(scope="project") == []
    assert len(store.list_entries(scope="global")) == 1


def test_超出上限淘汰低分经验(tmp_path):
    store = _store(tmp_path, ASTERCLI_EXPERIENCE_MAX_ENTRIES="2")
    now = time.time()
    old = store.save(_entry("旧经验A", created=now - 100))
    store.save(_entry("旧经验B", created=now - 50))
    # 显式创建时间保证同分（hits=1、同一时效档）时按更老优先淘汰
    store.save(_entry("新经验C", created=now))
    remaining = store.list_entries(scope="project")
    assert len(remaining) == 2
    assert old[0].id not in [e.id for e in remaining]


def test_索引文本包含经验并在空库时为空(tmp_path):
    store = _store(tmp_path)
    assert store.index_text() == ""
    store.save(_entry("不要在未确认目录时写文件"))
    text = store.index_text()
    assert "可用经验" in text
    assert "不要在未确认目录时写文件" in text


def test_任务级上下文只注入相关经验(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("写文件前先确认目录存在"))
    store.save(_entry("执行命令前确认网络可用"))
    assert store.context_for_query("写文件") != ""
    assert store.context_for_query("今天天气怎么样") == ""


def test_索引文本按预算截断(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("很长的经验" * 20))
    text = store.index_text(max_entries=20, max_chars=120)
    assert len(text) <= 120
    assert "截断" in text


def test_状态摘要统计项目与全局(tmp_path):
    store = _store(tmp_path)
    store.save(_entry("项目经验"))
    store.save(_entry("全局经验", scope="global"))
    summary = store.status_summary()
    assert "2 条" in summary
    assert "项目 1" in summary
