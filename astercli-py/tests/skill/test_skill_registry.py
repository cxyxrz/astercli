"""SkillRegistry 测试：三层加载 / 同名覆盖 / 禁用持久化 / 解析降级。"""

from __future__ import annotations

from astercli.skill import SkillRegistry, SkillStateStore
from astercli.skill.skill import Source


def write_skill(directory, name, body="正文", **fields):
    import textwrap

    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}"]
    if "description" in fields:
        lines.append(f"description: {fields['description']}")
    if "version" in fields:
        lines.append(f"version: {fields['version']}")
    if "tags" in fields:
        lines.append(f"tags: [{', '.join(fields['tags'])}]")
    lines.extend(["---", ""])
    content = "\n".join(lines) + body
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


class TestLoad:
    def test_空目录_无_skill_无警告(self, tmp_path):
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user",
            project_skills_dir=tmp_path / "project",
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills() == []
        assert registry.warnings() == []

    def test_项目级_加载(self, tmp_path):
        project = tmp_path / "project"
        write_skill(project, "java-style", description="Java 代码风格约定")
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user",
            project_skills_dir=project,
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        skills = registry.all_skills()
        assert len(skills) == 1
        assert skills[0].name == "java-style"
        assert skills[0].description == "Java 代码风格约定"
        assert skills[0].source == Source.PROJECT

    def test_用户级_加载(self, tmp_path):
        user = tmp_path / "user"
        write_skill(user, "git-flow", description="Git 工作流")
        registry = SkillRegistry(
            user_skills_dir=user,
            project_skills_dir=tmp_path / "project",
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills()[0].source == Source.USER

    def test_同名_项目级_覆盖_用户级(self, tmp_path):
        user = tmp_path / "user"
        project = tmp_path / "project"
        write_skill(user, "same-name", description="用户级版本", body="用户级正文")
        write_skill(project, "same-name", description="项目级版本", body="项目级正文")
        registry = SkillRegistry(
            user_skills_dir=user,
            project_skills_dir=project,
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        skills = registry.all_skills()
        assert len(skills) == 1  # 整体覆盖，不合并
        assert skills[0].description == "项目级版本"
        assert skills[0].body == "项目级正文"
        assert skills[0].source == Source.PROJECT

    def test_缺_name_回退目录名(self, tmp_path):
        project = tmp_path / "project"
        skill_dir = project / "dir-name-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: 只有描述没有名字\n---\n正文", encoding="utf-8"
        )
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user",
            project_skills_dir=project,
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills()[0].name == "dir-name-skill"

    def test_子目录缺_skill_md_跳过(self, tmp_path):
        project = tmp_path / "project"
        (project / "not-a-skill").mkdir(parents=True)
        (project / "not-a-skill" / "README.md").write_text("占位", encoding="utf-8")
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user",
            project_skills_dir=project,
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills() == []

    def test_目录不存在_静默跳过(self, tmp_path):
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "不存在",
            project_skills_dir=tmp_path / "也不存在",
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills() == []

    def test_tags_解析(self, tmp_path):
        project = tmp_path / "project"
        write_skill(project, "tagged", description="d", tags=["java", "maven"])
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user",
            project_skills_dir=project,
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills()[0].tags == ["java", "maven"]

    def test_reload_刷新(self, tmp_path):
        project = tmp_path / "project"
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user",
            project_skills_dir=project,
            state_store=SkillStateStore(tmp_path / "state.json"),
        )
        registry.reload()
        assert registry.all_skills() == []
        write_skill(project, "later", description="后来新增")
        registry.reload()
        assert registry.all_skills()[0].name == "later"


class TestDisabled:
    def test_禁用后_find_skill_返回_none(self, tmp_path):
        project = tmp_path / "project"
        write_skill(project, "off-skill", description="将被禁用")
        store = SkillStateStore(tmp_path / "state.json")
        registry = SkillRegistry(
            user_skills_dir=tmp_path / "user", project_skills_dir=project, state_store=store
        )
        registry.reload()
        assert registry.find_skill("off-skill") is not None
        store.set_disabled(["off-skill"])
        assert registry.find_skill("off-skill") is None
        # find_any_skill 仍可见（/skill show 用）
        assert registry.find_any_skill("off-skill") is not None
        # enabled 过滤
        assert registry.enabled_skills() == []

    def test_持久化_跨实例生效(self, tmp_path):
        store_path = tmp_path / "state.json"
        SkillStateStore(store_path).set_disabled(["a", "b"])
        # 另一个实例读取
        assert SkillStateStore(store_path).disabled() == {"a", "b"}

    def test_损坏状态文件_回退空集(self, tmp_path):
        store_path = tmp_path / "state.json"
        store_path.write_text("not-json{", encoding="utf-8")
        assert SkillStateStore(store_path).disabled() == set()
