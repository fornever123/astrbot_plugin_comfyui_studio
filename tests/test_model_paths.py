"""模型目录解析回归测试：extra_model_paths.yaml、手动覆盖、优先级与来源标注。

覆盖两条独立链路：
  1) paths.py 里的纯函数（YAML 子集解析、目录解析优先级、去重、别名兜底）；
  2) main.py 中把这两者接进插件的方法（_model_dirs / _path_sources /
     _describe_model_dirs / _model_folder_path）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


# ------------------------------------------------------------------ 测试替身

class _Stub:
    """只带 _get 的轻量替身，用来直接调用插件里的目录解析方法。"""

    def __init__(self, config: dict):
        self._config = config

    def _get(self, key: str, default=None):
        return self._config.get(key, default)


def _bind(stub: _Stub) -> _Stub:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    for name in (
        "_extra_model_paths_sections",
        "_model_dir_overrides",
        "_model_dirs",
        "_path_sources",
        "_describe_model_dirs",
        "_model_folder_path",
    ):
        setattr(stub, name, getattr(ComfyUIAIStudio, name).__get__(stub, _Stub))
    return stub


def _fake_comfyui(root: Path) -> Path:
    """造一个能被 looks_like_comfyui 认可的最小 ComfyUI 目录。"""
    (root / "models" / "loras").mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# stub\n", encoding="utf-8")
    return root


def _disable_root_detection(monkeypatch) -> None:
    """关掉自动探测，避免测试受本机真实 ComfyUI 目录影响。"""
    from astrbot_plugin_comfyui_ai_studio import main as plugin_main

    monkeypatch.setattr(plugin_main, "detect_comfyui_root", lambda explicit="": "")


# ------------------------------------------------------- parse_extra_model_paths

def test_parse_extra_model_paths_basic_sections():
    from astrbot_plugin_comfyui_ai_studio.paths import parse_extra_model_paths

    text = """
# 顶部注释
comfyui:
    base_path: D:/models
    loras: loras
    checkpoints: checkpoints
    vae: D:/other/vae

another_ui:
    base_path: E:/shared
    loras: extra_loras
"""
    sections = parse_extra_model_paths(text)

    assert set(sections) == {"comfyui", "another_ui"}
    assert sections["comfyui"]["base_path"] == ["D:/models"]
    assert sections["comfyui"]["loras"] == ["loras"]
    assert sections["comfyui"]["vae"] == ["D:/other/vae"]
    assert sections["another_ui"]["base_path"] == ["E:/shared"]


def test_parse_extra_model_paths_block_scalar():
    from astrbot_plugin_comfyui_ai_studio.paths import parse_extra_model_paths

    text = (
        "comfyui:\n"
        "    base_path: D:/models\n"
        "    loras: |\n"
        "        loras\n"
        "        shared/loras\n"
        "    vae: vae\n"
    )
    sections = parse_extra_model_paths(text)

    assert sections["comfyui"]["loras"] == ["loras", "shared/loras"]
    assert sections["comfyui"]["vae"] == ["vae"]


def test_parse_extra_model_paths_handles_quotes_and_inline_comments():
    from astrbot_plugin_comfyui_ai_studio.paths import parse_extra_model_paths

    text = (
        "comfyui:\n"
        "    base_path: 'D:/my models'   # 路径带空格\n"
        '    loras: "a/b#c"            # 引号里的 # 不是注释\n'
        "    vae: vae   # 尾部注释\n"
    )
    sections = parse_extra_model_paths(text)

    assert sections["comfyui"]["base_path"] == ["D:/my models"]
    assert sections["comfyui"]["loras"] == ["a/b#c"]
    assert sections["comfyui"]["vae"] == ["vae"]


def test_parse_extra_model_paths_skips_empty_sections_and_empty_values():
    from astrbot_plugin_comfyui_ai_studio.paths import parse_extra_model_paths

    text = "empty_section:\n\ncomfyui:\n    loras:\n    vae: vae\n"
    sections = parse_extra_model_paths(text)

    assert "empty_section" not in sections
    assert sections["comfyui"] == {"vae": ["vae"]}


def test_parse_extra_model_paths_tolerates_garbage():
    from astrbot_plugin_comfyui_ai_studio.paths import parse_extra_model_paths

    assert parse_extra_model_paths("") == {}
    assert parse_extra_model_paths("just some text\nno colons here") == {}
    # 段落里出现没有冒号的行也不应崩溃
    assert parse_extra_model_paths("comfyui:\n    loras: loras\n    坏行\n") == {
        "comfyui": {"loras": ["loras"]}
    }


# ------------------------------------------------------ 目录查找与读取

def test_find_extra_model_paths_file_prefers_root_then_parent(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import find_extra_model_paths_file

    root = _fake_comfyui(tmp_path / "ComfyUI")
    # 根目录没有 -> 找不到
    assert find_extra_model_paths_file(str(root)) == ""

    # 上一层有 -> 命中
    parent_yaml = tmp_path / "extra_model_paths.yaml"
    parent_yaml.write_text("comfyui:\n    loras: loras\n", encoding="utf-8")
    assert find_extra_model_paths_file(str(root)) == str(parent_yaml)

    # 根目录也有 -> 优先根目录
    root_yaml = root / "extra_model_paths.yaml"
    root_yaml.write_text("comfyui:\n    loras: loras\n", encoding="utf-8")
    assert find_extra_model_paths_file(str(root)) == str(root_yaml)


def test_find_extra_model_paths_file_accepts_yml_suffix(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import find_extra_model_paths_file

    root = _fake_comfyui(tmp_path / "ComfyUI")
    yml = tmp_path / "extra_model_paths.yml"
    yml.write_text("comfyui:\n    loras: loras\n", encoding="utf-8")
    assert find_extra_model_paths_file(str(root)) == str(yml)


def test_read_extra_model_paths_missing_file_returns_empty(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import read_extra_model_paths

    assert read_extra_model_paths("") == {}
    assert read_extra_model_paths(str(tmp_path / "不存在的.yaml")) == {}


# --------------------------------------------------------- resolve_model_dirs

def test_resolve_model_dirs_priority_override_extra_default(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    sections = {"comfyui": {"base_path": ["D:/models"], "loras": ["shared_loras"]}}

    entries = resolve_model_dirs(
        root, "loras", override="E:/only/here", sections=sections
    )

    assert [e.source for e in entries] == ["override", "extra", "default"]
    assert entries[0].path == Path("E:/only/here")
    assert entries[1].path == Path("D:/models/shared_loras")
    assert entries[2].path == Path(root) / "models" / "loras"


def test_resolve_model_dirs_without_extra_or_override(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    entries = resolve_model_dirs(root, "vae")

    assert len(entries) == 1
    assert entries[0].source == "default"
    assert entries[0].path == Path(root) / "models" / "vae"


def test_resolve_model_dirs_keeps_extra_paths_absolute(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    sections = {"x": {"base_path": ["D:/base"], "loras": ["E:/absolute/loras"]}}

    entries = resolve_model_dirs(root, "loras", sections=sections)

    # 绝对路径不拼 base_path
    assert entries[0].path == Path("E:/absolute/loras")


def test_resolve_model_dirs_without_base_path_keeps_value_as_is(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    sections = {"x": {"loras": ["only_loras"]}}

    entries = resolve_model_dirs(root, "loras", sections=sections)

    assert entries[0].path == Path("only_loras")


def test_resolve_model_dirs_dedupes_same_path(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    sections = {
        "a": {"base_path": ["D:/models"], "loras": ["loras"]},
        "b": {"base_path": ["D:/models"], "loras": ["loras"]},
    }

    # 不给 root，排除默认位置干扰，只看 extra 之间的去重
    entries = resolve_model_dirs("", "loras", sections=sections)

    assert len(entries) == 1
    assert entries[0].source == "extra"
    assert entries[0].path == Path("D:/models/loras")


def test_resolve_model_dirs_override_equal_to_default_is_not_duplicated(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    default = str(Path(root) / "models" / "loras")

    entries = resolve_model_dirs(root, "loras", override=default)

    # 手动覆盖与默认位置指向同一目录时只保留第一个（override）
    assert len(entries) == 1
    assert entries[0].source == "override"


def test_resolve_model_dirs_alias_fallback(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    sections = {"x": {"base_path": ["D:/m"], "unet": ["unet_dir"], "clip": ["clip_dir"]}}

    diffusion = resolve_model_dirs(root, "diffusion_models", sections=sections)
    text_enc = resolve_model_dirs(root, "text_encoders", sections=sections)

    # diffusion_models 缺省时用 unet；text_encoders 缺省时用 clip（旧名）
    assert diffusion[0].path == Path("D:/m/unet_dir")
    assert diffusion[0].source == "extra"
    assert text_enc[0].path == Path("D:/m/clip_dir")
    assert text_enc[0].source == "extra"


def test_resolve_model_dirs_ignores_non_dict_sections(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    entries = resolve_model_dirs(root, "loras", sections={"bad": "not-a-dict"})

    assert [e.source for e in entries] == ["default"]


def test_resolve_model_dirs_empty_everything():
    from astrbot_plugin_comfyui_ai_studio.paths import resolve_model_dirs

    assert resolve_model_dirs("", "loras") == []


def test_primary_model_dir_returns_first_entry(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import primary_model_dir

    root = str(_fake_comfyui(tmp_path / "ComfyUI"))
    picked = primary_model_dir(root, "loras", override="E:/o")

    assert picked is not None
    assert picked.path == Path("E:/o")
    assert picked.source == "override"
    assert primary_model_dir("", "loras") is None


# ------------------------------------------------------------ ModelDir 与别名

def test_model_dir_label_and_is_custom(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import SOURCE_LABELS, ModelDir

    for source, label in SOURCE_LABELS.items():
        entry = ModelDir(path=tmp_path, source=source)
        assert entry.label == label
        assert entry.is_custom == (source != "default")


def test_model_dir_unknown_source_falls_back_to_raw_value(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import ModelDir

    entry = ModelDir(path=tmp_path, source="mystery")
    assert entry.label == "mystery"
    assert entry.is_custom is True


def test_category_aliases_cover_old_and_new_folder_names():
    from astrbot_plugin_comfyui_ai_studio.paths import category_aliases

    assert category_aliases("diffusion_models") == ("diffusion_models", "unet")
    assert category_aliases("unet") == ("unet", "diffusion_models")
    assert category_aliases("text_encoders") == ("text_encoders", "clip")
    assert category_aliases("loras") == ("loras",)
    assert category_aliases("") == ()


def test_model_categories_use_new_names_and_exclude_legacy_clip():
    from astrbot_plugin_comfyui_ai_studio.paths import MODEL_CATEGORIES

    assert "clip" not in MODEL_CATEGORIES
    assert "text_encoders" in MODEL_CATEGORIES
    assert "unet" in MODEL_CATEGORIES
    # 不要出现重复项
    assert len(MODEL_CATEGORIES) == len(set(MODEL_CATEGORIES))


def test_detect_comfyui_root_accepts_explicit_valid_path(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import detect_comfyui_root

    root = _fake_comfyui(tmp_path / "ComfyUI")
    assert detect_comfyui_root(str(root)) == str(root)
    # 传上一层目录也能探测到
    assert detect_comfyui_root(str(tmp_path)) == str(root)
    # 无效路径不应误判为 ComfyUI
    assert detect_comfyui_root(str(tmp_path / "nope")) != str(tmp_path)


# ------------------------------------------------------- 插件方法接线

def _plugin(tmp_path, yaml_body: str | None = None, **extra):
    root = _fake_comfyui(tmp_path / "ComfyUI")
    config = {"comfyui_root": str(root), **extra}
    if yaml_body is not None:
        yaml_path = tmp_path / "extra_model_paths.yaml"
        yaml_path.write_text(yaml_body, encoding="utf-8")
        config["extra_model_paths_file"] = str(yaml_path)
    return _bind(_Stub(config)), root


def test_plugin_model_dirs_uses_extra_model_paths(tmp_path):
    stub, root = _plugin(
        tmp_path,
        "comfyui:\n    base_path: D:/models\n    loras: shared_loras\n",
    )

    entries = stub._model_dirs("loras")

    assert [e.source for e in entries] == ["extra", "default"]
    assert entries[0].path == Path("D:/models/shared_loras")
    assert entries[1].path == root / "models" / "loras"


def test_plugin_model_dirs_override_beats_extra_and_default(tmp_path):
    stub, _ = _plugin(
        tmp_path,
        "comfyui:\n    base_path: D:/models\n    loras: shared_loras\n",
        model_dir_overrides={"loras": "F:/my_loras"},
    )

    entries = stub._model_dirs("loras")

    assert entries[0].source == "override"
    assert entries[0].path == Path("F:/my_loras")


def test_plugin_model_dir_overrides_accepts_json_string(tmp_path):
    stub, _ = _plugin(
        tmp_path,
        None,
        model_dir_overrides='{"loras": "G:/json_loras", "vae": "G:/json_vae"}',
    )

    assert stub._model_dirs("loras")[0].path == Path("G:/json_loras")
    assert stub._model_dirs("vae")[0].path == Path("G:/json_vae")


def test_plugin_model_dir_overrides_ignores_broken_payload(tmp_path):
    stub, root = _plugin(tmp_path, None, model_dir_overrides="{不是合法 JSON")

    entries = stub._model_dirs("loras")

    # 坏配置不能把解析搞崩，退回默认位置
    assert [e.source for e in entries] == ["default"]
    assert entries[0].path == root / "models" / "loras"


def test_plugin_model_dir_overrides_ignores_blank_entries(tmp_path):
    stub, root = _plugin(
        tmp_path, None, model_dir_overrides={"loras": "   ", "vae": None}
    )

    assert stub._model_dirs("loras")[0].path == root / "models" / "loras"
    assert stub._model_dirs("vae")[0].path == root / "models" / "vae"


def test_plugin_extra_model_paths_sections_are_cached(tmp_path):
    stub, _ = _plugin(tmp_path, "comfyui:\n    loras: shared_loras\n")

    first = stub._extra_model_paths_sections()
    cached = stub._extra_model_paths_cache
    second = stub._extra_model_paths_sections()

    # 同一份文件命中缓存：内容一致且缓存条目未被替换
    assert first == second
    assert cached is stub._extra_model_paths_cache
    assert first[1]["comfyui"]["loras"] == ["shared_loras"]


def test_plugin_path_sources_marks_non_default(tmp_path):
    stub, _ = _plugin(
        tmp_path,
        "comfyui:\n    base_path: D:/models\n    loras: shared_loras\n",
        model_dir_overrides={"vae": "F:/my_vae"},
    )
    from astrbot_plugin_comfyui_ai_studio.paths import MODEL_CATEGORIES

    sources = stub._path_sources()

    assert sources["loras"] == "extra"
    assert sources["vae"] == "override"
    assert sources["checkpoints"] == "default"
    assert set(sources) == set(MODEL_CATEGORIES)


def test_plugin_describe_model_dirs_lists_every_directory(tmp_path):
    stub, root = _plugin(
        tmp_path,
        "comfyui:\n    base_path: D:/models\n    loras: shared_loras\n",
    )

    lines = stub._describe_model_dirs("loras", "LoRA 文件夹")

    assert len(lines) == 2
    # 用 Path 归一化，避免 Windows 反斜杠导致的字面量比较失败
    assert lines[0] == f"LoRA 文件夹：{Path('D:/models/shared_loras')}（额外路径）"
    assert str(root / "models" / "loras") in lines[1]
    assert "默认位置" in lines[1]


def test_plugin_describe_model_dirs_reports_missing(tmp_path, monkeypatch):
    _disable_root_detection(monkeypatch)
    stub = _bind(_Stub({"comfyui_root": ""}))
    stub._extra_model_paths_cache = (None, None, {})

    assert stub._describe_model_dirs("loras", "LoRA 文件夹") == ["LoRA 文件夹：未检测到"]


def test_plugin_model_folder_path_returns_resolved_dir(tmp_path):
    stub, root = _plugin(
        tmp_path,
        "comfyui:\n    base_path: D:/models\n    loras: shared_loras\n",
    )

    assert stub._model_folder_path("loras") == Path("D:/models/shared_loras")
    assert stub._model_folder_path("checkpoints") == root / "models" / "checkpoints"


def test_plugin_model_folder_path_rejects_unknown_kind(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.main import UsageError

    stub, _ = _plugin(tmp_path)

    with pytest.raises(UsageError):
        stub._model_folder_path("不存在的类别")


def test_plugin_model_folder_path_requires_detected_root(tmp_path, monkeypatch):
    from astrbot_plugin_comfyui_ai_studio.main import UsageError

    _disable_root_detection(monkeypatch)
    stub = _bind(_Stub({"comfyui_root": ""}))
    stub._extra_model_paths_cache = (None, None, {})

    with pytest.raises(UsageError):
        stub._model_folder_path("loras")


# ------------------------------------------------- 配置项 / 前端接线回归

def _source(name: str) -> str:
    return (PLUGIN_DIR / name).read_text(encoding="utf-8")


def test_schema_exposes_model_path_config():
    import json

    schema = json.loads(_source("_conf_schema.json"))

    assert "extra_model_paths_file" in schema
    assert schema["extra_model_paths_file"]["type"] == "string"
    assert "model_dir_overrides" in schema
    assert "comfyui_root" in schema
    # 默认必须留空，不能写死作者本机路径
    assert schema["extra_model_paths_file"].get("default", "") == ""


def test_writable_config_contains_model_path_keys():
    from astrbot_plugin_comfyui_ai_studio.main import WRITABLE_CONFIG

    assert "extra_model_paths_file" in WRITABLE_CONFIG
    assert "model_dir_overrides" in WRITABLE_CONFIG


def test_api_status_sends_path_sources():
    source = _source("main.py")

    assert '"path_sources": self._path_sources()' in source
    assert "model_paths = {" in source


def test_frontend_renders_path_source_badge():
    js = _source("app.js")
    css = _source("style.css")

    assert "status.path_sources" in js
    assert "class=\"path-source\"" in js
    assert "手动指定" in js and "额外路径" in js
    assert ".path-source" in css


def test_ui_mentions_model_dir_resolution_order():
    for name in ("index.html", "pages/console/index.html"):
        html = _source(name)
        assert html.count("模型目录来源自动识别") == 2
