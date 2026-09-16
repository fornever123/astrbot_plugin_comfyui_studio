"""「文件夹位置」面板回归测试：注册表、读写接口、检测函数与前端接线。

后端用继承插件类的轻量替身（只替换配置读写与探测），因此不依赖 AstrBot 运行时，
也不受本机真实 ComfyUI 布局影响。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


# ------------------------------------------------------------------ 测试替身

CONFIG_KEYS = (
    "comfyui_root",
    "extra_model_paths_file",
    "workflow_dir",
    "source_workflow",
    "image_font_regular",
    "image_font_bold",
    "anima_template_path",
    "comfyui_start_script",
)


def _plugin(config=None, sections=None, monkeypatch=None, root=None):
    """构造一个只带配置读写的插件替身。"""
    from astrbot_plugin_comfyui_ai_studio import main as plugin_main

    if monkeypatch is not None:
        monkeypatch.setattr(plugin_main, "detect_comfyui_root", lambda explicit="": str(root or ""))

    class _Stub(plugin_main.ComfyUIAIStudio):
        def __init__(self):
            self.cfg = {key: "" for key in CONFIG_KEYS}
            self.cfg["model_dir_overrides"] = {}
            self.cfg.update(config or {})
            self.sections = ("", {}) if sections is None else sections
            self.saves = 0
            self.output_dir = Path("E:/stub/out")
            self.data_dir = Path("E:/stub/data")
            self.plugin_dir = PLUGIN_DIR
            self._extra_model_paths_cache = None

        def _get(self, key, default=None):
            return self.cfg.get(key, default)

        def _set(self, key, value):
            self.cfg[key] = value

        def _save_config(self):
            self.saves += 1

        def _extra_model_paths_sections(self):
            return self.sections

    return _Stub()


def _fake_comfyui(tmp_path: Path) -> Path:
    root = tmp_path / "ComfyUI"
    (root / "models" / "loras").mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# stub\n", encoding="utf-8")
    return root


def _all_entries(payload) -> dict:
    return {entry["id"]: entry for group in payload["groups"] for entry in group["entries"]}


# --------------------------------------------------------------- 注册表结构

def test_registry_covers_every_model_category():
    from astrbot_plugin_comfyui_ai_studio.main import PATH_SETTING_GROUPS
    from astrbot_plugin_comfyui_ai_studio.paths import MODEL_CATEGORIES

    ids = [entry["id"] for group in PATH_SETTING_GROUPS for entry in group["entries"]]
    for category in MODEL_CATEGORIES:
        assert f"model:{category}" in ids


def test_registry_ids_are_unique():
    from astrbot_plugin_comfyui_ai_studio.main import PATH_SETTING_GROUPS

    ids = [entry["id"] for group in PATH_SETTING_GROUPS for entry in group["entries"]]
    assert len(ids) == len(set(ids))


def test_registry_covers_all_user_facing_paths():
    from astrbot_plugin_comfyui_ai_studio.main import PATH_SETTING_GROUPS

    ids = {entry["id"] for group in PATH_SETTING_GROUPS for entry in group["entries"]}
    for expected in (
        "comfyui_root",
        "extra_model_paths_file",
        "workflow_dir",
        "source_workflow",
        "image_font_regular",
        "image_font_bold",
        "anima_template_path",
        "comfyui_start_script",
        "output_dir",
        "data_dir",
    ):
        assert expected in ids, expected


def test_registry_readonly_entries_carry_no_write_target():
    from astrbot_plugin_comfyui_ai_studio.main import PATH_SETTING_GROUPS

    readonly = [e for g in PATH_SETTING_GROUPS for e in g["entries"] if e.get("readonly")]
    assert readonly
    for entry in readonly:
        assert "config_key" not in entry
        assert "category" not in entry


def test_targets_map_ids_to_write_scopes():
    stub = _plugin()

    targets = stub._path_setting_targets()

    assert targets["comfyui_root"] == ("config", "comfyui_root")
    assert targets["model:loras"] == ("model", "loras")
    # 只读项不进入可写集合
    assert "output_dir" not in targets
    assert "data_dir" not in targets


# ------------------------------------------------------------------ 读取

def test_payload_shape(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    payload = stub.path_settings_payload()

    assert payload["ok"] is True
    assert payload["groups"]
    assert all({"group", "note", "entries"} <= set(group) for group in payload["groups"])
    assert payload["writable_ids"]
    for entry in _all_entries(payload).values():
        assert {
            "id", "label", "kind", "readonly", "configured", "resolved",
            "source", "source_label", "exists", "files", "fallbacks", "suffixes",
        } <= set(entry)


def test_payload_model_entry_defaults_to_default_source(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "is_dir", Path.is_dir, raising=False)
    root = _fake_comfyui(tmp_path)
    stub = _plugin(config={"comfyui_root": str(root)}, monkeypatch=monkeypatch, root=root)

    entry = _all_entries(stub.path_settings_payload())["model:loras"]

    assert entry["source"] == "default"
    assert entry["source_label"] == "默认位置"
    assert entry["configured"] == ""
    assert Path(entry["resolved"]) == root / "models" / "loras"


def test_payload_model_entry_reports_override(tmp_path, monkeypatch):
    root = _fake_comfyui(tmp_path)
    stub = _plugin(
        config={"comfyui_root": str(root), "model_dir_overrides": {"loras": "E:/mine"}},
        monkeypatch=monkeypatch,
        root=root,
    )

    entry = _all_entries(stub.path_settings_payload())["model:loras"]

    assert entry["source"] == "override"
    assert entry["configured"] == "E:/mine"
    assert entry["resolved"] == str(Path("E:/mine"))


def test_payload_model_entry_reports_extra_paths(tmp_path, monkeypatch):
    root = _fake_comfyui(tmp_path)
    stub = _plugin(
        config={"comfyui_root": str(root)},
        sections=("D:/extra_model_paths.yaml", {"comfyui": {"base_path": ["D:/models"], "loras": ["shared"]}}),
        monkeypatch=monkeypatch,
        root=root,
    )

    entry = _all_entries(stub.path_settings_payload())["model:loras"]

    assert entry["source"] == "extra"
    assert entry["resolved"] == str(Path("D:/models/shared"))
    # 默认位置作为候选列出来
    assert entry["fallbacks"] == [str(root / "models" / "loras")]


def test_payload_marks_unconfigured_extras_as_builtin_or_missing(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    entries = _all_entries(stub.path_settings_payload())

    assert entries["workflow_dir"]["source"] == "builtin"
    assert entries["output_dir"]["source"] == "runtime"
    assert entries["output_dir"]["readonly"] is True
    assert entries["extra_model_paths_file"]["source"] == "none"
    assert entries["extra_model_paths_file"]["exists"] is False


def test_payload_counts_files_for_model_dirs(tmp_path, monkeypatch):
    root = _fake_comfyui(tmp_path)
    (root / "models" / "loras" / "a.safetensors").write_text("x", encoding="utf-8")
    (root / "models" / "loras" / "note.txt").write_text("x", encoding="utf-8")
    stub = _plugin(config={"comfyui_root": str(root)}, monkeypatch=monkeypatch, root=root)

    entry = _all_entries(stub.path_settings_payload())["model:loras"]

    # 只统计模型后缀，.txt 不计入
    assert entry["exists"] is True
    assert entry["files"] == 1


# ------------------------------------------------------------------ 保存

def test_save_writes_model_override(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    result = stub.save_path_settings({"model:loras": "  D:/mylora  "})

    assert result["ok"] is True
    assert result["changed"] == ["model:loras"]
    assert stub.cfg["model_dir_overrides"] == {"loras": "D:/mylora"}
    assert stub.saves == 1


def test_save_writes_plain_config_keys(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    stub.save_path_settings({"comfyui_root": "E:/1aaaaaai/ComfyUI-aki-v3/ComfyUI", "workflow_dir": "E:/wf"})

    assert stub.cfg["comfyui_root"] == "E:/1aaaaaai/ComfyUI-aki-v3/ComfyUI"
    assert stub.cfg["workflow_dir"] == "E:/wf"
    # 没有改模型目录就不该动 model_dir_overrides
    assert stub.cfg["model_dir_overrides"] == {}


def test_save_empty_value_clears_override(tmp_path, monkeypatch):
    stub = _plugin(config={"model_dir_overrides": {"loras": "E:/mine", "vae": "E:/vae"}}, monkeypatch=monkeypatch)

    stub.save_path_settings({"model:loras": ""})

    assert stub.cfg["model_dir_overrides"] == {"vae": "E:/vae"}


def test_save_strips_quotes_and_whitespace(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    stub.save_path_settings({"model:vae": '  "D:/my vae"  '})

    assert stub.cfg["model_dir_overrides"] == {"vae": "D:/my vae"}


def test_save_skips_unknown_and_readonly_ids(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    result = stub.save_path_settings({"output_dir": "D:/hack", "不存在": "x", "model:loras": "E:/ok"})

    assert result["skipped"] == ["output_dir", "不存在"]
    assert result["changed"] == ["model:loras"]
    assert "output_dir" not in stub.cfg


def test_save_returns_refreshed_payload(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    result = stub.save_path_settings({"model:loras": "E:/mine"})

    # 保存后直接带着最新状态返回，UI 不用再多发一次请求
    assert result["groups"]
    assert _all_entries(result)["model:loras"]["configured"] == "E:/mine"


def test_save_invalidates_extra_paths_cache(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)
    stub._extra_model_paths_cache = ("x", ("y", 1), {"old": {}})

    stub.save_path_settings({"model:loras": "E:/mine"})

    assert stub._extra_model_paths_cache is None


def test_normalize_path_value_accepts_list_and_none():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio.normalize_path_value(None) == ""
    assert ComfyUIAIStudio.normalize_path_value(["D:/a", "D:/b"]) == "D:/a"
    assert ComfyUIAIStudio.normalize_path_value("  D:/a  ") == "D:/a"


# ------------------------------------------------------------------ 恢复自动

def test_reset_clears_config_and_overrides(tmp_path, monkeypatch):
    stub = _plugin(
        config={
            "comfyui_root": "E:/x",
            "workflow_dir": "E:/wf",
            "model_dir_overrides": {"loras": "E:/mine"},
        },
        monkeypatch=monkeypatch,
    )

    result = stub.reset_path_settings()

    assert stub.cfg["model_dir_overrides"] == {}
    for key in CONFIG_KEYS:
        assert stub.cfg[key] == ""
    assert "model:loras" in result["cleared"]
    assert "comfyui_root" in result["cleared"]
    # 只读项不该出现在恢复列表里
    assert "output_dir" not in result["cleared"]


# ------------------------------------------------------------------ 检测

def test_check_only_checks_given_ids(tmp_path, monkeypatch):
    stub = _plugin(monkeypatch=monkeypatch)

    result = stub.check_path_settings({"model:loras": "E:/nope"})

    assert list(result["results"]) == ["model:loras"]
    assert result["results"]["model:loras"]["exists"] is False


def test_check_falls_back_to_resolved_when_value_blank(tmp_path, monkeypatch):
    root = _fake_comfyui(tmp_path)
    (root / "models" / "loras" / "a.safetensors").write_text("x", encoding="utf-8")
    stub = _plugin(config={"comfyui_root": str(root)}, monkeypatch=monkeypatch, root=root)

    result = stub.check_path_settings({"model:loras": ""})

    info = result["results"]["model:loras"]
    assert info["exists"] is True
    assert info["files"] == 1


def test_check_without_values_checks_everything(tmp_path, monkeypatch):
    from astrbot_plugin_comfyui_ai_studio.main import PATH_SETTING_GROUPS

    stub = _plugin(monkeypatch=monkeypatch)

    result = stub.check_path_settings()

    total = sum(len(group["entries"]) for group in PATH_SETTING_GROUPS)
    assert len(result["results"]) == total


def test_check_marks_file_entries_by_kind(tmp_path, monkeypatch):
    target = tmp_path / "font.ttf"
    target.write_text("x", encoding="utf-8")
    stub = _plugin(config={"image_font_regular": str(target)}, monkeypatch=monkeypatch)

    result = stub.check_path_settings({"image_font_regular": str(target)})

    info = result["results"]["image_font_regular"]
    assert info["kind"] == "file"
    assert info["exists"] is True
    assert info["files"] == 1


# ------------------------------------------------------------------ 浏览

def test_browse_lists_subdirectories(tmp_path):
    root = tmp_path / "tree"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / ".hidden").mkdir()
    (root / "file.txt").write_text("x", encoding="utf-8")

    result = _plugin().browse_path(str(root))

    names = [item["name"] for item in result["entries"]]
    assert names == ["a", "b"]
    assert ".hidden" not in names
    assert result["path"] == str(root)
    assert result["parent"] == str(root.parent)


def test_browse_falls_back_when_path_missing():
    result = _plugin().browse_path("Z:/definitely/not/here")

    assert result["ok"] is True
    assert result["path"]
    assert result["roots"]


def test_browse_root_has_no_parent():
    result = _plugin().browse_path("")

    if result["path"] in result["roots"]:
        assert result["parent"] == ""


def test_browse_tolerates_blank_and_quotes(tmp_path):
    root = tmp_path / "q"
    root.mkdir()

    assert _plugin().browse_path("")["ok"] is True
    assert _plugin().browse_path(f'  "{root}"  ')["path"] == str(root)


# ----------------------------------------------------- paths.py 检测函数

def test_scan_path_counts_only_expected_suffixes(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import scan_path

    (tmp_path / "sub").mkdir()
    (tmp_path / "a.safetensors").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "b.SAFETENSORS").write_text("x", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")

    exists, files = scan_path(tmp_path, suffixes=(".safetensors",))

    assert exists is True
    assert files == 2


def test_scan_path_skips_hidden_entries(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import scan_path

    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "a.safetensors").write_text("x", encoding="utf-8")
    (tmp_path / ".b.safetensors").write_text("x", encoding="utf-8")

    exists, files = scan_path(tmp_path, suffixes=(".safetensors",))

    assert (exists, files) == (True, 0)


def test_scan_path_without_suffixes_skips_full_walk(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import UNKNOWN_FILE_COUNT, scan_path

    (tmp_path / "many.txt").write_text("x", encoding="utf-8")

    exists, files = scan_path(tmp_path)

    assert exists is True
    assert files == UNKNOWN_FILE_COUNT


def test_scan_path_file_kind(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import scan_path

    target = tmp_path / "f.ttf"
    target.write_text("x", encoding="utf-8")

    assert scan_path(target, kind="file") == (True, 1)
    assert scan_path(tmp_path, kind="file") == (False, 0)
    assert scan_path("", kind="file") == (False, 0)


def test_scan_path_rejects_missing_and_blank():
    from astrbot_plugin_comfyui_ai_studio.paths import scan_path

    assert scan_path("") == (False, 0)
    assert scan_path("   ") == (False, 0)
    assert scan_path("Z:/nope") == (False, 0)
    assert scan_path(None) == (False, 0)


def test_scan_path_accepts_file_as_dir_target(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import scan_path

    target = tmp_path / "not-a-dir.bin"
    target.write_text("x", encoding="utf-8")

    assert scan_path(target) == (False, 0)


def test_scan_path_respects_limit(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.paths import scan_path

    for index in range(5):
        (tmp_path / f"m{index}.safetensors").write_text("x", encoding="utf-8")

    assert scan_path(tmp_path, suffixes=(".safetensors",), limit=3) == (True, 3)


# ------------------------------------------------------------ 类别名与后缀

def test_category_labels_cover_every_category():
    from astrbot_plugin_comfyui_ai_studio.paths import MODEL_CATEGORIES, MODEL_CATEGORY_LABELS

    assert set(MODEL_CATEGORY_LABELS) == set(MODEL_CATEGORIES)


def test_category_suffixes_cover_every_category():
    from astrbot_plugin_comfyui_ai_studio.paths import MODEL_CATEGORIES, MODEL_CATEGORY_SUFFIXES

    assert set(MODEL_CATEGORY_SUFFIXES) == set(MODEL_CATEGORIES)
    for category, suffixes in MODEL_CATEGORY_SUFFIXES.items():
        assert suffixes, category
        assert all(item.startswith(".") for item in suffixes), category


def test_source_labels_cover_panel_sources():
    from astrbot_plugin_comfyui_ai_studio.paths import SOURCE_LABELS

    for source in ("override", "extra", "default", "auto", "builtin", "runtime", "none"):
        assert source in SOURCE_LABELS


# ------------------------------------------------------------ 接口与前端接线

def test_paths_route_is_registered():
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")

    assert '"paths", self.api_path_settings' in source
    assert "async def api_path_settings" in source


def test_open_folder_accepts_arbitrary_path():
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")

    assert 'raw_path = str(data.get("path", "") or "").strip()' in source


def test_api_actions_are_covered():
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")

    for action in ('action == "save"', 'action == "reset"', 'action == "check"', 'action == "browse"'):
        assert action in source, action


def test_ui_exposes_path_settings_view():
    html = (PLUGIN_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="view-paths"' in html
    assert 'data-view-tab="paths"' in html
    assert 'data-view-panel="paths"' in html
    assert 'id="pointSettings"' not in html
    assert 'id="pathSettings"' in html
    assert 'id="pathSave"' in html
    assert 'id="pathBrowser"' in html


def test_ui_path_settings_view_is_synced_to_console():
    assert (PLUGIN_DIR / "index.html").read_bytes() == (PLUGIN_DIR / "pages" / "console" / "index.html").read_bytes()
    assert (PLUGIN_DIR / "app.js").read_bytes() == (PLUGIN_DIR / "pages" / "console" / "app.js").read_bytes()
    assert (PLUGIN_DIR / "style.css").read_bytes() == (PLUGIN_DIR / "pages" / "console" / "style.css").read_bytes()


def test_frontend_wires_every_path_action():
    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")

    for needle in (
        "async function loadPathSettings",
        "async function savePathSettings",
        "async function checkPathEntry",
        "async function checkAllPaths",
        "async function resetAllPaths",
        "async function renderPathBrowser",
        "function pickPathBrowser",
        "function initPathSettings",
        'action: "save"',
        'action: "check"',
        'action: "reset"',
        'action: "browse"',
    ):
        assert needle in js, needle
    # 启动与切换到该页时都会读取
    assert "loadPathSettings()" in js
    assert '[data-view-tab="paths"]' in js


def test_frontend_has_path_styles():
    css = (PLUGIN_DIR / "style.css").read_text(encoding="utf-8")

    for needle in (".path-setting-row", ".path-setting-input", ".path-chip", ".path-browser-box", ".path-toolbar"):
        assert needle in css, needle


def test_config_schema_documents_model_dir_overrides():
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))

    assert "model_dir_overrides" in schema
    assert schema["model_dir_overrides"]["type"] == "dict"
    assert "extra_model_paths_file" in schema
