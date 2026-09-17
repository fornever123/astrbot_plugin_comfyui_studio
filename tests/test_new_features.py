"""新增功能回归测试：批量导入 LoRA、免 API 的 C 站下载、图生图独立 LoRA 标注。"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


# ---------------------------------------------------------------- 免 API 解析

def test_resolver_finds_download_url_in_next_data():
    from astrbot_plugin_comfyui_ai_studio.main import CivitAIDownloadResolver

    payload = {
        "props": {
            "pageProps": {
                "trpcState": {
                    "json": {
                        "queries": [
                            {
                                "state": {
                                    "data": {
                                        "modelVersions": [
                                            {
                                                "id": 123456,
                                                "files": [
                                                    {
                                                        "name": "my_lora_v1.safetensors",
                                                        "downloadUrl": "https://civitai.com/api/download/models/123456",
                                                    }
                                                ],
                                            }
                                        ]
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    }
    url, name = CivitAIDownloadResolver.find_download(payload)
    assert url == "https://civitai.com/api/download/models/123456"
    assert name == "my_lora_v1.safetensors"


def test_extract_from_html_next_data_script():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    inner = {
        "props": {
            "pageProps": {
                "files": [
                    {
                        "name": "anime_style.safetensors",
                        "downloadUrl": "https://civitai.com/api/download/models/777",
                    }
                ]
            }
        }
    }
    html = (
        "<html><head>"
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(inner)
        + "</script></head><body></body></html>"
    )
    url, name = ComfyUIAIStudio._extract_civitai_download_from_html(html)
    assert url == "https://civitai.com/api/download/models/777"
    assert name == "anime_style.safetensors"


def test_extract_from_html_regex_fallback():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    html = '<a href="https://civitai.com/api/download/models/424242?type=Model">dl</a>'
    url, name = ComfyUIAIStudio._extract_civitai_download_from_html(html)
    assert url.startswith("https://civitai.com/api/download/models/424242")
    assert name == ""


def test_extract_returns_empty_for_unrelated_html():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._extract_civitai_download_from_html("<html>nope</html>") == ("", "")


def test_direct_download_id_detection():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._civitai_direct_download_id(
        "https://civitai.com/api/download/models/555"
    ) == "555"
    assert ComfyUIAIStudio._civitai_direct_download_id(
        "https://civitai.com/models/555"
    ) == ""
    assert ComfyUIAIStudio._civitai_direct_download_id("") == ""


def test_filename_from_headers_utf8_and_plain():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._civitai_filename_from_headers(
        {"content-disposition": "attachment; filename=\"cool_lora.safetensors\""},
        "fallback.safetensors",
    ) == "cool_lora.safetensors"
    assert ComfyUIAIStudio._civitai_filename_from_headers(
        {"content-disposition": "attachment; filename*=UTF-8''%E4%B8%AD%E6%96%87.safetensors"},
        "fallback.safetensors",
    ) == "中文.safetensors"
    assert ComfyUIAIStudio._civitai_filename_from_headers(
        {}, "fallback.safetensors"
    ) == "fallback.safetensors"


def test_download_mode_normalization():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)

    def make(value):
        plugin._get = lambda key, default=None: value if key == "civitai_download_mode" else default
        return plugin._civitai_download_mode()

    assert make("auto") == "auto"
    assert make("nokey") == "nokey"
    assert make("免api") == "nokey"
    assert make("token") == "token"
    assert make("api") == "token"
    assert make("") == "auto"
    assert make("garbage") == "auto"


def test_nokey_resolver_uses_direct_link_without_network():
    """直链场景必须在完全没有网络/API 的情况下解析成功。"""
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)
    url, name, version = asyncio.run(
        plugin._civitai_resolve_download_nokey(
            "https://civitai.com/api/download/models/999888"
        )
    )
    assert url == "https://civitai.com/api/download/models/999888"
    assert name == "civitai_999888.safetensors"
    assert version == 999888


def test_nokey_resolver_rejects_empty_link():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import UsageError

    plugin = object.__new__(ComfyUIAIStudio)
    with pytest.raises(UsageError):
        asyncio.run(plugin._civitai_resolve_download_nokey(""))


def test_nokey_headers_never_send_token():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)
    plugin._get = lambda key, default=None: {
        "civitai_base_url": "https://civitai.com",
        "civitai_token": "SECRET-TOKEN",
    }.get(key, default)
    with_token = plugin._civitai_headers("application/json")
    without = plugin._civitai_headers("application/json", use_token=False)
    assert with_token.get("Authorization") == "Bearer SECRET-TOKEN"
    assert "Authorization" not in without


# ---------------------------------------------------------------- 批量导入

def test_collect_upload_items_supports_multi_and_single():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    class Multi:
        def __init__(self, data):
            self.data = data

        def getlist(self, key):
            return self.data.get(key, [])

        def get(self, key, default=None):
            values = self.data.get(key, [])
            return values[0] if values else default

    a, b, c = object(), object(), object()
    multi = Multi({"files": [a, b], "file": [c]})
    picked = ComfyUIAIStudio._collect_upload_items(multi, ("files", "file"))
    assert picked == [a, b, c]

    single = Multi({"file": [c]})
    assert ComfyUIAIStudio._collect_upload_items(single, ("files", "file")) == [c]


def test_collect_upload_items_handles_empty_and_plain_dict():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._collect_upload_items(None, ("file",)) == []
    assert ComfyUIAIStudio._collect_upload_items({}, ("file",)) == []


def test_upload_field_names_and_suffixes():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert "files" in ComfyUIAIStudio.LORA_UPLOAD_FIELDS
    assert "file" in ComfyUIAIStudio.LORA_UPLOAD_FIELDS
    assert ".safetensors" in ComfyUIAIStudio.LORA_FILE_SUFFIXES


def test_save_single_upload_validates_suffix_and_duplicates(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import UsageError

    plugin = object.__new__(ComfyUIAIStudio)

    class Upload:
        def __init__(self, filename):
            self.filename = filename
            self.saved = None

        async def save(self, path):
            self.saved = path
            Path(path).write_bytes(b"x" * 2048)

    good = Upload("good_lora.safetensors")
    assert asyncio.run(plugin._save_single_lora_upload(good, tmp_path)) == "good_lora.safetensors"
    assert Path(good.saved).exists()

    with pytest.raises(UsageError):
        asyncio.run(plugin._save_single_lora_upload(Upload("bad.txt"), tmp_path))

    with pytest.raises(UsageError):
        asyncio.run(plugin._save_single_lora_upload(Upload("good_lora.safetensors"), tmp_path))


def test_batch_route_is_registered():
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert '"upload_lora_batch", self.api_upload_lora_batch' in source
    assert "async def api_upload_lora_batch" in source


# ---------------------------------------------------------------- 图生图标注

def test_img2img_notice_constant_and_field():
    from astrbot_plugin_comfyui_ai_studio.main import IMG2IMG_LORA_ISOLATION_NOTICE
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    assert "特别标注" in IMG2IMG_LORA_ISOLATION_NOTICE
    assert "独立 LoRA" in IMG2IMG_LORA_ISOLATION_NOTICE
    params = DrawParams()
    assert params.lora_isolation_notice == ""
    params.lora_isolation_notice = IMG2IMG_LORA_ISOLATION_NOTICE
    assert "文生图" in params.lora_isolation_notice


def test_img2img_notice_appended_to_start_reply():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio, IMG2IMG_LORA_ISOLATION_NOTICE
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    plugin = object.__new__(ComfyUIAIStudio)
    plugin._get = lambda key, default=None: default
    plugin._queue_notice_from_params = lambda params: ""

    params = DrawParams(mode="img2img")
    params.lora_isolation_notice = IMG2IMG_LORA_ISOLATION_NOTICE
    reply = ComfyUIAIStudio._draw_start_reply(plugin, "img2img", "改成夜景", params)
    assert IMG2IMG_LORA_ISOLATION_NOTICE in reply

    # 文生图不应带图生图标注
    txt = ComfyUIAIStudio._draw_start_reply(plugin, "txt2img", "海边少女", DrawParams())
    assert "特别标注" not in txt


def test_img2img_lora_isolation_uses_only_independent_lora():
    """图生图的长期 LoRA 只能来自 img2img_lora_name，不能读取 lora_list。"""
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    marker = "configured_img2img_loras: list[str] = []"
    assert marker in source
    anchor = source.index(marker)
    block = source[anchor : anchor + 2200]
    assert 'self._get("img2img_lora_name"' in block
    # 图生图分支必须使用独立 LoRA 列表，而不是 lora_list
    assert "if mode == \"img2img\"" in block
    assert "IMG2IMG_LORA_ISOLATION_NOTICE" in block


def test_frontend_exposes_batch_and_notice():
    html = (PLUGIN_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="loraFile" type="file" multiple' in html
    assert 'id="loraBatchCategory"' in html
    assert 'id="loraBatchProgress"' in html
    assert 'id="styleLoraFile" type="file" multiple' in html
    assert "lora-isolation-notice" in html
    assert html.count("【特别标注】") >= 2

    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")
    assert "async function importLoraBatch" in js
    assert "category_batch" in js

    css = (PLUGIN_DIR / "style.css").read_text(encoding="utf-8")
    assert ".lora-isolation-notice" in css
    assert ".lora-batch-failures" in css


def test_schema_has_download_mode():
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    mode = schema["civitai_download_mode"]
    assert mode["default"] == "auto"
    assert set(mode["enum"]) == {"auto", "nokey", "token"}

# ------------------------------------------------- 路径可配置（1.0.0 发布版）

BS = chr(92)  # 反斜杠字符，避免测试源码里出现难以阅读的多重转义


def _schema() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))


def test_schema_ships_no_machine_specific_path_defaults():
    """发布版不允许把作者本机盘符路径写成默认值。"""
    offenders = {
        key: value.get("default")
        for key, value in _schema().items()
        if isinstance(value, dict)
        and isinstance(value.get("default"), str)
        and (f"E:{BS}" in value["default"] or f"C:{BS}" in value["default"])
    }
    assert offenders == {}


def test_source_workflow_defaults_are_empty_for_portability():
    schema = _schema()
    for key in (
        "source_workflow",
        "wash_source_workflow",
        "outpaint_source_workflow",
        "multi_angle_source_workflow",
        "img2img_source_workflow",
        "img2img_flux2_source_workflow",
    ):
        assert schema[key]["default"] == "", key


def test_anima_knowledge_has_no_hardcoded_local_paths():
    source = (PLUGIN_DIR / "anima_knowledge.py").read_text(encoding="utf-8")
    assert f"E:{BS}" not in source
    assert f"C:{BS}" not in source


def test_no_drive_path_is_used_as_a_config_default():
    """任何 self._get(key, 默认值) 的默认值都不能是本机盘符路径。"""
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    pattern = re.compile(r'_get\(\s*"[A-Za-z_0-9]+"\s*,\s*r?"([^"]*)"')
    offenders = [
        match.group(1)
        for match in pattern.finditer(source)
        if re.match(r"^[A-Za-z]:", match.group(1))
    ]
    assert offenders == []


def test_drive_paths_live_only_in_the_legacy_fingerprint_block():
    """本机盘符路径只允许集中在旧值指纹块里，方便审计与后续清理。"""
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    start = source.index("LEGACY_AUTHOR_PATH_FINGERPRINTS = {")
    end = source.index("FLUX2_IMG2IMG_SOURCE_DEFAULT = ", start)
    remainder = source[:start] + source[end:]
    assert f"E:{BS}" not in remainder


def test_legacy_fingerprints_are_still_used_for_migration():
    """指纹块必须真的参与老配置清理，否则就是死代码。"""
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert 'LEGACY_AUTHOR_PATH_FINGERPRINTS["flux2_img2img"]' in source
    assert 'LEGACY_AUTHOR_PATH_FINGERPRINTS["qwen_img2img"]' in source
    assert "LEGACY_AUTHOR_PATH_FINGERPRINTS[mode]" in source


def test_image_font_candidates_prefers_configured_paths():
    from astrbot_plugin_comfyui_ai_studio.main import image_font_candidates

    regular, bold = image_font_candidates("/tmp/my-regular.ttf", "/tmp/my-bold.ttf")
    assert regular[0] == "/tmp/my-regular.ttf"
    assert bold[0] == "/tmp/my-bold.ttf"
    assert len(regular) > 1 and len(bold) > 1


def test_image_font_candidates_cover_windows_linux_and_macos():
    from astrbot_plugin_comfyui_ai_studio.main import image_font_candidates

    regular, bold = image_font_candidates()
    joined = " ".join([*regular, *bold])
    assert f"C:{BS}Windows{BS}Fonts" in joined
    assert "/usr/share/fonts" in joined
    assert "/System/Library/Fonts" in joined or "/Library/Fonts" in joined


def test_image_font_candidates_deduplicates_blank_values():
    from astrbot_plugin_comfyui_ai_studio.main import image_font_candidates

    regular, bold = image_font_candidates("", "   ")
    assert all(item.strip() for item in regular)
    assert all(item.strip() for item in bold)
    assert len(set(regular)) == len(regular)


def test_font_and_template_keys_are_writable_via_webui():
    main_source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    schema = _schema()
    for key in ("image_font_regular", "image_font_bold", "anima_template_path"):
        assert f'"{key}"' in main_source, key
        assert key in schema, key


def test_anima_template_path_can_be_overridden(tmp_path):
    from astrbot_plugin_comfyui_ai_studio import anima_knowledge

    custom_dir = tmp_path / "templates"
    custom_dir.mkdir()
    custom_file = custom_dir / "提示词模版.txt"
    custom_file.write_text("## 1. ROLE" + chr(10) + "自定义模板内容", encoding="utf-8")

    assert anima_knowledge.find_template_path(PLUGIN_DIR, str(custom_file)) == custom_file
    assert anima_knowledge.find_template_path(PLUGIN_DIR, str(custom_dir)) == custom_file


def test_bundled_template_is_used_when_nothing_is_configured():
    from astrbot_plugin_comfyui_ai_studio import anima_knowledge

    found = anima_knowledge.find_template_path(PLUGIN_DIR)
    assert found == PLUGIN_DIR / "knowledge" / "提示词模版.txt"
    assert "Anima3" in anima_knowledge.build_context(PLUGIN_DIR)


def test_bundled_workflows_cover_every_mode():
    """源工作流留空时，七种模式都必须有插件内置副本可回退。"""
    from astrbot_plugin_comfyui_ai_studio.main import MODE_FILES

    for mode, filename in MODE_FILES.items():
        assert (PLUGIN_DIR / "workflows" / filename).is_file(), mode


def test_plugin_version_is_1_1_4_everywhere():
    assert 'PLUGIN_VERSION = "1.1.4"' in (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert "version: 1.1.4" in (PLUGIN_DIR / "metadata.yaml").read_text(encoding="utf-8")
    assert "版本 v1.1.4" in (PLUGIN_DIR / "index.html").read_text(encoding="utf-8")
    assert "版本-1.1.4-" in (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
    assert "0.10.3" not in (PLUGIN_DIR / "translation.py").read_text(encoding="utf-8")


def test_brand_name_is_uniform_across_github_and_astrbot():
    """GitHub（README）与 AstrBot（metadata / 控制台 / 帮助）统一使用同一昵称。"""
    brand = "Anima 全能绘画台"
    stale = ("绘梦坊", "ComfyUI AI 绘画台", "Comfy 画图台", "幻境画坊")
    for name in (
        "README.md",
        "metadata.yaml",
        "index.html",
        "pages/console/index.html",
        "main.py",
        "style.css",
    ):
        text = (PLUGIN_DIR / name).read_text(encoding="utf-8")
        assert brand in text, name
        for old in stale:
            assert old not in text, name + " 仍残留旧品牌名 " + old


def test_text_files_keep_lf_line_endings():
    crlf = (chr(13) + chr(10)).encode("utf-8")
    for name in (
        "main.py",
        "anima_knowledge.py",
        "index.html",
        "style.css",
        "README.md",
        "metadata.yaml",
        "tests/test_new_features.py",
    ):
        assert crlf not in (PLUGIN_DIR / name).read_bytes(), name


# ------------------------------------------- 画风简洁卡：长期启用 / 纳入随机候选

def test_style_lora_group_has_include_all_button():
    """画风组顶部有「全部纳入随机候选」，且排在「保存画风选择」之前。"""
    html = (PLUGIN_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="styleLoraIncludeAll"' in html
    assert "全部纳入随机候选" in html
    assert html.index('id="styleLoraIncludeAll"') < html.index('id="saveStyleLora"')
    # 按钮必须落在画风组里，而不是普通 LoRA 组
    style_group = html[html.index('id="styleLoras"') - 4000:html.index('id="styleLoras"')]
    assert 'id="styleLoraIncludeAll"' in style_group


def test_compact_style_card_exposes_both_switches():
    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")
    card = js[js.index("function renderLoraCompactCard"):js.index("function renderLoraFullCard")]
    assert "lora-simple-switches" in card
    assert "data-lora-enabled-checkbox" in card and "长期启用" in card
    assert "data-style-lora-file" in card and "纳入随机候选" in card
    # 开关行只在画风简洁卡出现（整个渲染函数里只有一处）
    assert card.count("lora-simple-switches") == 1
    # 状态类由 JS 显式输出，不依赖 :has() —— 老 WebView 不支持
    assert "is-on" in card
    assert ":has(" not in card


def test_persistent_switch_writes_lora_list_without_duplicates():
    """「长期启用」写入 lora_list，保留既有项与权重，并去重。"""
    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")
    fn = js[js.index("async function setStyleLoraPersistent"):]
    fn = fn[:fn.index("\n}\n")]
    assert "currentLoraSelection()" in fn
    assert "${file}:${weight}" in fn
    assert "new Set(" in fn                    # 去重
    assert "lora_list" in fn
    assert "syncStyleLoraSwitchUI" in fn        # 就地同步状态，不整表重绘


def test_switch_saves_are_queued_and_never_dropped():
    """连续勾选必须串行排队逐个提交；不允许用锁把并发点击静默丢弃。"""
    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")
    assert "styleLoraSaveChain" in js
    assert "styleLoraSaveChain.then(run, run)" in js
    assert "let styleLoraSaving" not in js
    assert "if (styleLoraSaving) return false" not in js
    # 勾选即保存走 silent + 不重绘，靠 syncStyleLoraSwitchUI 就地更新
    assert "function syncStyleLoraSwitchUI" in js
    assert "if (needRerender) renderStyleLoras(); else syncStyleLoraSwitchUI();" in js


def test_include_all_only_collects_style_category():
    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")
    block = js[js.index('document.getElementById("styleLoraIncludeAll").onclick'):]
    block = block[:block.index("\n};")]
    assert '(item.category || "未分类") === "画风"' in block
    assert "candidates: files" in block


def test_style_switch_row_css_present():
    css = (PLUGIN_DIR / "style.css").read_text(encoding="utf-8")
    assert ".lora-simple-switches" in css
    # 选择器必须带 label. 前缀，否则被原版 label.check 的 !important 吃掉状态色
    assert "label.lora-simple-switch {" in css
    assert "label.lora-simple-switch.is-on" in css
    assert "label.lora-simple-switch:hover" in css


def _css_block(css: str, header: str) -> str:
    start = css.index(header)
    return css[start:css.index("}", start)]


def _hex(text: str, var: str) -> str:
    m = re.search(re.escape(var) + r"\s*:\s*(#[0-9A-Fa-f]{3,6})", text)
    assert m, "未在 CSS 块里找到 " + var
    return m.group(1)


def _rgba(text: str, var: str):
    m = re.search(re.escape(var) + r"\s*:\s*rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", text)
    assert m, "未在 CSS 块里找到 " + var
    return tuple(float(m.group(i)) for i in (1, 2, 3, 4))


def _srgb(channel: float) -> float:
    s = channel / 255.0
    return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4


def _lum(rgb) -> float:
    r, g, b = (_srgb(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg, bg) -> float:
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _unhex(value: str):
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _composite(over, under):
    r, g, b, a = over
    return tuple(over[i] * a + under[i] * (1 - a) for i in range(3))


def _skin_layer() -> str:
    """只取尾部的 T4-E 换肤层——原版第 60 行有同样的深色块头，不切层会读错块。"""
    css = (PLUGIN_DIR / "style.css").read_text(encoding="utf-8")
    start = css.index("/* ===== T4-E 换肤层")
    return css[start:css.index("/* ===== T4-E 换肤层 · END", start)]


def test_switch_label_color_meets_wcag_aa_in_both_themes():
    """未勾选开关的文字必须达到 AA（4.5:1）——浅色皮肤最容易在这里翻车。

    深色下曾整条压成浅色字面量导致几乎不可见；浅色下 muted 白底只有 3.4:1。
    这里直接读已发布的 style.css，按卡片真实表面色算对比度。
    """
    layer = _skin_layer()
    light = _css_block(layer, ':root, :root[data-ui-theme="light"], :root.theme-light')
    dark = _css_block(layer, ':root[data-ui-theme="dark"], :root.theme-dark')

    light_bg = _unhex(_hex(light, "--bg"))
    dark_bg = _unhex(_hex(dark, "--bg"))

    # 卡片用 --glass-strong 取表面色；深色是半透明，需合成到页面底色上
    light_surface = _unhex(_hex(light, "--glass-strong"))
    dark_surface = _composite(_rgba(dark, "--glass-strong"), dark_bg)

    for theme, block, surface in (
        ("light", light, light_surface),
        ("dark", dark, dark_surface),
    ):
        fg = _unhex(_hex(block, "--simple-switch-idle"))
        ratio_surface = _contrast(fg, surface)
        assert ratio_surface >= 4.5, "{} 开关文字对卡片表面仅 {:.2f}:1".format(theme, ratio_surface)
        ratio_bg = _contrast(fg, light_bg if theme == "light" else dark_bg)
        assert ratio_bg >= 4.5, "{} 开关文字对页面底色仅 {:.2f}:1".format(theme, ratio_bg)

        # 已勾选态用强调色，同样要达标
        ratio_accent = _contrast(_unhex(_hex(block, "--accent-strong")), surface)
        assert ratio_accent >= 4.5, "{} 已勾选态强调色仅 {:.2f}:1".format(theme, ratio_accent)


def test_style_layer_has_dark_counterpart_for_switch_row():
    """换肤层里凡带 !important 的浅色规则都要有深色跟随；开关行的取色必须走主题变量。"""
    layer = _skin_layer()
    row = layer[layer.index("label.lora-simple-switch"):]
    row = row[:row.index("input {")]
    assert "var(--simple-switch-idle)" in row
    assert "var(--accent-strong)" in row
    # 状态类由 JS 输出，这里不允许依赖 :has()
    assert ":has(" not in row
    # 不允许出现写死的颜色字面量（原版 CSS 里 :has() 另有 14 处，与此无关）
    assert not re.search(r":\s*#[0-9A-Fa-f]{3,6}\s*;", row)
    assert not re.search(r":\s*rgba?\(\d", row)
