from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


class FakeContext:
    def __init__(self) -> None:
        self.apis: list[tuple[str, list[str], str]] = []

    def register_web_api(self, route, view_handler, methods, desc=""):
        self.apis.append((route, methods, desc))


def test_web_api_handlers_use_dashboard_request_context() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    for handler in (
        "api_status",
        "api_models",
        "api_workflows",
        "api_config",
        "api_ai_models",
        "api_presets",
        "api_artist_presets",
        "api_recent",
    ):
        assert f"async def {handler}(self, request:" not in source
    assert "from astrbot.api.web import json_response, request" in source


def test_console_loads_astrbot_bridge_and_shows_version() -> None:
    page = (PLUGIN_DIR / "pages" / "console" / "index.html").read_text(encoding="utf-8")
    assert '/api/plugin/page/bridge-sdk.js' in page
    assert "版本 v0.8.0" in page


def test_anima_prompt_engineer_is_built_into_comfyui_plugin() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert "on_llm_request" in source
    assert "内置 Anima 提示词工程师" in source
    assert "api_characters" not in source
    assert not (PLUGIN_DIR / "anima3.py").exists()
    assert not (PLUGIN_DIR / "characters.py").exists()
    assert not (PLUGIN_DIR / "pages" / "characters").exists()
    assert (PLUGIN_DIR / "knowledge" / "提示词模版.txt").stat().st_size > 100000


def test_standalone_ai_switch_is_removed_from_prompt() -> None:
    from astrbot_plugin_comfyui_ai_studio.prompting import fill_params

    params = fill_params("ai 夏空站在海边", "txt2img")
    assert params.prompt == "夏空站在海边"
    assert params.ai is True

    quoted = fill_params("\"ai\" 夏空站在海边", "txt2img")
    assert quoted.prompt == "夏空站在海边"
    assert quoted.ai is True

    normal = fill_params("夏空站在海边", "txt2img")
    assert normal.prompt == "夏空站在海边"
    assert normal.ai is None

    noai = fill_params("noai 夏空站在海边", "txt2img")
    assert noai.prompt == "夏空站在海边"
    assert noai.ai is False

    prompt_words = fill_params("a chair painting", "txt2img")
    assert prompt_words.prompt == "a chair painting"
    assert prompt_words.ai is None


def test_console_exposes_reply_and_lora_management() -> None:
    page = (PLUGIN_DIR / "pages" / "console" / "index.html").read_text(encoding="utf-8")
    app = (PLUGIN_DIR / "pages" / "console" / "app.js").read_text(encoding="utf-8")
    assert 'id="draw_reply_mode"' in page
    assert 'id="draw_delivery_mode"' in page
    assert 'id="draw_reply_custom"' in page
    assert 'id="themeToggle"' in page
    assert 'data-open-folder="loras"' in page
    assert 'id="loraFile"' in page
    assert 'id="artistPresetActive"' in page
    assert 'id="default_positive"' in page
    assert 'id="civitaiDownloadUrl"' in page
    assert 'id="civitaiDownloadOverwrite"' in page
    assert 'id="loraDownloadProgress"' in page
    assert 'id="loraDownloadBar"' in page
    for view in ("workflow", "txt2img", "img2img", "ai", "presets", "artist"):
        assert f'data-view-tab="{view}"' in page
        assert f'data-view-panel="{view}"' in page
    assert "文生图" in page
    assert 'id="saveTxt2ImgSettings"' in page
    assert 'id="saveAi"' in page
    assert 'id="saveReply"' in page
    assert 'id="llm_prompt_source"' in page
    assert 'id="default_positive"' in page
    assert "initViews" in app
    assert 'id="loadAiModels"' in page
    assert 'id="testAiConnection"' in page
    assert 'id="aiConnectionStatus"' in page
    assert 'data-lora-action="save-lora"' in app
    assert 'data-lora-action="open-lora"' in app
    assert '打开 LoRA 文件位置' in app
    assert 'data-lora-action="delete-lora"' not in app
    assert 'data-lora-action="add-command-alias"' in app
    assert 'command_aliases:' in app
    assert 'post(`${API}/download_lora`' in app
    assert 'post(`${API}/download_lora_progress`' in app
    assert "自定义显示图片（每行一个链接）" not in app
    assert "data-civitai-images-file" not in app
    assert "AI 翻译" not in page
    assert "AI 翻译" not in app
    assert "prompt_ai_source" not in page
    assert "prompt_ai_source" not in app
    assert "draw_reply_mode" in app
    assert "draw_delivery_mode" in app
    assert "lora_info" in app
    assert 'id="loraDisplayToggle"' in page
    assert 'id="loraBulkActions"' in page
    assert 'id="saveTxt2ImgSettings"' in page
    assert 'id="seed"' in page
    assert "initLoraMode" in app
    assert "sourcePathElement" in app
    assert "saveTxt2ImgSettings" in app
    assert "payload.model_name" in app
    assert "请先获取模型列表并测试连接" in app


def test_help_image_and_download_order_are_available() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    requirements = (PLUGIN_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "def _help_image_path(self)" in source
    assert "Image.fromFileSystem(str(image_path))" in source
    assert "def _save_lora_download_order" in source
    assert "lora_download_order[filename]" in source
    assert "pillow>=10" in requirements.lower()
    assert "AstrBot-ComfyUI-AI-Studio/0.8.0" in (PLUGIN_DIR / "translation.py").read_text(encoding="utf-8")


def test_config_and_delete_paths_are_stateful_and_windows_safe() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert '"lora_list": list(self._get("lora_list", []) or [])' in source
    assert 'normalized_filename = raw_filename.replace("\\\\", "/")' in source
    assert 'async def api_delete_lora(self)' in source


def test_commands_do_not_use_legacy_suffix() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    for command in ("helpd", "文生图", "图生图", "高清放大", "模型", "预设", "工作流", "画图配置", "comfy状态"):
        assert f'@filter.command("{command}s"' not in source
    assert '@filter.command("loras"' not in source


def test_request_readers_support_callable_and_property_api() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    class CallableRequest:
        async def json(self, default=None):
            return {"mode": "callable"}

        async def files(self):
            return {"file": "callable"}

    class PropertyRequest:
        @property
        def json(self):
            async def read():
                return {"mode": "property"}

            return read()

        @property
        def files(self):
            async def read():
                return {"file": "property"}

            return read()

    async def read_all():
        callable_request = CallableRequest()
        property_request = PropertyRequest()
        return (
            await ComfyUIAIStudio._request_json(callable_request, {}),
            await ComfyUIAIStudio._request_files(callable_request),
            await ComfyUIAIStudio._request_json(property_request, {}),
            await ComfyUIAIStudio._request_files(property_request),
        )

    assert asyncio.run(read_all()) == (
        {"mode": "callable"},
        {"file": "callable"},
        {"mode": "property"},
        {"file": "property"},
    )


def test_placeholder_png_has_safe_dimensions() -> None:
    from astrbot_plugin_comfyui_ai_studio import main as plugin_main

    png = plugin_main.BLANK_PNG
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert int.from_bytes(png[16:20], "big") == 64
    assert int.from_bytes(png[20:24], "big") == 64


def test_generated_blank_png_has_requested_dimensions() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import _blank_png

    png = _blank_png(416, 608)
    assert int.from_bytes(png[16:20], "big") == 416
    assert int.from_bytes(png[20:24], "big") == 608


def test_preset_update_keeps_matching_translation_only() -> None:
    from astrbot_plugin_comfyui_ai_studio.prompting import PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_preset_test_") as temp:
        store = PresetStore(Path(temp) / "presets.json")
        store.add("夏空", "ciaccona")
        store.items["夏空"]["translated"] = "ciaccona, red hair"
        store.save()

        store.add("夏空", "ciaccona")
        assert store.effective("夏空") == "ciaccona, red hair"

        store.add("夏空", "ciaccona, green eyes")
        assert store.effective("夏空") == "ciaccona, green eyes"


def test_plain_translation_keeps_network_out_of_ai_path(monkeypatch) -> None:
    from astrbot_plugin_comfyui_ai_studio.translation import PlainTranslator

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [[["girl by the sea", "女孩在海边"]]]

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return Response()

    import astrbot_plugin_comfyui_ai_studio.translation as translation

    monkeypatch.setattr(translation.httpx, "AsyncClient", lambda **kwargs: Client())
    assert asyncio.run(PlainTranslator().translate("海边女孩")) == "girl by the sea"


def test_plain_translation_falls_back_after_google_rate_limit(monkeypatch) -> None:
    from astrbot_plugin_comfyui_ai_studio.translation import PlainTranslator

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, endpoint, *args, **kwargs):
            if "mymemory.translated.net" in endpoint:
                return Response({"responseData": {"translatedText": "girl bathing"}})
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("GET", endpoint),
                response=httpx.Response(429),
            )

        async def post(self, *args, **kwargs):
            raise AssertionError("Google 限流后不应先调用自定义 POST")

    import astrbot_plugin_comfyui_ai_studio.translation as translation

    monkeypatch.setattr(translation.httpx, "AsyncClient", lambda **kwargs: Client())
    assert asyncio.run(PlainTranslator().translate("洗澡的女孩")) == "girl bathing"


def test_lora_trigger_words_are_separate_from_named_presets() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.civitai_overrides = {
        "demo.safetensors": {"trigger_words": ["demo trigger", "character_trigger"]}
    }
    star.civitai_cache = {}
    assert star._lora_trigger_words("demo.safetensors") == ["demo trigger", "character_trigger"]


def test_lora_preset_alias_is_dynamic_and_civitai_words_are_not_prompt_values() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.lora_presets = {
        "demo.safetensors": [{"tag": "小小爱", "content": "little Aemeath (WuWa)"}],
    }
    star.lora_aliases = {}
    star.lora_command_aliases = {"demo.safetensors": []}
    assert star._resolve_lora("小小爱", ["demo.safetensors"]) == "demo.safetensors"
    assert star._lora_prompt_values("demo.safetensors") == ["little Aemeath (WuWa)"]

    star.lora_presets["demo.safetensors"][0]["tag"] = "新简称"
    assert star._resolve_lora("小小爱", ["demo.safetensors"]) is None
    assert star._resolve_lora("新简称", ["demo.safetensors"]) == "demo.safetensors"


def test_legacy_civitai_preset_fragments_are_removed_but_global_content_survives() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    with tempfile.TemporaryDirectory(prefix="astrbot_civitai_migration_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.lora_presets_path = Path(temp) / "lora_presets.json"
        star.lora_presets = {
            "demo.safetensors": [
                {
                    "tag": "小小爱",
                    "content": "little Aemeath (WuWa), auto_trigger",
                    "_global_content": "little Aemeath (WuWa)",
                    "_civitai_content": ["auto_trigger"],
                },
                {
                    "tag": "旧自动行",
                    "content": "auto_trigger",
                    "_civitai_content": ["auto_trigger"],
                },
            ]
        }
        assert star._remove_legacy_civitai_preset_fragments() is True
        assert star.lora_presets["demo.safetensors"] == [
            {
                "tag": "小小爱",
                "content": "little Aemeath (WuWa)",
                "_global_content": "little Aemeath (WuWa)",
            }
        ]


def test_civitai_trigger_words_are_display_only_in_prompt_assembly() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_civitai_display_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.presets = PresetStore(Path(temp) / "presets.json")
        star.artist_presets = PresetStore(Path(temp) / "artist_presets.json")
        star.config = {"default_positive": "", "default_negative": "", "artist_preset": "无"}
        async def fake_translate(*args, **kwargs):
            return "translated scene"

        star._translate_prompt = fake_translate
        params = DrawParams(prompt="中文动作", ai=True)
        positive, _, _ = asyncio.run(
            star._prompt_text(
                None,
                params,
                lora_trigger_words=["civitai_auto_trigger"],
                lora_prompt_values=["little Aemeath (WuWa)"],
            )
        )
        assert "little Aemeath (WuWa)" in positive
        assert "translated scene" in positive
        assert "civitai_auto_trigger" not in positive


def test_input_image_is_cached_before_temp_file_cleanup() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_input_cache_test_") as root:
            base = Path(root)
            source = base / "media_image_test.jpg"
            source.write_bytes(b"fake-image-data")
            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()

            cached = await star._persist_input_image(source)
            source.unlink()

            assert cached
            assert Path(cached).is_file()
            assert Path(cached).read_bytes() == b"fake-image-data"

    asyncio.run(run())


def test_forward_delivery_builds_merge_forward_nodes() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot.api.message_components import Image, Nodes, Plain

    class Event:
        def get_self_id(self):
            return "123456"

    star = object.__new__(ComfyUIAIStudio)
    chain = star._forward_result_chain(Event(), "绘图完成", [Path("one.png"), Path("two.png")])

    assert len(chain) == 1
    assert isinstance(chain[0], Nodes)
    assert len(chain[0].nodes) == 3
    assert isinstance(chain[0].nodes[0].content[0], Image)
    assert isinstance(chain[0].nodes[-1].content[0], Plain)
    assert chain[0].nodes[-1].content[0].text == "绘图完成"
    assert [node.uin for node in chain[0].nodes] == ["123456", "123456", "123456"]
    flattened = star._flatten_forward_chain(chain)
    assert isinstance(flattened[0], Image)
    assert isinstance(flattened[-1], Plain)
    assert len(flattened) == 3


def test_lora_search_and_preset_labels_are_alias_based() -> None:
    page = (PLUGIN_DIR / "pages" / "console" / "index.html").read_text(encoding="utf-8")
    app = (PLUGIN_DIR / "pages" / "console" / "app.js").read_text(encoding="utf-8")
    assert 'id="loraSearch"' in page
    assert "loraSearchQuery" in app
    assert "data-lora-preset-alias" in app
    assert "CivitAI tag" not in app


def test_completion_forward_quotes_original_message_and_puts_text_last() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot.api.message_components import Image, Nodes, Plain, Reply

    class Message:
        message_id = "qq-message-123"

    class Event:
        message_obj = Message()

        def get_self_id(self):
            return "123456"

    star = object.__new__(ComfyUIAIStudio)
    chain = star._forward_result_chain(Event(), "绘图完成", [Path("one.png")])
    assert isinstance(chain[0], Reply)
    assert chain[0].id == "qq-message-123"
    assert isinstance(chain[1], Nodes)
    assert isinstance(chain[1].nodes[0].content[0], Image)
    assert isinstance(chain[1].nodes[-1].content[0], Plain)


def test_civitai_trigger_words_sync_method_is_now_a_noop() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_lora_preset_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.lora_presets = {}
        asyncio.run(star._sync_lora_trigger_presets(
            "demo.safetensors",
            aliases=["一号lora", "角色简称"],
            trigger_words=["character_trigger", "red_hair"],
        ))
        assert star.lora_presets == {}

        asyncio.run(star._sync_lora_trigger_presets(
            "demo.safetensors",
            aliases=["新简称"],
            trigger_words=["new_trigger"],
        ))
        assert star.lora_presets == {}

        asyncio.run(star._sync_lora_trigger_presets(
            "demo.safetensors",
            aliases=["手动简称"],
            trigger_words=["should_not_replace"],
        ))
        assert star.lora_presets == {}


def test_lora_tags_and_private_presets_are_kept_separate_from_global_presets() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_lora_private_preset_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.presets = PresetStore(Path(temp) / "presets.json")
        star.presets.add("旧全局预设", "old_global_tag")
        star.civitai_overrides = {
            "demo.safetensors": {
                "tags": ["civitai_tag", "civitai_tag"],
            }
        }
        star.civitai_cache = {}
        star.lora_presets = {
            "demo.safetensors": [
                {"tag": "civitai_tag", "content": "character_tag, red_hair"},
                {"tag": "second_tag", "content": "blue_eyes"},
            ]
        }

        assert star._lora_civitai_tags("demo.safetensors") == ["civitai_tag"]
        assert star._lora_prompt_values("demo.safetensors") == [
            "character_tag, red_hair",
            "blue_eyes",
        ]
        assert star.presets.effective("旧全局预设") == "old_global_tag"


def test_civitai_tag_normalization_and_private_preset_validation() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._normalize_civitai_tags("one, two，one") == ["one", "two"]
    assert ComfyUIAIStudio._normalize_lora_preset_entries(
        {"entries": {"tag_a": "content_a", "tag_b": "content_b"}}
    ) == [
        {"tag": "tag_a", "content": "content_a"},
        {"tag": "tag_b", "content": "content_b"},
    ]


def test_civitai_download_links_and_windows_filename_are_supported() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._civitai_reference(
        "https://civitai.com/models/123456?modelVersionId=789012"
    ) == ("123456", "789012")
    assert ComfyUIAIStudio._civitai_reference(
        "https://civitai.com/api/download/models/789012?fileId=9"
    ) == ("", "789012")
    assert ComfyUIAIStudio._safe_lora_filename(
        "bad:name?.safetensors", "fallback.safetensors"
    ) == "bad_name_.safetensors"
    assert ComfyUIAIStudio._safe_lora_filename(
        "CON.safetensors", "fallback.safetensors"
    ) == "_CON.safetensors"


def test_llm_plugin_prompt_keeps_original_action_and_debug_output() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "帮我画一张洗澡的夏空"

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "llm_prompt_source": "plugin",
        "plugin_ai_debug": True,
    }

    received: list[str] = []

    async def fake_translate(event, text, **kwargs):
        received.append(text)
        assert kwargs["source_override"] == "plugin_llm"
        return "ciaccona, bathing, in bathtub, wet hair"

    star._translate_prompt = fake_translate
    params = DrawParams(prompt="夏空")
    translated = asyncio.run(star._prepare_llm_prompt(Event(), params, Event.message_str))

    assert "洗澡" in received[0]
    assert "夏空" in received[0]
    assert translated == params.prompt == "ciaccona, bathing, in bathtub, wet hair"
    assert "bathing" in star._llm_debug_reply("任务已提交", translated)


def test_console_exposes_separate_llm_and_command_ai_prompts() -> None:
    page = (PLUGIN_DIR / "pages" / "console" / "index.html").read_text(encoding="utf-8")
    app = (PLUGIN_DIR / "pages" / "console" / "app.js").read_text(encoding="utf-8")
    assert 'id="llm_prompt_source"' in page
    assert 'id="plugin_ai_llm_system_prompt"' in page
    assert 'id="plugin_ai_command_system_prompt"' in page
    assert 'id="plugin_ai_debug"' in page
    assert "plugin_ai_llm_system_prompt" in app
    assert "plugin_ai_command_system_prompt" in app


def test_img2img_uses_a_separate_llm_editor_and_preserves_explicit_controls() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "把身上衣服换成裙子 维里奈"

        def get_sender_id(self):
            return "10001"

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "img2img_llm_prompt_source": "plugin",
        "img2img_default_positive": "base edit quality",
        "img2img_default_negative": "",
    }

    received: list[str] = []

    async def fake_translate(event, text, **kwargs):
        received.append(text)
        assert kwargs["source_override"] == "plugin_img2img_llm"
        return "change the outfit to a dress"

    star._translate_prompt = fake_translate
    params = DrawParams(mode="img2img", prompt="换成裙子", presets=["维里奈"])
    translated = asyncio.run(star._prepare_llm_prompt(Event(), params, Event.message_str, "img2img"))

    assert translated == "change the outfit to a dress"
    assert params.img2img_edit_instruction_ready is True
    assert "换成裙子" in received[0]
    assert params.presets == ["维里奈"]


def test_img2img_llm_prompt_parts_are_editable() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    page = (PLUGIN_DIR / "pages" / "console" / "index.html").read_text(encoding="utf-8")
    app = (PLUGIN_DIR / "pages" / "console" / "app.js").read_text(encoding="utf-8")
    for key in (
        "img2img_llm_tool_prompt",
        "img2img_astrbot_llm_system_prompt",
        "img2img_astrbot_user_prompt_template",
        "img2img_plugin_ai_user_prompt_template",
        "img2img_plugin_ai_output_format",
    ):
        assert key in source
        assert key in page
        assert key in app
    assert "img2img_prompt_defaults" in source
    assert "{request}" in source


def test_img2img_astrbot_llm_source_also_generates_an_edit_instruction() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "把身上衣服换成裙子 维里奈"

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "img2img_llm_prompt_source": "astrbot",
        "img2img_plugin_ai_knowledge": "",
    }
    received: list[tuple[str, str, int]] = []

    async def fake_astrbot_generate(event, prompt, *, system_prompt=None, max_tokens=512, **kwargs):
        received.append((prompt, system_prompt or "", max_tokens))
        return "EDIT: change the outfit to a dress"

    star._astrbot_generate = fake_astrbot_generate
    params = DrawParams(mode="img2img", prompt="把衣服换成裙子", presets=["维里奈"])
    translated = asyncio.run(star._prepare_llm_prompt(Event(), params, Event.message_str, "img2img"))

    assert translated == params.prompt == "change the outfit to a dress"
    assert params.img2img_edit_instruction_ready is True
    assert received and "用户对已有图片的编辑要求" in received[0][0]
    assert "Qwen Image Edit" in received[0][1]
    assert received[0][2] == 96


def test_img2img_editor_strips_reasoning_and_keeps_only_instruction() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    reasoning = (
        "我们需要保留用户的明确改动。最直接的英文是“remove her clothes”"
        "或“take off her clothes”。但不要扩写其它内容。"
    )
    assert ComfyUIAIStudio._clean_img2img_edit_instruction(reasoning) == "remove her clothes"
    assert ComfyUIAIStudio._clean_img2img_edit_instruction(
        "EDIT: change the outfit to a dress"
    ) == "change the outfit to a dress"
    assert ComfyUIAIStudio._clean_img2img_edit_instruction(
        "分析用户要求并保留原图, EDIT: change the outfit to a dress"
    ) == "change the outfit to a dress"
    assert ComfyUIAIStudio._clean_img2img_edit_instruction(
        '{"edit_instruction": "move her to the beach"}'
    ) == "move her to the beach"


def test_llm_img2img_ignores_tool_injected_default_denoise() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "把她的衣服换成裙子"

        def get_messages(self):
            return []

    star = object.__new__(ComfyUIAIStudio)
    captured: dict[str, float] = {}

    def fake_llm_params(prompt, mode, **kwargs):
        return DrawParams(
            mode=mode,
            prompt=prompt,
            denoise=float(kwargs.get("denoise") or 0.0),
        )

    async def fake_llm_execute(event, params, mode, *, require_image):
        captured["denoise"] = params.denoise
        return "ok"

    star._llm_params = fake_llm_params
    star._llm_execute = fake_llm_execute
    result = asyncio.run(
        star.llm_edit(Event(), prompt="把她的衣服换成裙子", denoise=0.6)
    )

    assert result == "ok"
    assert captured["denoise"] == 0.0


def test_llm_img2img_keeps_user_requested_denoise() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "把她的衣服换成裙子 重绘强度 0.6"

        def get_messages(self):
            return []

    star = object.__new__(ComfyUIAIStudio)
    captured: dict[str, float] = {}

    def fake_llm_params(prompt, mode, **kwargs):
        return DrawParams(
            mode=mode,
            prompt=prompt,
            denoise=float(kwargs.get("denoise") or 0.0),
        )

    async def fake_llm_execute(event, params, mode, *, require_image):
        captured["denoise"] = params.denoise
        return "ok"

    star._llm_params = fake_llm_params
    star._llm_execute = fake_llm_execute
    result = asyncio.run(
        star.llm_edit(Event(), prompt="把她的衣服换成裙子", denoise=0.6)
    )

    assert result == "ok"
    assert captured["denoise"] == 0.6


def test_img2img_final_prompt_uses_locked_edit_instruction() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_img2img_final_prompt_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.presets = PresetStore(Path(temp) / "presets.json")
        star.artist_presets = PresetStore(Path(temp) / "artist_presets.json")
        star.config = {
            "img2img_default_positive": "base edit quality",
            "img2img_default_negative": "",
            "artist_preset": "无",
        }
        params = DrawParams(
            mode="img2img",
            prompt="AstrBot 的原始长画面描述，包含不应再次送入 Qwen 的内容",
            img2img_edit_instruction_ready=True,
            img2img_edit_instruction="change the outfit to a dress",
        )
        positive, _, _ = asyncio.run(star._prompt_text(None, params, mode="img2img"))

    assert positive == "base edit quality, change the outfit to a dress"
    assert "原始长画面描述" not in positive


def test_draw_limit_counts_images_and_skips_configured_admins() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, UsageError

    class Event:
        def __init__(self, sender_id: str):
            self.sender_id = sender_id

        def get_sender_id(self):
            return self.sender_id

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "draw_limit_count": 2,
        "draw_limit_window_seconds": 3600,
        "draw_limit_admin_ids": "99999",
    }
    star._draw_limit_lock = asyncio.Lock()
    star._draw_limit_entries = {}

    async def verify():
        first = DrawParams(batch=1)
        second = DrawParams(batch=1)
        third = DrawParams(batch=1)
        await star._reserve_draw_limit(Event("10001"), first)
        await star._reserve_draw_limit(Event("10001"), second)
        try:
            await star._reserve_draw_limit(Event("10001"), third)
        except UsageError:
            pass
        else:
            raise AssertionError("第三张图片应被用户限额拦截")
        await star._release_draw_limit(first)
        await star._reserve_draw_limit(Event("10001"), third)
        await star._reserve_draw_limit(Event("99999"), DrawParams(batch=4))

    asyncio.run(verify())


def test_group_nsfw_blacklist_blocks_only_adult_prompt_markers() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    class Config:
        def get(self, key, default=None):
            return {"nsfw_group_blacklist": "12345\n67890"}.get(key, default)

    class GroupEvent:
        message_str = "请画一张裸体角色"

        def get_group_id(self):
            return "12345"

    class SafeGroupEvent:
        message_str = "请画一张海边少女"

        def get_group_id(self):
            return "12345"

    class PrivateEvent:
        message_str = "请画一张裸体角色"

        def get_group_id(self):
            return ""

    star = object.__new__(ComfyUIAIStudio)
    star.config = Config()

    assert "nsfw" in star._apply_group_safety_negative(GroupEvent(), "low quality")
    star._assert_nsfw_allowed(SafeGroupEvent(), SafeGroupEvent.message_str)
    star._assert_nsfw_allowed(PrivateEvent(), PrivateEvent.message_str)


def test_prompt_attachment_uses_final_prompt_values() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    params = DrawParams(prompt="海边少女")
    params.generated_positive = "masterpiece, seaside girl"
    params.generated_negative = "low quality, blurry"
    result = ComfyUIAIStudio._prompt_attachment(params)

    assert "本次绘图提示词" in result
    assert "masterpiece, seaside girl" in result
    assert "low quality, blurry" in result


def test_plugin_import_and_registration() -> None:
    assert ASTRBOT_DIR.is_dir(), f"AstrBot 目录不存在：{ASTRBOT_DIR}"
    with tempfile.TemporaryDirectory(prefix="astrbot_ai_studio_test_") as temp:
        os.environ["ASTRBOT_ROOT"] = temp
        from astrbot.api import AstrBotConfig
        from astrbot.core.provider.register import llm_tools
        from astrbot_plugin_comfyui_ai_studio import main as plugin_main

        schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
        config = AstrBotConfig(config_path=os.path.join(temp, "config.json"), schema=schema)
        context = FakeContext()
        star = plugin_main.ComfyUIAIStudio(context, config)
        star._set("ai_api_key", "secret-test-key")
        assert "ai_api_key" not in star._safe_config()
        assert star._safe_config()["ai_api_key_configured"] is True

        routes = {route for route, _, _ in context.apis}
        expected = {
            "/astrbot_plugin_comfyui_ai_studio/status",
            "/astrbot_plugin_comfyui_ai_studio/models",
            "/astrbot_plugin_comfyui_ai_studio/workflows",
            "/astrbot_plugin_comfyui_ai_studio/config",
            "/astrbot_plugin_comfyui_ai_studio/ai_models",
            "/astrbot_plugin_comfyui_ai_studio/presets",
            "/astrbot_plugin_comfyui_ai_studio/artist_presets",
            "/astrbot_plugin_comfyui_ai_studio/lora_info",
            "/astrbot_plugin_comfyui_ai_studio/open_folder",
            "/astrbot_plugin_comfyui_ai_studio/upload_lora",
            "/astrbot_plugin_comfyui_ai_studio/download_lora",
            "/astrbot_plugin_comfyui_ai_studio/download_lora_progress",
            "/astrbot_plugin_comfyui_ai_studio/delete_lora",
            "/astrbot_plugin_comfyui_ai_studio/recent",
        }
        assert expected <= routes
        tool_names = {tool.name for tool in llm_tools.func_list}
        assert {
            "comfyui_ai_studio_status",
            "comfyui_ai_studio_generate",
            "comfyui_ai_studio_edit",
            "comfyui_ai_studio_upscale",
        } <= tool_names

        params = star._llm_params(
            "夏空中的少女",
            "txt2img",
            negative_prompt="低质量，水印",
            lora="first.safetensors:0.8,second.safetensors:0.5",
            width=832,
            height=1216,
        )
        assert params.prompt == "夏空中的少女"
        assert params.negative == "低质量，水印"
        assert params.loras == ["first.safetensors:0.8", "second.safetensors:0.5"]
        assert (params.width, params.height) == (832, 1216)

        command_params = plugin_main.fill_params(
            "一名少女 --负面=低质量 --宽=832 --高=1216 --步数=24 --cfg=1",
            "txt2img",
        )
        llm_params = star._llm_params(
            "一名少女",
            "txt2img",
            negative_prompt="低质量",
            width=832,
            height=1216,
            steps=24,
            cfg=1,
            seed=123,
        )
        assert (llm_params.prompt, llm_params.negative) == (command_params.prompt, command_params.negative)
        assert (llm_params.width, llm_params.height, llm_params.steps, llm_params.cfg) == (
            command_params.width,
            command_params.height,
            command_params.steps,
            command_params.cfg,
        )
        assert llm_params.seed == 123

        float_json_params = star._llm_params(
            "一名少女",
            "txt2img",
            width=832.0,
            height=1216.0,
            steps=24.0,
            seed=123.0,
        )
        assert (
            float_json_params.width,
            float_json_params.height,
            float_json_params.steps,
            float_json_params.seed,
        ) == (832, 1216, 24, 123)

        star.presets.add("夏空", "ciaccona")
        assert star.presets.effective("夏空") == "ciaccona"
        star.presets.items["夏空"]["translated"] = "ciaccona, 1girl, solo"
        star.presets.save()
        star.presets.load()
        assert "1girl" in star.presets.effective("夏空")

        class FakeLoraClient:
            async def models(self, category):
                assert category == "loras"
                return ["first.safetensors"]

        star.lora_command_aliases["first.safetensors"] = "1号lora"
        assert star._lora_command_aliases("first.safetensors") == ["1号lora"]
        star.lora_command_aliases["first.safetensors"] = ["1号lora", "夏空LoRA"]
        assert star._resolve_lora("夏空LoRA", ["first.safetensors"]) == "first.safetensors"
        star._client = lambda: FakeLoraClient()
        llm_params = star._llm_params("使用1号lora画夏空海边", "txt2img")
        asyncio.run(star._extract_inline_loras(llm_params, "帮我使用1号lora画夏空"))
        star._extract_inline_presets(llm_params, "帮我使用1号lora画夏空")
        assert llm_params.loras == ["first.safetensors:0.8"]
        assert llm_params.presets == ["夏空"]
        assert "1号lora" not in llm_params.prompt
        assert "夏空" not in llm_params.prompt
        assert llm_params.auto_ai is False
        assert llm_params.ai is False
        structured = star._llm_params(
            "海边少女",
            "txt2img",
            lora="1号lora:0.6",
            preset="夏空",
        )
        assert structured.loras == ["1号lora:0.6"]
        assert structured.presets == ["夏空"]
        assert structured.ai is False
        assert structured.auto_ai is False

        star.lora_command_aliases["first.safetensors"] = ["1号lora", "夏空"]
        both = star._llm_params(
            "使用夏空画风和夏空 LoRA 画海边少女",
            "txt2img",
            lora="夏空:0.7",
            preset="夏空",
            ai=True,
        )
        asyncio.run(star._extract_inline_loras(both))
        star._extract_inline_presets(both)
        assert both.loras == ["夏空:0.7"]
        assert both.presets == ["夏空"]
        assert "夏空" not in both.prompt
        assert both.ai is False
        assert both.auto_ai is False

        async def fake_translate(event, text, **kwargs):
            return "translated scene"

        star._translate_prompt = fake_translate
        translated = plugin_main.fill_params("ai 夏空海边", "txt2img")
        star._extract_inline_presets(translated)
        positive, _, note = asyncio.run(star._prompt_text(None, translated))
        assert "ciaccona" in positive
        assert "translated scene" in positive
        assert note

        star.lora_command_aliases["first.safetensors"] = ["第一简称", "第二简称"]
        original_only = star._llm_params("英文画面提示", "txt2img")
        asyncio.run(star._extract_inline_loras(original_only, "帮我用第二简称画一张图"))
        star._extract_inline_presets(original_only, "帮我用第二简称画夏空")
        assert original_only.loras == ["first.safetensors:0.8"]
        assert original_only.presets == ["夏空"]
        assert original_only.prompt == "英文画面提示"


def test_anima_prompt_engineer_is_not_a_second_plugin() -> None:
    output_root = PLUGIN_DIR.parent
    assert not (output_root / "astrbot_plugin_anima_prompt_engineer").exists()


def test_original_workflow_is_adapted_without_mutation() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_original, load_api_workflow

    source_path = Path(r"E:\难工作流.json")
    assert source_path.is_file(), f"原工作流不存在：{source_path}"
    before = source_path.read_bytes()
    source = load_api_workflow(source_path)
    adapted, report = adapt_original(
        source,
        mode="hires",
        model_name="anima_baseV10.safetensors",
        loras=["example.safetensors:0.7", "second.safetensors:0.4"],
        positive="masterpiece, ciaccona",
        negative="low quality",
        image_name="uploaded.png",
        width=832,
        height=1216,
        steps=20,
        cfg=1.0,
        seed=123,
        denoise=0.25,
        scale=2.0,
        upscale_model="realesrganX4plusAnime_v1.pt",
        filename_prefix="astrbot/test",
    )

    assert source_path.read_bytes() == before
    assert adapted["1054"]["class_type"] == "Power Lora Loader (rgthree)"
    assert adapted["1071:1075"]["inputs"]["sage_attention"] == "disabled"
    assert adapted["1049"]["inputs"]["images"] == ["1413", 0]
    assert adapted["1339:1322"]["inputs"]["select"] == 2
    assert adapted["1413"]["inputs"]["select"] == 2
    assert adapted["592"]["inputs"]["image"] == "uploaded.png"
    assert all(node.get("class_type") for node in adapted.values() if isinstance(node, dict))
    assert report


def test_txt2img_and_img2img_use_sampling_output() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_original, load_api_workflow

    source = load_api_workflow(Path(r"E:\难工作流.json"))
    common = {
        "model_name": "anima_baseV10.safetensors",
        "loras": [],
        "positive": "1girl",
        "negative": "low quality",
        "image_name": "astrbot_blank.png",
        "width": 512,
        "height": 512,
        "steps": 10,
        "cfg": 1.0,
        "seed": 123,
        "denoise": 0.7,
        "scale": 2.0,
        "upscale_model": "realesrganX4plusAnime_v1.pt",
    }
    for mode in ("txt2img", "img2img"):
        adapted, _ = adapt_original(source, mode=mode, **common)
        assert adapted["1413"]["inputs"]["select"] == 1
        assert adapted["astrbot_txt2img_sampler"]["class_type"] == "KSampler"
        assert adapted["1413"]["inputs"]["input1"] == ["astrbot_txt2img_decode", 0]


def test_qwen_img2img_uses_the_standalone_api_workflow() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_qwen_img2img, load_workflow

    source_path = Path(r"E:\112121121.json")
    source = load_workflow(source_path)
    adapted, report = adapt_qwen_img2img(
        source,
        positive="a character standing by the sea",
        negative="low quality",
        image_name="input.png",
        unet_name="Qwen-Rapid-NSFW-v23_Q3_K.gguf",
        clip_name="Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf",
        vae_name="qwen_image_vae.safetensors",
        lora_name="qwen/Qwen-Image-Edit-F2P.safetensors",
        lora_strength=0.7,
        pre_cfg=True,
        seed=123,
        filename_prefix="astrbot/test-qwen",
    )

    assert adapted["11"]["inputs"]["image"] == "input.png"
    assert adapted["1"]["inputs"]["prompt"] == "a character standing by the sea"
    assert adapted["39"]["inputs"]["prompt"] == "low quality"
    assert adapted["174"]["inputs"]["lora_1"]["lora"] == "qwen/Qwen-Image-Edit-F2P.safetensors"
    assert adapted["174"]["inputs"]["lora_1"]["strength"] == 0.7
    assert adapted["152"]["inputs"]["seed"] == 123
    assert adapted["14"]["inputs"]["filename_prefix"] == "astrbot/test-qwen"
    assert adapted["119"]["inputs"]["pixels"] == ["158", 0]
    assert adapted["152"]["inputs"]["latent_image"] == ["119", 0]
    assert adapted["152"]["inputs"]["positive"] == ["1", 0]
    assert adapted["152"]["inputs"]["negative"] == ["39", 0]
    assert adapted["13"]["inputs"]["samples"] == ["152", 0]
    assert adapted["160"]["inputs"]["unet_name"] == "Qwen-Rapid-NSFW-v23_Q3_K.gguf"
    assert adapted["161"]["inputs"]["clip_name"] == "Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf"
    assert adapted["10"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors"
    assert "image2" not in adapted["1"]["inputs"]
    assert all(node.get("class_type") for node in adapted.values())
    assert isinstance(report, list)


def test_ai_and_comfy_compatibility_fallbacks() -> None:
    from astrbot_plugin_comfyui_ai_studio.ai import AITranslator
    from astrbot_plugin_comfyui_ai_studio.comfy import ComfyClient

    assert AITranslator._content_value({"choices": [{"message": {"content": "one, two"}}]}) == "one, two"
    assert AITranslator._content_value({"choices": [{"message": {"reasoning_content": "three"}}]}) == "three"
    assert AITranslator._content_value({"choices": [{"text": "four"}]}) == "four"
    object_info = {
        "LoraLoaderModelOnly": {
            "input": {"required": {"lora_name": [["qwen\\edit.safetensors"], {}]}}
        },
        "UnetLoaderGGUF": {
            "input": {"required": {"unet_name": [["qwen.gguf"], {}]}}
        },
    }
    assert ComfyClient.models_from_object_info(object_info, "loras") == ["qwen\\edit.safetensors"]
    assert ComfyClient.models_from_object_info(object_info, "unet_gguf") == ["qwen.gguf"]
