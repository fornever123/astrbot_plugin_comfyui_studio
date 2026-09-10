from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path


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
    assert "版本 v0.6.4" in page


def test_anima_and_character_features_are_outside_comfyui_plugin() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert "on_llm_request" not in source
    assert "api_characters" not in source
    assert not (PLUGIN_DIR / "anima3.py").exists()
    assert not (PLUGIN_DIR / "characters.py").exists()
    assert not (PLUGIN_DIR / "pages" / "characters").exists()


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
    for view in ("workflow", "ai", "presets", "artist"):
        assert f'data-view-tab="{view}"' in page
        assert f'data-view-panel="{view}"' in page
    assert "initViews" in app
    assert 'id="loadAiModels"' in page
    assert 'id="testAiConnection"' in page
    assert 'id="aiConnectionStatus"' in page
    assert 'data-lora-action="save-lora"' in app
    assert 'data-lora-action="delete-lora"' in app
    assert 'data-lora-action="add-command-alias"' in app
    assert 'command_aliases:' in app
    assert 'post(`${API}/download_lora`' in app
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
    assert 'id="saveDrawSettings"' in page
    assert 'id="seed"' in page
    assert "initLoraMode" in app
    assert "sourcePathElement" in app


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


def test_lora_trigger_words_are_separate_from_named_presets() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.civitai_overrides = {
        "demo.safetensors": {"trigger_words": ["demo trigger", "character_trigger"]}
    }
    star.civitai_cache = {}
    assert star._lora_trigger_words("demo.safetensors") == ["demo trigger", "character_trigger"]


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
    from astrbot.api.message_components import Nodes, Plain

    class Event:
        def get_self_id(self):
            return "123456"

    star = object.__new__(ComfyUIAIStudio)
    chain = star._forward_result_chain(Event(), "绘图完成", [Path("one.png"), Path("two.png")])

    assert len(chain) == 1
    assert isinstance(chain[0], Nodes)
    assert len(chain[0].nodes) == 3
    assert isinstance(chain[0].nodes[0].content[0], Plain)
    assert chain[0].nodes[0].content[0].text == "绘图完成"
    assert [node.uin for node in chain[0].nodes] == ["123456", "123456", "123456"]
    flattened = star._flatten_forward_chain(chain)
    assert isinstance(flattened[0], Plain)
    assert len(flattened) == 3


def test_civitai_trigger_words_sync_to_command_alias_presets() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_lora_preset_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.presets = PresetStore(Path(temp) / "presets.json")
        asyncio.run(star._sync_lora_trigger_presets(
            "demo.safetensors",
            aliases=["一号lora", "角色简称"],
            trigger_words=["character_trigger", "red_hair"],
        ))
        assert star.presets.items["一号lora"]["content"] == "character_trigger, red_hair"
        assert star.presets.items["角色简称"]["content"] == "character_trigger, red_hair"
        assert star.presets.items["一号lora"]["_auto_source"] == "civitai_lora"

        asyncio.run(star._sync_lora_trigger_presets(
            "demo.safetensors",
            aliases=["新简称"],
            trigger_words=["new_trigger"],
        ))
        assert "一号lora" not in star.presets.items
        assert star.presets.items["新简称"]["content"] == "new_trigger"

        star.presets.add("手动简称", "user_owned")
        asyncio.run(star._sync_lora_trigger_presets(
            "demo.safetensors",
            aliases=["手动简称"],
            trigger_words=["should_not_replace"],
        ))
        assert star.presets.items["手动简称"]["content"] == "user_owned"


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


def test_independent_prompt_plugins_have_their_own_boundaries() -> None:
    output_root = PLUGIN_DIR.parent
    character_dir = output_root / "astrbot_plugin_wuthering_waves_character_knowledge"
    anima_dir = output_root / "astrbot_plugin_anima_prompt_engineer"
    assert "astrbot_plugin_wuthering_waves_character_knowledge" in (
        character_dir / "metadata.yaml"
    ).read_text(encoding="utf-8")
    assert "astrbot_plugin_anima_prompt_engineer" in (
        anima_dir / "metadata.yaml"
    ).read_text(encoding="utf-8")
    assert (character_dir / "pages" / "characters" / "index.html").is_file()
    assert (anima_dir / "knowledge" / "提示词模版.txt").stat().st_size > 100000
    assert "on_llm_request" in (character_dir / "main.py").read_text(encoding="utf-8")
    assert "on_llm_request" in (anima_dir / "main.py").read_text(encoding="utf-8")


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
