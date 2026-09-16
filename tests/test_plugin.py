from __future__ import annotations

import asyncio
import inspect
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
    assert "版本 v1.1.1" in page


@pytest.mark.parametrize(
    "message",
    [
        "你有什么功能",
        "你有什么画图功能",
        "这个插件能做什么",
        "机器人会画什么",
        "画图功能有哪些",
    ],
)
def test_feature_query_uses_one_sentence_summary(message: str) -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    class Event:
        message_str = message

    class Request:
        system_prompt = "当前人格"

    request = Request()
    asyncio.run(ComfyUIAIStudio.on_llm_request(object.__new__(ComfyUIAIStudio), Event(), request))
    assert "只回复这一句" in request.system_prompt
    assert "支持文生图、图生图、高清放大、洗图、扩图、多角度处理" in request.system_prompt
    assert "Qwen/Flux2" not in request.system_prompt


def test_llm_draw_sends_start_reply_without_event_result() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        def __init__(self) -> None:
            self.results: list[str] = []
            self.sent: list[str] = []

        def plain_result(self, text: str) -> str:
            return text

        def set_result(self, result: str) -> None:
            self.results.append(result)

        async def send(self, result: str) -> None:
            self.sent.append(result)

    async def never_finishes(*args, **kwargs):
        await asyncio.Future()

    star = object.__new__(ComfyUIAIStudio)
    star.config = {"draw_start_reply": "{mode}开始：{prompt}"}
    star._get = lambda key, default="": star.config.get(key, default)
    star._tasks = set()
    star._run = never_finishes
    event = Event()

    async def run() -> None:
        await star._llm_draw(event, DrawParams(prompt="一只猫"), "txt2img")
        await asyncio.sleep(0)
        for task in list(star._tasks):
            task.cancel()
        if star._tasks:
            await asyncio.gather(*list(star._tasks), return_exceptions=True)

    asyncio.run(run())
    assert event.results == []
    assert event.sent == ["文生图开始：一只猫"]


def test_llm_tool_arguments_are_unwrapped_and_prompt_can_fall_back_to_event() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "帮我画一只猫"

    star = object.__new__(ComfyUIAIStudio)
    captured: dict[str, object] = {}
    star._event_has_image_hint = lambda event: False
    star._looks_like_edit_request = lambda event, prompt: False

    def fake_llm_params(prompt, mode, **kwargs):
        captured["prompt"] = prompt
        captured["mode"] = mode
        captured["kwargs"] = kwargs
        return DrawParams(prompt=str(prompt))

    async def fake_llm_execute(event, params, mode, *, require_image):
        captured["require_image"] = require_image
        return "已提交"

    star._llm_params = fake_llm_params
    star._llm_execute = fake_llm_execute

    result = asyncio.run(
        star.llm_generate(
            Event(),
            arguments='{"arguments": {"prompt": "一只猫", "steps": 12}}',
        )
    )
    assert result == "已提交"
    assert captured["prompt"] == "一只猫"
    assert captured["mode"] == "txt2img"
    assert captured["kwargs"]["steps"] == 12

    asyncio.run(star.llm_generate(Event(), arguments={}))
    assert captured["prompt"] == "帮我画一只猫"


def test_structured_llm_prompt_never_runs_txt2img_ai_a_second_time() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    with tempfile.TemporaryDirectory(prefix="astrbot_llm_prompt_guard_") as root:
        from astrbot_plugin_comfyui_ai_studio.prompting import PresetStore

        star = object.__new__(ComfyUIAIStudio)
        star.presets = PresetStore(Path(root) / "presets.json")
        star.artist_presets = PresetStore(Path(root) / "artists.json")
        star._get = lambda key, default="": {
            "artist_preset": "无",
            "default_positive": "",
            "default_negative": "",
        }.get(key, default)

        params = star._llm_params("夏空在海边", "txt2img", ai=True)
        assert params.ai is False
        assert params.llm_invocation is True
        # 模拟上游模型错误保留 ai=true；结构化 LLM 标记仍必须阻止二次 AI。
        params.ai = True

        async def unexpected_ai(*args, **kwargs):
            raise AssertionError("结构化 LLM 绘图不应再次调用文生图 AI")

        star._translate_prompt = unexpected_ai
        positive, _, _ = asyncio.run(star._prompt_text(None, params, mode="txt2img"))
        assert "夏空在海边" in positive


def test_all_llm_tools_accept_hidden_arguments_compatibility_parameter() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    for method_name in (
        "llm_status",
        "llm_presets",
        "llm_generate",
        "llm_edit",
        "llm_edit_flux2",
        "llm_upscale",
    ):
        signature = inspect.signature(getattr(ComfyUIAIStudio, method_name))
        assert "arguments" in signature.parameters
        doc = getattr(ComfyUIAIStudio, method_name).__doc__ or ""
        assert "arguments(object)" not in doc


def test_img2img_command_path_yields_fixed_start_reply_before_background_task() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    marker = 'async def _command_img2img_mode('
    start = source.index(marker)
    end = source.index('\n    @filter.command("图生图"', start)
    command_body = source[start:end]
    assert 'start_reply = await self._draw_start_reply_for_command(event, params, mode)' in command_body
    assert 'yield event.plain_result(start_reply)' in command_body
    assert command_body.index('yield event.plain_result(start_reply)') < command_body.index('self._schedule_draw(')


def test_draw_completion_reply_is_fixed_and_does_not_call_ai() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "draw_reply_mode": "astrbot",
        "draw_reply_custom": "{mode}已完成，共 {count} 张。",
    }
    star._get = lambda key, default="": star.config.get(key, default)

    async def unexpected_ai_call(*args, **kwargs):
        raise AssertionError("完成回复不应再次调用 AI")

    star._astrbot_generate = unexpected_ai_call
    params = DrawParams(prompt="一只猫")
    result = asyncio.run(star._draw_reply(None, params, "txt2img", 1))

    assert result == "文生图已完成，共 1 张。"


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
    # 旧版在 app.js 里读一个早已从页面移除的元素；这段死代码已清理。
    assert "sourcePathElement" not in app
    # 加速 LoRA 管线（界面、保存、提示）已整体移除。
    assert "img2img_accel_lora_enabled" not in app
    assert "updateQwenAccelHint" not in app
    assert 'data-view-tab="paths"' in page
    assert "saveTxt2ImgSettings" in app
    assert "payload.model_name" in app
    assert "请先获取模型列表并测试连接" in app


def test_console_separates_normal_and_style_lora_regions() -> None:
    for relative_page, relative_app, relative_css in (
        ("index.html", "app.js", "style.css"),
        ("pages/console/index.html", "pages/console/app.js", "pages/console/style.css"),
    ):
        page = (PLUGIN_DIR / relative_page).read_text(encoding="utf-8")
        app = (PLUGIN_DIR / relative_app).read_text(encoding="utf-8")
        css = (PLUGIN_DIR / relative_css).read_text(encoding="utf-8")

        normal_start = page.index('<section class="band lora-management-band">')
        style_start = page.index('<section class="band style-lora-band">')
        assert normal_start < style_start
        normal_region = page[normal_start:page.index("</section>", normal_start)]
        # 标题内嵌了图标 svg，因此只校验标题文案本身，避免图标改动误伤这里。
        assert "普通 LoRA</h2>" in normal_region
        assert "style-lora-band" not in normal_region

        assert 'const normalItems = allItems.filter(item => (item.category || "未分类") !== "画风");' in app
        assert 'const styleCandidates = items.filter(item => (item.category || "未分类") === "画风");' in app
        # 卡片改为“简洁/展开”两态布局后，普通和画风各自使用独立的网格容器。
        assert ".lora-grid {" in css and "display: grid;" in css
        assert ".lora-card {" in css and "flex-direction: column;" in css
        assert ".style-lora-grid {" in css


def test_help_image_and_download_order_are_available() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    requirements = (PLUGIN_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "def _help_image_path(self)" in source
    assert "Image.fromFileSystem(str(image_path))" in source
    assert "def _save_lora_download_order" in source
    assert "lora_download_order[filename]" in source
    assert "pillow>=10" in requirements.lower()
    assert "AstrBot-Anima-Studio/1.0.0" in (PLUGIN_DIR / "translation.py").read_text(encoding="utf-8")
    assert "/洗图" in source and "/扩图" in source and "/多角度" in source
    assert "其余忽略" in source


def test_config_and_delete_paths_are_stateful_and_windows_safe() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert '"lora_list": list(self._get("lora_list", []) or [])' in source
    assert 'normalized_filename = raw_filename.replace("\\\\", "/")' in source
    assert 'async def api_delete_lora(self)' in source


def test_style_random_mode_falls_back_to_classified_loras() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "style_lora_list": [],
        "style_lora_aliases": {
            "style_a.safetensors": "画风1",
            "style_b.safetensors": "画风2",
        },
        "style_lora_weights": {},
        "style_lora_mode": "random",
        "style_lora_random_count": 1,
    }
    star.lora_categories = {
        "style_a.safetensors": "画风",
        "style_b.safetensors": "画风",
    }
    star.lora_category_entries = {"画风": True}
    star.lora_download_order = {}
    available = ["style_a.safetensors", "style_b.safetensors"]

    star._set = lambda key, value: star.config.__setitem__(key, value)
    star._save_config = lambda: None

    classified = star._style_lora_entries(available, selected_only=False)
    candidates = star._style_lora_candidates_without_persistent(
        classified,
        [],
        available,
        classified,
    )

    assert {item["file_name"] for item in candidates} == set(available)
    assert len(star._select_style_loras(candidates)) == 1


def test_style_usage_text_reports_random_selection_source() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    star = object.__new__(ComfyUIAIStudio)
    text = star._style_lora_usage_text(
        DrawParams(
            style_loras_used=[
                {
                    "alias": "画风2",
                    "display_name": "示例画风",
                    "file_name": "style_b.safetensors",
                    "source": "随机选择",
                    "weight": 0.8,
                }
            ]
        )
    )

    assert "画风2" in text
    assert "示例画风" in text
    assert "随机选择" in text


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


def test_llm_fuzzy_lora_alias_keeps_private_preset_and_command_exactness() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Presets:
        items = {}

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "style_lora_list": [],
        "style_lora_aliases": {},
        "style_lora_weights": {},
    }
    star.lora_aliases = {}
    star.lora_categories = {}
    star.lora_category_entries = {}
    star.lora_command_aliases = {
        "character.safetensors": [],
        "style-4.safetensors": ["画风4"],
        "style-5.safetensors": ["画风5"],
    }
    star.lora_presets = {
        "character.safetensors": [
            {"tag": "小小爱", "content": "little Aemeath (WuWa)"},
        ],
    }
    star.presets = Presets()

    async def available(*, force=False):
        return [
            "character.safetensors",
            "style-4.safetensors",
            "style-5.safetensors",
        ]

    star._available_loras = available

    fuzzy_params = DrawParams(prompt="a girl")
    asyncio.run(
        star._extract_inline_loras(
            fuzzy_params,
            "帮我画小小艾",
            allow_fuzzy=True,
        )
    )
    assert fuzzy_params.loras == ["character.safetensors:0.8"]
    assert fuzzy_params.lora_preset_tags == {
        "character.safetensors": ["小小爱"],
    }

    exact_params = DrawParams(prompt="a girl")
    asyncio.run(
        star._extract_inline_loras(
            exact_params,
            "请使用画风4绘图",
            allow_fuzzy=True,
        )
    )
    assert exact_params.loras == ["style-4.safetensors:0.8"]

    command_params = DrawParams(prompt="a girl")
    asyncio.run(
        star._extract_inline_loras(
            command_params,
            "帮我画小小艾",
        )
    )
    assert command_params.loras == []


def test_llm_fuzzy_alias_handles_multiword_english_without_partial_word_match() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    matches = ComfyUIAIStudio._fuzzy_lora_alias_matches(
        "please draw little Aemeat in a scene",
        [("little Aemeath", "character.safetensors")],
    )
    assert matches and matches[0][1] == "character.safetensors"
    assert ComfyUIAIStudio._fuzzy_lora_alias_matches(
        "please draw artist in a scene",
        [("art", "style.safetensors")],
    ) == []


def test_style_lora_candidates_are_limited_to_style_category() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "style_lora_list": [
            "style.safetensors:0.7",
            "character.safetensors:0.8",
            "unclassified.safetensors:0.9",
        ],
        "style_lora_weights": {},
    }
    star.lora_categories = {
        "style.safetensors": "画风",
        "character.safetensors": "鸣潮角色",
    }
    star.lora_category_entries = {"画风": True, "鸣潮角色": True}

    entries = star._style_lora_entries(
        ["style.safetensors", "character.safetensors", "unclassified.safetensors"]
    )
    assert entries == [{"file_name": "style.safetensors", "weight": 0.7}]


def test_style_lora_alias_is_triggerable_even_when_not_random_candidate() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "style_lora_list": [],
        "style_lora_aliases": {"style.safetensors": "画风4"},
        "style_lora_weights": {},
    }
    star.lora_categories = {"style.safetensors": "画风"}
    star.lora_category_entries = {"画风": True}
    star.lora_aliases = {}
    star.lora_command_aliases = {"style.safetensors": []}
    star.lora_presets = {
        "style.safetensors": [{"tag": "画风预设", "content": "style trigger"}],
    }

    available = ["style.safetensors", "other.safetensors"]
    assert star._style_lora_entries(available) == []
    assert star._style_lora_entries(available, selected_only=False) == [
        {"file_name": "style.safetensors", "weight": 0.8}
    ]
    assert star._resolve_lora("画风4", available) == "style.safetensors"
    assert star._resolve_lora("画风预设", available) == "style.safetensors"


def test_style_disable_control_is_per_task_and_does_not_match_keep_style_text() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    star = object.__new__(ComfyUIAIStudio)
    params = DrawParams(prompt="不用画风 夏空")
    assert star._extract_style_lora_disable(params, params.prompt) is True
    assert params.style_lora_disabled is True
    assert params.prompt == "夏空"

    keep_params = DrawParams(prompt="不要改变画风")
    assert star._extract_style_lora_disable(keep_params, keep_params.prompt) is False
    assert keep_params.style_lora_disabled is False
    assert keep_params.prompt == "不要改变画风"


def test_persistent_style_lora_does_not_consume_random_style_count() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    entries = [
        {"file_name": "persistent.safetensors", "weight": 0.8},
        {"file_name": "candidate-a.safetensors", "weight": 0.8},
        {"file_name": "candidate-b.safetensors", "weight": 0.8},
    ]
    available = [item["file_name"] for item in entries]
    pool = star._style_lora_candidates_without_persistent(
        entries,
        ["persistent.safetensors:0.8"],
        available,
        entries,
    )
    assert [item["file_name"] for item in pool] == [
        "candidate-a.safetensors",
        "candidate-b.safetensors",
    ]


def test_explicit_lora_alias_only_injects_the_matching_private_preset() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    star = object.__new__(ComfyUIAIStudio)
    star.lora_presets = {
        "style.safetensors": [
            {"tag": "画风预设", "content": "style trigger"},
            {"tag": "另一个预设", "content": "unrelated trigger"},
        ],
    }
    params = DrawParams(mode="txt2img", prompt="画风简称")
    star._remember_lora_preset_match(params, "style.safetensors", "画风简称")
    assert params.lora_preset_tags == {"style.safetensors": []}
    assert star._lora_prompt_values_for_task(params, "style.safetensors") == []

    star._remember_lora_preset_match(params, "style.safetensors", "画风预设")
    assert star._lora_prompt_values_for_task(params, "style.safetensors") == ["style trigger"]


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


def test_duplicate_structured_and_raw_image_segments_are_deduplicated() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_duplicate_input_test_") as root:
            base = Path(root)
            source = base / "same-image.png"
            source.write_bytes(b"same-image-data")
            segment = {"type": "image", "data": {"file": str(source)}}

            class Message:
                raw_message = {"message": [segment]}

            class Event:
                message_obj = Message()

                def get_messages(self):
                    return [segment]

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            images = await star._extract_images(Event())

            assert len(images) == 1
            assert len(list(star.input_cache_dir.iterdir())) == 1

    asyncio.run(run())


def test_forward_component_is_fetched_for_img2img() -> None:
    from astrbot.api.message_components import Forward
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_forward_input_test_") as root:
            base = Path(root)
            source = base / "forward.png"
            source.write_bytes(b"fake-forward-image-data")

            class Api:
                calls: list[tuple[str, dict[str, object]]] = []

                async def call_action(self, action: str, **params):
                    self.calls.append((action, params))
                    return {
                        "data": {
                            "messages": [
                                {
                                    "sender": {"nickname": "绘图机器人"},
                                    "content": [
                                        {"type": "image", "data": {"file": str(source)}}
                                    ],
                                }
                            ]
                        }
                    }

            class Bot:
                def __init__(self):
                    self.api = Api()

            class Event:
                bot = Bot()

                def get_messages(self):
                    return [Forward(id="forward-message-1")]

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            event = Event()
            assert star._event_has_image_hint(event) is True
            images = await star._extract_images(event)

            assert len(images) == 1
            assert Path(images[0]).is_file()
            assert Path(images[0]).read_bytes() == source.read_bytes()
            assert event.bot.api.calls[0][0] == "get_forward_msg"

    asyncio.run(run())


def test_reply_forward_is_fetched_when_adapter_keeps_only_reply_id() -> None:
    from astrbot.api.message_components import Reply
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_reply_forward_test_") as root:
            base = Path(root)
            source = base / "reply-forward.jpg"
            source.write_bytes(b"reply-forward-image-data")

            class Api:
                calls: list[str] = []

                async def call_action(self, action: str, **params):
                    self.calls.append(action)
                    if action == "get_msg":
                        return {
                            "data": {
                                "message": [
                                    {"type": "forward", "data": {"id": "forward-2"}}
                                ]
                            }
                        }
                    return {
                        "data": {
                            "messages": [
                                {
                                    "sender": {"nickname": "绘图机器人"},
                                    "content": [
                                        {"type": "image", "data": {"file": str(source)}}
                                    ],
                                }
                            ]
                        }
                    }

            class Bot:
                def __init__(self):
                    self.api = Api()

            class Event:
                bot = Bot()

                def get_messages(self):
                    return [Reply(id="quoted-message-1", chain=[])]

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            event = Event()

            assert star._event_has_image_hint(event) is True
            images = await star._extract_images(event)

            assert len(images) == 1
            assert Path(images[0]).read_bytes() == source.read_bytes()
            assert event.bot.api.calls[:2] == ["get_msg", "get_forward_msg"]

    asyncio.run(run())


def test_explicit_reply_forward_wins_over_a_different_direct_image() -> None:
    from astrbot.api.message_components import Image, Reply
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_reply_priority_test_") as root:
            base = Path(root)
            referenced = base / "referenced.jpg"
            wrong_attachment = base / "wrong-attachment.jpg"
            referenced.write_bytes(b"the-image-inside-the-quoted-forward")
            wrong_attachment.write_bytes(b"a-different-current-attachment")

            class Api:
                async def call_action(self, action: str, **params):
                    if action == "get_msg":
                        return {
                            "data": {
                                "message": [
                                    {"type": "forward", "data": {"id": "forward-target"}}
                                ]
                            }
                        }
                    return {
                        "data": {
                            "messages": [
                                {
                                    "sender": {"nickname": "ComfyUI 绘图"},
                                    "content": [
                                        {"type": "image", "data": {"file": str(referenced)}}
                                    ],
                                }
                            ]
                        }
                    }

            class Bot:
                api = Api()

            class Event:
                bot = Bot()

                def get_messages(self):
                    return [
                        Reply(id="quoted-message", chain=[]),
                        Image(file=str(wrong_attachment)),
                    ]

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            event = Event()

            images = await star._extract_images(event)

            assert len(images) == 1
            assert Path(images[0]).read_bytes() == referenced.read_bytes()
            assert Path(images[0]).read_bytes() != wrong_attachment.read_bytes()

    asyncio.run(run())


def test_llm_edit_never_uses_recent_image_when_explicit_reference_cannot_be_read() -> None:
    from astrbot.api.message_components import Reply
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "把引用的图换成白色裙子"

        def get_messages(self):
            return [Reply(id="missing-quoted-message", chain=[])]

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_no_recent_fallback_test_") as root:
            recent = Path(root) / "recent.png"
            recent.write_bytes(b"recent-output")
            star = object.__new__(ComfyUIAIStudio)
            star.config = {}
            star.last_images = {"origin": str(recent)}
            star._get = lambda key, default="": default
            star._origin = lambda event: "origin"
            star._extract_inline_presets = lambda params, text="": None

            async def fake_loras(params, text="", allow_fuzzy=False):
                return None

            async def fake_reserve(event, params):
                return None

            async def fake_prepare(event, params, original_text, mode):
                return None

            async def fake_queue(params, mode, event=None):
                return None

            async def should_not_draw(*args, **kwargs):
                raise AssertionError("不应在引用图片读取失败时提交绘图")

            star._extract_inline_loras = fake_loras
            star._reserve_draw_limit = fake_reserve
            star._prepare_llm_prompt = fake_prepare
            star._check_draw_queue = fake_queue
            star._extract_images = lambda event: _empty_images()
            star._llm_draw = should_not_draw

            result = await star._llm_execute(
                Event(),
                DrawParams(mode="img2img", prompt="换成白色裙子"),
                "img2img",
                require_image=True,
            )
            assert "未能读取你引用的图片" in str(result)
            assert "最近出图" not in str(result)

    async def _empty_images():
        return []

    asyncio.run(run())


def test_llm_img2img_requires_current_message_image_instead_of_recent_output() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Event:
        message_str = "把衣服换成白色裙子"

        def get_messages(self):
            return []

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_no_implicit_img2img_test_") as root:
            recent = Path(root) / "recent.png"
            recent.write_bytes(b"recent-output")
            star = object.__new__(ComfyUIAIStudio)
            star.config = {}
            star.last_images = {"origin": str(recent)}
            star._get = lambda key, default="": default
            star._origin = lambda event: "origin"
            star._extract_inline_presets = lambda params, text="": None

            async def fake_loras(params, text="", allow_fuzzy=False):
                return None

            async def should_not_reserve(*args, **kwargs):
                raise AssertionError("没有当前消息图片时不应进入绘图限额或后台任务")

            star._extract_inline_loras = fake_loras
            star._extract_images = lambda event: _empty_images()
            star._reserve_draw_limit = should_not_reserve

            result = await star._llm_execute(
                Event(),
                DrawParams(mode="img2img", prompt="把衣服换成白色裙子"),
                "img2img",
                require_image=True,
            )
            assert "同一条消息附图" in str(result)
            assert "最近出图" not in str(result)

    async def _empty_images():
        return []

    asyncio.run(run())


def test_raw_onebot_reply_forward_is_used_when_components_are_missing() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_raw_forward_test_") as root:
            base = Path(root)
            source = base / "raw-forward.webp"
            source.write_bytes(b"raw-forward-image")

            class Api:
                async def call_action(self, action: str, **params):
                    if action == "get_msg":
                        return {
                            "data": {
                                "message": [
                                    {"type": "forward", "data": {"id": "raw-forward-id"}}
                                ]
                            }
                        }
                    return {
                        "data": {
                            "messages": [
                                {
                                    "content": [
                                        {"type": "image", "data": {"file": str(source)}}
                                    ]
                                }
                            ]
                        }
                    }

            class Bot:
                api = Api()

            class Message:
                raw_message = {
                    "message": [
                        {"type": "reply", "data": {"id": "raw-quoted-message"}}
                    ]
                }

            class Event:
                bot = Bot()
                message_obj = Message()

                def get_messages(self):
                    return []

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            event = Event()

            assert star._event_has_image_hint(event) is True
            images = await star._extract_images(event)

            assert len(images) == 1
            assert Path(images[0]).read_bytes() == source.read_bytes()

    asyncio.run(run())


def test_event_level_reply_forward_is_used_for_img2img() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_event_reply_forward_test_") as root:
            base = Path(root)
            source = base / "event-reply-forward.png"
            source.write_bytes(b"event-level-reply-forward-image")

            class Api:
                calls: list[str] = []

                async def call_action(self, action: str, **params):
                    self.calls.append(action)
                    if action == "get_msg":
                        return {
                            "data": {
                                "message": [
                                    {"type": "forward", "data": {"id": "event-forward"}}
                                ]
                            }
                        }
                    return {
                        "data": {
                            "messages": [
                                {
                                    "content": [
                                        {"type": "image", "data": {"file": str(source)}}
                                    ]
                                }
                            ]
                        }
                    }

            class Bot:
                def __init__(self):
                    self.api = Api()

            class Event:
                bot = Bot()
                # Some adapters expose the reply target on the event instead
                # of adding a Reply component to get_messages().
                reply = {"id": "event-reply"}

                def get_messages(self):
                    return []

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            event = Event()

            assert star._event_has_image_hint(event) is True
            assert star._event_has_explicit_image_reference(event) is True
            images = await star._extract_images(event)

            assert len(images) == 1
            assert Path(images[0]).read_bytes() == source.read_bytes()
            assert event.bot.api.calls[:2] == ["get_msg", "get_forward_msg"]

    asyncio.run(run())


def test_quoted_forward_image_is_added_to_the_astrbot_llm_request() -> None:
    from astrbot.api.message_components import Forward
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    async def run() -> None:
        with tempfile.TemporaryDirectory(prefix="astrbot_llm_forward_image_test_") as root:
            base = Path(root)
            source = base / "llm-forward.png"
            source.write_bytes(b"llm-forward-image")

            class Api:
                async def call_action(self, action: str, **params):
                    return {
                        "data": {
                            "messages": [
                                {
                                    "content": [
                                        {"type": "image", "data": {"file": str(source)}}
                                    ]
                                }
                            ]
                        }
                    }

            class Bot:
                api = Api()

            class Event:
                bot = Bot()

                def get_messages(self):
                    return [Forward(id="llm-forward-id")]

            class Request:
                image_urls: list[str] = []

            star = object.__new__(ComfyUIAIStudio)
            star.input_cache_dir = base / "input_cache"
            star.input_cache_dir.mkdir()
            request = Request()

            added = await star._inject_event_images_to_llm_request(Event(), request)

            assert len(added) == 1
            assert len(request.image_urls) == 1
            assert Path(request.image_urls[0]).read_bytes() == source.read_bytes()

    asyncio.run(run())


def test_quoted_forward_send_falls_back_without_retrying_empty_reply() -> None:
    from astrbot.api.message_components import Nodes, Plain, Reply, Node
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    class Event:
        def __init__(self):
            self.sent: list[list[object]] = []

        def chain_result(self, chain):
            return chain

        async def send(self, chain):
            self.sent.append(chain)
            if len(self.sent) == 1:
                raise TimeoutError("WebSocket API call timeout")

    star = object.__new__(ComfyUIAIStudio)
    event = Event()
    chain = [
        Reply(id="original-message"),
        Nodes([Node(uin="1", name="ComfyUI 绘图", content=[Plain("结果")])]),
    ]

    asyncio.run(star._send_forward_chain(event, chain))

    assert len(event.sent) == 1
    assert len(event.sent[0]) == 1
    assert isinstance(event.sent[0][0], Nodes)
    assert all(not isinstance(item, Reply) for sent in event.sent for item in sent)


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


def test_forward_run_places_prompt_attachment_after_images() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams
    from astrbot.api.message_components import Image, Nodes, Plain

    class Event:
        def get_self_id(self):
            return "123456"

    class Config:
        def get(self, key, default=None):
            return {
                "draw_delivery_mode": "forward",
                "draw_attach_prompt": True,
            }.get(key, default)

    star = object.__new__(ComfyUIAIStudio)
    star.config = Config()
    star.semaphore = None
    params = DrawParams(prompt="海边的少女")
    params.generated_positive = "masterpiece, seaside girl"
    params.generated_negative = "low quality"

    async def fake_generate(*args, **kwargs):
        return [Path("one.png")]

    star._generate = fake_generate
    chain = asyncio.run(star._run(Event(), params, "txt2img"))

    assert len(chain) == 1
    assert isinstance(chain[0], Nodes)
    assert isinstance(chain[0].nodes[0].content[0], Image)
    assert isinstance(chain[0].nodes[-1].content[0], Plain)
    assert "本次绘图提示词" in chain[0].nodes[-1].content[0].text


def test_completion_reply_uses_fixed_template_without_llm() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class Config:
        def get(self, key, default=None):
            return {
                "draw_reply_mode": "astrbot",
                "draw_reply_custom": "{mode}已完成，共 {count} 张。",
            }.get(key, default)

    star = object.__new__(ComfyUIAIStudio)
    star.config = Config()

    async def unexpected_ai_call(*args, **kwargs):
        raise AssertionError("固定完成回复不应调用 AI")

    star._astrbot_generate = unexpected_ai_call
    params = DrawParams(prompt="夏空在海边")
    result = asyncio.run(star._draw_reply(None, params, "txt2img", 2))

    assert result == "文生图已完成，共 2 张。"


def test_completion_llm_reply_rejects_wrong_status() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    assert ComfyUIAIStudio._normalize_draw_reply("正在生成，请稍候", "txt2img", 1) == ""


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
    assert ComfyUIAIStudio._civitai_reference(
        "https://civitai.red/models/123456?modelVersionId=789012"
    ) == ("123456", "789012")
    assert ComfyUIAIStudio._civitai_reference(
        "https://civital.red/api/download/models/789012"
    ) == ("", "789012")
    assert ComfyUIAIStudio._safe_lora_filename(
        "bad:name?.safetensors", "fallback.safetensors"
    ) == "bad_name_.safetensors"
    assert ComfyUIAIStudio._safe_lora_filename(
        "CON.safetensors", "fallback.safetensors"
    ) == "_CON.safetensors"


def test_civitai_compatible_site_configuration_and_search_url() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = {"civitai_base_url": "https://civitai.red/api/v1/"}
    star._get = lambda key, default="": star.config.get(key, default)

    assert star._civitai_base_url() == "https://civitai.red"
    assert star._civitai_search_url("demo_lora.safetensors") == (
        "https://civitai.red/search/models?query=demo_lora"
    )
    assert star._civitai_headers("application/json")["Referer"] == "https://civitai.red/"
    assert star._civitai_link_api_base("https://civitai.red/models/123") == "https://civitai.red"

    star.config["civitai_base_url"] = "https://models.example.test/civitai"
    assert star._civitai_base_url() == "https://models.example.test/civitai"
    assert star._civitai_link_api_base(
        "https://models.example.test/civitai/models/123"
    ) == "https://models.example.test/civitai"
    assert ComfyUIAIStudio._civitai_reference(
        "https://models.example.test/civitai/models/123",
        {"models.example.test"},
    ) == ("123", "")


def test_image_reference_key_does_not_depend_on_urlunparse_symbol() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    value = ComfyUIAIStudio._image_ref_key("HTTPS://Example.com:443/assets/a.png?x=1#preview")
    assert value == "https://example.com:443/assets/a.png?x=1#preview"


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


def test_txt2img_plugin_ai_strips_reasoning_quality_words_and_duplicates() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "youhu，loli, petite, youhu, loli, petite, "
        "(masterpiece, best quality, amazing quality, very aesthetic, extremely detailed, "
        "absurdres, highres, score_9, score_8, year 2024), "
        "1. **分析用户请求**：输入‘一字马’。目标：转换为英文标签。"
    )
    assert ComfyUIAIStudio._clean_txt2img_prompt(raw) == "youhu, loli, petite"


def test_txt2img_plugin_ai_recovers_explicit_quoted_action_tag() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "youhu, loli, petite, 1. **分析用户请求**：用户要求一字马。"
        "核心标签是 `split`，不要输出解释。"
    )
    assert ComfyUIAIStudio._clean_txt2img_prompt(raw) == "youhu, loli, petite, split"


def test_txt2img_plugin_ai_keeps_action_and_reads_final_field() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "1. 分析用户请求：用户想要一字马。\n"
        "2. 确定核心标签：split。\n"
        "最终提示词：youhu, split, legs spread, split"
    )
    assert ComfyUIAIStudio._clean_txt2img_prompt(raw) == "youhu, split, legs spread"


def test_txt2img_prompt_cuts_chinese_reasoning_after_tag_prefix() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "verina, loli, petite, verina, "
        "(masterpiece, best quality, score_9), 用户要求输出一行英文 Danbooru 标签。"
        "需要包含：倒立、被绳子吊挂、劈叉。"
    )

    assert ComfyUIAIStudio._clean_txt2img_prompt(raw) == "verina, loli, petite"


def test_txt2img_prompt_cuts_the_reported_long_reasoning_response() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "verina, loli, petite, verina, "
        "(masterpiece, best quality, amazing quality, very aesthetic, extremely detailed, "
        "very detailed, absurdres, newest, highres, score_9, score_8, year 2024, newest,), "
        "用户要求输出一行英文 Danbooru 标签。需要包含：倒立、被绳子吊挂、劈叉、白色过膝袜。"
    )

    assert ComfyUIAIStudio._clean_txt2img_prompt(raw) == "verina, loli, petite"


def test_txt2img_prompt_removes_think_block_and_keeps_final_line() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "<think>分析用户要求并选择标签 split</think>\n"
        "verina, split, legs spread"
    )

    assert ComfyUIAIStudio._clean_txt2img_prompt(raw) == "verina, split, legs spread"


def test_llm_txt2img_sanitizer_falls_back_to_original_request() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = "分析用户请求：无法生成有效标签。"
    fallback = "帮我画一张海边的夏空"

    assert (
        ComfyUIAIStudio._sanitize_llm_txt2img_prompt(raw, fallback)
        == fallback
    )


def test_txt2img_plugin_ai_rejects_reasoning_without_final_prompt() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import AIError, ComfyUIAIStudio

    raw = "分析用户请求：需要转换动作。核心标签：split。让我继续检查标签。"
    with pytest.raises(AIError):
        ComfyUIAIStudio._clean_txt2img_prompt(raw)


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

    assert translated == "把身上衣服换成裙子，其他内容保持原图不变"
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

    assert translated == params.prompt == "把身上衣服换成裙子，其他内容保持原图不变"
    assert params.img2img_edit_instruction_ready is True
    assert received and "用户的图生图编辑要求" in received[0][0]
    assert "Qwen Image Edit" in received[0][1]
    assert received[0][2] == 256


def test_img2img_editor_keeps_concrete_pose_details() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    result = ComfyUIAIStudio._clean_img2img_edit_instruction(
        "EDIT: make her stand in a relaxed contrapposto pose, one hand touching her hair, "
        "the other arm resting naturally, looking toward the viewer"
    )

    assert "contrapposto" in result
    assert "touching her hair" in result
    assert "looking toward the viewer" in result


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

    assert positive == (
        "base edit quality, change the outfit to a dress; preserve the original "
        "character identity, face, hair, body proportions, pose, background, camera, "
        "composition, lighting and art style"
    )
    assert "原始长画面描述" not in positive


def test_plain_img2img_command_keeps_original_prompt_without_ai_or_translation() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, PresetStore

    with tempfile.TemporaryDirectory(prefix="astrbot_plain_img2img_command_test_") as temp:
        star = object.__new__(ComfyUIAIStudio)
        star.presets = PresetStore(Path(temp) / "presets.json")
        star.artist_presets = PresetStore(Path(temp) / "artist_presets.json")
        star.config = {
            "img2img_default_positive": "base edit quality",
            "img2img_default_negative": "",
            "artist_preset": "无",
            "img2img_plain_translate_enabled": True,
        }

        async def unexpected_ai(*args, **kwargs):
            raise AssertionError("普通图生图命令不应调用 AI")

        async def unexpected_translation(*args, **kwargs):
            raise AssertionError("普通图生图命令不应调用翻译")

        star._translate_prompt = unexpected_ai
        star._plain_translate_prompt = unexpected_translation
        params = DrawParams(
            mode="img2img",
            prompt="把衣服换成白色连衣裙",
            command_invocation=True,
        )
        positive, _, _ = asyncio.run(star._prompt_text(None, params, mode="img2img"))

    assert "把衣服换成白色连衣裙" in positive
    assert "base edit quality" in positive


def test_img2img_editor_drops_user_and_astrbot_metadata() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    raw = (
        "User's original words: Take off their clothes, other unchanged "
        "AstrBot LLM picture description: same two girls, pink long hair girl "
        "with purple eyes, orange twin braids girl with green eyes, masterpiece, "
        "best quality, highly detailed, anime style"
    )

    assert ComfyUIAIStudio._clean_img2img_edit_instruction(raw) == "Take off their clothes"
    final = ComfyUIAIStudio._finalize_img2img_instruction(
        ComfyUIAIStudio._clean_img2img_edit_instruction(raw)
    )
    assert final.startswith("Take off their clothes;")
    assert "User's original words" not in final
    assert "AstrBot LLM picture description" not in final
    assert "same two girls" not in final
    assert "masterpiece" not in final


def test_img2img_input_size_keeps_ratio_and_caps_oversized_source() -> None:
    from PIL import Image as PILImage

    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)
    plugin._bool_config = lambda key, default=False: default
    with tempfile.TemporaryDirectory(prefix="astrbot_img2img_size_test_") as temp:
        image_path = Path(temp) / "source.png"
        PILImage.new("RGB", (2048, 1024), (20, 40, 60)).save(image_path)

        width, height = plugin._img2img_output_size(
            str(image_path),
            512,
            960,
            use_input_size=True,
            max_edge=1152,
        )

    assert (width, height) == (1152, 576)


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

        # LoRA 简称可以通过 preset 参数传入，但没有同名全局预设时，
        # 不应在 _prompt_text 中报“预设不存在”；它应只临时加载 LoRA。
        star.lora_command_aliases["first.safetensors"] = []
        star.lora_presets["first.safetensors"] = [
            {"tag": "画风4", "content": "style four"},
        ]
        lora_alias_only = star._llm_params(
            "海边少女",
            "txt2img",
            preset="画风4",
        )
        asyncio.run(star._extract_inline_loras(lora_alias_only))
        assert lora_alias_only.loras == ["first.safetensors:0.8"]
        assert lora_alias_only.presets == []

        # 如果同名全局预设存在，简称和普通预设必须继续同时生效。
        star.presets.add("画风4", "artist-defined style")
        both_alias_and_preset = star._llm_params(
            "海边少女",
            "txt2img",
            preset="画风4",
        )
        asyncio.run(star._extract_inline_loras(both_alias_and_preset))
        assert both_alias_and_preset.loras == ["first.safetensors:0.8"]
        assert both_alias_and_preset.presets == ["画风4"]

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

    source_path = Path(r"E:\QwenImageEdit2511局部重绘替换万物 (1).json")
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


def test_qwen_rollback_ignores_acceleration_and_keeps_content_lora() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_qwen_img2img, load_workflow

    source = load_workflow(Path(r"E:\QwenImageEdit2511局部重绘替换万物 (1).json"))
    adapted, report = adapt_qwen_img2img(
        source,
        positive="保持人物并更换服装",
        negative="低质量",
        image_name="input.png",
        unet_name="Qwen-Image-Edit-2511-Q4_K_M.gguf",
        clip_name="Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf",
        vae_name="qwen_image_vae.safetensors",
        loras=["qwen/Qwen-Image-Edit-F2P.safetensors:0.7"],
        steps=4,
        cfg=1.0,
    )

    assert adapted["174"]["inputs"]["lora_1"]["on"] is True
    assert adapted["30"]["inputs"]["model"] == ["174", 0]
    assert "astrbot_qwen_accel_lora" not in adapted
    assert any("无加速 LoRA" in item for item in report)


def test_qwen_rollback_uses_regular_sampling_parameters() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_qwen_img2img, load_workflow

    source = load_workflow(Path(r"E:\QwenImageEdit2511局部重绘替换万物 (1).json"))
    adapted, report = adapt_qwen_img2img(
        source,
        positive="保持原图",
        negative="",
        image_name="input.png",
        unet_name="Qwen-Rapid-NSFW-v23_Q3_K.gguf",
        clip_name="Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf",
        vae_name="qwen_image_vae.safetensors",
        steps=8,
        cfg=1.25,
    )

    assert "astrbot_qwen_accel_lora" not in adapted
    assert adapted["30"]["inputs"]["model"] == ["174", 0]
    assert adapted["152"]["inputs"]["steps"] == 8
    assert adapted["152"]["inputs"]["cfg"] == 1.25
    assert any("无加速 LoRA" in item for item in report)


def test_flux2_img2img_uses_the_single_reference_source_workflow() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import (
        FLUX2_IMG2IMG_WORKFLOW_FILE,
        MODE_NAMES,
        WRITABLE_CONFIG,
    )
    from astrbot_plugin_comfyui_ai_studio.workflow import (
        WorkflowError,
        adapt_flux2_klein_img2img,
        load_workflow,
    )

    assert MODE_NAMES["img2img_flux2"] == "图生图 Flux2"
    assert "img2img_flux2_unet_name" in WRITABLE_CONFIG
    assert "img2img_flux2_lora_name" in WRITABLE_CONFIG
    assert "img2img_flux2_prompt_template" in WRITABLE_CONFIG
    assert (PLUGIN_DIR / "workflows" / FLUX2_IMG2IMG_WORKFLOW_FILE).is_file()

    source = load_workflow(PLUGIN_DIR / "workflows" / FLUX2_IMG2IMG_WORKFLOW_FILE)
    assert not any(
        node.get("class_type") == "Fast Groups Bypasser (rgthree)"
        for node in source.values()
    )
    adapted, report = adapt_flux2_klein_img2img(
        source,
        positive="中文编辑要求",
        negative="低质量",
        image_names=["image.png"],
        unet_name="flux-2-klein\\flux-2-klein-9b-fp8.safetensors",
        clip_name="qwen_3_8b_fp8mixed.safetensors",
        vae_name="flux2-vae.safetensors",
        lora_name="flux-2-klein\\flux-2-klein-NSFW.safetensors",
        lora_strength=0.7,
        size=1024,
        steps=8,
        cfg=1.2,
        seed=123,
        sampler_name="euler",
        scheduler="simple",
        denoise=0.6,
        batch=2,
        filename_prefix="astrbot/test-flux2",
    )

    assert adapted["19"]["inputs"]["text"] == "中文编辑要求"
    assert adapted["19"]["inputs"]["clip"] == ["14", 0]
    assert adapted["13"]["inputs"]["unet_name"] == "flux-2-klein\\flux-2-klein-9b-fp8.safetensors"
    assert adapted["14"]["inputs"]["clip_name"] == "qwen_3_8b_fp8mixed.safetensors"
    assert adapted["10"]["inputs"]["vae_name"] == "flux2-vae.safetensors"
    assert adapted["38"]["inputs"]["lora_name"] == "flux-2-klein\\flux-2-klein-NSFW.safetensors"
    assert adapted["38"]["inputs"]["strength_model"] == 0.7
    assert adapted["95"]["inputs"]["model"] == ["38", 0]
    assert not any(
        node.get("_meta", {}).get("title") == "Flux2 加速 LoRA"
        for node in adapted.values()
    )
    assert not any(
        node.get("class_type") in {"FluxKVCache", "Fast Groups Bypasser (rgthree)"}
        for node in adapted.values()
    )
    assert adapted["95"]["inputs"]["positive"] == ["74", 0]
    assert adapted["95"]["inputs"]["negative"] == ["73", 0]
    assert adapted["95"]["inputs"]["latent_image"] == ["83", 0]
    assert adapted["95"]["inputs"]["steps"] == 8
    assert adapted["95"]["inputs"]["cfg"] == 1.2
    assert adapted["95"]["inputs"]["seed"] == 123
    assert adapted["95"]["inputs"]["denoise"] == 0.6
    assert adapted["83"]["inputs"]["batch_size"] == 2
    assert adapted["62"]["inputs"]["filename_prefix"] == "astrbot/test-flux2"
    assert adapted["63"]["inputs"]["image"] == "image.png"
    assert adapted["90"]["inputs"]["缩放长度"] == 1024
    assert adapted["75"]["inputs"]["pixels"] == ["90", 0]
    assert adapted["74"]["inputs"]["latent"] == ["75", 0]
    assert adapted["73"]["inputs"]["latent"] == ["75", 0]
    assert any("最长边等比例缩放" in item for item in report)

    adapted_two, _ = adapt_flux2_klein_img2img(
        source,
        positive="中文编辑要求",
        negative="",
        image_names=["one.png", "two.png"],
        unet_name="model.safetensors",
        clip_name="clip.safetensors",
        vae_name="vae.safetensors",
    )
    assert adapted_two["64"]["inputs"]["image"] == "two.png"
    assert adapted_two["95"]["inputs"]["positive"] == ["85", 0]
    assert adapted_two["95"]["inputs"]["negative"] == ["86", 0]

    adapted_three, _ = adapt_flux2_klein_img2img(
        source,
        positive="中文编辑要求",
        negative="",
        image_names=["one.png", "two.png", "three.png"],
        unet_name="model.safetensors",
        clip_name="clip.safetensors",
        vae_name="vae.safetensors",
    )
    assert adapted_three["61"]["inputs"]["image"] == "three.png"
    assert adapted_three["95"]["inputs"]["positive"] == ["80", 0]
    assert adapted_three["95"]["inputs"]["negative"] == ["79", 0]

    with pytest.raises(WorkflowError, match="最多支持 3"):
        adapt_flux2_klein_img2img(
            source,
            positive="中文编辑要求",
            negative="",
            image_names=["one.png", "two.png", "three.png", "four.png"],
            unet_name="model.safetensors",
            clip_name="clip.safetensors",
            vae_name="vae.safetensors",
        )


def test_flux2_uses_source_model_chain_without_acceleration_nodes() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_flux2_klein_img2img, load_workflow

    source = load_workflow(PLUGIN_DIR / "workflows" / "图生图_flux2_klein.json")
    adapted, report = adapt_flux2_klein_img2img(
        source,
        positive="保持原图",
        negative="",
        image_names=["image.png"],
        unet_name="flux-2-klein\\flux-2-klein-9b-fp8.safetensors",
        clip_name="qwen_3_8b_fp8mixed.safetensors",
        vae_name="flux2-vae.safetensors",
    )

    # 源工作流自带的 LoRA（节点 38）属于模型链，独立 LoRA 留空时必须保留；
    # 但绝不能注入「Flux2 加速 LoRA」这类额外节点。
    assert adapted["95"]["inputs"]["model"] == ["38", 0]
    assert adapted["38"]["inputs"]["lora_name"] == "flux-2-klein\\flux-2-klein-NSFW.safetensors"
    assert adapted["95"]["inputs"]["denoise"] == 1.0
    assert not any(
        node.get("_meta", {}).get("title") == "Flux2 加速 LoRA"
        for node in adapted.values()
    )
    assert any("不添加加速 LoRA 或 KV Cache" in item for item in report)


def test_flux2_does_not_inject_kv_cache() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_flux2_klein_img2img, load_workflow

    source = load_workflow(PLUGIN_DIR / "workflows" / "图生图_flux2_klein.json")
    adapted, report = adapt_flux2_klein_img2img(
        source,
        positive="保持人物和构图",
        negative="",
        image_names=["image.png"],
        unet_name="flux-2-klein-9b-kv-fp8.safetensors",
        clip_name="qwen_3_8b_fp8mixed.safetensors",
        vae_name="flux2-vae.safetensors",
        steps=4,
        cfg=1.0,
    )

    # 采样器仍然走源工作流的模型链（含它自带的 LoRA），但不得出现 KV Cache 节点。
    assert adapted["95"]["inputs"]["model"] == ["38", 0]
    assert not any(node.get("class_type") == "FluxKVCache" for node in adapted.values())
    assert adapted["95"]["inputs"]["positive"] == ["74", 0]
    assert any("不添加加速 LoRA 或 KV Cache" in item for item in report)


def test_flux2_configuration_does_not_use_qwen_values() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))

    assert "img2img_flux2" in source
    assert "img2img_flux2_output_format" in schema
    assert schema["img2img_engine"]["enum"] == ["qwen", "flux2"]
    assert 'self._get("img2img_lora_name", "")' in source
    assert 'self._get("img2img_flux2_lora_name", FLUX2_IMG2IMG_LORA_DEFAULT)' in source
    assert 'f"{img2img_config_prefix}_default_positive"' in source


def test_flux2_tool_workflows_keep_default_loras_and_write_tool_inputs() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_flux2_tool, load_workflow

    workflow_dir = PLUGIN_DIR / "workflows"
    tool_files = {
        "wash": next(workflow_dir.glob("*.json")),
        "outpaint": next(path for path in workflow_dir.glob("*.json") if "扩图" in path.name),
        "multi_angle": next(path for path in workflow_dir.glob("*.json") if "多角度" in path.name),
    }
    wash_path = next(path for path in workflow_dir.glob("*.json") if "洗图" in path.name)
    tool_files["wash"] = wash_path

    for mode, path in tool_files.items():
        source = load_workflow(path)
        adapted, report = adapt_flux2_tool(
            source,
            tool=mode,
            positive="用户中文要求",
            negative="低质量",
            image_name="input.png",
            size=768,
            steps=9,
            cfg=1.5,
            seed=123,
            denoise=0.5,
            sampler_name="euler",
            scheduler="simple",
            filename_prefix="astrbot/test",
            left=16,
            top=24,
            right=32,
            bottom=40,
            feathering=8,
            horizontal_angle=12,
            vertical_angle=-5,
            zoom=2.2,
            batch=1,
            caption_tokens=256,
        )

        load_nodes = [
            node for node in adapted.values()
            if node.get("class_type") == "LoadImage"
        ]
        save_nodes = [
            node for node in adapted.values()
            if node.get("class_type") == "SaveImage"
        ]
        assert load_nodes and load_nodes[0]["inputs"]["image"] == "input.png"
        assert save_nodes and save_nodes[0]["inputs"]["filename_prefix"] == "astrbot/test"
        assert any(node.get("class_type") == "KSampler" for node in adapted.values())
        assert all(node.get("class_type") for node in adapted.values())
        assert any("不使用全局 LoRA" in item for item in report)

        if mode in {"outpaint", "multi_angle"}:
            latent_nodes = [
                node for node in adapted.values()
                if node.get("class_type") == "EmptyFlux2LatentImage"
            ]
            assert latent_nodes and latent_nodes[0]["inputs"]["batch_size"] == 1

        if mode == "wash":
            assert any(
                node.get("class_type") == "Florence2Run"
                and node["inputs"].get("max_new_tokens") == 256
                for node in adapted.values()
            )
        elif mode == "outpaint":
            pad = next(node for node in adapted.values() if node.get("class_type") == "ImagePadForOutpaint")
            assert pad["inputs"]["left"] == 16
            assert pad["inputs"]["top"] == 24
            assert pad["inputs"]["right"] == 32
            assert pad["inputs"]["bottom"] == 40
            assert pad["inputs"]["feathering"] == 8
        else:
            camera = next(node for node in adapted.values() if node.get("class_type") == "QwenMultiangleCameraNode")
            assert camera["inputs"]["horizontal_angle"] == 12
            assert camera["inputs"]["vertical_angle"] == -5
            assert camera["inputs"]["zoom"] == 2.2


def test_flux2_tool_workflow_source_conversion_preserves_disabled_lora() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import load_workflow

    for path in (
        Path(r"E:\0000001\工作流们（7个）\▶flux-2-klein-图生图洗图流.json"),
        Path(r"E:\0000001\工作流们（7个）\▶flux-2-klein-图像扩展流.json"),
        Path(r"E:\0000001\工作流们（7个）\▶flux-2-klein-多角度转换流.json"),
    ):
        if not path.is_file():
            pytest.skip(f"源工作流不存在：{path}")
        workflow = load_workflow(path)
        assert all(
            not (
                node.get("class_type") == "LoraLoaderModelOnly"
                and node.get("inputs", {}).get("lora_name") == "flux-2-klein\\flux-2-klein-NSFW.safetensors"
            )
            for node in workflow.values()
        )


def test_img2img_edit_ai_rejects_literal_horse_translation() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    repaired = ComfyUIAIStudio._repair_img2img_edit_instruction(
        "Make her stand on a horse",
        "让她站立一字马",
    )
    assert repaired == "让人物站立完成一字马（劈叉）动作，其他内容保持原图不变"

    repaired = ComfyUIAIStudio._repair_img2img_edit_instruction(
        "change her outfit to a dress",
        "把她的衣服换成白色连衣裙",
    )
    assert repaired == "把她的衣服换成白色连衣裙，其他内容保持原图不变"


def test_qwen_img2img_connects_second_reference_image() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_qwen_img2img, load_workflow

    source = load_workflow(Path(r"E:\112121121.json"))
    adapted, report = adapt_qwen_img2img(
        source,
        positive="把衣服换成参考图中的服装",
        negative="low quality",
        image_name="source.png",
        second_image_name="clothes.png",
        unet_name="Qwen-Rapid-NSFW-v23_Q3_K.gguf",
        clip_name="Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf",
        vae_name="qwen_image_vae.safetensors",
    )

    assert adapted["astrbot_img2img_second_input"]["inputs"]["image"] == "clothes.png"
    assert adapted["1"]["inputs"]["image2"] == ["astrbot_img2img_second_input", 0]
    assert adapted["39"]["inputs"]["image2"] == ["astrbot_img2img_second_input", 0]
    assert any("第二张" in item for item in report)


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


def test_environment_keeps_flux2_text_encoders_separate_from_qwen_gguf() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)
    plugin._lora_names_cache = None
    plugin._order_loras = lambda names: list(names)

    class _Client:
        async def object_info(self):
            return {}

        async def models(self, category):
            values = {
                "diffusion_models": [],
                "checkpoints": [],
                "loras": [],
                "upscale_models": [],
                "clip_gguf": ["Qwen2.5-VL.Q4_K_M.gguf"],
                "vae": [],
                "unet_gguf": [],
                "controlnet": [],
                "ipadapter": [],
                "clip_vision": [],
                "text_encoders": ["qwen_3_8b_fp8mixed.safetensors"],
            }
            return values[category]

    environment = asyncio.run(plugin._environment(_Client()))

    assert environment["flux2_text_encoders"] == ["qwen_3_8b_fp8mixed.safetensors"]
    assert environment["clip_gguf"] == ["Qwen2.5-VL.Q4_K_M.gguf"]
    assert set(environment["text_encoders"]) == {
        "qwen_3_8b_fp8mixed.safetensors",
        "Qwen2.5-VL.Q4_K_M.gguf",
    }


def test_lora_resolver_accepts_saved_backslash_path_against_api_slash_path() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.lora_aliases = {}
    plugin.lora_command_aliases = {}
    plugin.lora_presets = {}

    available = ["flux-2-klein/flux-2-klein-NSFW.safetensors"]

    assert (
        plugin._resolve_lora(
            r"flux-2-klein\flux-2-klein-NSFW.safetensors",
            available,
        )
        == available[0]
    )


def test_flux2_model_resolver_preserves_comfyui_returned_separator() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    available = [r"flux-2-klein\flux-2-klein-9b-fp8.safetensors"]

    assert (
        ComfyUIAIStudio._resolve_flux2_model(
            "flux-2-klein/flux-2-klein-9b-fp8.safetensors",
            available,
            "核心模型",
        )
        == available[0]
    )


def test_lora_order_preserves_comfyui_returned_separator() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.lora_download_order = {}
    available = [r"flux-2-klein\flux-2-klein-NSFW.safetensors"]

    assert plugin._order_loras(available) == available


def test_comfy_queue_info_reads_running_and_pending_entries() -> None:
    from astrbot_plugin_comfyui_ai_studio.comfy import ComfyClient

    client = ComfyClient("http://127.0.0.1:8188")

    async def fake_request(method, path, **kwargs):
        assert method == "GET"
        assert path == "/queue"
        return {"queue_running": [[1, "running", {}]], "queue_pending": [[2, "pending", {}]]}

    client.request = fake_request
    result = asyncio.run(client.queue_info())

    assert len(result["queue_running"]) == 1
    assert len(result["queue_pending"]) == 1


def test_queue_snapshot_estimates_batch_images_and_formats_notice() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.config = {"draw_queue_notice_enabled": True}
    plugin._get = lambda key, default="": plugin.config.get(key, default)

    async def fake_snapshot():
        return {
            "available": True,
            "running_tasks": 1,
            "pending_tasks": 2,
            "running_images": 2,
            "pending_images": 3,
            "queued_tasks": 3,
            "queued_images": 5,
        }

    plugin._queue_snapshot = fake_snapshot
    params = DrawParams(batch=2)
    asyncio.run(plugin._check_draw_queue(params, "txt2img"))

    assert params.queue_available is True
    assert params.queue_images_before == 5
    assert params.queue_tasks_before == 3
    assert "当前前面已有 5 张图排队" in params.queue_notice_text
    assert "运行中 2 张" in params.queue_notice_text
    assert "等待中 3 张" in params.queue_notice_text


def test_queue_limit_rejects_when_existing_queue_plus_batch_is_too_large() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, UsageError

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.config = {
        "draw_queue_notice_enabled": True,
        "draw_queue_limit_enabled": True,
        "draw_queue_limit_count": 3,
    }
    plugin._get = lambda key, default="": plugin.config.get(key, default)

    async def fake_snapshot():
        return {
            "available": True,
            "running_tasks": 1,
            "pending_tasks": 1,
            "running_images": 1,
            "pending_images": 1,
            "queued_tasks": 2,
            "queued_images": 2,
        }

    plugin._queue_snapshot = fake_snapshot
    with pytest.raises(UsageError, match="超过允许的最多排队 3 张"):
        asyncio.run(plugin._check_draw_queue(DrawParams(batch=2), "txt2img"))


def test_queue_limit_allows_configured_admin() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    class AdminEvent:
        def get_sender_id(self) -> str:
            return "admin-1"

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.config = {
        "draw_queue_notice_enabled": True,
        "draw_queue_limit_enabled": True,
        "draw_queue_limit_count": 1,
        "draw_limit_admin_ids": "admin-1",
    }
    plugin._get = lambda key, default="": plugin.config.get(key, default)

    async def fake_snapshot():
        return {
            "available": True,
            "running_tasks": 2,
            "pending_tasks": 2,
            "running_images": 2,
            "pending_images": 2,
            "queued_tasks": 4,
            "queued_images": 4,
        }

    plugin._queue_snapshot = fake_snapshot
    params = DrawParams(batch=1)
    asyncio.run(plugin._check_draw_queue(params, "txt2img", AdminEvent()))

    assert params.queue_images_before == 4
    assert "当前前面已有 4 张图排队" in params.queue_notice_text


def test_llm_start_reply_uses_persona_and_keeps_exact_queue_notice() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.config = {"draw_queue_notice_enabled": True}
    plugin._get = lambda key, default="": plugin.config.get(key, default)

    async def fake_generate(*args, **kwargs):
        assert kwargs["use_event_context"] is True
        return "我来准备这张图啦！"

    plugin._astrbot_generate = fake_generate
    params = DrawParams(
        mode="txt2img",
        prompt="海边的夏空",
        queue_available=True,
        queue_images_before=4,
        queue_tasks_before=2,
        queue_running_images=1,
        queue_pending_images=3,
    )
    result = asyncio.run(plugin._ai_draw_start_reply(object(), params, "txt2img"))

    assert result.startswith("我来准备这张图啦")
    assert "当前前面已有 4 张图排队" in result


def test_llm_start_reply_does_not_treat_an_unrelated_number_as_queue_notice() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    plugin = object.__new__(ComfyUIAIStudio)
    plugin.config = {"draw_queue_notice_enabled": True}
    plugin._get = lambda key, default="": plugin.config.get(key, default)

    async def fake_generate(*args, **kwargs):
        return "我会在第 4 步开始处理这张图。"

    plugin._astrbot_generate = fake_generate
    params = DrawParams(
        mode="txt2img",
        prompt="海边的夏空",
        queue_available=True,
        queue_images_before=5,
        queue_tasks_before=2,
        queue_running_images=2,
        queue_pending_images=3,
    )
    result = asyncio.run(plugin._ai_draw_start_reply(object(), params, "txt2img"))

    assert result.endswith("当前前面已有 5 张图排队（运行中 2 张，等待中 3 张，共 2 个任务）。")
