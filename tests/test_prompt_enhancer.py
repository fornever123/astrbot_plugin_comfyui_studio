from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR.parents[2]))


def test_templates_select_model_format_and_preserve_user_configuration():
    from prompt_enhancer import build_system_prompt

    qwen = build_system_prompt({}, "txt2img", "qwen_image_2.1_bf16.safetensors")
    anima = build_system_prompt({}, "txt2img", "anima-2.9b.safetensors")
    edit = build_system_prompt({}, "img2img", "qwen_image_2.1_bf16.safetensors")
    assert "中文" in qwen and "自然语言" in qwen
    assert "Danbooru" in anima and "英文" in anima
    assert "<Picture 1>" in edit and "未指定" in edit
    assert "预设" in edit and "LoRA" in edit
    custom = build_system_prompt(
        {"prompt_enhancer_img2img_template": "服装编辑时保持人物身份。"},
        "img2img",
        "qwen_image_2.1_bf16.safetensors",
    )
    assert "服装编辑时保持人物身份。" in custom
    assert "思考" in custom


@pytest.mark.parametrize(
    "raw",
    [
        "<think>分析过程</think>把外套换成蓝色夹克，保持脸部与背景不变。",
        '```json\n{"rewritten_prompt": "把外套换成蓝色夹克，保持脸部与背景不变。", "thinking": "分析"}\n```',
        "最终提示词：把外套换成蓝色夹克，保持脸部与背景不变。",
    ],
)
def test_extracts_only_final_prompt(raw):
    from prompt_enhancer import clean_enhanced_prompt

    assert clean_enhanced_prompt(raw) == "把外套换成蓝色夹克，保持脸部与背景不变。"


@pytest.mark.parametrize(
    "raw",
    [
        "<think>还在分析，没有最终答案",
        "分析用户请求：我们需要根据图片判断人物的衣服。",
        '{"thinking": "只有思考过程"}',
        "I cannot help with that request.",
        "",
    ],
)
def test_rejects_reasoning_or_non_prompt_answers(raw):
    from prompt_enhancer import clean_enhanced_prompt

    with pytest.raises(ValueError):
        clean_enhanced_prompt(raw)


def make_star(config=None):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = config or {}
    return star


def test_qwen_ai_uses_enhancer_and_keeps_chinese_output(monkeypatch):
    from astrbot_plugin_comfyui_ai_studio import main

    star = make_star({"model_name": "qwen_image_2.1_bf16.safetensors"})
    calls = []

    async def fake_generate(self, prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "<think>分析</think>一只白猫坐在窗台上，窗外是花园。"

    monkeypatch.setattr(main.AITranslator, "generate", fake_generate)
    result = asyncio.run(
        star._translate_prompt(None, "窗台上的白猫", source_override="plugin")
    )
    assert result == "一只白猫坐在窗台上，窗外是花园。"
    assert len(calls) == 1
    assert calls[0][1]["preserve_newlines"] is True
    assert "自然语言" in calls[0][1]["system_prompt"]
    assert "没有收到参考图片" in calls[0][1]["system_prompt"]


def test_format_error_retries_without_sending_reasoning_to_model(monkeypatch):
    from astrbot_plugin_comfyui_ai_studio import main

    star = make_star({"model_name": "qwen_image_2.1_bf16.safetensors"})
    calls = []

    async def fake_generate(self, prompt, **kwargs):
        calls.append(prompt)
        return "分析用户请求：只有分析" if len(calls) == 1 else "一只白猫坐在窗台上。"

    monkeypatch.setattr(main.AITranslator, "generate", fake_generate)
    result = asyncio.run(
        star._translate_prompt(None, "窗台上的白猫", source_override="plugin")
    )
    assert result == "一只白猫坐在窗台上。"
    assert len(calls) == 2
    assert "只有分析" not in calls[1]


def test_enhancer_preserves_multi_image_edit_tokens(monkeypatch):
    from astrbot_plugin_comfyui_ai_studio import main

    star = make_star({"img2img_unet_name": "qwen_image_2.1_bf16.safetensors"})

    async def fake_generate(self, prompt, **kwargs):
        assert "<Picture 1>" in kwargs["system_prompt"]
        return (
            "把<Picture 2>的蓝色外套换到<Picture 1>的人物身上，其他内容保持原图不变。"
        )

    monkeypatch.setattr(main.AITranslator, "generate", fake_generate)
    result = asyncio.run(
        star._translate_prompt(
            None,
            "把第二张图的外套换到第一张图",
            source_override="plugin_img2img_llm",
        )
    )
    assert "<Picture 1>" in result and "<Picture 2>" in result


def test_qwen_ai_command_preserves_preset_and_lora_and_noai_skips_ai(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, PresetStore

    star = make_star(
        {"model_name": "qwen_image_2.1_bf16.safetensors", "artist_preset": ""}
    )
    star.presets = PresetStore(tmp_path / "presets.json")
    star.presets.add("角色甲", "character_alpha")
    star.artist_presets = PresetStore(tmp_path / "artist.json")
    calls = []

    async def translate(event, text, **kwargs):
        calls.append(text)
        return "坐在窗台上看花园"

    star._translate_prompt = translate
    star._apply_group_safety_negative = lambda event, value: value
    params = DrawParams(
        prompt="角色甲 坐在窗台", ai=True, loras=["hero.safetensors:0.6"]
    )
    positive, _, _ = asyncio.run(
        star._prompt_text(None, params, lora_prompt_values=["blue_jacket"])
    )
    assert calls == ["坐在窗台"]
    assert positive.count("character_alpha") == 1
    assert "blue_jacket" in positive
    assert params.loras == ["hero.safetensors:0.6"]
    calls.clear()
    params = DrawParams(prompt="白猫 坐在窗台", ai=False, command_invocation=True)
    asyncio.run(star._prompt_text(None, params))
    assert calls == []


def test_reasoning_only_response_is_not_used_as_a_prompt():
    from astrbot_plugin_comfyui_ai_studio.ai import AITranslator

    assert (
        AITranslator._content_value(
            {"choices": [{"message": {"reasoning_content": "分析"}}]}
        )
        == "分析"
    )
    assert (
        AITranslator._content_value(
            {"choices": [{"message": {"reasoning_content": "分析"}}]},
            allow_reasoning=False,
        )
        == ""
    )


def test_webui_preview_awaits_control_parsing_and_does_not_save(monkeypatch, tmp_path):
    import json

    from astrbot_plugin_comfyui_ai_studio.prompting import PresetStore

    import astrbot.api.web as web

    star = make_star({"model_name": "qwen_image_2.1_bf16.safetensors"})
    star.presets = PresetStore(tmp_path / "presets.json")
    star.presets.add("角色甲", "character_alpha")
    original_file = (tmp_path / "presets.json").read_bytes()
    parsed = []

    class Request:
        async def json(self, **kwargs):
            return {"mode": "txt2img", "prompt": "角色甲 坐在窗台"}

    async def extract(params):
        parsed.append(params.prompt)
        params.loras = ["hero.safetensors:0.6"]

    async def enhance(event, text, **kwargs):
        assert text == "坐在窗台"
        return "坐在窗台上看花园"

    monkeypatch.setattr(web, "request", Request())
    star._extract_inline_loras = extract
    star._enhance_prompt = enhance
    response = asyncio.run(star.api_prompt_enhancer())
    body = json.loads(response.body)
    assert parsed == ["坐在窗台"]
    assert body["presets"] == ["角色甲"]
    assert body["loras"] == ["hero.safetensors:0.6"]
    assert body["preset_content"] == "character_alpha"
    assert body["prompt"] == "坐在窗台上看花园"
    assert (tmp_path / "presets.json").read_bytes() == original_file
