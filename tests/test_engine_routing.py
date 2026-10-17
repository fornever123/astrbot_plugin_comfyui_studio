"""引擎路由：Anima / Krea 2 / Qwen 2.1 各走各的工作流与 LoRA 列表。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


def make_star(config=None):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = config or {}
    star.plugin_dir = PLUGIN_DIR
    return star


def test_engine_aliases_normalize_to_three_engines():
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    normalize = ComfyUIAIStudio._normalize_engine_key
    for value in ("krea2", "Krea 2", "克雷亚", "水粉"):
        assert normalize(value) == "krea2"
    for value in ("qwen21", "qwen", "Qwen Image 2.1", "千问"):
        assert normalize(value) == "qwen21"
    for value in ("", None, "anima", "默认"):
        assert normalize(value) == ""
    assert normalize("unknown-engine") == ""


def test_engine_workflow_paths_point_at_bundled_graphs():
    star = make_star()
    assert star._engine_workflow_path("krea2", "txt2img").name == "krea2_t2i_api.json"
    assert star._engine_workflow_path("qwen21", "txt2img").name == "qwen21_t2i_api.json"
    # 空引擎 = 完全沿用 WebUI 当前配置，不做任何改写
    assert star._engine_workflow_path("", "txt2img") == star._workflow_path("txt2img")
    # 图生图不另做引擎图，复用图生图工作流
    assert star._engine_workflow_path("qwen21", "img2img").name == "qwen21_edit_api.json"


def test_qwen21_bundled_workflow_has_required_node():
    graph = json.loads((PLUGIN_DIR / "workflows" / "qwen21_t2i_api.json").read_text(encoding="utf-8"))
    classes = {node.get("class_type") for node in graph.values() if isinstance(node, dict)}
    assert "TextEncodeQwenImage21" in classes
    assert "UnetLoaderGGUF" not in classes


def test_llm_params_carries_engine_into_draw_params():
    star = make_star()
    assert star._llm_params("一只猫", "txt2img", engine="krea2").engine == "krea2"
    assert star._llm_params("一只猫", "txt2img", engine="千问").engine == "qwen21"
    assert star._llm_params("一只猫", "txt2img").engine == ""
    # 未知引擎不应把任务带偏
    assert star._llm_params("一只猫", "txt2img", engine="dalle").engine == ""


def test_engine_lora_lists_are_independent_config_keys():
    from astrbot_plugin_comfyui_ai_studio.main import ENGINE_LORA_CONFIG_KEYS

    assert ENGINE_LORA_CONFIG_KEYS["krea2"] == "krea2_lora_list"
    assert ENGINE_LORA_CONFIG_KEYS["qwen21"] == "qwen21_lora_list"
    assert "lora_list" not in ENGINE_LORA_CONFIG_KEYS.values()


def test_schema_declares_engine_lora_and_workflow_keys():
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    for key in (
        "krea2_lora_list",
        "qwen21_lora_list",
        "workflow_krea2",
        "workflow_qwen21",
        "plugin_ai_krea2_context",
        "plugin_ai_qwen21_context",
    ):
        assert key in schema, key


def test_console_has_two_extra_lora_sections():
    html = (PLUGIN_DIR / "index.html").read_text(encoding="utf-8")
    for section in ("krea2", "qwen21"):
        assert f'data-lora-section="{section}"' in html
        assert f'data-engine-lora-panel="{section}"' in html
    js = (PLUGIN_DIR / "app.js").read_text(encoding="utf-8")
    assert "krea2_lora_list" in js and "qwen21_lora_list" in js


def test_nsfw_prompt_gate_requires_switch_and_respects_blacklist():
    """成人向提示词必须同时满足：开关打开 + 当前群不在黑名单。黑名单优先。"""
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    class _Gate:
        def __init__(self, cfg, group="", blacklist=()):
            self.cfg = cfg
            self.group = group
            self.blacklist = set(blacklist)

        def _get(self, key, default=None):
            return self.cfg.get(key, default)

        def _group_id(self, event):
            return self.group

        def _nsfw_group_ids(self):
            return self.blacklist

    active = ComfyUIAIStudio._nsfw_prompt_active

    # 默认关闭：没配开关就是 False
    assert active(_Gate({}), None) is False
    # 开关打开、私聊/白名单群：可用
    assert active(_Gate({"nsfw_prompt_enabled": True}), None) is True
    assert active(_Gate({"nsfw_prompt_enabled": True}, "g2", {"g1"}), None) is True
    # 开关打开但在黑名单群：必须关掉
    assert active(_Gate({"nsfw_prompt_enabled": True}, "g1", {"g1"}), None) is False
    # 字符串 "false" 不能被当成真值
    assert active(_Gate({"nsfw_prompt_enabled": "false"}), None) is False
    assert active(_Gate({"nsfw_prompt_enabled": "true"}), None) is True


def test_schema_and_console_expose_nsfw_prompt_switch():
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["nsfw_prompt_enabled"]["type"] == "bool"
    assert schema["nsfw_prompt_enabled"]["default"] is False
    for rel in ("index.html", "pages/console/index.html"):
        html = (PLUGIN_DIR / rel).read_text(encoding="utf-8")
        assert 'id="nsfw_prompt_enabled"' in html
    for rel in ("app.js", "pages/console/app.js"):
        js = (PLUGIN_DIR / rel).read_text(encoding="utf-8")
        assert "nsfw_prompt_enabled" in js


def test_qwen21_engine_keeps_chinese_and_drops_anima_quality(tmp_path):
    """engine=qwen21 时不能被 Anima 的翻译链改写成标签，也不能追加其默认正面词。"""
    import asyncio

    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams, PresetStore

    star = make_star({
        "model_name": "anima-base-v1.0.safetensors",
        "img2img_unet_name": "qwen_image_2.1_int8_convrot.safetensors",
        "default_positive": "masterpiece, best quality",
        "artist_preset": "画师甲",
    })
    star.presets = PresetStore(tmp_path / "presets.json")
    star.artist_presets = PresetStore(tmp_path / "artist.json")
    star.artist_presets.add("画师甲", "some_artist_name")
    star._apply_group_safety_negative = lambda event, value: value

    async def forbidden(*args, **kwargs):
        raise AssertionError("Qwen 2.1 引擎不应走普通翻译链")

    star._plain_translate_prompt = forbidden

    params = DrawParams(prompt="雨后的山城街道", engine="qwen21", command_invocation=True)
    positive, _, _ = asyncio.run(star._prompt_text(None, params))
    assert "雨后的山城街道" in positive
    assert "masterpiece" not in positive
    assert "some_artist_name" not in positive

