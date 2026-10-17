from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(PLUGIN_DIR.parents[2]))


def test_catalog_includes_models_workflows_and_no_private_lora_defaults():
    from bundled_resources import builtin_catalog

    entries = builtin_catalog(PLUGIN_DIR)
    assert {entry["id"] for entry in entries} >= {
        "anima-base",
        "anima-29b",
        "qwen21",
        "qwen21-turbo",
    }
    qwen = next(entry for entry in entries if entry["id"] == "qwen21")
    assert qwen["workflows"]["img2img"] == "qwen21_edit_api.json"
    assert qwen["parameters"]["steps"] == 25
    assert qwen["parameters"]["cfg"] == 1
    assert all(entry["bundled"] for entry in entries)
    assert all(not entry["settings"].get("lora_list") for entry in entries)


def test_readiness_resolves_actual_subfolders_and_does_not_mutate_catalog():
    from bundled_resources import builtin_catalog, inspect_bundle

    entry = next(item for item in builtin_catalog(PLUGIN_DIR) if item["id"] == "qwen21")
    before = copy.deepcopy(entry)
    graph = json.loads((PLUGIN_DIR / "workflows" / "qwen21_edit_api.json").read_text())
    info = {node["class_type"]: {"input": {"required": {}}} for node in graph.values()}
    info["UNETLoader"]["input"]["required"]["unet_name"] = [
        ["自定义目录/qwen_image_2.1_int8_convrot.safetensors"]
    ]
    info["CLIPLoader"]["input"]["required"].update(
        clip_name=[["编码器/qwen3vl_8b_w4a8.safetensors"]],
        type=[["qwen_image"]],
    )
    info["VAELoader"]["input"]["required"]["vae_name"] = [
        ["VAE/qwen_image_2.1_vae_bf16.safetensors"]
    ]
    result = inspect_bundle(entry, PLUGIN_DIR, info, "img2img")
    assert result["ready"] is True
    assert (
        result["resolved"]["model"]
        == "自定义目录/qwen_image_2.1_int8_convrot.safetensors"
    )
    assert result["resolved"]["clip"] == "编码器/qwen3vl_8b_w4a8.safetensors"
    assert entry == before


def test_missing_nodes_and_models_are_reported_separately():
    from bundled_resources import builtin_catalog, inspect_bundle

    entry = next(item for item in builtin_catalog(PLUGIN_DIR) if item["id"] == "qwen21")
    result = inspect_bundle(entry, PLUGIN_DIR, {}, "img2img")
    assert result["ready"] is False
    assert "TextEncodeQwenImage21" in result["missing_nodes"]
    assert {item["role"] for item in result["missing_models"]} == {
        "model",
        "clip",
        "vae",
    }
    assert all(item["url"].startswith("https://") for item in result["missing_models"])


def test_duplicate_basename_is_not_chosen_arbitrarily():
    from bundled_resources import builtin_catalog, inspect_bundle

    entry = next(item for item in builtin_catalog(PLUGIN_DIR) if item["id"] == "qwen21")
    info = {
        "UNETLoader": {
            "input": {
                "required": {
                    "unet_name": [
                        [
                            "A/qwen_image_2.1_int8_convrot.safetensors",
                            "B/qwen_image_2.1_int8_convrot.safetensors",
                        ]
                    ]
                }
            }
        }
    }
    result = inspect_bundle(entry, PLUGIN_DIR, info, "txt2img")
    assert "model" not in result["resolved"]
    assert next(item for item in result["missing_models"] if item["role"] == "model")[
        "ambiguous"
    ]


def test_unsupported_encoder_type_is_not_marked_ready():
    from bundled_resources import builtin_catalog, inspect_bundle

    entry = next(item for item in builtin_catalog(PLUGIN_DIR) if item["id"] == "qwen21")
    graph = json.loads((PLUGIN_DIR / "workflows" / "qwen21_edit_api.json").read_text())
    info = {node["class_type"]: {"input": {"required": {}}} for node in graph.values()}
    for role, class_type, key, filename in (
        ("model", "UNETLoader", "unet_name", "qwen_image_2.1_int8_convrot.safetensors"),
        ("clip", "CLIPLoader", "clip_name", "qwen3vl_8b_w4a8.safetensors"),
        ("vae", "VAELoader", "vae_name", "qwen_image_2.1_vae_bf16.safetensors"),
    ):
        info[class_type]["input"]["required"][key] = [[filename]]
    info["CLIPLoader"]["input"]["required"]["type"] = [["stable_diffusion"]]
    result = inspect_bundle(entry, PLUGIN_DIR, info, "img2img")
    assert result["ready"] is False
    assert any("qwen_image" in item for item in result["unsupported_options"])


@pytest.mark.parametrize("action", ["check", "apply"])
def test_bundle_api_never_changes_settings_when_dependency_check_fails(
    monkeypatch, tmp_path, action
):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    import astrbot.api.web as web

    star = object.__new__(ComfyUIAIStudio)
    star.config = {"model_name": "my-model", "workflow_txt2img": "custom.json"}
    star.plugin_dir = PLUGIN_DIR
    star.model_profiles_path = tmp_path / "profiles.json"
    before = copy.deepcopy(star.config)

    class Request:
        method = "POST"

        async def json(self, **kwargs):
            return {"action": action, "id": "qwen21", "mode": "img2img"}

    class Client:
        async def object_info(self):
            return {}

    monkeypatch.setattr(web, "request", Request())
    star._client = lambda: Client()
    response = asyncio.run(star.api_bundles())
    data = json.loads(response.body)
    assert star.config == before
    assert not star.model_profiles_path.exists()
    if action == "apply":
        assert response.status_code == 400
        assert data["error"]
    else:
        assert response.status_code == 200
        assert data["connected"] is True


def test_bundle_apply_updates_only_selected_mode_and_preserves_saved_sampling(
    monkeypatch, tmp_path
):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio
    from astrbot_plugin_comfyui_ai_studio.model_profiles import save_profile

    import astrbot.api.web as web

    star = object.__new__(ComfyUIAIStudio)
    star.config = {
        "model_name": "my-anima",
        "workflow_txt2img": "custom.json",
        "ai_api_key": "private",
    }
    star.plugin_dir = PLUGIN_DIR
    star.model_profiles_path = tmp_path / "profiles.json"
    model = "Qwen/qwen_image_2.1_int8_convrot.safetensors"
    save_profile(
        star.model_profiles_path,
        model,
        "img2img",
        {"steps": 19, "workflow": "old.json"},
    )
    graph = json.loads((PLUGIN_DIR / "workflows" / "qwen21_edit_api.json").read_text())
    info = {node["class_type"]: {"input": {"required": {}}} for node in graph.values()}
    info["UNETLoader"]["input"]["required"]["unet_name"] = [[model]]
    info["CLIPLoader"]["input"]["required"]["clip_name"] = [
        ["qwen3vl_8b_w4a8.safetensors"]
    ]
    info["VAELoader"]["input"]["required"]["vae_name"] = [
        ["qwen_image_2.1_vae_bf16.safetensors"]
    ]

    class Request:
        method = "POST"

        async def json(self, **kwargs):
            return {"action": "apply", "id": "qwen21", "mode": "img2img"}

    class Client:
        async def object_info(self):
            return info

    monkeypatch.setattr(web, "request", Request())
    star._client = lambda: Client()
    star._save_config = lambda: None
    response = asyncio.run(star.api_bundles())
    assert response.status_code == 200
    assert star.config["img2img_unet_name"] == model
    assert star.config["img2img_steps"] == 19
    assert star.config["workflow_img2img"] == "bundled:qwen21_edit_api.json"
    assert star.config["workflow_txt2img"] == "custom.json"
    assert star.config["model_name"] == "my-anima"
    assert star.config["ai_api_key"] == "private"


def test_bundled_workflow_identifier_survives_installation_move_and_shadowing(tmp_path):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    for installation in (tmp_path / "original", tmp_path / "restored"):
        bundled = installation / "workflows" / "qwen21_edit_api.json"
        bundled.parent.mkdir(parents=True)
        bundled.write_text("{}", encoding="utf-8")
        user_dir = installation / "my_workflows"
        user_dir.mkdir()
        (user_dir / bundled.name).write_text('{"custom":true}', encoding="utf-8")
        star = object.__new__(ComfyUIAIStudio)
        star.plugin_dir = installation
        star.config = {
            "workflow_dir": str(user_dir),
            "workflow_img2img": "bundled:qwen21_edit_api.json",
        }
        assert star._workflow_path("img2img") == bundled
