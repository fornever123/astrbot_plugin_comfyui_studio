from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from model_profiles import (  # noqa: E402
    apply_profile_defaults,
    profile_key,
    recommended_profile,
    save_profile,
)


def test_qwen21_normal_and_turbo_profiles_are_separate() -> None:
    model = "qwen_image_2.1_int8_convrot.safetensors"
    normal = recommended_profile(model, "txt2img")
    turbo = recommended_profile(model, "txt2img", accelerated=True)
    assert normal["recommended"]["steps"] == 25
    assert normal["recommended"]["cfg"] == 1
    assert normal["recommended"]["width"] == 1024
    assert normal["recommended"]["scheduler"] == "simple"
    assert turbo["recommended"]["steps"] == 8
    assert turbo["recommended"]["scheduler"] == "manual_sigmas"
    assert turbo["recommended"]["turbo_lora"]
    assert profile_key(model, "txt2img", False) != profile_key(model, "txt2img", True)


def test_integrated_turbo_model_does_not_recommend_a_turbo_lora() -> None:
    profile = recommended_profile("qwen_image_2.1_turbo.safetensors", "img2img")
    assert profile["integrated_acceleration"] is True
    assert profile["recommended"]["steps"] == 8
    assert profile["recommended"]["turbo_lora"] == ""


def test_anima_model_switch_selects_native_workflow_and_correct_components():
    for model in ("anima-base-v1.0.safetensors", "Anima-2.9B-preview-v1.safetensors"):
        profile = recommended_profile(model, "txt2img")
        assert profile["recommended"]["workflow"] == "anima_native_api.json"
        assert profile["recommended"]["clip_name"] == "qwen_3_06b_base.safetensors"
        assert profile["recommended"]["vae_name"] == "qwen_image_vae.safetensors"


def test_anima_text_workflow_is_never_recommended_for_qwen_image_edit_slot():
    for model in ("anima-base-v1.0.safetensors", "Anima-2.9B-preview-v1.safetensors"):
        profile = recommended_profile(model, "img2img")
        assert profile["recognized"] is False
        assert profile["recommended"] == {}


def test_saving_one_profile_preserves_other_model_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    save_profile(
        path, "qwen_image_2.1_int8_convrot.safetensors", "txt2img", {"steps": 28}
    )
    save_profile(
        path, "qwen_image_2.1_int8_convrot.safetensors", "img2img", {"steps": 19}, True
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (
        data[profile_key("qwen_image_2.1_int8_convrot.safetensors", "txt2img")]["steps"]
        == 28
    )
    assert (
        data[profile_key("qwen_image_2.1_int8_convrot.safetensors", "img2img", True)][
            "steps"
        ]
        == 8
    )
    merged = apply_profile_defaults(
        "qwen_image_2.1_int8_convrot.safetensors", "txt2img", {"steps": 28}
    )
    assert merged["recommended"]["steps"] == 28
    turbo_merged = apply_profile_defaults(
        "qwen_image_2.1_int8_convrot.safetensors",
        "img2img",
        {"steps": 19},
        accelerated=True,
    )
    assert turbo_merged["recommended"]["steps"] == 8
