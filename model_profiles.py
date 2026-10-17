"""Portable model-family recommendations and profile persistence.

This module intentionally uses only the Python standard library. Model files
remain on the ComfyUI host; AstrBot can resolve their names through ComfyUI's
object-info API without mounting the model directory or installing torch.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

QWEN21_TURBO_SIGMAS = (
    "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"
)


def explicit_numeric_overrides(
    text: str, allowed: tuple[str, ...] | None = None
) -> dict[str, int | float]:
    """Parse only values explicitly stated by the user, never values guessed by an LLM."""
    source = str(text or "")
    patterns: dict[str, tuple[str, ...]] = {
        "steps": (
            r"(?:--steps\s*=\s*|(?:步数|steps?)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(\d+)\s*(?:步)?",
            r"(?<!\d)(\d+)\s*步(?!\w)",
        ),
        "cfg": (
            r"(?:--cfg\s*=\s*|(?:cfg|引导系数)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(\d+(?:\.\d+)?)",
        ),
        "seed": (
            r"(?:--seed\s*=\s*|(?:种子|seed)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(-?\d+)",
        ),
        "width": (
            r"(?:--width\s*=\s*|(?:宽度|宽|width)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(\d{2,5})",
        ),
        "height": (
            r"(?:--height\s*=\s*|(?:高度|高|height)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(\d{2,5})",
        ),
        "denoise": (
            r"(?:--denoise\s*=\s*|(?:重绘幅度|denoise)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(\d+(?:\.\d+)?)",
        ),
        "batch": (
            r"(?:--batch\s*=\s*|(?:批量数|生成数量|batch)\s*(?:为|是|设为|设置为|到|=|:|：)?\s*)(\d+)",
        ),
    }
    result: dict[str, int | float] = {}
    selected = set(allowed or patterns)
    for key, pattern_list in patterns.items():
        if key not in selected:
            continue
        match = next(
            (
                match
                for pattern in pattern_list
                if (match := re.search(pattern, source, re.IGNORECASE))
            ),
            None,
        )
        if not match:
            continue
        raw = next(group for group in match.groups() if group is not None)
        result[key] = float(raw) if "." in raw else int(raw)
    dimensions = re.search(
        r"(?<!\d)(\d{2,5})\s*[x×*]\s*(\d{2,5})(?!\d)", source, re.IGNORECASE
    )
    if dimensions and {"width", "height"}.issubset(selected):
        result["width"], result["height"] = (
            int(dimensions.group(1)),
            int(dimensions.group(2)),
        )
    return result


def profile_key(model_name: str, mode: str, accelerated: bool = False) -> str:
    """Build a stable profile key from a ComfyUI filename and draw mode."""
    normalized = str(model_name or "").replace("\\", "/").strip("/").casefold()
    return f"{mode}:{normalized}:{'turbo' if accelerated else 'normal'}"


def recommended_profile(
    model_name: str,
    mode: str = "txt2img",
    accelerated: bool = False,
) -> dict[str, Any]:
    """Return known sampling defaults without guessing unknown architectures."""
    normalized = str(model_name or "").replace("-", "_").casefold()
    is_qwen21 = "qwen_image_2.1" in normalized or "qwen_image_2_1" in normalized
    is_qwen21_turbo_model = is_qwen21 and "turbo" in normalized
    if is_qwen21:
        if accelerated or is_qwen21_turbo_model:
            return {
                "family": "Qwen Image 2.1",
                "recognized": True,
                "verified": False,
                "official_recommendation": True,
                "mode": mode,
                "accelerated": True,
                "integrated_acceleration": is_qwen21_turbo_model,
                "recommended": {
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "manual_sigmas",
                    **({"width": 1024, "height": 1024} if mode == "txt2img" else {}),
                    "denoise": 1.0,
                    "sigmas": QWEN21_TURBO_SIGMAS,
                    "turbo_lora": ""
                    if is_qwen21_turbo_model
                    else "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
                    "turbo_lora_strength": 0.0 if is_qwen21_turbo_model else 1.0,
                    "clip_name": "qwen3vl_8b_w4a8.safetensors",
                    "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
                    "workflow": (
                        (
                            "qwen21_t2i_turbo_model_api.json"
                            if is_qwen21_turbo_model
                            else "qwen21_t2i_turbo_api.json"
                        )
                        if mode == "txt2img"
                        else (
                            "qwen21_edit_turbo_model_api.json"
                            if is_qwen21_turbo_model
                            else "qwen21_edit_turbo_api.json"
                        )
                    ),
                },
                "description": "Turbo 使用官方 8 段 sigma 序列并以 0 收尾；Turbo 主模型已内置蒸馏，不应再次叠加 Turbo LoRA。CFG 固定为 1。",
            }
        return {
            "family": "Qwen Image 2.1",
            "recognized": True,
            "verified": False,
            "official_recommendation": True,
            "mode": mode,
            "accelerated": False,
            "recommended": {
                "steps": 25,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "resolution": 1024,
                "width": 1024,
                "height": 1024,
                "denoise": 1.0,
                "clip_name": "qwen3vl_8b_w4a8.safetensors",
                "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
                "workflow": "qwen21_t2i_api.json"
                if mode == "txt2img"
                else "qwen21_edit_api.json",
            },
            "description": "官方普通模式采用 25 步、CFG 1、Euler/simple、分辨率 1024；编辑模式保持参考图宽高比。CFG 1 时负面提示词不参与引导。",
        }

    anima_29b_markers = ("anima29b", "anima_2.9b", "anima_29b")
    if mode == "txt2img" and any(marker in normalized for marker in anima_29b_markers):
        return {
            "family": "Anima-2.9B",
            "recognized": True,
            "verified": False,
            "official_recommendation": False,
            "mode": mode,
            "accelerated": False,
            "recommended": {
                "steps": 35,
                "cfg": 3.5,
                "workflow": "anima_native_api.json",
                "clip_name": "qwen_3_06b_base.safetensors",
                "vae_name": "qwen_image_vae.safetensors",
                "sampler_name": "euler",
                "scheduler": "sgm_uniform",
                "denoise": 1.0,
            },
            "description": "需要 Anima-2.9B 动态层数支持；这是推荐初值，已保存的模型档案优先。",
        }

    anima_base_markers = ("anima-base", "anima_base", "anima base")
    if mode == "txt2img" and any(
        marker.replace("-", "_") in normalized for marker in anima_base_markers
    ):
        return {
            "family": "Anima Base",
            "recognized": True,
            "verified": False,
            "official_recommendation": False,
            "mode": mode,
            "accelerated": False,
            "recommended": {
                "steps": 30,
                "cfg": 5.0,
                "workflow": "anima_native_api.json",
                "clip_name": "qwen_3_06b_base.safetensors",
                "vae_name": "qwen_image_vae.safetensors",
                "sampler_name": "er_sde",
                "scheduler": "simple",
                "denoise": 1.0,
            },
            "description": "Anima Base 推荐初值；已保存的模型档案优先。",
        }

    return {
        "family": "未识别模型",
        "recognized": False,
        "verified": False,
        "mode": mode,
        "accelerated": accelerated,
        "recommended": {},
        "description": "未识别架构，不自动套用其它模型参数；可保存自己的模型档案。",
    }


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    """Read user-owned profiles, returning an empty mapping when absent."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, dict)
    }


def save_profile(
    path: Path,
    model_name: str,
    mode: str,
    profile: dict[str, Any],
    accelerated: bool = False,
) -> None:
    """Persist one model profile without discarding other user profiles."""
    profiles = load_profiles(path)
    saved = dict(profile)
    family = recommended_profile(model_name, mode, accelerated)
    if family.get("family") == "Qwen Image 2.1" and family.get("accelerated"):
        # Turbo's official sigma vector has exactly eight transitions. A KSampler
        # step count override cannot alter ManualSigmas, so persist the real value.
        saved["steps"] = 8
        saved["scheduler"] = "manual_sigmas"
    profiles[profile_key(model_name, mode, accelerated)] = saved
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def apply_profile_defaults(
    model_name: str,
    mode: str,
    saved: dict[str, Any] | None = None,
    accelerated: bool = False,
) -> dict[str, Any]:
    """Apply precedence: saved model profile over known family recommendations."""
    profile = recommended_profile(model_name, mode, accelerated)
    recommended = profile.get("recommended", {})
    saved_values = saved if isinstance(saved, dict) else {}
    profile["recommended"] = {**recommended, **saved_values}
    if profile.get("family") == "Qwen Image 2.1" and profile.get("accelerated"):
        profile["recommended"]["steps"] = 8
        profile["recommended"]["scheduler"] = "manual_sigmas"
    profile["saved"] = bool(saved_values)
    return profile
