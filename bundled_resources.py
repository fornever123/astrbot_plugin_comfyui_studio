"""Portable built-in recipes and read-only ComfyUI dependency inspection."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    from .workflow import prune_unreachable
except ImportError:
    from workflow import prune_unreachable

COMFY_UPDATE_URL = "https://github.com/Comfy-Org/ComfyUI"
ANIMA_URL = "https://huggingface.co/circlestone-labs/Anima"
ANIMA_29B_URL = "https://huggingface.co/Gazingstars123/Anima-2.9B"
QWEN_URL = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1"


def builtin_catalog(plugin_dir: Path) -> list[dict[str, Any]]:
    """Describe bundled recipes without querying ComfyUI or user settings.

    Args:
        plugin_dir: Installed plugin source directory.

    Returns:
        Independent recipes with workflow files, starting parameters and links.
    """
    entries = []
    for key, title, models, steps, cfg, sampler, scheduler, url in (
        (
            "anima-base",
            "Anima Base · 原生文生图",
            ["anima-base-v1.0.safetensors", "anima_baseV10.safetensors"],
            30,
            5.0,
            "er_sde",
            "simple",
            ANIMA_URL,
        ),
        (
            "anima-29b",
            "Anima 2.9B · 原生文生图",
            [
                "Anima-2.9B-preview-v1.safetensors",
                "Anima-2.9B-preview-v1_int8_convrot.safetensors",
            ],
            35,
            3.5,
            "euler",
            "sgm_uniform",
            ANIMA_29B_URL,
        ),
    ):
        entries.append(
            {
                "id": key,
                "title": title,
                "workflows": {"txt2img": "anima_native_api.json"},
                "parameters": {
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "width": 832,
                    "height": 1216,
                    "denoise": 1.0,
                },
                "settings": {"qwen21_turbo_enabled": False},
                "components": {
                    "model": {
                        "node": "anima_model",
                        "input": "unet_name",
                        "names": models,
                        "folder": "diffusion_models",
                        "url": url,
                    },
                    "clip": {
                        "node": "anima_clip",
                        "input": "clip_name",
                        "names": ["qwen_3_06b_base.safetensors"],
                        "folder": "text_encoders",
                        "url": ANIMA_URL,
                    },
                    "vae": {
                        "node": "anima_vae",
                        "input": "vae_name",
                        "names": ["qwen_image_vae.safetensors"],
                        "folder": "vae",
                        "url": ANIMA_URL,
                    },
                },
                "notes": "使用原生节点，支持多个 LoRA；参数是可修改初值。模型权重遵守作者许可证。"
                + (
                    " 2.9B 需要新版 ComfyUI 的模型加载支持；旧版可按作者说明安装扩展。"
                    if key == "anima-29b"
                    else ""
                ),
                "url": url,
            }
        )
    for turbo in (False, True):
        components = {
            "model": {
                "node": "1",
                "input": "unet_name",
                "names": [
                    "qwen_image_2.1_int8_convrot.safetensors",
                    "qwen_image_2.1_bf16.safetensors",
                ],
                "folder": "diffusion_models",
                "url": QWEN_URL + "/tree/main/diffusion_models",
            },
            "clip": {
                "node": "2",
                "input": "clip_name",
                "names": ["qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_bf16.safetensors"],
                "folder": "text_encoders",
                "url": QWEN_URL + "/tree/main/text_encoders",
            },
            "vae": {
                "node": "3",
                "input": "vae_name",
                "names": ["qwen_image_2.1_vae_bf16.safetensors"],
                "folder": "vae",
                "url": QWEN_URL + "/tree/main/vae",
            },
        }
        if turbo:
            components["turbo_lora"] = {
                "node": "9",
                "input": "lora_name",
                "names": ["qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"],
                "folder": "loras",
                "url": QWEN_URL + "/tree/main/loras",
            }
        entries.append(
            {
                "id": "qwen21-turbo" if turbo else "qwen21",
                "title": "Qwen Image 2.1 · " + ("Turbo 8 步" if turbo else "普通模式"),
                "workflows": {
                    "txt2img": "qwen21_t2i" + ("_turbo" if turbo else "") + "_api.json",
                    "img2img": "qwen21_edit"
                    + ("_turbo" if turbo else "")
                    + "_api.json",
                },
                "parameters": {
                    "steps": 8 if turbo else 25,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "manual_sigmas" if turbo else "simple",
                    "width": 1024,
                    "height": 1024,
                    "resolution": 1024,
                    "denoise": 1.0,
                },
                "settings": {"qwen21_turbo_lora_strength": 1.0},
                "components": components,
                "notes": "中文自然语言，图生图支持 1–3 张参考图并保持第一张图宽高比。需要新版 ComfyUI 的 TextEncodeQwenImage21。CFG 1 时负面词不参与引导。"
                + (
                    " 使用基础模型加 Turbo LoRA，不与已蒸馏的 Turbo 主模型叠加。"
                    if turbo
                    else ""
                ),
                "url": QWEN_URL,
            }
        )
    for entry in entries:
        entry["bundled"] = all(
            (plugin_dir / "workflows" / name).is_file()
            for name in entry["workflows"].values()
        )
    return entries


def inspect_bundle(
    entry: dict[str, Any], plugin_dir: Path, object_info: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Inspect the recipe's actual loader choices and reachable node classes.

    Args:
        entry: Built-in recipe, not a caller-provided workflow.
        plugin_dir: Installed source directory.
        object_info: One snapshot of ComfyUI's /object_info response.
        mode: Mode included in the recipe.

    Returns:
        Dependency diagnostics and exact server-side filenames. A passing check
        does not claim GPU memory, file contents or generation quality verified.
    """
    result: dict[str, Any] = {
        "ready": False,
        "resolved": {},
        "missing_nodes": [],
        "missing_models": [],
        "unsupported_options": [],
        "workflow_error": "",
    }
    try:
        graph = json.loads(
            (plugin_dir / "workflows" / entry["workflows"][mode]).read_text(
                encoding="utf-8-sig"
            )
        )
        graph = copy.deepcopy(graph)
        if mode == "img2img" and "4" in graph:
            graph["4"]["inputs"]["images.image_1"] = ["15", 0]
        for node in graph.values():
            if node.get("class_type") == "KSampler":
                node["inputs"].update(
                    {
                        key: entry["parameters"][key]
                        for key in ("sampler_name", "scheduler")
                    }
                )
        graph, _ = prune_unreachable(graph)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["workflow_error"] = f"内置工作流不可用：{exc}"
        return result
    result["missing_nodes"] = sorted(
        {node["class_type"] for node in graph.values()} - set(object_info)
    )
    for role, component in entry["components"].items():
        node = graph.get(component["node"], {})
        class_type = node.get("class_type", "")
        required = object_info.get(class_type, {}).get("input", {}).get("required", {})
        spec = required.get(component["input"], [])
        choices = (
            spec[0]
            if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], list)
            else []
        )
        choices = [name for name in choices if isinstance(name, str)]
        resolved = ""
        ambiguous = False
        for expected in component["names"]:
            normalized = expected.replace("\\", "/").casefold()
            exact = [
                name
                for name in choices
                if name.replace("\\", "/").casefold() == normalized
            ]
            matches = exact or [
                name
                for name in choices
                if Path(name.replace("\\", "/")).name.casefold()
                == Path(normalized).name
            ]
            if len(matches) == 1:
                resolved = matches[0]
                break
            if len(matches) > 1:
                ambiguous = True
                break
        if resolved:
            result["resolved"][role] = resolved
        else:
            result["missing_models"].append(
                {
                    "role": role,
                    "names": component["names"],
                    "folder": "models/" + component["folder"],
                    "url": component["url"],
                    "ambiguous": ambiguous,
                }
            )
    for node in graph.values():
        schema = object_info.get(node["class_type"], {}).get("input", {})
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for key in ("type", "sampler_name", "scheduler"):
            value = node.get("inputs", {}).get(key)
            spec = inputs.get(key, [])
            options = (
                spec[0]
                if isinstance(spec, (list, tuple))
                and spec
                and isinstance(spec[0], list)
                else None
            )
            if value is not None and options is not None and value not in options:
                result["unsupported_options"].append(
                    f"{node['class_type']}.{key} 不支持 {value}"
                )
    result["ready"] = not any(
        result[key]
        for key in (
            "missing_nodes",
            "missing_models",
            "unsupported_options",
            "workflow_error",
        )
    )
    return result
