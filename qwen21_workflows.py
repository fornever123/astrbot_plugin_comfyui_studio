"""Small native API graphs for Qwen Image 2.1, independent of torch/AstrBot."""

from __future__ import annotations

import copy
from typing import Any

try:
    from .model_profiles import QWEN21_TURBO_SIGMAS
except ImportError:
    from model_profiles import QWEN21_TURBO_SIGMAS


def build_template(*, edit: bool = False, turbo: bool = False, integrated_turbo: bool = False) -> dict[str, Any]:
    """Return a ComfyUI API graph using native Qwen 2.1 nodes."""
    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": "", "negative_prompt": "", "vae": ["3", 0], "resolution": 0 if edit else 1024}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "astrbot/qwen21"}},
    }
    if edit:
        graph["15"] = {"class_type": "LoadImage", "inputs": {"image": "input.png"}}
    else:
        # Qwen 2.1's text encoder latent is resolution-based and square. Use
        # Comfy's native empty latent so callers can request an explicit aspect
        # ratio; fix_empty_latent_channels adjusts Qwen's latent channel count.
        graph["16"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    if turbo:
        if not integrated_turbo:
            graph["9"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors", "strength_model": 1.0}}
        graph["10"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}}
        graph["11"] = {"class_type": "CFGGuider", "inputs": {"model": ["1" if integrated_turbo else "9", 0], "positive": ["4", 0], "negative": ["4", 1], "cfg": 1.0}}
        graph["12"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        graph["13"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN21_TURBO_SIGMAS}}
        graph["14"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["16", 0] if not edit else ["4", 2]}}
        graph["7"]["inputs"]["samples"] = ["14", 0]
    elif edit:
        graph["5"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["4", 2], "seed": 0, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    else:
        graph["5"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["16", 0], "seed": 0, "steps": 25, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    return copy.deepcopy(graph)


def adapt_template(
    source: dict[str, Any], *, prompt: str, negative: str, model: str, clip: str, vae: str,
    seed: int, resolution: int = 1024, width: int = 1024, height: int = 1024, steps: int = 25, cfg: float = 1.0,
    sampler_name: str = "euler", scheduler: str = "simple", denoise: float = 1.0,
    image_name: str = "", image_names: list[str] | None = None, loras: list[tuple[str, float]] | None = None,
    turbo_lora: str = "", turbo_strength: float = 1.0, filename_prefix: str = "astrbot/qwen21",
) -> dict[str, Any]:
    """Apply user-selected values to a bundled graph without touching host paths."""
    graph = copy.deepcopy(source)
    graph["1"]["inputs"]["unet_name"] = model
    graph["2"]["inputs"]["clip_name"] = clip
    graph["3"]["inputs"]["vae_name"] = vae
    text = graph["4"]["inputs"]
    text.update(prompt=prompt, negative_prompt=negative)
    references = [str(name) for name in (image_names or ([image_name] if image_name else [])) if str(name).strip()][:3]
    if references:
        text.pop("images", None)
        for index, name in enumerate(references, start=1):
            node_id = str(14 + index)
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            # Native Autogrow API fields are flattened with dot notation.
            text[f"images.image_{index}"] = [node_id, 0]
    else:
        for key in tuple(text):
            if key == "images" or key.startswith("images."):
                text.pop(key, None)
    if references:
        text["resolution"] = 0
    else:
        text["resolution"] = max(0, int(resolution))
        if "16" in graph and graph["16"].get("class_type") == "EmptyLatentImage":
            graph["16"]["inputs"].update(width=max(64, int(width)), height=max(64, int(height)))
    graph["8"]["inputs"]["filename_prefix"] = filename_prefix
    if "KSampler" == graph.get("5", {}).get("class_type"):
        graph["5"]["inputs"].update(
            seed=int(seed), steps=int(steps), cfg=float(cfg),
            sampler_name=str(sampler_name), scheduler=str(scheduler), denoise=float(denoise),
        )
    else:
        graph["10"]["inputs"]["noise_seed"] = int(seed)
        graph["11"]["inputs"]["cfg"] = float(cfg)
        graph["12"]["inputs"]["sampler_name"] = str(sampler_name)
        if turbo_lora and "9" in graph:
            graph["9"]["inputs"].update(lora_name=turbo_lora, strength_model=float(turbo_strength))
    # Ordinary LoRAs are layered in order. Keep their names as Comfy object-info
    # identifiers; this module never assumes the Comfy model directory is local.
    entries = [(str(name), float(weight)) for name, weight in (loras or []) if str(name).strip()]
    previous_model: list[Any] = ["9" if "9" in graph else "1", 0]
    previous_clip: list[Any] = ["2", 0]
    for index, (name, weight) in enumerate(entries, start=20):
        graph[str(index)] = {"class_type": "LoraLoader", "inputs": {"model": previous_model, "clip": previous_clip, "lora_name": name, "strength_model": weight, "strength_clip": weight}}
        previous_model = [str(index), 0]
        previous_clip = [str(index), 1]
    if entries:
        if "KSampler" in graph.get("5", {}).get("class_type", ""):
            graph["5"]["inputs"]["model"] = previous_model
        else:
            graph["11"]["inputs"]["model"] = previous_model
        graph["4"]["inputs"]["clip"] = previous_clip
    return graph


if __name__ == "__main__":
    import json
    from pathlib import Path

    output = Path(__file__).resolve().parent / "workflows"
    output.mkdir(parents=True, exist_ok=True)
    for suffix, edit, turbo, integrated in (
        ("t2i", False, False, False),
        ("edit", True, False, False),
        ("t2i_turbo", False, True, False),
        ("edit_turbo", True, True, False),
        ("t2i_turbo_model", False, True, True),
        ("edit_turbo_model", True, True, True),
    ):
        path = output / f"qwen21_{suffix}_api.json"
        path.write_text(json.dumps(build_template(edit=edit, turbo=turbo, integrated_turbo=integrated), ensure_ascii=False, indent=2), encoding="utf-8")
