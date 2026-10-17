"""Portable native ComfyUI defaults, separate from advanced user workflows."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def build_anima_template() -> dict[str, Any]:
    """Build the native Anima Base/2.9B text-to-image graph.

    Returns:
        A fresh API graph with Base defaults and no custom nodes or LoRAs.
    """
    return {
        "anima_model": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "anima-base-v1.0.safetensors",
                "weight_dtype": "default",
            },
        },
        "anima_clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_06b_base.safetensors",
                "type": "qwen_image",
                "device": "cpu",
            },
        },
        "anima_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["anima_clip", 0], "text": ""},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["anima_clip", 0], "text": ""},
        },
        "anima_sampling": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["anima_model", 0], "shift": 3.0},
        },
        "anima_latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "anima_sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["anima_sampling", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["anima_latent", 0],
                "seed": 0,
                "steps": 30,
                "cfg": 5.0,
                "sampler_name": "er_sde",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "anima_decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["anima_sampler", 0], "vae": ["anima_vae", 0]},
        },
        "anima_save": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["anima_decode", 0],
                "filename_prefix": "astrbot/anima",
            },
        },
    }


def adapt_anima_template(
    source: dict[str, Any],
    *,
    model: str,
    clip: str,
    vae: str,
    positive: str,
    negative: str,
    loras: list[tuple[str, float]],
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    sampler_name: str,
    scheduler: str,
    batch: int = 1,
    filename_prefix: str,
) -> dict[str, Any]:
    """Apply selections to a deep copy, chaining native model and CLIP LoRAs.

    Args:
        source: Native graph returned by build_anima_template.
        model: ComfyUI diffusion model identifier for Base or 2.9B.
        clip: ComfyUI text encoder identifier.
        vae: ComfyUI VAE identifier.
        positive: Positive prompt text.
        negative: Negative prompt text.
        loras: Ordered pairs of LoRA identifiers and model/CLIP strengths.
        width: Output width in pixels.
        height: Output height in pixels.
        steps: Number of sampling steps.
        cfg: Classifier-free guidance scale.
        seed: Sampling seed.
        sampler_name: Native sampler selection.
        scheduler: Native scheduler selection.
        batch: Number of output images.
        filename_prefix: ComfyUI SaveImage prefix.

    Returns:
        An independent API graph with all selections on its active output path.
    """
    graph = copy.deepcopy(source)
    graph["anima_model"]["inputs"]["unet_name"] = model
    graph["anima_clip"]["inputs"]["clip_name"] = clip
    graph["anima_vae"]["inputs"]["vae_name"] = vae
    model_output: list[Any] = ["anima_model", 0]
    clip_output: list[Any] = ["anima_clip", 0]
    for index, (name, strength) in enumerate(loras, start=1):
        node_id = f"anima_lora_{index}"
        graph[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_output,
                "clip": clip_output,
                "lora_name": name,
                "strength_model": float(strength),
                "strength_clip": float(strength),
            },
        }
        model_output = [node_id, 0]
        clip_output = [node_id, 1]
    graph["positive"]["inputs"].update(text=positive, clip=clip_output.copy())
    graph["negative"]["inputs"].update(text=negative, clip=clip_output.copy())
    # Match the existing advanced Anima adapter: flow sampling follows LoRAs.
    graph["anima_sampling"]["inputs"].update(model=model_output, shift=3.0)
    graph["anima_latent"]["inputs"].update(
        width=int(width), height=int(height), batch_size=int(batch)
    )
    graph["anima_sampler"]["inputs"].update(
        model=["anima_sampling", 0],
        positive=["positive", 0],
        negative=["negative", 0],
        latent_image=["anima_latent", 0],
        steps=int(steps),
        cfg=float(cfg),
        seed=int(seed),
        sampler_name=sampler_name,
        scheduler=scheduler,
    )
    graph["anima_save"]["inputs"]["filename_prefix"] = filename_prefix
    return graph


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "workflows"
    for filename, graph in (("anima_native_api.json", build_anima_template()),):
        (output / filename).write_text(
            json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
