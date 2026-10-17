"""Native Krea 2 API graph for the ComfyUI plugin (no torch / AstrBot imports).

Krea 2 Turbo is a Qwen-Image-family distilled model: it needs the Qwen3VL-4B
text encoder, the Qwen-Image VAE and ``CLIPLoader(type="krea2")``.  When it is
stored as GGUF it must be loaded through ``UnetLoaderGGUF`` instead of
``UNETLoader``, which is also how the executor recognises this graph.
"""

from __future__ import annotations

import copy
from typing import Any

KREA2_UNET_DEFAULT = "krea2_turbo-Q4_K_M.gguf"
KREA2_CLIP_DEFAULT = "qwen3vl_4b_fp8_scaled.safetensors"
KREA2_VAE_DEFAULT = "qwen_image_vae.safetensors"
KREA2_STEPS_DEFAULT = 8
KREA2_CFG_DEFAULT = 1.0
KREA2_SAMPLER_DEFAULT = "euler"
KREA2_SCHEDULER_DEFAULT = "simple"


def build_template(
    *,
    model: str = KREA2_UNET_DEFAULT,
    clip: str = KREA2_CLIP_DEFAULT,
    vae: str = KREA2_VAE_DEFAULT,
    filename_prefix: str = "astrbot/krea2",
) -> dict[str, Any]:
    """Return a ComfyUI API graph for Krea 2 Turbo text-to-image."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model}},
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        # cfg=1 时负面分支不参与引导，仍要求非空 conditioning，
        # 所以用 ConditioningZeroOut 而不是直接接空文本编码。
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": KREA2_STEPS_DEFAULT,
                "cfg": KREA2_CFG_DEFAULT,
                "sampler_name": KREA2_SAMPLER_DEFAULT,
                "scheduler": KREA2_SCHEDULER_DEFAULT,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": filename_prefix},
        },
    }


def adapt_template(
    source: dict[str, Any],
    *,
    prompt: str,
    negative: str = "",
    model: str = "",
    clip: str = "",
    vae: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = KREA2_STEPS_DEFAULT,
    cfg: float = KREA2_CFG_DEFAULT,
    sampler_name: str = KREA2_SAMPLER_DEFAULT,
    scheduler: str = KREA2_SCHEDULER_DEFAULT,
    denoise: float = 1.0,
    loras: list[tuple[str, float]] | None = None,
    filename_prefix: str = "astrbot/krea2",
) -> dict[str, Any]:
    """Fill a bundled Krea 2 graph with one request's values.

    LoRA names are kept exactly as ComfyUI reports them; this module never
    assumes the model directory is reachable from the plugin process.
    """
    graph = copy.deepcopy(source)

    if "1" in graph and graph["1"].get("class_type") == "UnetLoaderGGUF" and model:
        graph["1"]["inputs"]["unet_name"] = model
    if "2" in graph and clip:
        graph["2"]["inputs"]["clip_name"] = clip
        graph["2"]["inputs"].setdefault("type", "krea2")
        graph["2"]["inputs"].setdefault("device", "default")
    if "3" in graph and vae:
        graph["3"]["inputs"]["vae_name"] = vae

    if "4" in graph:
        graph["4"]["inputs"]["text"] = str(prompt or "")
    if "5" in graph:
        graph["5"]["inputs"]["text"] = str(negative or "")

    latent = graph.get("7")
    if isinstance(latent, dict) and latent.get("class_type") == "EmptyLatentImage":
        latent["inputs"].update(
            width=max(64, int(width or 1024)),
            height=max(64, int(height or 1024)),
            batch_size=1,
        )

    entries = [(str(name), float(weight)) for name, weight in (loras or []) if str(name).strip()]
    previous_model: list[Any] = ["1", 0]
    for index, (name, weight) in enumerate(entries, start=20):
        graph[str(index)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": previous_model, "lora_name": name, "strength_model": weight},
        }
        previous_model = [str(index), 0]

    sampler = graph.get("8")
    if isinstance(sampler, dict) and sampler.get("class_type") == "KSampler":
        sampler["inputs"].update(
            seed=int(seed),
            steps=max(1, int(steps or KREA2_STEPS_DEFAULT)),
            cfg=float(cfg),
            sampler_name=str(sampler_name or KREA2_SAMPLER_DEFAULT),
            scheduler=str(scheduler or KREA2_SCHEDULER_DEFAULT),
            denoise=float(denoise),
            model=previous_model,
        )

    if "10" in graph:
        graph["10"]["inputs"]["filename_prefix"] = filename_prefix
    return graph


if __name__ == "__main__":
    import json
    from pathlib import Path

    output = Path(__file__).resolve().parent / "workflows"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "krea2_t2i_api.json"
    # write_bytes：仓库有 LF 守卫，Path.write_text 在 Windows 会翻成 CRLF。
    path.write_bytes(
        json.dumps(build_template(), ensure_ascii=False, indent=2).encode("utf-8")
    )
    print(f"written {path}")
