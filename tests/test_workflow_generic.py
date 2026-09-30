from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR.parent))


def _graph() -> dict[str, dict]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old.safetensors"}},
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old positive", "clip": ["1", 1]},
            "_meta": {"title": "Positive Prompt"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old negative", "clip": ["1", 1]},
            "_meta": {"title": "Negative Prompt"},
        },
        "4": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0], "clip": ["1", 1],
                "lora_name": "old_lora.safetensors", "strength_model": 0.7, "strength_clip": 0.7,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["5", 0], "seed": 1, "steps": 20, "cfg": 5.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "out"}},
    }


def test_generic_inspection_does_not_classify_sampler_connections_as_text() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import inspect_workflow

    summary = inspect_workflow(_graph())
    positive = {(item["node_id"], item["input"]) for item in summary["candidate_inputs"]["positive_text"]}
    negative = {(item["node_id"], item["input"]) for item in summary["candidate_inputs"]["negative_text"]}
    assert positive == {("2", "text")}
    assert negative == {("3", "text")}
    assert ("6", "positive") not in positive
    assert ("6", "negative") not in negative


def test_generic_adapter_rewrites_inputs_and_preserves_links() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_generic, infer_workflow_mapping

    source = _graph()
    original = copy.deepcopy(source)
    mapping = infer_workflow_mapping(source, "txt2img")
    adapted, report = adapt_generic(
        source,
        mode="txt2img",
        positive="一名站在海边的少女",
        negative="模糊",
        model_name="new.safetensors",
        loras=["new_lora.safetensors:0.8"],
        width=768,
        height=512,
        steps=12,
        cfg=2.0,
        seed=123,
        mapping=mapping,
    )
    assert source == original
    assert adapted["2"]["inputs"]["text"] == "一名站在海边的少女"
    assert adapted["3"]["inputs"]["text"] == "模糊"
    assert adapted["6"]["inputs"]["positive"] == ["2", 0]
    assert adapted["6"]["inputs"]["negative"] == ["3", 0]
    assert adapted["1"]["inputs"]["ckpt_name"] == "new.safetensors"
    assert adapted["4"]["inputs"]["lora_name"] == "new_lora.safetensors"
    assert adapted["4"]["inputs"]["strength_model"] == 0.8
    assert adapted["5"]["inputs"]["width"] == 768
    assert adapted["5"]["inputs"]["height"] == 512
    assert adapted["6"]["inputs"]["steps"] == 12
    assert adapted["6"]["inputs"]["cfg"] == 2.0
    assert any("通用适配" in item for item in report)


def test_generic_img2img_requires_and_writes_reference_image() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_generic, infer_workflow_mapping

    source = _graph()
    source["9"] = {"class_type": "LoadImage", "inputs": {"image": "old.png"}}
    source["10"] = {"class_type": "ImageToLatent", "inputs": {"image": ["9", 0]}}
    with pytest.raises(Exception, match="参考图片"):
        adapt_generic(
            source, mode="img2img", positive="换成白色裙子",
            mapping=infer_workflow_mapping(source, "img2img"), image_names=[],
        )
    adapted, _ = adapt_generic(
        source,
        mode="img2img",
        positive="换成白色裙子",
        image_names=["uploaded.png"],
        mapping=infer_workflow_mapping(source, "img2img"),
    )
    assert adapted["9"]["inputs"]["image"] == "uploaded.png"


def test_generic_mapping_rejects_missing_output() -> None:
    from astrbot_plugin_comfyui_ai_studio.workflow import adapt_generic

    source = _graph()
    source.pop("8")
    with pytest.raises(Exception, match="输出"):
        adapt_generic(source, mode="txt2img", positive="一只猫")
