"""Portable native graph contracts, exercised through the real adapters."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from workflow import (  # noqa: E402
    load_workflow,
    prune_unreachable,
)

NATIVE_TYPES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ModelSamplingAuraFlow",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "VAEEncode",
    "SaveImage",
    "LoadImage",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "ConditioningZeroOut",
    "ReferenceLatent",
    "ImageScaleToTotalPixels",
    "GetImageSize",
}


def _bundled():
    return importlib.import_module("bundled_workflows")


def _assert_native_links(graph: dict) -> None:
    assert graph
    for node_id, node in graph.items():
        assert node["class_type"] in NATIVE_TYPES
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert len(value) == 2
                assert value[0] in graph, f"Dangling link from {node_id}: {value}"
                assert isinstance(value[1], int) and value[1] >= 0


def _adapt_anima(source: dict, **overrides) -> dict:
    options = {
        "model": "anima-base-v1.0.safetensors",
        "clip": "selected-clip.safetensors",
        "vae": "selected-vae.safetensors",
        "positive": "A seaside portrait",
        "negative": "blur",
        "loras": [],
        "width": 832,
        "height": 1216,
        "steps": 27,
        "cfg": 4.5,
        "seed": 12345,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "batch": 2,
        "filename_prefix": "astrbot/test-anima",
    }
    options.update(overrides)
    return _bundled().adapt_anima_template(source, **options)


def test_anima_builder_has_native_loaders_and_complete_output_path() -> None:
    graph = _bundled().build_anima_template()
    _assert_native_links(graph)
    assert graph["anima_model"]["inputs"]["unet_name"] == "anima-base-v1.0.safetensors"
    assert graph["anima_clip"]["inputs"] == {
        "clip_name": "qwen_3_06b_base.safetensors",
        "type": "qwen_image",
        "device": "cpu",
    }
    assert graph["anima_vae"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors"
    assert graph["positive"]["class_type"] == "CLIPTextEncode"
    assert graph["negative"]["class_type"] == "CLIPTextEncode"
    sampler = graph["anima_sampler"]["inputs"]
    sampling = graph[sampler["model"][0]]
    assert sampling["class_type"] == "ModelSamplingAuraFlow"
    assert sampling["inputs"] == {"model": ["anima_model", 0], "shift": 3.0}
    assert sampler["positive"] == ["positive", 0]
    assert sampler["negative"] == ["negative", 0]
    assert graph["anima_decode"]["inputs"] == {
        "samples": ["anima_sampler", 0],
        "vae": ["anima_vae", 0],
    }
    assert graph["anima_save"]["inputs"]["images"] == ["anima_decode", 0]
    pruned, removed = prune_unreachable(graph)
    assert not removed
    assert pruned == graph


@pytest.mark.parametrize(
    "model", ["anima-base-v1.0.safetensors", "anima29b.safetensors"]
)
def test_anima_adapter_applies_values_to_actual_sampler_without_mutation(
    model: str,
) -> None:
    source = _bundled().build_anima_template()
    before = copy.deepcopy(source)
    graph = _adapt_anima(source, model=model)
    assert source == before
    _assert_native_links(graph)
    assert graph["anima_model"]["inputs"]["unet_name"] == model
    assert graph["anima_clip"]["inputs"]["clip_name"] == "selected-clip.safetensors"
    assert graph["anima_vae"]["inputs"]["vae_name"] == "selected-vae.safetensors"
    assert graph["positive"]["inputs"]["text"] == "A seaside portrait"
    assert graph["negative"]["inputs"]["text"] == "blur"
    sampler = graph["anima_sampler"]["inputs"]
    assert {
        key: sampler[key]
        for key in (
            "steps",
            "cfg",
            "seed",
            "sampler_name",
            "scheduler",
            "denoise",
        )
    } == {
        "steps": 27,
        "cfg": 4.5,
        "seed": 12345,
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "denoise": 1.0,
    }
    assert graph[sampler["latent_image"][0]]["inputs"] == {
        "width": 832,
        "height": 1216,
        "batch_size": 2,
    }
    assert graph["anima_save"]["inputs"]["filename_prefix"] == "astrbot/test-anima"
    graph["positive"]["inputs"]["clip"][0] = "changed"
    assert source == before


def test_anima_multiple_loras_feed_both_conditioning_and_sampling_chains() -> None:
    source = _bundled().build_anima_template()
    before = copy.deepcopy(source)
    graph = _adapt_anima(
        source, loras=[("style.safetensors", 0.6), ("character.safetensors", -0.2)]
    )
    assert source == before
    sampling = graph[graph["anima_sampler"]["inputs"]["model"][0]]
    last_id = sampling["inputs"]["model"][0]
    last = graph[last_id]
    assert last["class_type"] == "LoraLoader"
    assert last["inputs"]["lora_name"] == "character.safetensors"
    first_id = last["inputs"]["model"][0]
    assert last["inputs"]["clip"] == [first_id, 1]
    assert last["inputs"]["strength_model"] == -0.2
    assert last["inputs"]["strength_clip"] == -0.2
    assert graph[first_id]["inputs"] == {
        "model": ["anima_model", 0],
        "clip": ["anima_clip", 0],
        "lora_name": "style.safetensors",
        "strength_model": 0.6,
        "strength_clip": 0.6,
    }
    assert graph["positive"]["inputs"]["clip"] == [last_id, 1]
    assert graph["negative"]["inputs"]["clip"] == [last_id, 1]
    pruned, removed = prune_unreachable(graph)
    assert not removed
    _assert_native_links(pruned)


def test_anima_no_lora_uses_bare_loaders_and_default_batch() -> None:
    graph = _bundled().adapt_anima_template(
        _bundled().build_anima_template(),
        model="anima-base-v1.0.safetensors",
        clip="qwen_3_06b_base.safetensors",
        vae="qwen_image_vae.safetensors",
        positive="portrait",
        negative="blur",
        loras=[],
        width=1024,
        height=1024,
        steps=30,
        cfg=5.0,
        seed=0,
        sampler_name="er_sde",
        scheduler="simple",
        filename_prefix="astrbot/anima",
    )
    assert not any(node["class_type"] == "LoraLoader" for node in graph.values())
    assert graph["positive"]["inputs"]["clip"] == ["anima_clip", 0]
    assert graph["negative"]["inputs"]["clip"] == ["anima_clip", 0]
    assert graph["anima_latent"]["inputs"]["batch_size"] == 1


@pytest.mark.parametrize(
    "filename,builder",
    [
        ("anima_native_api.json", "build_anima_template"),
    ],
)
def test_shipping_api_graph_matches_builder_and_loads_without_conversion(
    filename: str, builder: str
) -> None:
    path = PLUGIN_ROOT / "workflows" / filename
    graph = load_workflow(path)
    assert graph == getattr(_bundled(), builder)()
    assert graph == json.loads(path.read_text(encoding="utf-8"))
    _assert_native_links(graph)
