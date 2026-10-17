from __future__ import annotations

from pathlib import Path

import sys

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from qwen21_workflows import adapt_template, build_template  # noqa: E402
from workflow import prune_unreachable  # noqa: E402


def _pruned(graph: dict) -> dict:
    result, _ = prune_unreachable(graph)
    ids = set(result)
    for node_id, node in result.items():
        for value in node.get("inputs", {}).values():
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in ids, f"dangling connection {node_id}: {value}"
    assert set(result) == _reachable(result)
    return result


def _reachable(graph: dict) -> set[str]:
    output_ids = {
        node_id for node_id, node in graph.items()
        if node.get("class_type") == "SaveImage"
    }
    reached: set[str] = set()
    stack = list(output_ids)
    while stack:
        node_id = stack.pop()
        if node_id in reached or node_id not in graph:
            continue
        reached.add(node_id)
        for value in graph[node_id].get("inputs", {}).values():
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str):
                stack.append(value[0])
    return reached


def _adapt(source: dict, **overrides) -> dict:
    args = {
        "prompt": "中文画面描述",
        "negative": "",
        "model": "qwen_image_2.1_int8_convrot.safetensors",
        "clip": "qwen3vl_8b_w4a8.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
        "seed": 123,
        "width": 832,
        "height": 1216,
        "loras": [],
    }
    args.update(overrides)
    return adapt_template(source, **args)


def test_text_to_image_uses_custom_aspect_ratio_and_prunes_cleanly() -> None:
    graph = _pruned(_adapt(
        build_template(),
        steps=31,
        cfg=1.2,
        sampler_name="euler_ancestral",
        scheduler="normal",
        width=832,
        height=1216,
    ))
    assert graph["16"]["class_type"] == "EmptyLatentImage"
    assert graph["16"]["inputs"]["width"] == 832
    assert graph["16"]["inputs"]["height"] == 1216
    assert graph["5"]["inputs"]["latent_image"] == ["16", 0]
    assert graph["5"]["inputs"]["steps"] == 31
    assert graph["5"]["inputs"]["cfg"] == 1.2
    assert graph["5"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert graph["5"]["inputs"]["scheduler"] == "normal"
    assert "images" not in graph["4"]["inputs"]


def test_edit_images_use_native_flattened_autogrow_links_and_survive_pruning() -> None:
    graph = _pruned(_adapt(
        build_template(edit=True),
        image_names=["ref-a.png", "ref-b.png", "ref-c.png"],
    ))
    text_inputs = graph["4"]["inputs"]
    assert text_inputs["images.image_1"] == ["15", 0]
    assert text_inputs["images.image_2"] == ["16", 0]
    assert text_inputs["images.image_3"] == ["17", 0]
    assert text_inputs["resolution"] == 0
    assert {graph[node_id]["inputs"]["image"] for node_id in ("15", "16", "17")} == {
        "ref-a.png", "ref-b.png", "ref-c.png"
    }


def test_turbo_with_regular_loras_keeps_both_model_chains() -> None:
    graph = _pruned(_adapt(
        build_template(turbo=True),
        loras=[("style.safetensors", 0.6), ("character.safetensors", 0.8)],
        turbo_lora="qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
        turbo_strength=0.75,
    ))
    assert graph["9"]["inputs"]["lora_name"] == "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"
    assert graph["12"]["inputs"]["sampler_name"] == "euler"
    assert graph["20"]["inputs"]["model"] == ["9", 0]
    assert graph["21"]["inputs"]["model"] == ["20", 0]
    assert graph["11"]["inputs"]["model"] == ["21", 0]
    assert graph["4"]["inputs"]["clip"] == ["21", 1]
    assert graph["16"]["inputs"]["width"] == 832


def test_integrated_turbo_model_does_not_try_to_load_turbo_lora() -> None:
    graph = _pruned(_adapt(
        build_template(turbo=True, integrated_turbo=True),
        turbo_lora="must-not-be-added.safetensors",
        loras=[("style.safetensors", 0.7)],
    ))
    assert "9" not in graph
    assert not any(node.get("class_type") == "LoraLoaderModelOnly" for node in graph.values())
    assert graph["20"]["inputs"]["model"] == ["1", 0]
    assert graph["11"]["inputs"]["model"] == ["20", 0]


def test_normal_edit_with_loras_and_three_references_is_reachable() -> None:
    graph = _pruned(_adapt(
        build_template(edit=True),
        image_names=["one.png", "two.png", "three.png"],
        loras=[("style.safetensors", 0.5)],
    ))
    assert graph["5"]["inputs"]["model"] == ["20", 0]
    assert graph["4"]["inputs"]["clip"] == ["20", 1]
    assert all(f"images.image_{index}" in graph["4"]["inputs"] for index in range(1, 4))
