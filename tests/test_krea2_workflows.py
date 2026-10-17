from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from krea2_workflows import (  # noqa: E402
    KREA2_CFG_DEFAULT,
    KREA2_CLIP_DEFAULT,
    KREA2_STEPS_DEFAULT,
    KREA2_UNET_DEFAULT,
    KREA2_VAE_DEFAULT,
    adapt_template,
    build_template,
)
from workflow import prune_unreachable  # noqa: E402

WORKFLOW_FILE = PLUGIN_ROOT / "workflows" / "krea2_t2i_api.json"


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


def _pruned(graph: dict) -> dict:
    result, _ = prune_unreachable(graph)
    ids = set(result)
    for node_id, node in result.items():
        for value in node.get("inputs", {}).values():
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in ids, f"dangling connection {node_id}: {value}"
    assert set(result) == _reachable(result)
    return result


def test_bundled_workflow_file_matches_builder() -> None:
    assert WORKFLOW_FILE.is_file(), "Krea 2 内置工作流缺失，/水粉 会直接报错"
    bundled = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
    assert bundled == build_template()


def test_bundled_workflow_file_keeps_lf() -> None:
    raw = WORKFLOW_FILE.read_bytes()
    assert b"\r\n" not in raw, "内置工作流必须保持 LF 行尾（仓库有守卫测试）"


def test_template_uses_krea2_native_nodes() -> None:
    graph = build_template()
    assert graph["1"]["class_type"] == "UnetLoaderGGUF", "Krea 2 GGUF 必须走 UnetLoaderGGUF"
    assert graph["1"]["inputs"]["unet_name"] == KREA2_UNET_DEFAULT
    assert graph["2"]["class_type"] == "CLIPLoader"
    assert graph["2"]["inputs"]["type"] == "krea2", "Krea 2 文本编码器类型必须是 krea2"
    assert graph["2"]["inputs"]["clip_name"] == KREA2_CLIP_DEFAULT
    assert graph["3"]["inputs"]["vae_name"] == KREA2_VAE_DEFAULT
    # cfg=1 时负面分支不参与引导，但仍需非空 conditioning。
    assert graph["6"]["class_type"] == "ConditioningZeroOut"
    assert graph["8"]["inputs"]["steps"] == KREA2_STEPS_DEFAULT
    assert graph["8"]["inputs"]["cfg"] == KREA2_CFG_DEFAULT


def test_template_is_fully_connected() -> None:
    _pruned(build_template())


def test_adapt_applies_request_values() -> None:
    graph = adapt_template(
        build_template(),
        prompt="云海上的中式楼阁",
        negative="模糊",
        model="krea2_turbo-Q5_K_M.gguf",
        clip="qwen3vl_4b_fp8_scaled.safetensors",
        vae="qwen_image_vae.safetensors",
        seed=4242,
        width=768,
        height=1344,
        steps=4,
        cfg=1.0,
        filename_prefix="astrbot/krea2_test",
    )
    assert graph["1"]["inputs"]["unet_name"] == "krea2_turbo-Q5_K_M.gguf"
    assert graph["4"]["inputs"]["text"] == "云海上的中式楼阁"
    assert graph["5"]["inputs"]["text"] == "模糊"
    assert graph["7"]["inputs"] == {"width": 768, "height": 1344, "batch_size": 1}
    assert graph["8"]["inputs"]["seed"] == 4242
    assert graph["8"]["inputs"]["steps"] == 4
    assert graph["10"]["inputs"]["filename_prefix"] == "astrbot/krea2_test"
    _pruned(graph)


def test_adapt_latent_falls_back_and_clamps() -> None:
    # 过小值夹到 64；0 视为「未指定」，回落到默认 1024。
    graph = adapt_template(build_template(), prompt="x", width=8, height=0)
    assert graph["7"]["inputs"]["width"] == 64
    assert graph["7"]["inputs"]["height"] == 1024


def test_adapt_chains_loras_before_sampler() -> None:
    graph = adapt_template(
        build_template(),
        prompt="x",
        loras=[
            ("krea2_turbo_4step_rank_64_lora_comfyui.safetensors", 1.0),
            ("krea2_softwatercolor.safetensors", 0.9),
        ],
    )
    assert graph["20"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["20"]["inputs"]["model"] == ["1", 0]
    assert graph["21"]["inputs"]["model"] == ["20", 0]
    assert graph["21"]["inputs"]["strength_model"] == 0.9
    assert graph["8"]["inputs"]["model"] == ["21", 0], "采样器必须接在 LoRA 链末端"
    _pruned(graph)


def test_adapt_without_loras_keeps_direct_model_link() -> None:
    graph = adapt_template(build_template(), prompt="x", loras=[])
    assert graph["8"]["inputs"]["model"] == ["1", 0]
    assert "20" not in graph
