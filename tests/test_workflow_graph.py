"""工作流图结构回归测试。

这里的核心是 v1.1.0 的修复：ComfyUI 会校验提交 payload 里的**每一个**节点，
包括输出用不到的节点。内置 Flux2 工作流仍带着作者的示例图片名与示例 GGUF
模型名，孤儿节点因此会让「单图参考」任务在校验阶段就被整单拒绝
（表现为图生图 Flux2 提交了却像没生效）。提交前只保留输出真正依赖的子图即可。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))

WORKFLOW_DIR = PLUGIN_DIR / "workflows"
FLUX2_FILE = "图生图_flux2_klein.json"
# 源工作流里作者留下的示例素材：这些文件在别人机器上并不存在。
EXAMPLE_INPUTS = ("jimeng-2025-11-01-3812-一条牛仔裤，白色背景.png", "11 (56).png", "11 (92).png")
EXAMPLE_GGUF = ("flux-2-klein\\flux-2-klein-9b-Q4_K_M.gguf", "qwen3-8b-abliterated-q5_k_m.gguf")


def _workflow_module():
    from astrbot_plugin_comfyui_ai_studio import workflow

    return workflow


def _links(node):
    for name, value in (node.get("inputs") or {}).items():
        if isinstance(value, list) and value and isinstance(value[0], str):
            yield name, str(value[0])


def _reachable(graph):
    ids = set(graph)
    stack = [
        node_id
        for node_id, node in graph.items()
        if node.get("class_type") in {"SaveImage", "PreviewImage"}
    ]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen or current not in ids:
            continue
        seen.add(current)
        for _, source in _links(graph.get(current, {})):
            stack.append(source)
    return seen


def _file_deps(graph):
    keys = {
        "image", "unet_name", "clip_name", "vae_name", "lora_name",
        "ckpt_name", "model_name",
    }
    found = []
    for node_id, node in graph.items():
        for name, value in (node.get("inputs") or {}).items():
            if name in keys and isinstance(value, str) and value.strip():
                found.append(value)
    return found


def _adapt_flux2(image_names, **overrides):
    workflow = _workflow_module()
    source = workflow.load_workflow(WORKFLOW_DIR / FLUX2_FILE, preserve_disabled=True)
    options = dict(
        positive="把背景换成海边",
        negative="",
        image_names=image_names,
        unet_name="flux-2-klein/flux-2-klein-9b-fp8.safetensors",
        clip_name="qwen_3_8b_fp8mixed.safetensors",
        vae_name="flux2-vae.safetensors",
        lora_name="",
        size=1280,
        steps=6,
        cfg=1.0,
        seed=12345,
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
        batch=1,
    )
    options.update(overrides)
    return workflow.adapt_flux2_klein_img2img(source, **options)


# --------------------------------------------------------------- prune 基本行为


def test_prune_unreachable_drops_orphan_nodes() -> None:
    workflow = _workflow_module()
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "used.png"}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "9": {"class_type": "KSampler", "inputs": {"latent_image": ["100", 0]}},
        "100": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}},
        "77": {"class_type": "LoadImage", "inputs": {"image": "unused.png"}},
    }
    pruned, removed = workflow.prune_unreachable(graph)
    assert removed == ["1", "77"]
    assert set(pruned) == {"2", "3", "9", "100"}
    # 原图不被就地修改
    assert "77" in graph


def test_prune_unreachable_returns_same_mapping_when_nothing_to_drop() -> None:
    workflow = _workflow_module()
    graph = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {"class_type": "KSampler", "inputs": {}},
    }
    pruned, removed = workflow.prune_unreachable(graph)
    assert removed == []
    assert pruned is graph


def test_prune_unreachable_keeps_graph_without_output_node() -> None:
    workflow = _workflow_module()
    graph = {"1": {"class_type": "KSampler", "inputs": {}}}
    pruned, removed = workflow.prune_unreachable(graph)
    assert removed == []
    assert pruned == graph


def test_prune_unreachable_ignores_empty_and_broken_input() -> None:
    workflow = _workflow_module()
    assert workflow.prune_unreachable({}) == ({}, [])
    graph = {"1": {"class_type": "SaveImage", "inputs": {"images": ["404", 0]}}}
    pruned, removed = workflow.prune_unreachable(graph)
    assert list(pruned) == ["1"]
    assert removed == []


# ------------------------------------------------------- Flux2 图生图：核心回归


@pytest.mark.parametrize(
    "image_names, expected_inputs",
    [
        (["uploaded_one.png"], 1),
        (["uploaded_one.png", "uploaded_two.png"], 2),
        (["uploaded_one.png", "uploaded_two.png", "uploaded_three.png"], 3),
    ],
)
def test_flux2_single_and_multi_reference_drop_example_assets(image_names, expected_inputs) -> None:
    """1/2/3 张参考图都不能把作者的示例图片名带进提交图。"""
    workflow = _workflow_module()
    patched, _ = _adapt_flux2(image_names)
    assert any(node.get("class_type") == "SaveImage" for node in patched.values())

    pruned, removed = workflow.prune_unreachable(patched)
    assert removed, "应有可剔除的孤儿节点"

    for dependency in _file_deps(pruned):
        assert dependency not in EXAMPLE_INPUTS, dependency
        assert dependency not in EXAMPLE_GGUF, dependency

    used = [
        node.get("inputs", {}).get("image")
        for node in pruned.values()
        if node.get("class_type") == "LoadImage"
    ]
    # 节点在字典里按 id 排序，不保证与附图顺序一致，因此比较集合。
    assert sorted(used) == sorted(image_names)
    assert len(used) == expected_inputs


def test_flux2_pruned_graph_has_no_dangling_links() -> None:
    workflow = _workflow_module()
    patched, _ = _adapt_flux2(["only.png"])
    pruned, _ = workflow.prune_unreachable(patched)
    ids = set(pruned)
    for node_id, node in pruned.items():
        for name, source in _links(node):
            assert source in ids, f"{node_id}.{name} -> {source} 悬空"


def test_flux2_pruned_graph_keeps_reference_chain_intact() -> None:
    """修剪不能把参考图条件链剪掉，否则图生图就真的「无效」了。"""
    workflow = _workflow_module()
    patched, _ = _adapt_flux2(["only.png"])
    pruned, _ = workflow.prune_unreachable(patched)
    sampler = next(node for node in pruned.values() if node.get("class_type") == "KSampler")

    # 沿正面条件的**全部**上游走一遍：ReferenceLatent 同时挂 conditioning 与 latent，
    # 只跟一条分支会漏掉参考图编码链。
    ancestors: set[str] = set()
    stack = [str(sampler["inputs"]["positive"][0])]
    while stack:
        current = stack.pop()
        if current in ancestors or current not in pruned:
            continue
        ancestors.add(current)
        for _, source in _links(pruned[current]):
            stack.append(source)

    kinds = {pruned[node_id].get("class_type") for node_id in ancestors}
    assert "ReferenceLatent" in kinds
    assert "VAEEncode" in kinds
    assert "LoadImage" in kinds
    # 参考图的编码链必须真的连到 ImageScale 上，而不是悬空
    kinds_all = {node.get("class_type") for node in pruned.values()}
    assert "ImageScaleToTotalPixels" in kinds or "EasySizeSimpleImage" in kinds_all


def test_flux2_three_references_keep_all_branches() -> None:
    workflow = _workflow_module()
    patched, _ = _adapt_flux2(["a.png", "b.png", "c.png"])
    pruned, removed = workflow.prune_unreachable(patched)
    assert len(pruned) == len(patched) - len(removed)
    assert _reachable(pruned) == set(pruned)


# --------------------------------------------------- 其它模式也必须能安全修剪


def test_all_adapters_produce_prunable_graphs() -> None:
    workflow = _workflow_module()
    produced = []

    for mode, filename in (("txt2img", "文生图.json"), ("hires", "高清放大.json")):
        source = workflow.load_workflow(WORKFLOW_DIR / filename)
        graph, _ = workflow.adapt_original(
            source, mode=mode, model_name="anima-base-v1.0.safetensors", loras=[],
            positive="1girl", negative="low quality", image_name="input.png",
            width=1024, height=1024, steps=20, cfg=5.0, seed=1, denoise=0.6,
            scale=2.0, upscale_model="r.pt", filename_prefix="astrbot/test",
        )
        produced.append((f"original/{mode}", graph))

    qwen = workflow.load_workflow(WORKFLOW_DIR / "图生图_qwen_edit.json")
    graph, _ = workflow.adapt_qwen_img2img(
        qwen, positive="改背景", negative="", image_name="input.png",
        unet_name="u.gguf", clip_name="c.gguf", vae_name="v.safetensors",
        steps=4, cfg=1.0, seed=1,
    )
    produced.append(("qwen_img2img", graph))

    for tool, filename in (("wash", "洗图.json"), ("outpaint", "扩图.json"),
                           ("multi_angle", "多角度.json")):
        source = workflow.load_workflow(WORKFLOW_DIR / filename)
        graph, _ = workflow.adapt_flux2_tool(
            source, tool=tool, positive="测试", negative="", image_name="input.png",
            model_name="m.safetensors", clip_name="c.safetensors",
            vae_name="v.safetensors", steps=6, cfg=1.0, seed=1, denoise=0.6,
        )
        produced.append((f"flux2_tool/{tool}", graph))

    for label, graph in produced:
        pruned, _ = workflow.prune_unreachable(graph)
        assert any(node.get("class_type") == "SaveImage" for node in pruned.values()), label
        ids = set(pruned)
        for node_id, node in pruned.items():
            for name, source in _links(node):
                assert source in ids, f"{label}: {node_id}.{name} -> {source} 悬空"
        # 剩下的一定是从输出可达的
        assert _reachable(pruned) == ids, label


def test_flux2_tool_graphs_lose_unused_gguf_loaders() -> None:
    """洗图/扩图/多角度里未连接的 GGUF 加载器会被剔除，省一次校验失败的风险。"""
    workflow = _workflow_module()
    for tool, filename in (("wash", "洗图.json"), ("outpaint", "扩图.json"),
                           ("multi_angle", "多角度.json")):
        source = workflow.load_workflow(WORKFLOW_DIR / filename)
        graph, _ = workflow.adapt_flux2_tool(
            source, tool=tool, positive="测试", negative="", image_name="input.png",
            steps=6, cfg=1.0, seed=1, denoise=0.6,
        )
        pruned, removed = workflow.prune_unreachable(graph)
        assert removed, tool
        loader_gguf = [
            node.get("inputs", {}).get("unet_name")
            for node in pruned.values()
            if node.get("class_type") == "UnetLoaderGGUF"
        ]
        assert "flux-2-klein\\flux-2-klein-9b-Q4_K_M.gguf" not in loader_gguf, tool


# --------------------------------------------------------- 提交入口统一收口


def test_comfy_client_prepare_workflow_prunes() -> None:
    from astrbot_plugin_comfyui_ai_studio.comfy import ComfyClient

    client = ComfyClient("http://127.0.0.1:1")
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "ghost.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        "3": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "4": {"class_type": "KSampler", "inputs": {}},
    }
    prepared = client.prepare_workflow(graph)
    assert "1" not in prepared
    assert set(prepared) == {"2", "3", "4"}


def test_queue_is_the_only_submission_helper() -> None:
    """确保所有提交都经过 prepare_workflow，不会绕过修剪。"""
    source = (PLUGIN_DIR / "comfy.py").read_text(encoding="utf-8")
    assert "prepare_workflow(workflow)" in source
    assert '/prompt", json={"prompt": prompt' in source
    main_source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    assert "client.request(\"POST\", \"/prompt\"" not in main_source


# ------------------------------------------------------------ 打包资源一致性


def test_shipped_workflows_match_every_mode_exactly() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import MODE_FILES

    expected = set(MODE_FILES.values())
    actual = {path.name for path in WORKFLOW_DIR.glob("*.json")}
    assert actual == expected
    assert len(actual) == len(MODE_FILES)


def test_removed_workflow_files_are_gone() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import REMOVED_WORKFLOW_FILES

    for name in REMOVED_WORKFLOW_FILES:
        assert not (WORKFLOW_DIR / name).exists(), name
    # 图生图已改用 Qwen 工作流，不再生成原工作流的图生图副本
    assert "图生图.json" in REMOVED_WORKFLOW_FILES


def test_stale_workflow_selection_is_migrated() -> None:
    """配置指向已删除的工作流时，启动会改回该模式的默认工作流。"""
    from astrbot_plugin_comfyui_ai_studio import main as plugin_main

    class _Stub(plugin_main.ComfyUIAIStudio):
        def __init__(self):
            self.cfg = {
                "workflow_img2img_flux2": "图生图_flux2.json",
                "img2img_flux2_source_workflow": "",
            }
            self.saves = 0

        def _get(self, key, default=None):
            return self.cfg.get(key, default)

        def _set(self, key, value):
            self.cfg[key] = value

        def _save_config(self):
            self.saves += 1

        def _workflow_dir(self):
            return WORKFLOW_DIR

    stub = _Stub()
    stub._refresh_img2img_flux2_workflow()
    assert stub.cfg["workflow_img2img_flux2"] == FLUX2_FILE


def test_adapter_for_removed_workflow_is_gone() -> None:
    workflow = _workflow_module()
    assert not hasattr(workflow, "adapt_flux2_img2img")
    source = (PLUGIN_DIR / "workflow.py").read_text(encoding="utf-8")
    assert "adapt_flux2_img2img" not in source


# ------------------------------------------------------------------ 前端一致性


def _html(name="index.html"):
    return (PLUGIN_DIR / name).read_text(encoding="utf-8")


def test_every_control_id_app_js_binds_exists_in_markup() -> None:
    import re

    for app_name, html_name in (
        ("app.js", "index.html"),
        ("pages/console/app.js", "pages/console/index.html"),
    ):
        app = (PLUGIN_DIR / app_name).read_text(encoding="utf-8")
        page = (PLUGIN_DIR / html_name).read_text(encoding="utf-8")
        needed = set(re.findall(
            r'getElementById\(\s*["\']([A-Za-z0-9_\-]+)["\']\s*\)', app))
        present = set(re.findall(r'\bid="([A-Za-z0-9_\-]+)"', page))
        assert not (needed - present), sorted(needed - present)


def test_sidebar_is_grouped_into_four_sections() -> None:
    page = _html()
    for title in ("绘制", "图像工具", "LoRA 与预设", "系统"):
        assert f"<span>{title}</span>" in page, title
    assert page.count('class="nav-group"') == 4


def test_every_panel_has_a_page_header() -> None:
    import re

    page = _html()
    panels = re.findall(r'data-view-panel="([a-z0-9_]+)"', page)
    assert len(panels) == len(set(panels))
    for key in panels:
        assert f'<div id="view-{key}"' in page
    assert page.count('class="page-head"') == len(set(panels))


def test_default_img2img_engine_lives_on_the_img2img_page() -> None:
    page = _html()
    assert page.count('id="img2img_engine"') == 1
    img2img_panel = page.split('data-view-panel="img2img"', 1)[1]
    img2img_panel = img2img_panel.split('data-view-panel="img2img_flux2"', 1)[0]
    assert 'id="img2img_engine"' in img2img_panel


def test_removed_acceleration_controls_are_not_referenced_anymore() -> None:
    for name in ("index.html", "pages/console/index.html"):
        page = _html(name)
        assert "accel" not in page, name
    for name in ("app.js", "pages/console/app.js"):
        app = (PLUGIN_DIR / name).read_text(encoding="utf-8")
        assert "accel" not in app, name


def test_style_defines_page_head() -> None:
    css = (PLUGIN_DIR / "style.css").read_text(encoding="utf-8")
    assert ".page-head {" in css
    assert ".page-head h2" in css
    assert ".page-head p" in css


# ------------------------------------------------------------- 配置项无死键

DEAD_CONFIG_KEYS = (
    "img2img_megapixels",
    "img2img_resolution_steps",
    "img2img_reference_method",
    "img2img_cfg_norm_strength",
    "img2img_pre_cfg",
    "img2img_tile_size",
    "img2img_tile_overlap",
    "img2img_temporal_size",
    "img2img_temporal_overlap",
    "img2img_second_image",
    "anima_teacache",
    "draw_reply_timeout",
)


def test_dead_config_keys_left_the_writable_set() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import WRITABLE_CONFIG

    for key in DEAD_CONFIG_KEYS:
        assert key not in WRITABLE_CONFIG, key


def test_dead_config_keys_left_the_schema() -> None:
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    for key in DEAD_CONFIG_KEYS:
        assert key not in schema, key


def test_every_writable_config_key_is_reachable_from_the_ui_or_schema() -> None:
    """每个可写配置键要么有界面控件，要么是 AstrBot 配置页专用项。"""
    from astrbot_plugin_comfyui_ai_studio.main import WRITABLE_CONFIG

    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))
    # 这些键由 AstrBot 配置页或客户端派生，不需要 WebUI 控件。
    schema_only = {
        "comfyui_url", "comfyui_root", "source_workflow", "workflow_dir",
        "extra_model_paths_file", "model_dir_overrides", "image_font_regular",
        "image_font_bold", "anima_template_path", "comfyui_start_script",
        "workflow_txt2img", "workflow_hires", "workflow_img2img",
        "workflow_img2img_flux2", "workflow_wash", "workflow_outpaint",
        "workflow_multi_angle", "img2img_source_workflow", "model_name",
        "lora_list", "style_lora_list", "style_lora_aliases", "style_lora_weights",
        "quality_prefix", "artist_preset", "civitai_download_mode",
        "max_concurrent", "draw_reply_timeout",
    }
    orphans = {
        key for key in WRITABLE_CONFIG
        if key not in schema and key not in schema_only
    }
    assert not orphans, sorted(orphans)
