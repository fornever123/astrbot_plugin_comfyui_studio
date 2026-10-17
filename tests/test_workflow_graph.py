"""工作流图结构回归测试。

覆盖**提交前剪除不可达节点**（``prune_unreachable``）。实测 ComfyUI 0.3.x 的
两条校验规则并不一样：不可达节点的**文件类输入**不会被校验（带着指向
不存在图片的孤儿 ``LoadImage`` 依然返回 200 并正常出图），但**节点类型**
会被逐个校验——只要有一个节点用了本机没装的自定义节点类型，``/prompt``
会直接以 ``missing_node_type`` 拒绝整包。内置与自带工作流里都留着依赖
ComfyUI-GGUF（``UnetLoaderGGUF`` / ``CLIPLoaderGGUF``）和 rgthree
（``Image Comparer`` / ``Power Lora Loader``）的孤儿节点，没装这些扩展的
用户会让整次绘图直接失败。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))

WORKFLOW_DIR = PLUGIN_DIR / "workflows"


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

    for label, graph in produced:
        pruned, _ = workflow.prune_unreachable(graph)
        assert any(node.get("class_type") == "SaveImage" for node in pruned.values()), label
        ids = set(pruned)
        for node_id, node in pruned.items():
            for name, source in _links(node):
                assert source in ids, f"{label}: {node_id}.{name} -> {source} 悬空"
        # 剩下的一定是从输出可达的
        assert _reachable(pruned) == ids, label


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
    qwen21_templates = {
        "qwen21_t2i_api.json", "qwen21_edit_api.json",
        "qwen21_t2i_turbo_api.json", "qwen21_edit_turbo_api.json",
        "qwen21_t2i_turbo_model_api.json", "qwen21_edit_turbo_model_api.json",
    }
    assert expected | qwen21_templates <= actual


def test_removed_workflow_files_are_gone() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import REMOVED_WORKFLOW_FILES

    for name in REMOVED_WORKFLOW_FILES:
        assert not (WORKFLOW_DIR / name).exists(), name
    # 图生图已改用 Qwen 工作流，不再生成原工作流的图生图副本
    assert "图生图.json" in REMOVED_WORKFLOW_FILES


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


def test_sidebar_is_grouped_into_three_sections() -> None:
    """原「图像工具」组（洗图 / 扩图 / 多角度）已随 Flux2 一并移除，侧栏只剩三组。"""
    page = _html()
    for title in ("绘制", "LoRA 与预设", "系统"):
        assert f"<span>{title}</span>" in page, title
    assert "图像工具" not in page
    assert page.count('class="nav-group"') == 3


def test_every_panel_has_a_page_header() -> None:
    import re

    page = _html()
    panels = re.findall(r'data-view-panel="([a-z0-9_]+)"', page)
    assert len(panels) == len(set(panels))
    for key in panels:
        assert f'<div id="view-{key}"' in page
    assert page.count('class="page-head"') == len(set(panels))


def test_img2img_engine_selector_is_gone_from_both_webui_pages() -> None:
    """图生图已固定为 Qwen Image 2.1：引擎下拉与后端配置键一起移除，不留死控件。"""
    for name in ("index.html", "pages/console/index.html"):
        page = _html(name)
        assert 'id="img2img_engine"' not in page, name
    for name in ("app.js", "pages/console/app.js"):
        app = (PLUGIN_DIR / name).read_text(encoding="utf-8")
        assert "img2img_engine" not in app, name


def test_qwen21_turbo_controls_are_present_in_both_webui_pages() -> None:
    for name in ("index.html", "pages/console/index.html"):
        page = _html(name)
        assert 'id="qwen21_turbo_enabled"' in page, name
        assert 'id="qwen21_img2img_turbo_enabled"' in page, name
    for name in ("app.js", "pages/console/app.js"):
        app = (PLUGIN_DIR / name).read_text(encoding="utf-8")
        assert "chooseQwen21Workflow" in app, name
        assert "profileForModel" in app, name
        assert "model_profiles_turbo" in app, name
        assert "saveModelProfile" in app, name
        assert "qwen21_turbo_lora_strength" in app, name


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
    # 随 Flux2 / 洗图 / 扩图 / 多角度 / 旧 Qwen 图生图一并移除的功能键
    "img2img_engine",
    "img2img_source_workflow",
    "qwen_img2img_source_workflow",
    "workflow_img2img_flux2",
    "outpaint_feathering",
    "workflow_wash",
    "workflow_outpaint",
    "workflow_multi_angle",
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
        "model_name",
        "lora_list", "style_lora_list", "style_lora_aliases", "style_lora_weights",
        "quality_prefix", "artist_preset", "civitai_download_mode",
        "max_concurrent", "draw_reply_timeout",
        "qwen21_turbo_enabled", "qwen21_img2img_turbo_enabled", "qwen21_turbo_lora_strength",
    }
    orphans = {
        key for key in WRITABLE_CONFIG
        if key not in schema and key not in schema_only
    }
    assert not orphans, sorted(orphans)
