from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class WorkflowError(Exception):
    pass


# Anima-2.9B is a 40-layer expansion of the 28-layer Anima base model.  The
# filename used by the CivitAI release contains ``29B`` even though the model
# is approximately 2.9B parameters, so do not treat it as a 29B model.
ANIMA_29B_MARKERS = ("anima29b", "anima-2.9b", "anima_2.9b")

# These nodes only control groups in the ComfyUI editor. They are not
# executable graph nodes and must never be sent through /prompt, especially
# when rgthree is not installed on the user's ComfyUI instance.
EDITOR_ONLY_NODE_TYPES = {
    "Fast Groups Bypasser (rgthree)",
}


def is_anima_29b_model(model_name: str) -> bool:
    normalized = str(model_name or "").replace("-", "").replace("_", "").lower()
    return any(marker.replace("-", "").replace("_", "") in normalized for marker in ANIMA_29B_MARKERS)


def anima_sampling_defaults(model_name: str) -> dict[str, Any]:
    """Return the tested defaults for the selected Anima model family."""
    if is_anima_29b_model(model_name):
        return {
            "steps": 35,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "sgm_uniform",
            "denoise": 1.0,
        }
    return {
        "steps": 30,
        "cfg": 5.0,
        "sampler_name": "er_sde",
        "scheduler": "simple",
        "denoise": 1.0,
    }


def load_api_workflow(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WorkflowError(f"工作流不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"工作流 JSON 无法读取：{path}") from exc
    if not isinstance(data, dict) or not any(
        isinstance(node, dict) and node.get("class_type") for node in data.values()
    ):
        raise WorkflowError("工作流不是 ComfyUI API 格式，请使用 API 导出 JSON")
    return data


def convert_ui_workflow(
    data: dict[str, Any],
    *,
    preserve_disabled: bool = False,
) -> dict[str, Any]:
    """Convert a ComfyUI editor workflow into the API prompt format.

    ComfyUI's ``Save (API Format)`` export is not the only format users save.
    The Qwen edit workflow supplied with the model pack is an editor JSON, so
    keep the conversion local and deterministic instead of asking the user to
    export it again by hand.
    """
    raw_nodes = data.get("nodes")
    raw_links = data.get("links")
    if not isinstance(raw_nodes, list) or not isinstance(raw_links, list):
        raise WorkflowError("工作流不是可识别的 ComfyUI API 或编辑器格式")

    links: dict[int, tuple[str, int, str, int]] = {}
    for raw in raw_links:
        if not isinstance(raw, list) or len(raw) < 5:
            continue
        try:
            link_id = int(raw[0])
            source_id = str(raw[1])
            source_slot = int(raw[2])
            target_id = str(raw[3])
            target_slot = int(raw[4])
        except (TypeError, ValueError):
            continue
        links[link_id] = (source_id, source_slot, target_id, target_slot)

    node_by_id: dict[str, dict[str, Any]] = {}
    disabled_by_id: dict[str, dict[str, Any]] = {}
    for node in raw_nodes:
        if not isinstance(node, dict) or "id" not in node or not node.get("type"):
            continue
        if str(node.get("type")) in {
            "Note",
            "Reroute",
            "PrimitiveNode",
            *EDITOR_ONLY_NODE_TYPES,
        }:
            continue
        # Most editor workflows use mode=4 for an intentionally bypassed
        # branch. Flux2's supplied editor graph uses that same flag for its
        # optional second/third reference branches and its model LoRA. The
        # Flux2 adapter needs those nodes and their original connections, so
        # it opts into preserving them explicitly.
        if int(node.get("mode", 0) or 0) == 4 and not preserve_disabled:
            disabled_by_id[str(node["id"])] = node
            continue
        node_by_id[str(node["id"])] = node

    # Editor JSON omits disabled nodes from execution.  Follow their first
    # connected input when an active node still points at a disabled node, so
    # a disabled LoRA/RAM cleanup node becomes a transparent bypass instead
    # of breaking the model or SaveImage chain.
    incoming: dict[str, list[tuple[str, int]]] = {}
    for raw in raw_links:
        if not isinstance(raw, list) or len(raw) < 5:
            continue
        try:
            source_id = str(raw[1])
            source_slot = int(raw[2])
            target_id = str(raw[3])
            link_id = int(raw[0])
        except (TypeError, ValueError):
            continue
        link = links.get(link_id)
        if link is not None:
            incoming.setdefault(target_id, []).append((source_id, source_slot))

    def resolve_source(
        source_id: str, source_slot: int, visited: set[str] | None = None
    ) -> tuple[str, int] | None:
        if source_id in node_by_id:
            return source_id, source_slot
        if source_id not in disabled_by_id:
            return None
        visited = set(visited or ())
        if source_id in visited:
            return None
        visited.add(source_id)
        for upstream_id, upstream_slot in incoming.get(source_id, ()):
            resolved = resolve_source(upstream_id, upstream_slot, visited)
            if resolved is not None:
                return resolved
        return None

    result: dict[str, Any] = {}
    for node_id, node in node_by_id.items():
        class_type = str(node["type"])
        inputs: dict[str, Any] = {}
        for ui_input in node.get("inputs", []) or []:
            if not isinstance(ui_input, dict):
                continue
            name = str(ui_input.get("name", ""))
            link_id = ui_input.get("link")
            if not name or link_id in (None, ""):
                continue
            try:
                source_id, source_slot, _, _ = links[int(link_id)]
            except (KeyError, TypeError, ValueError):
                continue
            resolved = resolve_source(source_id, source_slot)
            if resolved is not None:
                inputs[name] = [resolved[0], resolved[1]]

        widget_values = list(node.get("widgets_values") or [])
        widget_inputs = [
            item for item in (node.get("inputs") or [])
            if isinstance(item, dict)
            and isinstance(item.get("widget"), dict)
            and item.get("name")
        ]
        widget_names = [str(item["name"]) for item in widget_inputs]
        # widgets_values keeps values for linked widgets too, so retain the
        # original positions and only skip an already-resolved linked input.
        # KSampler has an editor-only control_after_generate value between
        # seed and steps; Florence2Run has a trailing editor-only control
        # value. Remove those values before mapping real API inputs.
        if class_type == "KSampler" and len(widget_values) == len(widget_names) + 1:
            widget_values.pop(1)
        elif class_type == "Florence2Run" and len(widget_values) == len(widget_names) + 1:
            widget_values.pop()
        for index, name in enumerate(widget_names):
            if index >= len(widget_values) or name in inputs:
                continue
            inputs[name] = widget_values[index]
        # Editor JSON often omits widget metadata for newer core nodes. The
        # explicit map above is the compatibility fallback for those nodes.
        result[node_id] = {
            "inputs": inputs,
            "class_type": class_type,
            "_meta": {"title": str(node.get("title") or class_type)},
        }
    if not any(node.get("class_type") == "SaveImage" for node in result.values()):
        raise WorkflowError("编辑器工作流中没有启用 SaveImage 输出节点")
    return result


def load_workflow(path: Path, *, preserve_disabled: bool = False) -> dict[str, Any]:
    """Load either an API workflow or a normal ComfyUI editor workflow."""
    if not path.is_file():
        raise WorkflowError(f"工作流不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"工作流 JSON 无法读取：{path}") from exc
    if not isinstance(data, dict):
        raise WorkflowError("工作流 JSON 顶层必须是对象")
    if any(isinstance(node, dict) and node.get("class_type") for node in data.values()):
        # API exports can also contain editor-only nodes when they were saved
        # by a frontend helper. Drop those before the graph reaches ComfyUI;
        # they have no computational output and require optional UI packages.
        return {
            node_id: node
            for node_id, node in data.items()
            if not (
                isinstance(node, dict)
                and node.get("class_type") in EDITOR_ONLY_NODE_TYPES
            )
        }
    return convert_ui_workflow(data, preserve_disabled=preserve_disabled)


# ComfyUI validates every node inside the submitted ``/prompt`` payload, not
# only the nodes that the output needs.  A leftover editor node therefore keeps
# its original file references alive: the bundled Flux2 graphs still carry the
# author's example PNG names and the example GGUF model names, so a
# single-reference edit is rejected with "invalid image file" even though that
# image never takes part in the run.  Submitting exactly the executable graph
# keeps validation aligned with execution, and it also stops the sampler from
# allocating VRAM for unused loaders.
OUTPUT_NODE_TYPES = (
    "SaveImage",
    "PreviewImage",
    "SaveAnimatedWEBP",
    "SaveAnimatedPNG",
    "VHS_VideoCombine",
)


def prune_unreachable(workflow: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return the executable subgraph plus the ids that were dropped.

    Only nodes an output depends on are kept.  When the graph exposes no output
    node at all the mapping is returned untouched, so a malformed workflow still
    reaches ComfyUI and reports its own error instead of silently submitting an
    empty prompt.
    """
    if not isinstance(workflow, dict) or not workflow:
        return workflow, []
    outputs = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") in OUTPUT_NODE_TYPES
    ]
    if not outputs:
        return workflow, []
    keep: set[str] = set()
    stack: list[str] = list(outputs)
    while stack:
        current = stack.pop()
        if current in keep or current not in workflow:
            continue
        keep.add(current)
        node = workflow.get(current)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                stack.append(value[0])
    removed = [node_id for node_id in workflow if node_id not in keep]
    if not removed:
        return workflow, []
    return {node_id: node for node_id, node in workflow.items() if node_id in keep}, removed


def _node(workflow: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    value = workflow.get(node_id)
    return value if isinstance(value, dict) else None


def _inputs(workflow: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    node = _node(workflow, node_id)
    value = node.get("inputs") if node else None
    return value if isinstance(value, dict) else None


def _parse_lora(raw: str) -> tuple[str, float]:
    if ":" not in raw:
        return raw.strip(), 1.0
    name, strength = raw.rsplit(":", 1)
    try:
        return name.strip(), max(0.0, min(2.0, float(strength)))
    except ValueError:
        return name.strip(), 1.0


def _repair_missing_lora_node(workflow: dict[str, Any], report: list[str]) -> None:
    """原图的 1054 是丢失自定义节点类型的 Power Lora Loader。

    其 UNKNOWN_* 是前端动态 LoRA 槽位，转回 rgthree 的标准 API 输入后，原图的
    两个 MODEL/CLIP 输出和后续 Switch 连接都可以继续使用。
    """
    node = _node(workflow, "1054")
    if not node or node.get("class_type"):
        return
    old = node.get("inputs", {})
    if not isinstance(old, dict):
        old = {}
    new_inputs: dict[str, Any] = {
        "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"}
    }
    for index in range(1, 31):
        old_key = "UNKNOWN" if index == 1 else f"UNKNOWN_{index - 1}"
        raw = old.get(old_key)
        if raw is None:
            continue
        value: Any = raw
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except ValueError:
                value = {"on": bool(raw), "lora": raw, "strength": 1.0}
        if not isinstance(value, dict) or value.get("lora") in {None, "null", "None"}:
            value = {"on": False, "lora": "", "strength": 1.0}
        new_inputs[f"lora_{index}"] = {
            "on": bool(value.get("on", False)),
            "lora": str(value.get("lora", "")),
            "strength": float(value.get("strength", 1.0) or 1.0),
        }
    for key in ("model", "clip"):
        if key in old:
            new_inputs[key] = old[key]
    node["inputs"] = new_inputs
    node["class_type"] = "Power Lora Loader (rgthree)"
    node["_meta"] = {"title": "原工作流多 LoRA（兼容修复）"}
    report.append("已修复原工作流节点 1054 的缺失类型")


def _power_lora_nodes(workflow: dict[str, Any]) -> list[str]:
    return [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == "Power Lora Loader (rgthree)"
    ]


def _reference(value: Any) -> tuple[str, int] | None:
    """Return a ComfyUI API link such as ["1383", 0]."""
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], str):
        try:
            return value[0], int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _selected_upstream_refs(workflow: dict[str, Any], node_id: str) -> list[tuple[str, int]]:
    """Follow only the branch selected by ImpactSwitch/Crystools switches."""
    node = _node(workflow, node_id)
    inputs = node.get("inputs") if node else None
    if not isinstance(inputs, dict):
        return []
    class_type = str(node.get("class_type", ""))
    if class_type == "ImpactSwitch" and "select" in inputs:
        try:
            selected = int(inputs.get("select", 1) or 1)
        except (TypeError, ValueError):
            selected = 1
        reference = _reference(inputs.get(f"input{selected}"))
        return [reference] if reference else []
    if class_type.startswith("Switch any") and "boolean" in inputs:
        key = "on_true" if bool(inputs.get("boolean")) else "on_false"
        reference = _reference(inputs.get(key))
        return [reference] if reference else []
    references: list[tuple[str, int]] = []
    for value in inputs.values():
        reference = _reference(value)
        if reference:
            references.append(reference)
    return references


def _active_model_lora_nodes(workflow: dict[str, Any]) -> list[str]:
    """Find active Power Lora Loader nodes in upstream-to-downstream order."""
    roots = [node_id for node_id in ("1071:1081", "1071:1075") if _node(workflow, node_id)]
    visited: set[str] = set()
    found: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        node = _node(workflow, node_id)
        if not node:
            return
        if node.get("class_type") == "Power Lora Loader (rgthree)":
            found.append(node_id)
        for upstream_id, _ in _selected_upstream_refs(workflow, node_id):
            visit(upstream_id)

    for root in roots:
        visit(root)
    found.reverse()
    return found


def _apply_loras(workflow: dict[str, Any], loras: list[str], report: list[str]) -> None:
    all_nodes = _power_lora_nodes(workflow)
    nodes = _active_model_lora_nodes(workflow)
    if not nodes:
        nodes = [node_id for node_id in all_nodes if node_id != "1054"] or all_nodes
    slots: list[tuple[str, str]] = []
    for node_id in all_nodes:
        inputs = _inputs(workflow, node_id) or {}
        for key, value in inputs.items():
            if key.startswith("lora_") and isinstance(value, dict):
                value["on"] = False
    for node_id in nodes:
        inputs = _inputs(workflow, node_id) or {}
        for key, value in inputs.items():
            if key.startswith("lora_") and isinstance(value, dict):
                slots.append((node_id, key))
    parsed = [_parse_lora(item) for item in loras if item.strip()]
    for (node_id, key), (name, strength) in zip(slots, parsed):
        inputs = _inputs(workflow, node_id)
        if inputs is None:
            continue
        inputs[key] = {"on": True, "lora": name, "strength": strength}
    if len(parsed) > len(slots):
        report.append(f"原工作流只有 {len(slots)} 个 LoRA 槽位，剩余 {len(parsed) - len(slots)} 个未应用")
    if parsed:
        report.append(f"LoRA 已写入当前模型链：{'、'.join(nodes)}")


def _set_text_source(workflow: dict[str, Any], node_id: str, value: str) -> bool:
    inputs = _inputs(workflow, node_id)
    if inputs is None:
        return False
    if "prompt" in inputs:
        inputs["prompt"] = value
        return True
    if "String" in inputs:
        inputs["String"] = value
        return True
    if "text" in inputs and isinstance(inputs["text"], str):
        inputs["text"] = value
        return True
    return False


def _apply_prompts(workflow: dict[str, Any], positive: str, negative: str, report: list[str]) -> None:
    # 这些 ID 来自 E:\难工作流.json；同时保留按节点标题的回退，便于用户切换同源工作流。
    positive_ids = ["1356"]
    negative_ids = ["97"]
    positive_ok = any(_set_text_source(workflow, node_id, positive) for node_id in positive_ids)
    negative_ok = any(_set_text_source(workflow, node_id, negative) for node_id in negative_ids)
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        title = str((node.get("_meta") or {}).get("title", "")).lower()
        if node.get("class_type") == "String":
            if any(word in title for word in ("正面", "positive")):
                positive_ok = _set_text_source(workflow, node_id, positive) or positive_ok
            if any(word in title for word in ("负面", "negative")):
                negative_ok = _set_text_source(workflow, node_id, negative) or negative_ok
    if not positive_ok:
        report.append("原工作流没有找到正面提示词输入")
    if not negative_ok:
        report.append("原工作流没有找到负面提示词输入")


def _apply_images(workflow: dict[str, Any], image_name: str, report: list[str]) -> None:
    count = 0
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "LoadImage":
            inputs = node.setdefault("inputs", {})
            inputs["image"] = image_name
            count += 1
    if not count:
        report.append("原工作流没有 LoadImage 节点")


def _apply_mode(workflow: dict[str, Any], mode: str, report: list[str]) -> None:
    # 原图的 1339:1322 在 VAEEncodeForInpaint、VAEEncode 与两个空 Latent 之间切换。
    switch = _inputs(workflow, "1339:1322")
    if switch is not None and "select" in switch:
        # 高清放大也必须从用户附图编码，否则会错误地从空 Latent 开始。
        switch["select"] = 2 if mode in {"img2img", "hires"} else 4
    output_switch = _inputs(workflow, "1413")
    if output_switch is not None and "select" in output_switch:
        # 1413 的 input1 是采样结果，input2 是放大分支。文生图和图生图
        # 都必须直接输出采样结果；只有高清放大才应使用 input2。
        output_switch["select"] = 2 if mode == "hires" else 1
    # 文生图和图生图使用标准采样支路，不应让原工作流的 LLLite
    # 图像控制分支读取黑色占位图；高清放大仍保留原工作流的控制分支。
    if mode in {"txt2img", "img2img"}:
        lllite_switch = _inputs(workflow, "513:591")
        if lllite_switch is not None and "boolean" in lllite_switch:
            lllite_switch["boolean"] = False
    if mode == "hires" and not _node(workflow, "915:282"):
        report.append("原工作流缺少 UltimateSDUpscale 节点")


def _apply_sampling_sampler(
    workflow: dict[str, Any],
    *,
    model_name: str,
    steps: int,
    cfg: float,
    seed: int,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    report: list[str],
) -> None:
    """为原工作流补上标准文生图/图生图采样支路。

    原图主要是图像控制/高清放大工作流，文生图没有可直接复用的 KSampler。
    这里复用原模型、CLIP、VAE、提示词和 Latent 选择器，只新增标准 KSampler；
    图生图传入 VAE 编码后的输入图，高清放大仍走原始 UltimateSDUpscale。
    """
    required = ("1071:1075", "1071:1063", "1071:1062", "1339:1322", "1071:1074")
    if any(_node(workflow, node_id) is None for node_id in required):
        report.append("原工作流缺少文生图采样所需的模型、提示词或 VAE 节点")
        return

    # Anima is a flow model.  KSampler must receive the AuraFlow-adjusted
    # model; connecting the raw Anima loader directly produces colored noise.
    sampling_node = "astrbot_anima_model_sampling"
    workflow[sampling_node] = {
        "inputs": {
            "model": ["1071:1075", 0],
            "shift": 3.0,
        },
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "Anima AuraFlow 采样适配"},
    }
    sampler_node = "astrbot_txt2img_sampler"
    decode_node = "astrbot_txt2img_decode"
    workflow[sampler_node] = {
        "inputs": {
            "model": [sampling_node, 0],
            "positive": ["1071:1063", 0],
            "negative": ["1071:1062", 0],
            "latent_image": ["1339:1322", 0],
            "seed": int(seed),
            "steps": max(1, int(steps)),
            "cfg": float(cfg if cfg > 0 else 5.0),
            "sampler_name": sampler_name or str(anima_sampling_defaults(model_name)["sampler_name"]),
            "scheduler": scheduler or str(anima_sampling_defaults(model_name)["scheduler"]),
            "denoise": float(denoise if denoise > 0 else 1.0),
        },
        "class_type": "KSampler",
        "_meta": {"title": "AstrBot 标准采样"},
    }
    workflow[decode_node] = {
        "inputs": {"samples": [sampler_node, 0], "vae": ["1071:1074", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "AstrBot 标准解码"},
    }
    output_inputs = _inputs(workflow, "1413")
    if output_inputs is not None:
        output_inputs["input1"] = [decode_node, 0]
    report.append(
        "已启用 Anima AuraFlow 采样适配（2.9B）"
        if is_anima_29b_model(model_name)
        else "已启用 Anima AuraFlow 采样适配（Base 1.0）"
    )


def _disable_optional_sage_attention(workflow: dict[str, Any], report: list[str]) -> None:
    node = _node(workflow, "1071:1075")
    inputs = node.get("inputs") if node else None
    if not isinstance(inputs, dict) or "sage_attention" not in inputs:
        return
    if inputs.get("sage_attention") != "disabled":
        inputs["sage_attention"] = "disabled"
        report.append("已禁用缺少依赖的 SageAttention，改用 ComfyUI 默认注意力")


def _qwen_inputs(workflow: dict[str, Any], node_id: str) -> dict[str, Any]:
    node = _node(workflow, node_id)
    if node is None:
        raise WorkflowError(f"Qwen 图生图工作流缺少节点：{node_id}")
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}
        node["inputs"] = inputs
    return inputs


def adapt_qwen_img2img(
    source: dict[str, Any],
    *,
    positive: str,
    negative: str,
    image_name: str,
    second_image_name: str = "",
    unet_name: str,
    clip_name: str,
    vae_name: str,
    lora_name: str = "",
    lora_strength: float = 0.0,
    loras: list[str] | None = None,
    width: int = 512,
    height: int = 960,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = 0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    denoise: float = 0.58,
    scale_method: str = "lanczos",
    largest_size: int = 1152,
    crop: str = "disabled",
    reference_method: str = "index_timestep_zero",
    sampling_shift: float = 3.1,
    cfg_norm_strength: float = 1.0,
    pre_cfg: bool = False,
    tile_size: int = 512,
    tile_overlap: int = 64,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
    filename_prefix: str = "astrbot/img2img_qwen",
) -> tuple[dict[str, Any], list[str]]:
    """Apply the API workflow exported from the configured Qwen baseline.

    The current Qwen Image Edit workflow uses stable nodes 1/10/11/13/14/
    30/39/119/138/152/158/160/161/174.  Keep the mapping explicit so the
    plugin never routes this graph through the old Flux2 or old-Qwen adapter.
    """
    workflow = copy.deepcopy(source)
    report: list[str] = []
    required = ("1", "10", "11", "13", "14", "30", "39", "119", "138", "152", "158", "160", "161")
    missing = [node_id for node_id in required if _node(workflow, node_id) is None]
    if missing:
        raise WorkflowError(f"Qwen 图生图工作流缺少节点：{', '.join(missing)}")
    if not image_name.strip():
        raise WorkflowError("Qwen 图生图没有收到已上传的输入图片")

    _qwen_inputs(workflow, "11")["image"] = image_name
    _qwen_inputs(workflow, "138").update(
        image=["11", 0],
        upscale_method=scale_method or "lanczos",
        largest_size=max(64, int(largest_size or 1152)),
    )
    if _node(workflow, "129") is not None:
        _qwen_inputs(workflow, "129").update(pixels=["138", 0], vae=["10", 0])
    if _node(workflow, "130") is not None:
        _qwen_inputs(workflow, "130").update(samples=["129", 0], vae=["10", 0])
    if _node(workflow, "132") is not None:
        _qwen_inputs(workflow, "132")["image"] = ["130", 0]
    scale_inputs = _qwen_inputs(workflow, "158")
    scale_inputs.update(
        image=["11", 0],
        upscale_method=scale_method or "lanczos",
        crop=crop or "center",
    )
    if width > 0 and height > 0:
        scale_inputs.update(width=max(64, int(width)), height=max(64, int(height)))
    else:
        scale_inputs.update(width=["132", 0], height=["132", 1])

    _qwen_inputs(workflow, "119").update(pixels=["158", 0], vae=["10", 0])
    positive_inputs = _qwen_inputs(workflow, "1")
    positive_inputs.update(clip=["161", 0], vae=["10", 0], image1=["158", 0], prompt=positive)
    _qwen_inputs(workflow, "39").update(clip=["161", 0], vae=["10", 0], prompt=negative or "")
    # image2 是 TextEncodeQwenImageEditPlus 的真实可选输入。只有用户确实
    # 提供第二张图时才添加 LoadImage 和 image2，单图任务保持原工作流不变。
    positive_inputs.pop("image2", None)
    positive_inputs.pop("image3", None)
    if second_image_name:
        second_load_id = "astrbot_img2img_second_input"
        workflow[second_load_id] = {
            "inputs": {"image": second_image_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Qwen 图生图第二参考图"},
        }
        positive_inputs["image2"] = [second_load_id, 0]
        negative_inputs = _qwen_inputs(workflow, "39")
        negative_inputs["image2"] = [second_load_id, 0]
        report.append("Qwen 图生图已接入第二张参考图：image2")

    _qwen_inputs(workflow, "160")["unet_name"] = unet_name
    _qwen_inputs(workflow, "161").update(clip_name=clip_name, type="qwen_image")
    _qwen_inputs(workflow, "10")["vae_name"] = vae_name
    # The acceleration LoRA is intentionally a separate loader.  It must not
    # consume one of the user's content-LoRA slots in the Power Lora Loader.
    workflow.pop("astrbot_qwen_accel_lora", None)

    requested_loras = list(loras or [])
    if lora_name and float(lora_strength or 0.0) != 0.0 and not loras:
        requested_loras.insert(0, f"{lora_name}:{float(lora_strength):g}")
    if _node(workflow, "174") is not None:
        lora_inputs = _qwen_inputs(workflow, "174")
        lora_inputs["model"] = ["160", 0]
        slots = sorted(
            [key for key, value in lora_inputs.items() if key.startswith("lora_") and isinstance(value, dict)],
            key=lambda item: int(item.split("_", 1)[1]) if item.split("_", 1)[1].isdigit() else 999,
        )
        for key in slots:
            value = lora_inputs.get(key)
            if isinstance(value, dict):
                value["on"] = False
        parsed = [_parse_lora(item) for item in requested_loras if str(item).strip()]
        for key, (name, strength) in zip(slots, parsed):
            lora_inputs[key] = {"on": True, "lora": name, "strength": strength}
        if len(parsed) > len(slots):
            report.append(f"Qwen 图生图只有 {len(slots)} 个 LoRA 槽位，剩余 {len(parsed) - len(slots)} 个未应用")
        if parsed:
            report.append("Qwen 图生图 LoRA 已写入 Power Lora Loader")
        else:
            report.append("Qwen 图生图 LoRA 未启用")
    elif requested_loras:
        report.append("Qwen 图生图工作流缺少 Power Lora Loader，LoRA 未应用")

    model_output = ["174", 0] if _node(workflow, "174") else ["160", 0]
    # The supplied baseline graph has no acceleration branch.  Older callers
    # cannot add one because the compatibility arguments were removed.
    report.append("Qwen 图生图已回退到无加速 LoRA 工作流")

    _qwen_inputs(workflow, "30").update(
        shift=max(0.0, float(sampling_shift)),
        model=model_output,
    )

    sampler = _qwen_inputs(workflow, "152")
    sampler.update(
        model=["30", 0], positive=["1", 0], negative=["39", 0], latent_image=["119", 0],
        seed=max(0, int(seed)), steps=max(1, int(steps)), cfg=max(0.0, float(cfg)),
        sampler_name=sampler_name or "euler", scheduler=scheduler or "simple",
        denoise=max(0.0, min(1.0, float(denoise))),
    )
    _qwen_inputs(workflow, "13").update(samples=["152", 0], vae=["10", 0])
    _qwen_inputs(workflow, "14").update(images=["13", 0], filename_prefix=filename_prefix)
    if _node(workflow, "51") is not None:
        _qwen_inputs(workflow, "51").update(image_a=["11", 0], image_b=["13", 0])
    report.append(
        "Qwen 图生图实际参数：普通 LoRA；步数={}；CFG={}".format(
            max(1, int(steps)),
            f"{max(0.0, float(cfg)):g}",
        )
    )
    if width > 0 and height > 0:
        report.append(f"Qwen 图生图输出尺寸已写入 ImageScale：{int(width)}x{int(height)}")
    _ = (reference_method, cfg_norm_strength, pre_cfg, tile_size, tile_overlap, temporal_size, temporal_overlap)
    return workflow, report



def adapt_flux2_klein_img2img(
    source: dict[str, Any],
    *,
    positive: str,
    negative: str,
    image_names: list[str],
    unet_name: str,
    clip_name: str,
    vae_name: str,
    lora_name: str = "",
    lora_strength: float = 0.9,
    # Kept for compatibility with older callers. Flux2 Klein is intentionally
    # restored to the original workflow and no acceleration LoRA is injected.
    size: int = 1280,
    steps: int = 6,
    cfg: float = 1.0,
    seed: int = 0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    denoise: float = 1.0,
    batch: int = 1,
    filename_prefix: str = "astrbot/img2img_flux2_klein",
) -> tuple[dict[str, Any], list[str]]:
    """Adapt the Flux2 Klein editor workflow with one to three references.

    The supplied multi-reference graph contains three independent image
    branches which are chained through ReferenceLatent nodes:

    ``63 -> 90 -> 75 -> 73/74`` (first image)
    ``64 -> 96 -> 87 -> 86/85`` (second image)
    ``61 -> 97 -> 81 -> 79/80`` (third image)

    The sampler receives the last branch selected by the number of supplied
    images.  A single-reference custom workflow remains supported; requesting
    more images from it produces a clear workflow capability error.
    """
    workflow = copy.deepcopy(source)
    report: list[str] = []

    def required(node_id: str, label: str) -> dict[str, Any]:
        inputs = _inputs(workflow, node_id)
        if inputs is None:
            raise WorkflowError(f"Flux2 Klein 图生图工作流缺少{label}节点：{node_id}")
        return inputs

    required_nodes = {
        "10": "VAE",
        "13": "核心模型",
        "14": "文本编码器",
        "19": "正面提示词",
        "38": "LoRA",
        "62": "输出",
        "83": "输出画布",
        "95": "采样器",
    }
    for node_id, label in required_nodes.items():
        required(node_id, label)

    image_names = [str(value or "").strip() for value in image_names]
    if not image_names:
        raise WorkflowError("Flux2 Klein 图生图至少需要 1 张参考图")
    if len(image_names) > 3:
        raise WorkflowError("Flux2 Klein 图生图最多支持 3 张参考图")
    if any(not value for value in image_names):
        raise WorkflowError("Flux2 Klein 图生图收到空的参考图名称")

    # The second and third branches are optional in the source graph.  Keep
    # the mapping explicit because their conditioning order is not the same
    # as the visual node order: branch 2 chains after branch 1, and branch 3
    # chains after branch 2.
    branches = (
        {
            "image": "63",
            "scale": "90",
            "encode": "75",
            "negative": "73",
            "positive": "74",
        },
        {
            "image": "64",
            "scale": "96",
            "encode": "87",
            "negative": "86",
            "positive": "85",
        },
        {
            "image": "61",
            "scale": "97",
            "encode": "81",
            "negative": "79",
            "positive": "80",
        },
    )
    available_branches = 0
    for branch in branches:
        if all(_inputs(workflow, node_id) is not None for node_id in branch.values()):
            available_branches += 1
        else:
            break
    if len(image_names) > available_branches:
        if available_branches <= 1:
            raise WorkflowError(
                "当前 Flux2 Klein 工作流只配置了 1 个参考图分支；"
                "请在 WebUI 切换支持 3 张参考图的 Flux2 工作流"
            )
        raise WorkflowError(
            f"当前 Flux2 Klein 工作流最多支持 {available_branches} 张参考图，"
            f"本次收到 {len(image_names)} 张"
        )

    active_branch = branches[len(image_names) - 1]
    for index, image_name in enumerate(image_names):
        branch = branches[index]
        required(branch["image"], f"第 {index + 1} 张参考图")["image"] = image_name
        scale = required(branch["scale"], f"第 {index + 1} 张参考图缩放")
        if int(size or 0) > 0:
            if "缩放长度" in scale:
                scale["缩放长度"] = max(64, int(size))
            elif "size" in scale:
                scale["size"] = max(64, int(size))
            elif "megapixels" in scale:
                # ImageScaleToTotalPixels keeps the source aspect ratio.  A
                # square with the requested longest edge is the equivalent
                # pixel budget, while the node derives the other edge from
                # the actual input aspect ratio.
                edge = max(64, int(size))
                scale["megapixels"] = max(0.01, (edge * edge) / 1_000_000)
            else:
                raise WorkflowError(
                    f"Flux2 Klein 第 {index + 1} 个参考图缩放节点缺少可调尺寸输入"
                )

    required("13", "核心模型")["unet_name"] = unet_name
    required("14", "文本编码器").update(clip_name=clip_name, type="flux2")
    required("10", "VAE")["vae_name"] = vae_name
    required("19", "正面提示词").update(text=positive, clip=["14", 0])

    # The source workflow includes one model-only LoRA node. An empty setting
    # bypasses it by connecting the sampler directly to the selected UNet;
    # a selected content LoRA uses the original node 38. Do not create any
    # extra acceleration node: this module must remain the original workflow.
    sampler = required("95", "采样器")
    lora = required("38", "LoRA")
    model_output: list[Any] = ["13", 0]
    if lora_name.strip():
        lora.update(
            lora_name=lora_name,
            strength_model=max(-2.0, min(2.0, float(lora_strength))),
            model=["13", 0],
        )
        model_output = ["38", 0]
        report.append(f"已启用 Flux2 Klein 独立 LoRA：{lora_name}")
    else:
        # The supplied graph contains an example LoRA node. Leaving that
        # node in the submitted prompt can fail ComfyUI validation when the
        # example file is absent, even though the sampler bypasses it.
        workflow.pop("38", None)
        report.append("Flux2 Klein 独立 LoRA 已关闭")

    # The sampler is connected directly to the source workflow's original
    # model chain. No acceleration LoRA, KV cache, or other node is added.
    sampler["model"] = model_output
    report.append("Flux2 使用源工作流模型链，不添加加速 LoRA 或 KV Cache")
    if int(size or 0) > 0:
        report.append(f"参考图最大边：{max(64, int(size))}")
    else:
        report.append("参考图尺寸沿用工作流默认值")

    sampler.update(
        positive=[active_branch["positive"], 0],
        negative=[active_branch["negative"], 0],
        seed=max(0, int(seed)),
        steps=max(1, int(steps)),
        cfg=max(0.0, float(cfg)),
        sampler_name=sampler_name or "euler",
        scheduler=scheduler or "simple",
        # Keep the workflow's denoise control available to the user. The
        # configured default remains 1.0, but lower values are valid when a
        # gentler Flux2 edit is needed.
        denoise=max(0.0, min(1.0, float(denoise))),
    )
    required("83", "输出画布")["batch_size"] = max(1, min(4, int(batch)))
    required("62", "输出")["filename_prefix"] = filename_prefix
    report.append(
        f"已接入 {len(image_names)} 张参考图："
        + "；".join(
            f"{branches[index]['image']} → {branches[index]['scale']} → "
            f"{branches[index]['encode']} → {branches[index]['negative']}/"
            f"{branches[index]['positive']}"
            for index in range(len(image_names))
        )
        + " → 95"
    )
    report.append("参考图使用最长边等比例缩放，输出画布沿用同一宽高比")
    report.append("负面条件沿用工作流 ConditioningZeroOut；用户提示词写入节点 19")
    _ = negative
    return workflow, report


def _first_node_id(workflow: dict[str, Any], class_types: tuple[str, ...]) -> str:
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") in class_types:
            return str(node_id)
    return ""


def _follow_input_node(
    workflow: dict[str, Any],
    node_id: str,
    input_name: str,
    class_types: tuple[str, ...],
    visited: set[str] | None = None,
) -> str:
    """Find a loader upstream of an input without relying on editor IDs."""
    if not node_id:
        return ""
    visited = set(visited or ())
    if node_id in visited:
        return ""
    visited.add(node_id)
    node = _node(workflow, node_id)
    if not node:
        return ""
    if node.get("class_type") in class_types:
        return node_id
    value = (_inputs(workflow, node_id) or {}).get(input_name)
    if isinstance(value, list) and len(value) >= 1:
        return _follow_input_node(workflow, str(value[0]), input_name, class_types, visited)
    return ""


def _set_named_input(inputs: dict[str, Any], names: tuple[str, ...], value: Any) -> bool:
    for name in names:
        if name in inputs:
            inputs[name] = value
            return True
    return False


def _join_tool_prompt(default_positive: str, user_prompt: str) -> str:
    values = [str(value or "").strip() for value in (default_positive, user_prompt)]
    return ", ".join(value for value in values if value)


def adapt_flux2_tool(
    source: dict[str, Any],
    *,
    tool: str,
    positive: str,
    negative: str,
    image_name: str,
    model_name: str = "",
    clip_name: str = "",
    vae_name: str = "",
    width: int = 0,
    height: int = 0,
    size: int = 0,
    steps: int = 6,
    cfg: float = 1.0,
    seed: int = 0,
    denoise: float = 0.6,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "astrbot/flux2_tool",
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    feathering: int = 0,
    horizontal_angle: int = 71,
    vertical_angle: int = 42,
    zoom: float = 4.5,
    default_prompts: bool = True,
    camera_view: bool = False,
    caption_tokens: int = 1024,
    batch: int = 1,
) -> tuple[dict[str, Any], list[str]]:
    """Adapt one of the standalone Flux.2 Klein image tools.

    These workflows are intentionally kept separate from the normal drawing
    path. Their active LoRA nodes are copied exactly as supplied by the user;
    disabled editor nodes are absent after conversion and are never enabled
    here. The text entered by the user is put into the workflow's own Chinese
    prompt field without AI translation or global preset/LoRA expansion.
    """
    tool = str(tool or "").strip().lower()
    if tool not in {"wash", "outpaint", "multi_angle"}:
        raise WorkflowError(f"未知 Flux2 工具模式：{tool}")
    workflow = copy.deepcopy(source)
    report: list[str] = []
    if not str(image_name or "").strip():
        raise WorkflowError("Flux2 工具没有收到输入图片")

    sampler_id = _first_node_id(workflow, ("KSampler",))
    save_id = _first_node_id(workflow, ("SaveImage",))
    image_id = _first_node_id(workflow, ("LoadImage",))
    positive_id = _first_node_id(workflow, ("CLIPTextEncode",))
    if tool in {"wash", "multi_angle"}:
        prompt_nodes = [
            node_id for node_id, node in workflow.items()
            if isinstance(node, dict) and node.get("class_type") == "CR Prompt Text"
        ]
        if prompt_nodes:
            positive_id = prompt_nodes[0] if tool == "wash" else prompt_nodes[-1]
    if not sampler_id or not save_id or not image_id or not positive_id:
        raise WorkflowError("Flux2 工具工作流缺少输入图、提示词、采样器或输出节点")
    sampler_inputs = _inputs(workflow, sampler_id) or {}
    positive_inputs = _inputs(workflow, positive_id) or {}
    if not positive_inputs:
        raise WorkflowError("Flux2 工具工作流的提示词节点没有输入")

    # Resolve the actual loaders used by the sampler/conditioning graph. This
    # keeps alternate GGUF branches in the source workflow untouched when they
    # are not connected.
    model_ref = sampler_inputs.get("model")
    model_loader_id = (
        str(model_ref[0])
        if isinstance(model_ref, list) and model_ref
        else ""
    )
    while model_loader_id:
        node = _node(workflow, model_loader_id)
        if not node:
            break
        if node.get("class_type") in {"UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple"}:
            break
        upstream = (_inputs(workflow, model_loader_id) or {}).get("model")
        model_loader_id = str(upstream[0]) if isinstance(upstream, list) and upstream else ""
    clip_loader_id = ""
    # CR Prompt Text is a string helper, so it does not contain the CLIP
    # connection itself. Find the actual encoder downstream of it before
    # creating an optional negative encoder.
    clip_source_inputs = positive_inputs
    if not clip_source_inputs.get("clip"):
        encoder_id = _first_node_id(workflow, ("CLIPTextEncode",))
        clip_source_inputs = _inputs(workflow, encoder_id) or {}
    clip_ref = clip_source_inputs.get("clip")
    if isinstance(clip_ref, list) and clip_ref:
        clip_loader_id = _follow_input_node(
            workflow, str(clip_ref[0]), "clip", ("CLIPLoader", "CLIPLoaderGGUF")
        )
    vae_loader_id = ""
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") not in {"VAEEncode", "VAEDecode"}:
            continue
        vae_ref = (_inputs(workflow, str(node_id)) or {}).get("vae")
        if isinstance(vae_ref, list) and vae_ref:
            vae_loader_id = _follow_input_node(
                workflow, str(vae_ref[0]), "vae", ("VAELoader",)
            )
            if vae_loader_id:
                break

    def set_loader(node_id: str, names: tuple[str, ...], value: str, label: str) -> None:
        if not value:
            return
        inputs = _inputs(workflow, node_id)
        if inputs is None or not _set_named_input(inputs, names, value):
            report.append(f"未找到{label}输入，保留工作流原值")

    set_loader(model_loader_id, ("unet_name", "ckpt_name", "model_name"), str(model_name or "").strip(), "核心模型")
    set_loader(clip_loader_id, ("clip_name",), str(clip_name or "").strip(), "文本编码器")
    set_loader(vae_loader_id, ("vae_name",), str(vae_name or "").strip(), "VAE")

    _inputs(workflow, image_id)["image"] = image_name  # type: ignore[index]
    prompt = _join_tool_prompt(positive, "")
    if not _set_named_input(positive_inputs, ("prompt", "text", "String"), prompt):
        raise WorkflowError("Flux2 工具工作流的提示词字段无法写入")

    # A zeroed negative branch is useful in the editor preview, but it cannot
    # carry a user-configured negative prompt. Insert one encoder while
    # retaining the workflow's reference-latent topology.
    if str(negative or "").strip():
        negative_id = f"astrbot_{tool}_negative"
        clip_value = positive_inputs.get("clip") or ([clip_loader_id, 0] if clip_loader_id else None)
        if not isinstance(clip_value, list):
            raise WorkflowError("Flux2 工具工作流无法连接负面提示词的文本编码器")
        workflow[negative_id] = {
            "inputs": {"clip": clip_value, "text": str(negative).strip()},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Flux2 独立负面提示词"},
        }
        zero_nodes = [
            node_id for node_id, node in workflow.items()
            if isinstance(node, dict) and node.get("class_type") == "ConditioningZeroOut"
        ]
        if zero_nodes:
            for node_id in zero_nodes:
                (_inputs(workflow, node_id) or {})["conditioning"] = [negative_id, 0]
        else:
            sampler_inputs["negative"] = [negative_id, 0]

    sampler_inputs.update(
        seed=max(0, int(seed)),
        steps=max(1, int(steps)),
        cfg=max(0.0, float(cfg)),
        sampler_name=str(sampler_name or "euler"),
        scheduler=str(scheduler or "simple"),
        denoise=max(0.0, min(1.0, float(denoise))),
    )
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "EmptyFlux2LatentImage":
            inputs = _inputs(workflow, str(node_id)) or {}
            _set_named_input(inputs, ("batch_size",), max(1, min(4, int(batch or 1))))

    if tool == "wash":
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or node.get("class_type") != "Florence2Run":
                continue
            inputs = _inputs(workflow, str(node_id)) or {}
            if "max_new_tokens" in inputs:
                inputs["max_new_tokens"] = max(64, min(4096, int(caption_tokens or 1024)))
        report.append("洗图保留工作流自带的 Florence2 图像描述链")
    elif tool == "outpaint":
        pad_id = _first_node_id(workflow, ("ImagePadForOutpaint",))
        if not pad_id:
            raise WorkflowError("扩图工作流缺少 ImagePadForOutpaint 节点")
        pad_inputs = _inputs(workflow, pad_id) or {}
        pad_inputs.update(
            left=max(0, int(left)), top=max(0, int(top)),
            right=max(0, int(right)), bottom=max(0, int(bottom)),
            feathering=max(0, int(feathering)),
        )
        report.append(f"扩图边距已写入：左 {left}、上 {top}、右 {right}、下 {bottom}")

    elif tool == "multi_angle":
        camera_id = _first_node_id(workflow, ("QwenMultiangleCameraNode",))
        if not camera_id:
            raise WorkflowError("多角度工作流缺少 QwenMultiangleCameraNode 节点")
        camera_inputs = _inputs(workflow, camera_id) or {}
        camera_inputs.update(
            horizontal_angle=int(horizontal_angle),
            vertical_angle=int(vertical_angle),
            zoom=max(0.1, float(zoom)),
            default_prompts=bool(default_prompts),
            camera_view=bool(camera_view),
        )
        report.append(f"多角度参数已写入：水平 {horizontal_angle}、垂直 {vertical_angle}、缩放 {zoom}")

    if size > 0:
        changed_size = False
        for node in workflow.values():
            if not isinstance(node, dict) or node.get("class_type") != "EasySizeSimpleImage":
                continue
            changed_size = _set_named_input(
                _inputs(workflow, str(next(k for k, v in workflow.items() if v is node))) or {},
                ("缩放长度", "resize_length", "length"),
                max(64, int(size)),
            ) or changed_size
        if not changed_size:
            report.append("工作流没有可调的缩放长度输入，保留原始尺寸设置")

    if width > 0 or height > 0:
        report.append("该 Flux2 工作流使用自身的画布尺寸链；宽高参数未强行覆盖")
    _inputs(workflow, save_id)["filename_prefix"] = filename_prefix  # type: ignore[index]

    lora_nodes = [
        (str(node_id), node)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly"
    ]
    if lora_nodes:
        names = [
            str((_inputs(workflow, node_id) or {}).get("lora_name", "")).strip()
            for node_id, _ in lora_nodes
        ]
        names = [name for name in names if name]
        report.append("仅保留工作流原本启用的 LoRA：" + ("、".join(names) if names else "无"))
    else:
        report.append("工作流没有可执行的 LoRA 节点")
    report.append("本模式不使用全局 LoRA、临时 LoRA、预设扩展或 AI 提示词优化")
    return workflow, report


def adapt_original(
    source: dict[str, Any],
    *,
    mode: str,
    model_name: str,
    loras: list[str],
    positive: str,
    negative: str,
    image_name: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float,
    scale: float,
    upscale_model: str,
    sampler_name: str = "",
    scheduler: str = "",
    batch: int = 1,
    filename_prefix: str = "",
) -> tuple[dict[str, Any], list[str]]:
    workflow = copy.deepcopy(source)
    report: list[str] = []
    _repair_missing_lora_node(workflow, report)
    _apply_loras(workflow, loras, report)
    _apply_prompts(workflow, positive, negative, report)
    _apply_images(workflow, image_name, report)
    _apply_mode(workflow, mode, report)
    _disable_optional_sage_attention(workflow, report)

    model_inputs = _inputs(workflow, "1071:1077")
    if model_inputs is not None and "model_name" in model_inputs:
        model_inputs["model_name"] = model_name
    else:
        report.append("原工作流核心模型节点 1071:1077 不存在")

    for node_id in ("1339:1193", "1339:1196"):
        inputs = _inputs(workflow, node_id)
        if inputs is not None:
            inputs.update(width=width, height=height, batch_size=1)
    scale_inputs = _inputs(workflow, "915:930")
    if scale_inputs is not None and "Number" in scale_inputs:
        scale_inputs["Number"] = str(scale if scale > 0 else 2.0)
    upscale_inputs = _inputs(workflow, "915:279")
    if upscale_inputs is not None and upscale_model:
        upscale_inputs["model_name"] = upscale_model
    sampler_inputs = _inputs(workflow, "915:282")
    if sampler_inputs is not None:
        sampler_inputs.update(
            seed=seed,
            steps=steps,
            cfg=cfg if cfg > 0 else 5.0,
            denoise=denoise if denoise > 0 else 0.2,
            upscale_by=scale if scale > 0 else 2.0,
        )
        if sampler_name:
            sampler_inputs["sampler_name"] = sampler_name
        if scheduler:
            sampler_inputs["scheduler"] = scheduler
        sampler_inputs["batch_size"] = max(1, min(4, int(batch or 1)))
    if mode in {"txt2img", "img2img"}:
        sampler_values = sampler_inputs or {}
        sampler_defaults = _inputs(workflow, "915:282") or {}
        _apply_sampling_sampler(
            workflow,
            model_name=model_name,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=str(sampler_values.get("sampler_name", sampler_defaults.get("sampler_name", "er_sde"))),
            scheduler=str(sampler_values.get("scheduler", sampler_defaults.get("scheduler", "normal"))),
            denoise=1.0 if mode == "txt2img" else (denoise if denoise > 0 else 0.6),
            report=report,
        )
    save = _inputs(workflow, "1049")
    if save is not None:
        save["images"] = ["1413", 0]
        save["filename_prefix"] = filename_prefix
    else:
        report.append("原工作流缺少 SaveImage 节点")

    # 所有实际提交节点必须带 class_type；未知的布局元数据不参与提交。
    invalid = [node_id for node_id, node in workflow.items() if isinstance(node, dict) and not node.get("class_type")]
    for node_id in invalid:
        workflow.pop(node_id, None)
    return workflow, report


def copy_and_repair_original(source_path: Path, destination_dir: Path) -> dict[str, Path]:
    """把原工作流复制成所需的副本，并只做静态结构修复，不写回源文件。

    图生图已经改用独立的 Qwen Image Edit 工作流，所以这里不再生成原工作流的
    图生图副本——过去它会被生成却从不被提交，只是多出一份容易误选的同名文件。
    """
    source = load_api_workflow(source_path)
    destination_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for mode, name in (("txt2img", "文生图.json"), ("hires", "高清放大.json")):
        workflow = copy.deepcopy(source)
        report: list[str] = []
        _repair_missing_lora_node(workflow, report)
        # 原始工作流带有示例角色/画风 LoRA。插件启动时只复制结构，默认必须全部关闭，
        # 否则用户没有通过 WebUI 或 /loraon 选择 LoRA 时也会被强行套用。
        _apply_loras(workflow, [], report)
        save = _inputs(workflow, "1049")
        if save is not None:
            save["images"] = ["1413", 0]
            save["filename_prefix"] = f"astrbot/{mode}"
        scale = _inputs(workflow, "915:279")
        if scale is not None:
            scale["model_name"] = "realesrganX4plusAnime_v1.pt"
        _apply_mode(workflow, mode, report)
        if mode in {"txt2img", "img2img"}:
            sampler = _inputs(workflow, "915:282") or {}
            _apply_sampling_sampler(
                workflow,
                model_name="anima-base-v1.0.safetensors",
                steps=int(sampler.get("steps", 30) or 30),
                cfg=5.0,
                seed=int(sampler.get("seed", 0) or 0),
                sampler_name="er_sde",
                scheduler="simple",
                denoise=1.0 if mode == "txt2img" else 0.6,
                report=report,
            )
        _disable_optional_sage_attention(workflow, report)
        invalid = [node_id for node_id, node in workflow.items() if isinstance(node, dict) and not node.get("class_type")]
        for node_id in invalid:
            workflow.pop(node_id, None)
        target = destination_dir / name
        target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        result[mode] = target
    return result
