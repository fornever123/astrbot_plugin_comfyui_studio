from __future__ import annotations

import copy
import json
import random
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
        # branch. Some supplied editor graphs use that same flag for optional
        # reference branches and their model LoRA, so those nodes and their
        # original connections can be preserved explicitly.
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
# its original file references alive: bundled editor graphs can still carry the
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


# ---------------------------------------------------------------------------
# 通用工作流识别与适配

# ComfyUI 自定义节点的命名很多，但输入字段通常仍遵循这些约定。通用层
# 只修改实际存在的字段，未知节点和未知输入会原样保留。
_TEXT_INPUT_NAMES = {
    "text", "prompt", "positive_prompt", "negative_prompt", "string", "value",
}
_OUTPUT_CLASS_MARKERS = ("saveimage", "previewimage", "saveanimated", "videocombine")
_SAMPLER_CLASS_MARKERS = ("ksampler", "samplercustom", "sampler")
_MODEL_INPUT_NAMES = {"ckpt_name", "unet_name", "model_name", "diffusion_model"}
_CLIP_INPUT_NAMES = {"clip_name", "text_encoder", "text_encoder_name"}
_VAE_INPUT_NAMES = {"vae_name"}
_SEED_INPUT_NAMES = {"seed", "noise_seed", "random_seed"}


def _generic_title(node: dict[str, Any]) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict) and meta.get("title"):
        return str(meta["title"])
    return str(node.get("class_type", ""))


def _generic_candidate(
    node_id: str,
    input_name: str,
    *,
    title: str,
    value: Any,
    reason: str,
) -> dict[str, Any]:
    current = value
    if isinstance(current, (list, tuple)):
        current = f"连接到节点 {current[0] if current else '?'}"
    elif isinstance(current, dict):
        current = "对象输入"
    else:
        current = str(current)[:180]
    return {
        "node_id": str(node_id),
        "input": str(input_name),
        "title": title,
        "current": current,
        "reason": reason,
    }


def _generic_text_kind(title: str, input_name: str) -> str:
    lowered = f"{title} {input_name}".casefold()
    if any(word in lowered for word in ("negative", "负面", "反向", "负提示")):
        return "negative_text"
    if any(word in lowered for word in ("positive", "正面", "prompt", "提示", "正向")):
        return "positive_text"
    return "text"


def _generic_is_output_node(node: dict[str, Any]) -> bool:
    class_type = str(node.get("class_type", ""))
    title = _generic_title(node)
    lowered = f"{class_type} {title}".casefold()
    if any(marker in lowered for marker in _OUTPUT_CLASS_MARKERS):
        return True
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return False
    has_image_output = any(str(name).casefold() in {"image", "images", "output_image"} for name in inputs)
    return has_image_output and any(marker in lowered for marker in ("output", "输出", "save", "preview"))


def inspect_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """生成适合展示和交给 AI 判断的工作流摘要。

    这是只读操作，不会修改 ``workflow``。摘要刻意省略大段节点数据，避免
    把整个工作流原文塞进模型上下文；AI 只需要节点 ID、类别、输入名和当前值。
    """
    if not isinstance(workflow, dict) or not workflow:
        raise WorkflowError("工作流为空，无法分析")

    candidates: dict[str, list[dict[str, Any]]] = {
        "positive_text": [],
        "negative_text": [],
        "images": [],
        "model": [],
        "clip": [],
        "vae": [],
        "loras": [],
        "samplers": [],
        "sizes": [],
        "outputs": [],
    }
    text_candidates: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    for node_id, raw_node in workflow.items():
        if not isinstance(raw_node, dict) or not raw_node.get("class_type"):
            continue
        class_type = str(raw_node.get("class_type", ""))
        title = _generic_title(raw_node)
        inputs = raw_node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        class_lower = class_type.casefold()
        title_lower = title.casefold()
        node_info = {
            "id": str(node_id),
            "class_type": class_type,
            "title": title,
            "inputs": {
                str(name): (
                    f"连接到节点 {value[0]}"
                    if isinstance(value, (list, tuple)) and value
                    else str(value)[:180]
                )
                for name, value in inputs.items()
            },
        }
        nodes.append(node_info)

        is_output = _generic_is_output_node(raw_node)
        if is_output:
            candidates["outputs"].append({"node_id": str(node_id), "title": title})

        for raw_name, value in inputs.items():
            input_name = str(raw_name)
            input_lower = input_name.casefold()
            candidate = None

            is_text_field = (
                input_lower in _TEXT_INPUT_NAMES
                or input_lower.endswith("_prompt")
                or input_lower in {"positive", "negative"} and not isinstance(value, (list, tuple))
            )
            if not is_output and (
                "cliptextencode" in class_lower
                or "textencode" in class_lower
                or is_text_field
                or "prompt" in title_lower
            ) and is_text_field and not isinstance(value, (list, tuple)):
                candidate = _generic_candidate(
                    str(node_id), input_name, title=title, value=value,
                    reason="文本编码或提示词输入",
                )
                text_candidates.append(candidate)
                kind = _generic_text_kind(title, input_name)
                if kind in {"positive_text", "negative_text"}:
                    candidates[kind].append(candidate)
                continue

            if not is_output and (
                "loadimage" in class_lower
                or input_lower == "image"
                or (input_lower.startswith("image") and input_lower[5:].isdigit())
            ):
                # 只有文件型输入才是可替换的参考图；图像连接不能被文件名覆盖。
                if isinstance(value, str) or "loadimage" in class_lower:
                    candidates["images"].append(_generic_candidate(
                        str(node_id), input_name, title=title, value=value,
                        reason="参考图输入",
                    ))

            if input_lower in _MODEL_INPUT_NAMES and not any(
                marker in class_lower for marker in ("upscale", "lora", "controlnet")
            ):
                candidates["model"].append(_generic_candidate(
                    str(node_id), input_name, title=title, value=value,
                    reason="核心模型输入",
                ))
            if input_lower in _CLIP_INPUT_NAMES or (
                "cliploader" in class_lower and input_lower.endswith("name")
            ):
                candidates["clip"].append(_generic_candidate(
                    str(node_id), input_name, title=title, value=value,
                    reason="文本编码器输入",
                ))
            if input_lower in _VAE_INPUT_NAMES or (
                "vaeloader" in class_lower and input_lower.endswith("name")
            ):
                candidates["vae"].append(_generic_candidate(
                    str(node_id), input_name, title=title, value=value,
                    reason="VAE 输入",
                ))
            if "lora" in class_lower and (
                input_lower == "lora_name" or input_lower.startswith("lora_")
            ):
                candidates["loras"].append(_generic_candidate(
                    str(node_id), input_name, title=title, value=value,
                    reason="LoRA 输入",
                ))

        sampler_fields = {
            name for name in inputs
            if str(name).casefold() in (
                _SEED_INPUT_NAMES | {"steps", "cfg", "sampler_name", "scheduler", "denoise", "noise"}
            )
        }
        if any(marker in class_lower for marker in _SAMPLER_CLASS_MARKERS) or sampler_fields:
            candidates["samplers"].append({
                "node_id": str(node_id),
                "title": title,
                "inputs": sorted(str(name) for name in sampler_fields),
            })

        size_fields = {
            name for name in inputs
            if str(name).casefold() in {"width", "height", "batch_size", "count"}
        }
        if size_fields and (
            "latent" in class_lower or "resize" in class_lower
            or {str(name).casefold() for name in size_fields} >= {"width", "height"}
        ):
            candidates["sizes"].append({
                "node_id": str(node_id),
                "title": title,
                "inputs": sorted(str(name) for name in size_fields),
            })

    # 很多工作流没有写 Positive/Negative 标题，按文本节点出现顺序做保守兜底。
    if not candidates["positive_text"]:
        for item in text_candidates:
            if item not in candidates["negative_text"]:
                candidates["positive_text"].append(item)
                break
    if not candidates["negative_text"] and len(text_candidates) > 1:
        for item in text_candidates[1:]:
            candidates["negative_text"].append(item)

    return {
        "nodes": nodes,
        "candidate_inputs": candidates,
        "node_count": len(nodes),
        "output_count": len(candidates["outputs"]),
    }


def infer_workflow_mapping(workflow: dict[str, Any], mode: str = "txt2img") -> dict[str, Any]:
    """不调用 AI 的通用映射推断，作为一键适配的离线兜底。"""
    summary = inspect_workflow(workflow)
    mapping = copy.deepcopy(summary["candidate_inputs"])
    mapping["mode"] = str(mode or "txt2img")
    mapping["warnings"] = []
    if not mapping["positive_text"]:
        mapping["warnings"].append("没有识别到正面提示词输入，请在工作流页面手动修正映射")
    if not mapping["outputs"]:
        mapping["warnings"].append("没有识别到图片输出节点")
    if str(mode).lower() == "img2img" and not mapping["images"]:
        mapping["warnings"].append("没有识别到参考图输入，图生图无法提交")
    return mapping


def validate_workflow_mapping(
    workflow: dict[str, Any], mapping: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """校验 AI 或用户编辑的映射，过滤不存在的节点和输入字段。"""
    raw = mapping if isinstance(mapping, dict) else {}
    valid: dict[str, Any] = {}
    warnings: list[str] = []
    list_categories = (
        "positive_text", "negative_text", "images", "model", "clip", "vae", "loras",
    )
    for category in list_categories:
        accepted: list[dict[str, Any]] = []
        values = raw.get(category, [])
        if not isinstance(values, list):
            values = [values]
        for entry in values:
            if not isinstance(entry, dict):
                continue
            node_id = str(entry.get("node_id", ""))
            input_name = str(entry.get("input", ""))
            node = _node(workflow, node_id)
            inputs = node.get("inputs") if node else None
            if not isinstance(inputs, dict) or input_name not in inputs:
                warnings.append(f"映射已忽略：节点 {node_id}.{input_name} 不存在")
                continue
            accepted.append({"node_id": node_id, "input": input_name})
        valid[category] = accepted

    sampler_values = raw.get("samplers", [])
    if not isinstance(sampler_values, list):
        sampler_values = [sampler_values]
    valid_samplers: list[dict[str, Any]] = []
    for entry in sampler_values:
        if not isinstance(entry, dict):
            continue
        node_id = str(entry.get("node_id", ""))
        node = _node(workflow, node_id)
        inputs = node.get("inputs") if node else None
        if not isinstance(inputs, dict):
            warnings.append(f"映射已忽略：采样器节点 {node_id} 不存在")
            continue
        names = entry.get("inputs", [])
        if not isinstance(names, list):
            names = list(inputs)
        names = [str(name) for name in names if str(name) in inputs]
        if names:
            valid_samplers.append({"node_id": node_id, "inputs": names})
    valid["samplers"] = valid_samplers

    size_values = raw.get("sizes", [])
    if not isinstance(size_values, list):
        size_values = [size_values]
    valid_sizes: list[dict[str, Any]] = []
    for entry in size_values:
        if not isinstance(entry, dict):
            continue
        node_id = str(entry.get("node_id", ""))
        node = _node(workflow, node_id)
        inputs = node.get("inputs") if node else None
        if not isinstance(inputs, dict):
            continue
        names = entry.get("inputs", [])
        if not isinstance(names, list):
            names = list(inputs)
        names = [str(name) for name in names if str(name) in inputs]
        if names:
            valid_sizes.append({"node_id": node_id, "inputs": names})
    valid["sizes"] = valid_sizes

    outputs = raw.get("outputs", [])
    if not isinstance(outputs, list):
        outputs = [outputs]
    valid_outputs: list[str] = []
    for raw_output in outputs:
        node_id = raw_output.get("node_id") if isinstance(raw_output, dict) else raw_output
        node_id = str(node_id or "")
        node = workflow.get(node_id)
        if (
            isinstance(node, dict)
            and _generic_is_output_node(node)
        ):
            valid_outputs.append(node_id)
    valid["outputs"] = valid_outputs
    if not valid["positive_text"]:
        warnings.append("没有有效的正面提示词映射")
    if not valid["outputs"]:
        warnings.append("没有有效的输出节点映射")
    return valid, warnings


def _generic_set_input(
    workflow: dict[str, Any], node_id: str, input_name: str, value: Any,
) -> bool:
    inputs = _inputs(workflow, node_id)
    if inputs is None or input_name not in inputs:
        return False
    inputs[input_name] = value
    return True


def _generic_set_sampler_input(inputs: dict[str, Any], field: str, values: dict[str, Any]) -> bool:
    lowered = str(field).casefold()
    if lowered in _SEED_INPUT_NAMES:
        inputs[field] = values["seed"]
    elif lowered == "steps":
        inputs[field] = values["steps"]
    elif lowered == "cfg":
        inputs[field] = values["cfg"]
    elif lowered == "sampler_name":
        inputs[field] = values["sampler_name"]
    elif lowered == "scheduler":
        inputs[field] = values["scheduler"]
    elif lowered == "denoise":
        inputs[field] = values["denoise"]
    else:
        return False
    return True


def adapt_generic(
    source: dict[str, Any],
    *,
    mode: str,
    positive: str,
    negative: str = "",
    image_names: list[str] | None = None,
    model_name: str = "",
    clip_name: str = "",
    vae_name: str = "",
    loras: list[str] | None = None,
    width: int = 0,
    height: int = 0,
    steps: int = 20,
    cfg: float = 5.0,
    seed: int = -1,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 1.0,
    batch: int = 1,
    filename_prefix: str = "astrbot/generic",
    mapping: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """把常见绘图输入写入任意 API 工作流。

    ``mapping`` 可以来自 ``infer_workflow_mapping`` 或 AI 一键适配结果。
    除了映射到的输入，原始图结构、节点参数和自定义节点都不被重写。
    """
    if not isinstance(source, dict) or not source:
        raise WorkflowError("工作流为空，无法适配")
    workflow = copy.deepcopy(source)
    inferred = infer_workflow_mapping(workflow, mode)
    chosen = mapping if isinstance(mapping, dict) else inferred
    valid, warnings = validate_workflow_mapping(workflow, chosen)
    # AI 可能只返回部分字段，缺失字段使用离线推断；AI 返回的有效字段优先。
    for category in ("positive_text", "negative_text", "images", "model", "clip", "vae", "loras", "samplers", "sizes", "outputs"):
        if not valid.get(category):
            valid[category] = inferred.get(category, [])
    report = list(warnings)
    if not valid.get("positive_text"):
        raise WorkflowError("通用适配没有找到正面提示词输入，请在工作流页面使用 AI 一键适配或手动编辑映射")
    if not valid.get("outputs"):
        raise WorkflowError("通用适配没有找到 SaveImage/PreviewImage 输出节点")

    positive_text = str(positive or "").strip()
    if not positive_text:
        raise WorkflowError("请提供正面提示词或画面描述")
    for entry in valid["positive_text"]:
        if _generic_set_input(workflow, entry["node_id"], entry["input"], positive_text):
            pass
    if str(negative or "").strip():
        for entry in valid.get("negative_text", []):
            _generic_set_input(workflow, entry["node_id"], entry["input"], str(negative).strip())

    images = [str(name).strip() for name in (image_names or []) if str(name).strip()]
    if str(mode or "").casefold() == "img2img" and not images:
        raise WorkflowError("图生图至少需要一张参考图片")
    image_entries = valid.get("images", [])
    for entry, image_name in zip(image_entries, images):
        if _generic_set_input(workflow, entry["node_id"], entry["input"], image_name):
            pass
    if len(images) > len(image_entries):
        report.append(f"当前工作流只识别到 {len(image_entries)} 个参考图输入，额外的 {len(images) - len(image_entries)} 张图片未接入")

    def set_loader(category: str, value: str, label: str) -> None:
        if not str(value or "").strip():
            return
        entries = valid.get(category, [])
        changed = False
        for entry in entries:
            changed = _generic_set_input(workflow, entry["node_id"], entry["input"], str(value).strip()) or changed
        if not changed:
            report.append(f"没有找到{label}输入，保留工作流原设置")

    set_loader("model", model_name, "核心模型")
    set_loader("clip", clip_name, "文本编码器")
    set_loader("vae", vae_name, "VAE")

    parsed_loras = [_parse_lora(item) for item in (loras or []) if str(item).strip()]
    lora_entries = valid.get("loras", [])
    for entry, (name, strength) in zip(lora_entries, parsed_loras):
        node_id, input_name = entry["node_id"], entry["input"]
        inputs = _inputs(workflow, node_id) or {}
        current = inputs.get(input_name)
        if isinstance(current, dict):
            replacement = dict(current)
            replacement.update(on=True, lora=name, strength=strength)
            inputs[input_name] = replacement
        else:
            inputs[input_name] = name
            if "strength_model" in inputs:
                inputs["strength_model"] = strength
            if "strength_clip" in inputs:
                inputs["strength_clip"] = strength
    if parsed_loras and len(parsed_loras) > len(lora_entries):
        report.append(f"当前工作流只识别到 {len(lora_entries)} 个 LoRA 输入，剩余 LoRA 未接入")

    sampler_values = {
        "seed": int(seed if int(seed) >= 0 else random.randint(0, 2**63 - 1)),
        "steps": max(1, int(steps or 20)),
        "cfg": max(0.0, float(cfg)),
        "sampler_name": str(sampler_name or "euler"),
        "scheduler": str(scheduler or "normal"),
        "denoise": max(0.0, min(1.0, float(denoise))),
    }
    for entry in valid.get("samplers", []):
        inputs = _inputs(workflow, entry["node_id"])
        if not inputs:
            continue
        for field in entry.get("inputs", []):
            _generic_set_sampler_input(inputs, field, sampler_values)
    for entry in valid.get("sizes", []):
        inputs = _inputs(workflow, entry["node_id"])
        if not inputs:
            continue
        for field in entry.get("inputs", []):
            lowered = field.casefold()
            if lowered == "width" and int(width or 0) > 0:
                inputs[field] = max(64, int(width))
            elif lowered == "height" and int(height or 0) > 0:
                inputs[field] = max(64, int(height))
            elif lowered in {"batch_size", "count"}:
                inputs[field] = max(1, min(16, int(batch or 1)))
    for output_id in valid.get("outputs", []):
        inputs = _inputs(workflow, output_id)
        if inputs and filename_prefix and "filename_prefix" in inputs:
            inputs["filename_prefix"] = str(filename_prefix)

    report.append(f"已使用通用适配：识别到 {len(valid['positive_text'])} 个正面提示词输入、{len(valid['outputs'])} 个输出节点")
    if str(mode or "").casefold() == "img2img":
        report.append(f"已接入 {min(len(images), len(image_entries))} 张参考图")
    return workflow, report


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
