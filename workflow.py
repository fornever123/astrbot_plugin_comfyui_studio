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


def convert_ui_workflow(data: dict[str, Any]) -> dict[str, Any]:
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
    for node in raw_nodes:
        if not isinstance(node, dict) or "id" not in node or not node.get("type"):
            continue
        if str(node.get("type")) in {"Note", "Reroute", "PrimitiveNode"}:
            continue
        # Disabled editor nodes must not be executed. The Qwen workflow uses
        # this for its optional second reference image and optional LoRA.
        if int(node.get("mode", 0) or 0) == 4:
            continue
        node_by_id[str(node["id"])] = node

    widget_maps: dict[str, tuple[str, ...]] = {
        "TextEncodeQwenImageEditPlus": ("prompt",),
        "TextEncodeQwenImageEdit": ("prompt",),
        "LoadImage": ("image",),
        "ImageScaleToTotalPixels": (
            "upscale_method", "megapixels", "resolution_steps"
        ),
        "FluxKontextMultiReferenceLatentMethod": ("reference_latents_method",),
        "ModelSamplingAuraFlow": ("shift",),
        "CFGNorm": ("strength", "pre_cfg"),
        "EmptySD3LatentImage": ("width", "height", "batch_size"),
        "CLIPLoaderGGUF": ("clip_name", "type"),
        "UnetLoaderGGUF": ("unet_name",),
        "LoraLoaderModelOnly": ("lora_name", "strength_model"),
        "VAELoader": ("vae_name",),
        "KSampler": (
            "seed", "control_after_generate", "steps", "cfg",
            "sampler_name", "scheduler", "denoise",
        ),
        "VAEDecodeTiled": (
            "tile_size", "overlap", "temporal_size", "temporal_overlap"
        ),
        "SaveImage": ("filename_prefix",),
    }
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
            if source_id in node_by_id:
                inputs[name] = [source_id, source_slot]

        widget_values = list(node.get("widgets_values") or [])
        names = widget_maps.get(class_type, ())
        for index, name in enumerate(names):
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


def load_workflow(path: Path) -> dict[str, Any]:
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
        return data
    return convert_ui_workflow(data)


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
    denoise: float = 1.0,
    scale_method: str = "lanczos",
    largest_size: int = 1152,
    crop: str = "center",
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
    """Apply the API workflow exported from E:/112121121.json.

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
    # 112121121.json 的 TextEncodeQwenImageEditPlus 只有 image1 输入。
    # 旧版适配器曾动态添加 image2，这会让 ComfyUI 拒绝整个 API 请求；
    # 保留参数只是为了兼容旧调用，但不向这个工作流伪造不存在的端口。
    positive_inputs.pop("image2", None)
    positive_inputs.pop("image3", None)
    if second_image_name:
        report.append("当前 Qwen 工作流只支持一张参考图，已忽略第二参考图")

    _qwen_inputs(workflow, "160")["unet_name"] = unet_name
    _qwen_inputs(workflow, "161").update(clip_name=clip_name, type="qwen_image")
    _qwen_inputs(workflow, "10")["vae_name"] = vae_name
    _qwen_inputs(workflow, "30").update(shift=max(0.0, float(sampling_shift)), model=["174", 0] if _node(workflow, "174") else ["160", 0])

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
    if width > 0 and height > 0:
        report.append(f"Qwen 图生图输出尺寸已写入 ImageScale：{int(width)}x{int(height)}")
    _ = (reference_method, cfg_norm_strength, pre_cfg, tile_size, tile_overlap, temporal_size, temporal_overlap)
    return workflow, report


def adapt_flux2_img2img(
    source: dict[str, Any],
    *,
    positive: str,
    negative: str,
    image_name: str,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    loras: list[str] | None = None,
    width: int = 720,
    height: int = 1280,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = 0,
    sampler_name: str = "euler",
    scale_method: str = "lanczos",
    megapixels: float = 0.8,
    resolution_steps: int = 1,
    filename_prefix: str = "astrbot/img2img_flux2",
) -> tuple[dict[str, Any], list[str]]:
    """Enable real Flux.2 Klein reference-image editing on the supplied workflow.

    The supplied ``11111图生图.json`` is the text-to-image state of the same
    workflow: its EmptyFlux2LatentImage is connected to the sampler and its
    reference-image branch is absent.  Flux.2 Klein expects the output canvas
    to remain an empty Flux2 latent, while the source image is VAE encoded and
    attached to the positive and negative conditionings as reference_latents.
    """
    workflow = copy.deepcopy(source)
    report: list[str] = []

    if not any(
        isinstance(node, dict) and node.get("class_type") == "EmptyFlux2LatentImage"
        for node in workflow.values()
    ):
        raise WorkflowError("Flux2 图生图工作流缺少 EmptyFlux2LatentImage 输出画布节点")
    if not image_name.strip():
        raise WorkflowError("Flux2 图生图没有收到已上传的输入图片")

    # Keep the original node chain as the source of truth.  These IDs are the
    # stable API nodes in E:\\11111图生图.json; the fallback lookup also makes
    # an uploaded equivalent workflow usable after a harmless node re-number.
    def find_node(class_type: str, preferred: str = "") -> str:
        if preferred and _node(workflow, preferred):
            return preferred
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                return str(node_id)
        return ""

    positive_id = find_node("CLIPTextEncode", "135")
    negative_zero_id = find_node("ConditioningZeroOut", "685")
    clip_loader_id = find_node("CLIPLoaderGGUF", "731")
    vae_loader_id = find_node("VAELoader", "127")
    lora_node_ids = _power_lora_nodes(workflow)
    model_loader_id = find_node("UNETLoader", "126")
    model_patch_id = find_node("FluxKVCache", "139")
    guider_id = find_node("CFGGuider", "138")
    sampler_id = find_node("SamplerCustomAdvanced", "123")
    noise_id = find_node("RandomNoise", "125")
    sampler_select_id = find_node("KSamplerSelect", "122")
    scheduler_id = find_node("Flux2Scheduler", "137")
    latent_id = find_node("EmptyFlux2LatentImage", "129")
    decode_id = find_node("VAEDecode", "124")
    save_id = find_node("SaveImage", "94")
    required = {
        "正面提示词": positive_id,
        "负面条件": negative_zero_id,
        "文本编码器": clip_loader_id,
        "VAE": vae_loader_id,
        "核心模型": model_loader_id,
        "Flux KV 缓存": model_patch_id,
        "CFG 引导": guider_id,
        "采样器": sampler_id,
        "噪声": noise_id,
        "采样器选择": sampler_select_id,
        "Flux2 调度器": scheduler_id,
        "输出画布": latent_id,
        "VAE 解码": decode_id,
        "保存图片": save_id,
    }
    missing = [label for label, node_id in required.items() if not node_id]
    if missing:
        raise WorkflowError(f"Flux2 图生图工作流缺少节点：{', '.join(missing)}")

    _inputs(workflow, positive_id).update(text=positive, clip=[clip_loader_id, 0])

    _inputs(workflow, model_loader_id)["unet_name"] = unet_name
    _inputs(workflow, clip_loader_id).update(clip_name=clip_name, type="flux2")
    _inputs(workflow, vae_loader_id)["vae_name"] = vae_name
    if loras is not None:
        _apply_loras(workflow, loras, report)

    # The source workflow's KV cache is the reference-image optimization and
    # must remain immediately downstream of the model/LoRA chain.
    model_inputs = _inputs(workflow, model_patch_id)
    model_inputs["model"] = [lora_node_ids[-1], 0] if lora_node_ids else [model_loader_id, 0]
    _inputs(workflow, guider_id).update(
        model=[model_patch_id, 0],
        positive=["707:703", 0],
        negative=[negative_zero_id, 0],
        cfg=max(0.0, float(cfg)),
    )
    _inputs(workflow, noise_id)["noise_seed"] = max(0, int(seed))
    _inputs(workflow, sampler_select_id)["sampler_name"] = sampler_name or "euler"
    _inputs(workflow, scheduler_id).update(
        steps=max(1, int(steps)),
        width=max(16, int(width)),
        height=max(16, int(height)),
    )
    _inputs(workflow, latent_id).update(
        width=max(16, int(width)),
        height=max(16, int(height)),
        batch_size=1,
    )
    _inputs(workflow, decode_id)["vae"] = [vae_loader_id, 0]
    _inputs(workflow, save_id)["filename_prefix"] = filename_prefix

    # This reproduces the enabled-reference branch in E:\\222.json.  Do not
    # connect this latent to sampler.latent_image: Flux2 Klein uses it as a
    # reference condition and keeps the EmptyFlux2 latent as the canvas.
    load_id = "76"
    size_id = "128"
    scale_id = "707:130"
    encode_id = "707:702"
    positive_ref_id = "707:703"
    workflow[load_id] = {
        "inputs": {"image": image_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Flux2 图生图输入图片"},
    }
    workflow[size_id] = {
        "inputs": {"image": [scale_id, 0]},
        "class_type": "GetImageSize",
        "_meta": {"title": "Flux2 图生图参考图尺寸"},
    }
    workflow[scale_id] = {
        "inputs": {
            "image": [load_id, 0],
            "upscale_method": scale_method or "lanczos",
            "megapixels": max(0.01, min(16.0, float(megapixels or 0.8))),
            "resolution_steps": max(1, int(resolution_steps or 1)),
        },
        "class_type": "ImageScaleToTotalPixels",
        "_meta": {"title": "Flux2 图生图参考图缩放"},
    }
    workflow[encode_id] = {
        "inputs": {"pixels": [scale_id, 0], "vae": [vae_loader_id, 0]},
        "class_type": "VAEEncode",
        "_meta": {"title": "Flux2 图生图参考图编码"},
    }
    workflow[positive_ref_id] = {
        "inputs": {
            "conditioning": [positive_id, 0],
            "latent": [encode_id, 0],
        },
        "class_type": "ReferenceLatent",
        "_meta": {"title": "Flux2 正面参考图条件"},
    }
    width_inputs = _inputs(workflow, "727")
    height_inputs = _inputs(workflow, "728")
    if width_inputs is not None and height_inputs is not None:
        width_inputs["any_01"] = [size_id, 0]
        width_inputs["any_02"] = ["725", 0]
        height_inputs["any_01"] = [size_id, 1]
        height_inputs["any_02"] = ["726", 0]
    report.append("已按 222.json 启用 Flux2 Klein 参考图：LoadImage → 缩放 → VAEEncode → ReferenceLatent")
    report.append("采样器仍使用 EmptyFlux2LatentImage 作为输出画布，输入图不会被当作文生图开关")
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
    """把原工作流复制成三份，并只做静态结构修复，不写回源文件。"""
    source = load_api_workflow(source_path)
    destination_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for mode, name in (("txt2img", "文生图.json"), ("img2img", "图生图.json"), ("hires", "高清放大.json")):
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
