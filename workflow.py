from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class WorkflowError(Exception):
    pass


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

    sampler_node = "astrbot_txt2img_sampler"
    decode_node = "astrbot_txt2img_decode"
    workflow[sampler_node] = {
        "inputs": {
            "model": ["1071:1075", 0],
            "positive": ["1071:1063", 0],
            "negative": ["1071:1062", 0],
            "latent_image": ["1339:1322", 0],
            "seed": int(seed),
            "steps": max(1, int(steps)),
            "cfg": float(cfg if cfg > 0 else 5.0),
            "sampler_name": sampler_name or "er_sde",
            "scheduler": scheduler or "normal",
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
    report.append("已启用原工作流模型链的标准采样支路")


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
                steps=int(sampler.get("steps", 30) or 30),
                cfg=5.0,
                seed=int(sampler.get("seed", 0) or 0),
                sampler_name="er_sde",
                scheduler="normal",
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
