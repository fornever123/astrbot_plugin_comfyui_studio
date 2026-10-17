from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

try:  # 包内导入；单文件导入（测试）时退回同级模块名。
    from .prompt_profiles import (
        build_enhancer_extra,
        build_enhancer_format_rule,
        is_natural_language,
        resolve_prompt_profile,
    )
except ImportError:  # pragma: no cover - 仅用于脱离包环境加载本文件
    from prompt_profiles import (  # type: ignore[no-redef]
        build_enhancer_extra,
        build_enhancer_format_rule,
        is_natural_language,
        resolve_prompt_profile,
    )

# Independently written adaptations of the upstream model-specific workflow.
# No upstream code, uncensoring directives, or local llama-server dependency.
SOURCE_URL = "https://github.com/MaxKruse/ComfyUI-PromptEnhancer"

TXT2IMG_TEMPLATE = """你是图像提示词优化师，理解用户要画的内容，再写可直接交给图像模型的最终提示词。
先区分用户明确指定的主体、数量、动作、服装、表情、构图、场景和文字，以及没有指定的部分。
用户明确指定的内容必须保留；不要把动作遗漏、改成另一动作，或把人物改成另一角色。
按主体与动作、空间关系、环境、画面表现的顺序组织描述。必要时补少量有助于表达的视觉细节，不添加无关人物、服装或剧情。
采用画面中实际呈现的内容来描述，避免“需要画、用户要求、请生成”等任务说明。
图片内需要写出的文字保留原文、标点与语言，用双引号标出；人名与已有专有词保持原样。
预设和 LoRA 的简称已由插件优先识别，内容会在优化后合并。不要猜测、补写、替换这些控制项，不写 LoRA 权重或调用代码。
分辨率、宽高比、步数、CFG 由工作流设置，不写入提示词。只返回最终提示词。"""

IMG2IMG_TEMPLATE = """你是图像编辑提示词优化师，将用户需求整理成明确、可执行的编辑指令。
先确定用户要改变的对象、属性与目标效果。以编辑动作开头，具体写明改哪里、改成什么。
只改用户指定的属性，未指定的人物身份、脸部、姿势、构图、画风与背景保持原图不变。
不要重新描述整张原图，避免让编辑变成重新生成；不添加未要求的服装、人物、场景或修复任务。
用户已给出明确目标时保留它；要求“换个表情”等开放编辑时选择一种具体且符合上下文的结果，不能只写“改变表情”。
中文习语按画面含义理解，例如“一字马”是劈叉动作。
多图编辑保留用户对各图的用途：用<Picture 1>表示主图，<Picture 2>和<Picture 3>表示参考图，写清从哪张取哪种内容；未提到的图片不要编造用途。单图不需要图片编号。
图片中文字保留用户指定的原文与语言，用双引号标出。
预设与 LoRA 简称由插件优先识别、最后合并，不猜测或补写它们，不输出权重、模型名、宽高比和分辨率。
只返回一段最终编辑指令，以一句概括的保留说明保护原图，不逐项重画原图细节。"""

OUTPUT_RULES = """只输出可以直接交给绘图工作流的最终提示词；不要分析、思考过程、步骤、说明、字段名、Markdown 或代码块。
如果内部进行了思考，思考不能进入正文。不要用“分析用户请求”“最终提示词”等前缀。
本次提示词优化没有收到参考图片的视觉内容。只能依据用户文字和明确的图片编号关系处理，不声称看见或编造原图中的细节。"""


def build_system_prompt(
    config: Mapping[str, Any], mode: str, model: str, engine: str = ""
) -> str:
    """Build an editable, model-specific prompt enhancement instruction.

    Args:
        config: Saved settings or unsaved settings from the preview form.
        mode: Drawing mode.
        model: Selected diffusion model filename.
        engine: Optional engine key such as ``krea2``; takes precedence over the
            filename because a Krea 2 GGUF name carries no ``qwen``/``flux`` hint.

    Returns:
        Complete system instruction with an enforced final-answer contract.
    """
    is_edit = mode == "img2img"
    key = (
        f"prompt_enhancer_{mode}_template"
        if is_edit
        else "prompt_enhancer_txt2img_template"
    )
    template = str(config.get(key, "") or "").strip()
    template = template or (IMG2IMG_TEMPLATE if is_edit else TXT2IMG_TEMPLATE)
    profile = resolve_prompt_profile(model, engine)
    # 图生图天然是编辑指令；文生图交给模型家族档案判定（Anima 标签 / 其余自然语言）。
    natural_language = is_natural_language(model, engine, mode)
    format_rule = build_enhancer_format_rule(model, engine, mode)
    extra = build_enhancer_extra(model, mode, engine)
    parts = [template]
    if extra.strip():
        parts.append(extra.strip())
    parts.append(format_rule)
    parts.append(OUTPUT_RULES)
    if not natural_language and profile.key == "anima":
        # 保留一行可读的家族标记，便于日志与测试断言。
        parts.insert(1, "目标模型家族：Anima（英文 Danbooru 标签）。")
    return "\n\n".join(parts)


def clean_enhanced_prompt(value: str) -> str:
    """Extract the final answer without accepting a reasoning-only response.

    Args:
        value: Raw completion text, with optional reasoning tags or JSON.

    Returns:
        A single-paragraph final prompt, retaining quotes and image markers.

    Raises:
        ValueError: No reliable final prompt was returned.
    """
    raw = str(value or "").strip()
    raw = re.sub(
        r"<(think|analysis|reasoning)\b[^>]*>.*?</\1>", "", raw, flags=re.I | re.S
    )
    # Some gateways remove the opening reasoning tag but retain its closer.
    raw = re.split(r"</(?:think|analysis|reasoning)>", raw, flags=re.I)[-1].strip()
    if re.search(r"<(?:think|analysis|reasoning)\b", raw, flags=re.I):
        raise ValueError("AI 只返回了未完成的思考过程")
    raw = re.sub(
        r"```(?:thinking|analysis|reasoning)\s*.*?```", "", raw, flags=re.I | re.S
    ).strip()
    fenced = re.fullmatch(
        r"```(?:json|text|plaintext)?\s*\n?(.*?)\n?```", raw, flags=re.I | re.S
    )
    if fenced:
        raw = fenced.group(1).strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("AI 返回的提示词 JSON 不完整") from exc
        if not isinstance(data, dict):
            raise ValueError("AI 返回格式无法解析")
        raw = next(
            (
                data[key].strip()
                for key in (
                    "rewritten_prompt",
                    "enhanced_prompt",
                    "edit_instruction",
                    "instruction",
                    "prompt",
                    "final",
                    "result",
                )
                if isinstance(data.get(key), str) and data[key].strip()
            ),
            "",
        )
    final = re.search(
        r"(?:^|\n)\s*(?:最终(?:编辑指令|提示词|结果)|final(?:\s+prompt)?|enhanced_prompt|rewritten_prompt)\s*[:：]\s*(.+)$",
        raw,
        flags=re.I | re.S,
    )
    if final:
        raw = final.group(1).strip()
    raw = re.sub(r"^(?:edit|instruction|prompt)\s*[:：]\s*", "", raw, flags=re.I)
    if re.search(
        r"(?:分析用户|我们需要|首先[，,:：]|思考过程|推理过程|输出协议|系统提示词|用户要求输出|"
        r"\b(?:let'?s think|we need to|chain.of.thought|reasoning|analysis)\b|"
        r"(?:^|\n)\s*\d+[.)、]\s|\b(?:cannot|can't|unable to)\s+(?:help|assist|comply))",
        raw,
        flags=re.I,
    ):
        raise ValueError("AI 返回了分析或说明，未提取到最终提示词")
    result = re.sub(r"\s+", " ", raw).strip()
    if not result or len(result) > 6000 or "```" in result:
        raise ValueError("AI 未返回有效的最终提示词")
    return result
