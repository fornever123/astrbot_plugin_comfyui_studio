from __future__ import annotations

import asyncio
import base64
import copy
import difflib
import hashlib
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import struct
import time
import unicodedata
import uuid
import zlib
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Forward, Image, Json, Node, Nodes, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .ai import DANBOORU_SYSTEM_PROMPT, AIError, AITranslator
from .anima_knowledge import (
    build_context as build_anima_context,
    find_template_path,
    is_drawing_request,
)
from .comfy import ComfyClient, ComfyError
from .moderation import (
    ImageModerator,
    ModerationConfig,
    ModerationVerdict,
)
from .paths import (
    MODEL_CATEGORIES,
    MODEL_CATEGORY_LABELS,
    MODEL_CATEGORY_SUFFIXES,
    ModelDir,
    category_aliases,
    detect_comfyui_root,
    find_extra_model_paths_file,
    read_extra_model_paths,
    SOURCE_LABELS,
    resolve_model_dirs,
    scan_path,
)
from .prompting import (
    DrawParams,
    PresetStore,
    UsageError,
    contains_chinese,
    fill_params,
)
from .translation import PlainTranslationError, PlainTranslator
from .workflow import (
    WorkflowError,
    adapt_qwen_img2img,
    adapt_flux2_klein_img2img,
    adapt_flux2_tool,
    adapt_original,
    anima_sampling_defaults,
    copy_and_repair_original,
    is_anima_29b_model,
    load_api_workflow,
    load_workflow,
)

PLUGIN_NAME = "astrbot_plugin_comfyui_ai_studio"
PLUGIN_VERSION = "1.1.1"


def _log_path(value: object) -> str:
    """Keep Windows console logs encodable when a path contains ``▶``."""
    return str(value).replace("▶", ">")


DEFAULT_IMG2IMG_DENOISE = 1.0
IMG2IMG_PRESERVE_TEXT = "未要求改变的内容保持原图不变"
IMG2IMG_PRESERVE_TEXT_EN = (
    "preserve the original character identity, face, hair, body proportions, pose, "
    "background, camera, composition, lighting and art style"
)
DEFAULT_ARTIST_PRESET_NAME = "画风001"
DEFAULT_ARTIST_PRESET_TAGS = (
    "@yukisiannn, @kani biimu, @ixy, @shnva, @shiromochi sakura, @stmast,"
)
CHARACTER_PRESET_SEED = {
    "漂泊者": "rover_(wuthering_waves)",
    "炽霞": "chixia_(wuthering_waves)",
    "忌炎": "jiyan_(wuthering_waves)",
    "吟霖": "yinlin_(wuthering_waves)",
    "卡卡罗": "calcharo_(wuthering_waves)",
    "今汐": "jinhsi_(wuthering_waves)",
    "长离": "changli_(wuthering_waves)",
    "守岸人": "shorekeeper_(wuthering_waves)",
    "椿": "camellya_(wuthering_waves)",
    "珂莱塔": "carlotta_(wuthering_waves)",
    "菲比": "phoebe_(wuthering_waves)",
    "洛可可": "roccia_(wuthering_waves)",
    "布兰特": "brant_(wuthering_waves)",
    "坎特蕾拉": "cantarella_(wuthering_waves)",
    "卡提希娅": "cartethyia_(wuthering_waves)",
    "露帕": "lupa_(wuthering_waves)",
    "尤诺": "iuno_(wuthering_waves)",
    "奥古斯塔": "augusta_(wuthering_waves)",
    "莫宁": "mornye_(wuthering_waves)",
    "秧秧": "yangyang_(wuthering_waves)",
    "散华": "sanhua_(wuthering_waves)",
    "凌阳": "lingyang_(wuthering_waves)",
    "折枝": "zhezhi_(wuthering_waves)",
    "相里要": "xiangli_yao_(wuthering_waves)",
    "釉瑚": "youhu_(wuthering_waves)",
    "灯灯": "lumi_(wuthering_waves)",
    "仇远": "qiuyuan_(wuthering_waves)",
    "卜灵": "buling_(wuthering_waves)",
    "嘉贝莉娜": "galbrena_(wuthering_waves)",
    "千咲": "chisa_(wuthering_waves)",
    "琳奈": "lynae_(wuthering_waves)",
    "绯雪": "hiyuki_(wuthering_waves)",
    "达妮娅": "denia_(wuthering_waves)",
    "白芷": "baizhi_(wuthering_waves)",
    "秋水": "aalto_(wuthering_waves)",
    "桃祈": "taoqi_(wuthering_waves)",
    "丹瑾": "danjin_(wuthering_waves)",
    "渊武": "yuanwu_(wuthering_waves)",
    "清宵": "qingxiao_(wuthering_waves)",
    "景燃": "jingran_(wuthering_waves)",
    "穗穗": "suisui_(wuthering_waves)",
    "锁暝": "suoming_(wuthering_waves)",
    "维里奈": "verina_(wuthering_waves)",
    "安可": "encore_(wuthering_waves)",
    "心": "hsin_(wuthering_waves)",
}
MODE_NAMES = {
    "txt2img": "文生图",
    "img2img": "图生图",
    "img2img_flux2": "图生图 Flux2",
    "hires": "高清放大",
    "wash": "洗图",
    "outpaint": "扩图",
    "multi_angle": "多角度",
}
MODE_ALIASES = {
    "文生图": "txt2img", "图生图": "img2img", "高清放大": "hires", "高清": "hires",
    "图生图Flux2": "img2img_flux2", "Flux2图生图": "img2img_flux2", "图生图2": "img2img_flux2",
    "洗图": "wash", "wash": "wash",
    "扩图": "outpaint", "outpaint": "outpaint",
    "多角度": "multi_angle", "multi-angle": "multi_angle",
}
# 图生图独立 LoRA 语义的「特别标注」。图生图只受各自页面配置的独立 LoRA
# 影响：不读取文生图的默认 LoRA 列表，也不参与画风 LoRA 的随机/全部自动
# 选择。该文案同时用在 WebUI 提示和开始回复中，保证前后端口径一致。
IMG2IMG_LORA_ISOLATION_NOTICE = (
    "【特别标注】图生图只受「独立 LoRA」影响："
    "它不会加载文生图的默认 LoRA 列表，也不会参与画风 LoRA 的随机/全部自动选择；"
    "如需在本次图生图中叠加其它 LoRA，请在指令中显式写出该 LoRA 简称。"
)
# 旧版本曾把作者本机的绝对路径写成配置默认值。下面这组「指纹」只在启动时用于
# 识别并清空历史遗留的本机路径，让插件改用自己的 workflows 目录；它们不会被写入
# _conf_schema.json，也不会出现在任何用户可见的文案里。
LEGACY_AUTHOR_PATH_FINGERPRINTS = {
    "flux2_img2img": r"E:\0000001\工作流们（7个）\▶flux-2-klein-单-多图编辑.json",
    "flux2_img2img_legacy": (r"E:\▶flux-2-klein-单-多图编辑.json",),
    "wash": r"E:\0000001\工作流们（7个）\▶flux-2-klein-图生图洗图流.json",
    "outpaint": r"E:\0000001\工作流们（7个）\▶flux-2-klein-图像扩展流.json",
    "multi_angle": r"E:\0000001\工作流们（7个）\▶flux-2-klein-多角度转换流.json",
    "qwen_img2img": r"E:\QwenImageEdit2511局部重绘替换万物 (1).json",
    "qwen_img2img_legacy": (
        r"E:\112121121.json",
        r"E:\222.json",
        r"E:\11111图生图.json",
        r"E:\QwenImageEdit2511局部重绘替换万物.json",
        r"E:\▶QwenImageEdit2511-AIO图生图10G.json",
        r"E:\No2图生图模型\8G-10G显存下载\工作流\▶QwenImageEdit2511-AIO图生图10G.json",
    ),
}
FLUX2_IMG2IMG_SOURCE_DEFAULT = LEGACY_AUTHOR_PATH_FINGERPRINTS["flux2_img2img"]
FLUX2_IMG2IMG_SOURCE_LEGACY = LEGACY_AUTHOR_PATH_FINGERPRINTS["flux2_img2img_legacy"]
FLUX2_IMG2IMG_WORKFLOW_FILE = "图生图_flux2_klein.json"
MODE_FILES = {
    "txt2img": "文生图.json",
    "img2img": "图生图_qwen_edit.json",
    "img2img_flux2": FLUX2_IMG2IMG_WORKFLOW_FILE,
    "hires": "高清放大.json",
    "wash": "洗图.json",
    "outpaint": "扩图.json",
    "multi_angle": "多角度.json",
}
FLUX2_TOOL_SOURCE_DEFAULTS = {
    mode: LEGACY_AUTHOR_PATH_FINGERPRINTS[mode]
    for mode in ("wash", "outpaint", "multi_angle")
}
# 这些名字曾随插件附带，但已经有更好的替代，且不在 MODE_FILES 里。文件已从包内
# 删除；配置若仍指向它们，启动时统一改回该模式的默认工作流。
REMOVED_WORKFLOW_FILES = {
    "图生图.json",
    "图生图_flux2.json",
    "img2img_qwen.json",
    "img2img_flux2.json",
}
FLUX2_TOOL_DEFAULT_POSITIVE = {
    "wash": "皮肤白皙，柔和明亮的光线，",
    "outpaint": "移除绿色区域",
    "multi_angle": "姿势保持不变",
}
QWEN_IMG2IMG_SOURCE_DEFAULT = LEGACY_AUTHOR_PATH_FINGERPRINTS["qwen_img2img"]
FLUX2_IMG2IMG_UNET_DEFAULT = r"flux-2-klein\flux-2-klein-9b-fp8.safetensors"
FLUX2_IMG2IMG_CLIP_DEFAULT = "qwen_3_8b_fp8mixed.safetensors"
FLUX2_IMG2IMG_VAE_DEFAULT = "flux2-vae.safetensors"
# The source graph contains an example NSFW LoRA, but it must not be silently
# applied to every reference edit. Users can still select a content LoRA in
# the independent Flux2 WebUI section.
FLUX2_IMG2IMG_LORA_DEFAULT = ""
FLUX2_IMG2IMG_LEGACY_LORA_DEFAULT = r"flux-2-klein\flux-2-klein-NSFW.safetensors"
FLUX2_IMG2IMG_SIZE_DEFAULT = 1280
FLUX2_IMG2IMG_STEPS_DEFAULT = 6
FLUX2_IMG2IMG_CFG_DEFAULT = 1.0
FLUX2_IMG2IMG_DENOISE_DEFAULT = 1.0
FLUX2_IMG2IMG_DEFAULT_POSITIVE = ""
FLUX2_IMG2IMG_DEFAULT_NEGATIVE = ""
QWEN_IMG2IMG_SOURCE_LEGACY = LEGACY_AUTHOR_PATH_FINGERPRINTS["qwen_img2img_legacy"]
QWEN_IMG2IMG_UNET_DEFAULT = "Qwen-Rapid-NSFW-v23_Q3_K.gguf"
QWEN_IMG2IMG_CLIP_DEFAULT = "Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf"
QWEN_IMG2IMG_VAE_DEFAULT = "qwen_image_vae.safetensors"
QWEN_IMG2IMG_LORA_DEFAULT = ""
QWEN_IMG2IMG_STEPS_DEFAULT = 4
QWEN_IMG2IMG_CFG_DEFAULT = 1.0
QWEN_IMG2IMG_DENOISE_DEFAULT = 1.0
# 图片字体候选：先读 WebUI 配置，再回退到系统常见中文字体。覆盖 Windows、
# Linux 和 macOS，避免把某一台机器的字体路径写死在代码里。
IMAGE_FONT_SYSTEM_REGULAR = (
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)
IMAGE_FONT_SYSTEM_BOLD = (
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)


def image_font_candidates(regular: str = "", bold: str = "") -> tuple[list[str], list[str]]:
    """返回 (常规候选, 粗体候选)；顺序为「WebUI 配置 -> 系统常见字体」。"""

    def _dedupe(values: "list[str]") -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    regular_paths = _dedupe([regular, *IMAGE_FONT_SYSTEM_REGULAR])
    bold_paths = _dedupe([bold, *IMAGE_FONT_SYSTEM_BOLD, *IMAGE_FONT_SYSTEM_REGULAR])
    return regular_paths, bold_paths
LLM_TOOL_NAMES = {
    "status": "comfyui_ai_studio_status",
    "generate": "comfyui_ai_studio_generate",
    "edit": "comfyui_ai_studio_edit",
    "edit_flux2": "comfyui_ai_studio_edit_flux2",
    "upscale": "comfyui_ai_studio_upscale",
    "presets": "comfyui_ai_studio_presets",
}
PATH_SETTING_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "ComfyUI 核心",
        "note": "留空时自动探测。根目录是默认模型位置的基准；配置文件指向 ComfyUI 的 extra_model_paths.yaml。",
        "entries": (
            {"id": "comfyui_root", "label": "ComfyUI 根目录", "kind": "dir", "config_key": "comfyui_root"},
            {"id": "extra_model_paths_file", "label": "额外模型路径配置文件", "kind": "file", "config_key": "extra_model_paths_file"},
        ),
    },
    {
        "group": "模型目录",
        "note": "留空时按「手动覆盖 → extra_model_paths.yaml → 默认位置」解析。填写后该类别优先用你指定的目录，导入/删除也落在那里。",
        "entries": tuple(
            {
                "id": f"model:{category}",
                "label": MODEL_CATEGORY_LABELS.get(category, category),
                "kind": "model",
                "category": category,
                "suffixes": MODEL_CATEGORY_SUFFIXES.get(category, ()),
            }
            for category in MODEL_CATEGORIES
        ),
    },
    {
        "group": "工作流",
        "note": "指向你自己维护的工作流 JSON；留空时使用插件内置副本，不会覆盖你的原文件。",
        "entries": (
            {"id": "workflow_dir", "label": "可切换 API 工作流目录", "kind": "dir", "config_key": "workflow_dir"},
            {"id": "source_workflow", "label": "原始工作流文件", "kind": "file", "config_key": "source_workflow"},
        ),
    },
    {
        "group": "字体与模板",
        "note": "留空时自动探测系统字体、使用插件内置提示词模板。",
        "entries": (
            {"id": "image_font_regular", "label": "常规字体文件", "kind": "file", "config_key": "image_font_regular"},
            {"id": "image_font_bold", "label": "粗体字体文件", "kind": "file", "config_key": "image_font_bold"},
            {"id": "anima_template_path", "label": "Anima 提示词模板", "kind": "file", "config_key": "anima_template_path"},
        ),
    },
    {
        "group": "脚本",
        "note": "留空时在 ComfyUI 根目录及上一层寻找 run_nvidia_gpu.bat / run.bat。",
        "entries": (
            {"id": "comfyui_start_script", "label": "ComfyUI 启动脚本", "kind": "file", "config_key": "comfyui_start_script"},
        ),
    },
    {
        "group": "运行时目录（只读）",
        "note": "由插件根据数据目录派生，不支持在此修改。",
        "entries": (
            {"id": "output_dir", "label": "图片输出目录", "kind": "dir", "readonly": True},
            {"id": "data_dir", "label": "插件数据目录", "kind": "dir", "readonly": True},
        ),
    },
)

WRITABLE_CONFIG = {
    "comfyui_url", "comfyui_root", "source_workflow", "workflow_dir", "model_name",
    "workflow_txt2img", "workflow_img2img", "workflow_img2img_flux2", "workflow_hires",
    "img2img_engine", "img2img_flux2_source_workflow", "img2img_flux2_unet_name",
    "img2img_flux2_clip_name", "img2img_flux2_vae_name", "img2img_flux2_lora_name",
    "img2img_flux2_lora_strength", "img2img_flux2_default_positive",
    "img2img_flux2_default_negative", "img2img_flux2_prompt_template",
    "img2img_flux2_llm_prompt_source", "img2img_flux2_ai_system_prompt",
    "img2img_flux2_plugin_ai_knowledge", "img2img_flux2_plugin_ai_debug",
    "img2img_flux2_output_format",
    "img2img_flux2_plain_translate_enabled", "img2img_flux2_plain_translate_url",
    "img2img_flux2_size", "img2img_flux2_steps", "img2img_flux2_cfg",
    "img2img_flux2_seed", "img2img_flux2_sampler_name", "img2img_flux2_scheduler",
    "img2img_flux2_denoise", "img2img_flux2_batch", "img2img_flux2_filename_prefix",
    "workflow_wash", "workflow_outpaint", "workflow_multi_angle",
    "wash_source_workflow", "outpaint_source_workflow", "multi_angle_source_workflow",
    "wash_default_positive", "wash_default_negative", "wash_model_name", "wash_clip_name", "wash_vae_name",
    "wash_steps", "wash_cfg", "wash_seed", "wash_denoise", "wash_sampler_name", "wash_scheduler",
    "wash_size", "wash_batch", "wash_caption_tokens", "wash_filename_prefix",
    "outpaint_default_positive", "outpaint_default_negative", "outpaint_model_name", "outpaint_clip_name", "outpaint_vae_name",
    "outpaint_steps", "outpaint_cfg", "outpaint_seed", "outpaint_denoise", "outpaint_sampler_name", "outpaint_scheduler",
    "outpaint_size", "outpaint_batch", "outpaint_left", "outpaint_top", "outpaint_right", "outpaint_bottom",
    "outpaint_feathering", "outpaint_filename_prefix",
    "multi_angle_default_positive", "multi_angle_default_negative", "multi_angle_model_name", "multi_angle_clip_name", "multi_angle_vae_name",
    "multi_angle_steps", "multi_angle_cfg", "multi_angle_seed", "multi_angle_denoise", "multi_angle_sampler_name", "multi_angle_scheduler",
    "multi_angle_size", "multi_angle_batch", "multi_angle_horizontal_angle", "multi_angle_vertical_angle", "multi_angle_zoom",
    "multi_angle_default_prompts", "multi_angle_camera_view", "multi_angle_filename_prefix",
    "img2img_source_workflow", "img2img_unet_name", "img2img_clip_name", "img2img_vae_name",
    "img2img_lora_name", "img2img_lora_strength", "img2img_default_positive", "img2img_default_negative",
    "img2img_width", "img2img_height", "img2img_keep_aspect_ratio", "img2img_steps", "img2img_cfg", "img2img_seed",
    "img2img_sampler_name", "img2img_scheduler", "img2img_denoise", "img2img_scale_method",
    "img2img_largest_size", "img2img_crop", "img2img_sampling_shift",
    "img2img_filename_prefix",
    "lora_list", "default_positive", "default_negative", "quality_prefix", "artist_preset", "ai_base_url",
    "style_lora_mode", "style_lora_random_count", "style_lora_list", "style_lora_aliases", "style_lora_weights",
    "ai_api_key", "ai_model", "civitai_base_url", "civitai_token", "civitai_download_mode",
    "width", "height", "steps", "cfg", "seed",
    "sampler_name", "scheduler", "denoise", "hires_scale",
    "hires_steps", "hires_denoise", "hires_upscale_model", "max_concurrent",
    "draw_start_reply", "draw_reply_mode", "draw_reply_custom", "draw_delivery_mode",
    "draw_attach_prompt", "nsfw_group_blacklist",
    "llm_draw_start_reply_mode", "draw_queue_notice_enabled", "draw_queue_notice_ai",
    "draw_queue_limit_enabled", "draw_queue_limit_count",
    "plain_translate_enabled", "plain_translate_url",
    "img2img_plain_translate_enabled", "img2img_plain_translate_url", "img2img_match_input_size",
    "llm_prompt_source", "plugin_ai_command_system_prompt", "plugin_ai_llm_system_prompt",
    "plugin_ai_anima_context", "plugin_ai_debug", "llm_wait_timeout",
    "img2img_llm_prompt_source", "img2img_ai_system_prompt", "img2img_plugin_ai_llm_system_prompt",
    "img2img_plugin_ai_knowledge", "img2img_plugin_ai_debug",
    "img2img_llm_tool_prompt", "img2img_plugin_ai_user_prompt_template",
    "img2img_plugin_ai_output_format",
    "img2img_astrbot_llm_system_prompt", "img2img_astrbot_user_prompt_template",
    "draw_limit_count", "draw_limit_window_seconds", "draw_limit_admin_ids",
    "comfyui_start_script", "image_font_regular", "image_font_bold", "anima_template_path",
    "extra_model_paths_file", "model_dir_overrides",
    "moderation_enabled", "moderation_input_groups", "moderation_output_groups",
    "moderation_base_url", "moderation_api_key", "moderation_model",
    "moderation_strictness", "moderation_timeout", "moderation_fail_open",
    "moderation_max_side",
}

# 图片安全审核的全部可配置项。新增配置只需改这一处 + _conf_schema.json。
MODERATION_CONFIG_KEYS = (
    "moderation_enabled",
    "moderation_input_groups",
    "moderation_output_groups",
    "moderation_base_url",
    "moderation_api_key",
    "moderation_model",
    "moderation_strictness",
    "moderation_timeout",
    "moderation_fail_open",
    "moderation_max_side",
)

# 检测名单里出现这些值表示“所有会话都检测”。
MODERATION_ALL_MARKERS = {"*", "all", "全部", "所有", "不限"}

MODERATION_CATEGORY_LABELS = {
    "nsfw": "成人内容",
    "guro": "血腥猎奇",
    "nsfw+guro": "成人内容 + 血腥猎奇",
    "error": "审核服务异常",
}

MODERATION_INPUT_WARNING = (
    "【图片安全审核拦截】你发送的图片未通过内容安全审核，已拒绝进入绘图流程。\n"
    "命中类型：{category}\n"
    "判定结果：{detail}\n"
    "请撤回或更换图片后重试。"
)

MODERATION_OUTPUT_WARNING = (
    "【图片安全审核拦截】本次生成结果未通过内容安全审核，已拦截，不对外发送。\n"
    "命中类型：{category}\n"
    "判定结果：{detail}\n"
    "可以调整提示词后重新生成。"
)

NSFW_PROMPT_MARKERS = (
    "色情", "涩图", "成人内容", "露骨", "裸体", "全裸", "裸身", "裸胸", "露点",
    "露乳", "乳头", "阴部", "阴茎", "阴道", "生殖器", "性交", "做爱", "自慰",
    "手淫", "口交", "肛交", "射精", "精液", "淫荡", "无修正",
)
NSFW_ENGLISH_MARKERS = (
    "nsfw", "nude", "naked", "topless", "nipples", "areola", "pussy", "penis",
    "vagina", "sexual", "sexually", "erotic", "porn", "hentai", "blowjob",
    "masturbat", "intercourse", "orgasm", "semen", "penetration", "genital",
    "anal", "uncensored", "lewd",
)
NSFW_GROUP_NEGATIVE = (
    "nsfw", "nude", "naked", "topless", "bottomless", "nipples", "areola",
    "explicit", "sexual", "sexual content", "erotic", "porn", "pornographic content",
    "hentai", "lewd", "uncensored", "genitals", "pussy", "penis", "vagina",
    "intercourse", "blowjob", "masturbation", "orgasm", "semen", "cum",
)

DEFAULT_PLUGIN_AI_LLM_SYSTEM_PROMPT = """你是 ComfyUI Anima 工作流的提示词工程师，负责把用户的完整绘图要求转换成可直接用于 Anima 的英文 Danbooru 标签。
你会同时收到用户原话和 AstrBot LLM 提取的画面描述。必须以用户原话为最高优先级，补回 LLM 遗漏的关键内容。
完整保留并具体表达：角色、主体、动作、姿势、正在进行的行为、表情、服装、镜头、构图、场景、时间、天气和光线。
例如用户说“洗澡的夏空”，必须输出与洗澡动作相关的标签，不能只输出角色名；用户说“坐在浴缸里洗澡”，必须保留坐姿、浴缸和洗澡行为。
不要把“帮我画一张”“请生成图片”等聊天套话写进提示词，也不要臆造用户没有要求的角色、服装或场景。
角色名、作品名优先转换为稳定的 Danbooru 标签；已经存在的英文标签保留且不要重复。
输出顺序遵循：人数与性别、角色与作品、外观、服装与状态、动作与姿势、表情、镜头与构图、场景环境、细节氛围。
输出协议是硬性要求：只输出一行小写英文、逗号分隔的最终提示词；第一个字符必须是标签，最后一个字符也必须是标签。
不要输出分析、思考、推理、分类、草稿、字段名、Markdown、代码块、换行、<think> 或 </think>，不要输出“最终提示词：”等前缀。
如果模型内部需要思考，请把思考留在内部，不要写进回复。用户原话中用于控制预设和 LoRA 的名称由绘图插件单独处理，不要因为翻译而删除或改写它们。"""

DEFAULT_IMG2IMG_PLUGIN_AI_KNOWLEDGE = """这是 Qwen Image Edit 图生图，原图是唯一基础，不是文生图。
只写用户明确要求改变的内容，使用一条简短、具体的中文编辑句子；不要重新描述整张原图。
主要处理换衣服和换面部表情，也要处理用户明确提出的动作、姿势、物体或局部变化。
“换个姿势”“改一下动作”不能原样照抄，必须根据用户要求或上下文写成可执行的具体动作；但不能凭空添加用户没有要求的服装、场景或人物设定。
“一字马”“劈叉”表示人物完成 split/劈叉动作，不是 horse（马）；“侧劈叉”“横劈叉”“前劈叉”要保留方向。
未要求改变的角色身份、脸、发型、身体比例、背景、镜头、构图、光线和画风都保持原图不变。
不要输出原图摘要、质量词、LoRA、预设、权重、Denoise、反向提示词、解释或 Markdown。"""

DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT = """你是 Qwen Image Edit 的专用图生图编辑器。
你的任务不是翻译整张图，也不是生成文生图提示词，而是把用户的修改要求整理成一条简短、具体、可执行的中文编辑指令。

严格规则：
1. 只处理用户明确要改变的内容，不要复述人物、背景、镜头、构图或画风。
2. 换衣服要写清目标服装；换表情要写清目标表情；动作或姿势要写清人物如何做，不能只写“改姿势”或“change her pose”。
3. “一字马/劈叉”必须理解为腿部伸展的劈叉动作，写成“完成一字马（劈叉）”，绝不能按字面理解成站在马身上。
4. 未要求改变的内容保持原图不变。不要添加角色、服装、场景或细节。
5. 最终只输出一行中文短句，不要输出 EDIT、英文翻译、分析、解释、Markdown、质量词、LoRA、预设、权重或 Denoise。

示例：
用户要求“把她的女仆装换成白色连衣裙”
把她的衣服换成白色连衣裙，其他内容保持原图不变

用户要求“让她站立一字马”
让人物站立完成一字马（劈叉）动作，其他内容保持原图不变

用户要求“让她露出温柔的微笑”
让她露出温柔的微笑，其他内容保持原图不变"""

DEFAULT_IMG2IMG_ASTRBOT_LLM_SYSTEM_PROMPT = DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT
DEFAULT_IMG2IMG_LLM_TOOL_PROMPT = """当前请求是图生图编辑。用户提供图片并要求换衣服或改变面部表情时，必须调用 comfyui_ai_studio_edit，不能调用文生图工具，也不能只文字回复。
图生图以原图为基础，只修改用户明确要求的内容；不要把完整原图描述塞进 prompt。prompt 写具体的修改目标。
预设名和 LoRA 指令简称分别填写 preset 和 lora，二者可以同时填写。"""
DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE = """用户的图生图编辑要求：
{request}

只写用户明确想改变的内容，输出一行具体、简短的中文编辑句。
不要重新描述原图，不要翻译成英文，不要补充用户没有要求的服装、场景或人物设定。
“一字马/劈叉”必须理解为腿部伸展的劈叉动作，不是站在马身上。
未要求改变的内容保持原图不变。"""
DEFAULT_IMG2IMG_ASTRBOT_USER_PROMPT_TEMPLATE = DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE
DEFAULT_IMG2IMG_OUTPUT_FORMAT = """最终只输出一行中文短句。
只写用户明确想改变的内容，并说明未要求改变的内容保持原图不变。
“一字马/劈叉”必须写成“完成一字马（劈叉）动作”，不得翻译成站在马身上。
禁止输出英文翻译、分析、解释、Markdown、原图摘要、质量词、角色标签、LoRA、预设、权重、Denoise 或其它字段。
禁止只输出“改姿势”“改衣服”“改表情”或 change her pose 等没有具体目标的空泛短语。"""
DEFAULT_DRAW_REPLY_SYSTEM_PROMPT = """你是绘图完成后的聊天回复助手。绘图任务已经完成，必须基于下面给出的事实回复，不能重新判断任务状态。

硬性事实：
- 绘图模式和实际图片数量必须完全照抄用户消息中提供的字段。
- 只能说已经完成，不能说正在生成、等待生成、生成失败或没有图片。
- 用户原始需求只用于自然地提及画面内容；不确定的细节不要补充，不要编造角色、动作、场景或参数。
- 这是完成后的聊天回复，不是提示词，不要输出英文标签、正面提示词、负面提示词、系统说明或分析过程。

只输出一到两句自然中文，语气符合当前 AstrBot 人格；回复中必须明确包含“已完成”和实际图片数量。"""
BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxIAAAsSAdLdfvwAAACaSURBVHhe5cgxDQAACMAw/JsGBMxBl/TZfIvLKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpySnJKckpyYmYPVh88OKn4LsIAAAAAElFTkSuQmCC"
)


def _blank_png(width: int, height: int) -> bytes:
    """生成指定尺寸的黑色 RGB PNG，作为原工作流的文生图初始图。"""
    width = max(64, int(width))
    height = max(64, int(height))
    row = b"\x00" + b"\x00\x00\x00" * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class CivitAIDownloadResolver:
    """从 CivitAI 页面的内嵌 JSON 中递归寻找模型下载地址。

    该解析器完全离线工作，只对已抓取到的 JSON 结构做遍历，不发起任何
    网络请求，也不读取 API Key，因此可以安全地用在「免 API 下载」流程中。
    """

    # 出现在文件名/下载地址字段里的候选键名。
    _DOWNLOAD_KEYS = ("downloadUrl", "download_url", "downloadURL", "url")
    _NAME_KEYS = ("name", "fileName", "file_name", "filename")

    @classmethod
    def find_download(cls, data: Any) -> tuple[str, str]:
        """返回 ``(下载地址, 文件名)``；找不到时返回空字符串。"""
        return cls._find_file_entry(data)

    @classmethod
    def _find_file_entry(cls, node: Any) -> tuple[str, str]:
        if isinstance(node, list):
            for item in node:
                url, name = cls._find_file_entry(item)
                if url:
                    return url, name
            return "", ""
        if not isinstance(node, dict):
            return "", ""

        for key in cls._DOWNLOAD_KEYS:
            value = node.get(key)
            if isinstance(value, str) and "/download/models/" in value:
                file_name = ""
                for name_key in cls._NAME_KEYS:
                    candidate = node.get(name_key)
                    if isinstance(candidate, str) and candidate.strip():
                        file_name = candidate.strip()
                        break
                return value, file_name

        # 深度优先遍历其余字段，兼容不同站点/版本的 JSON 结构。
        for key, value in node.items():
            if key in cls._DOWNLOAD_KEYS:
                continue
            url, name = cls._find_file_entry(value)
            if url:
                return url, name
        return "", ""


@register(
    PLUGIN_NAME,
    "fornever123",
    "全新独立的 ComfyUI 中文绘画插件，基于原始难工作流",
    PLUGIN_VERSION,
)
class ComfyUIAIStudio(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self._migrate_img2img_ai_config()
        self.plugin_dir = Path(__file__).resolve().parent
        self.data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # AstrBot 的 data/temp 图片会被后台清理。图生图任务进入队列前，
        # 必须把输入图片复制到插件自己的数据目录，避免后台任务拿到失效路径。
        self.input_cache_dir = self.data_dir / "input_cache"
        self.input_cache_dir.mkdir(parents=True, exist_ok=True)
        # 合并转发节点会被 AstrBot 编码成 base64。大图直接放进 OneBot
        # 请求容易超过接口限制，因此只为转发创建临时压缩副本。
        self.forward_cache_dir = self.data_dir / "forward_cache"
        self.forward_cache_dir.mkdir(parents=True, exist_ok=True)
        self.presets = PresetStore(self.data_dir / "presets.json")
        self.artist_presets = PresetStore(self.data_dir / "artist_presets.json")
        if not self.artist_presets.path.exists():
            self.artist_presets.add(DEFAULT_ARTIST_PRESET_NAME, DEFAULT_ARTIST_PRESET_TAGS)
        self.lora_aliases_path = self.data_dir / "lora_aliases.json"
        self.lora_command_aliases_path = self.data_dir / "lora_command_aliases.json"
        self.lora_categories_path = self.data_dir / "lora_categories.json"
        self.lora_category_entries_path = self.data_dir / "lora_category_entries.json"
        self.civitai_cache_path = self.data_dir / "civitai_lora_cache.json"
        self.civitai_links_path = self.data_dir / "civitai_lora_links.json"
        self.civitai_overrides_path = self.data_dir / "civitai_lora_overrides.json"
        self.lora_download_order_path = self.data_dir / "lora_download_order.json"
        # LoRA 独立预设与全局 presets.json 分开保存；旧版自动条目只做一次迁移，
        # 用户手动预设不覆盖、不删除。
        self.lora_presets_path = self.data_dir / "lora_presets.json"
        self.lora_aliases = self._load_map(self.lora_aliases_path)
        self.lora_command_aliases = self._load_map(self.lora_command_aliases_path)
        self.lora_categories = self._load_map(self.lora_categories_path)
        self.lora_category_entries = self._load_map(self.lora_category_entries_path)
        self.civitai_cache = self._load_map(self.civitai_cache_path)
        self.civitai_links = self._load_map(self.civitai_links_path)
        self.civitai_overrides = self._load_map(self.civitai_overrides_path)
        self.lora_presets = self._load_map(self.lora_presets_path)
        self.lora_download_order = self._load_map(self.lora_download_order_path)
        self._migrate_legacy_preset_data()
        self._ensure_visible_character_presets()
        # 预设和指令简称统一到 LoRA 专属预设；旧版 CivitAI 自动片段仍保留
        # 在数据文件中用于兼容/回退，但由可见内容读取器过滤，不再参与绘图。
        self._remove_legacy_civitai_preset_fragments()
        self._sync_all_preset_links()
        # 将自动画风简称固化到配置，避免新增文件或下载排序改变旧的“画风数字”。
        self._ensure_style_lora_aliases()
        # WebUI 返回的简称列表包含“专属预设左侧名称”这一组自动简称。
        # 启动时把它们从手动文件中剥离，防止改名/删除专属预设后旧简称残留。
        if self._strip_lora_preset_aliases_from_manual():
            self._save_lora_command_aliases()
            self._save_lora_presets()
        self._civitai_cache_lock = asyncio.Lock()
        self.last_images: dict[str, str] = {}
        self.semaphore: asyncio.Semaphore | None = None
        self._draw_limit_lock = asyncio.Lock()
        self._draw_limit_entries: dict[str, list[tuple[str, float, int]]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._download_jobs: dict[str, dict[str, Any]] = {}
        self._lora_names_cache: tuple[float, list[str]] | None = None
        # 同一张附件可能同时出现在 AstrBot 结构化消息链、原始 OneBot
        # 消息段和引用解析结果中。缓存原始引用及内容指纹，避免一次输入
        # 被复制成多个不同的 input_cache 文件后误判为多张参考图。
        self._image_ref_cache: dict[str, str] = {}
        # 读取 LoRA safetensors 头部得到的架构信息。只缓存文件名对应的结果，
        # 不读取权重正文；这样每次绘图不会因为检查兼容性而重复打开大文件。
        self._lora_profile_cache: dict[str, str] = {}
        # 图片安全审核客户端按配置指纹复用。审核模型地址/密钥/尺度改动后
        # 指纹变化会自动重建，这样 WebUI 保存后无需重启插件即可生效。
        self._moderator_instance: ImageModerator | None = None
        self._moderator_signature: tuple[Any, ...] | None = None
        self._register_web_api()

    def _migrate_img2img_ai_config(self) -> None:
        """清理旧版图生图提示词模板，避免旧配置覆盖统一编辑规则。"""
        current = str(self._get("img2img_ai_system_prompt", "") or "").strip()
        old_system = str(self._get("img2img_plugin_ai_llm_system_prompt", "") or "").strip()
        old_tool = str(self._get("img2img_llm_tool_prompt", "") or "").strip()
        current_knowledge = str(self._get("img2img_plugin_ai_knowledge", "") or "").strip()
        old_user_templates = (
            "img2img_plugin_ai_user_prompt_template",
            "img2img_astrbot_user_prompt_template",
        )
        old_output_format = str(self._get("img2img_plugin_ai_output_format", "") or "").strip()
        legacy_markers = (
            "正向提示词", "反向提示词", "建议 denoise", "建议 Denoise",
            "权重说明", "现在是2380年", "输出以下四项",
            "英文编辑指令", "英文提示词", "English editing instruction",
        )
        changed = False
        # 旧版配置即使已经保存过，也会继续覆盖新的内置编辑规则。只要
        # 检测到旧版“正反向提示词/原图描述/Denoise 建议”模板，就迁移到
        # 当前专门处理换衣服和换表情的规则；用户以后仍可在 WebUI 修改。
        if not current or any(marker in current for marker in legacy_markers):
            # 当前统一提示词为空时使用代码内置中文规则；旧版插件 AI
            # 系统提示词不能再回填，否则会把旧英文/完整描述规则带回来。
            self._set("img2img_ai_system_prompt", "")
            changed = True
        if old_system:
            self._set("img2img_plugin_ai_llm_system_prompt", "")
            changed = True
        for key in old_user_templates:
            value = str(self._get(key, "") or "").strip()
            if value:
                self._set(key, "")
                changed = True
        if old_output_format:
            self._set("img2img_plugin_ai_output_format", "")
            changed = True
        if old_tool and (
            "禁止补充" in old_tool
            or "只填写用户明确想改动" in old_tool
            or "只填写“换成裙子”" in old_tool
        ):
            self._set("img2img_llm_tool_prompt", "")
            changed = True
        if current_knowledge and (
            "换物体" in current_knowledge
            or "换背景" in current_knowledge
            or "不要改变和生成任何有关" in current_knowledge
        ) and "主要编辑范围只有两类" not in current_knowledge:
            self._set("img2img_plugin_ai_knowledge", "")
            changed = True
        # The rollback baseline uses denoise=1.0. Do not rewrite it to the
        # former gentle-edit default; the value remains user-configurable.
        try:
            configured_denoise = float(self._get("img2img_denoise", DEFAULT_IMG2IMG_DENOISE))
        except (TypeError, ValueError):
            configured_denoise = DEFAULT_IMG2IMG_DENOISE
        # Keep an explicitly configured value, including the baseline 1.0.
        if changed:
            self._save_config()

    def _ensure_visible_character_presets(self) -> None:
        """只补充缺失的 WebUI 可见预设，绝不改动用户已有条目。

        角色名单只是首次安装时的预设种子，不是运行时知识库，也不参与
        隐式提示词注入。用户在 WebUI 中保存过的内容始终拥有最高优先级。
        """
        changed = False
        for name, content in CHARACTER_PRESET_SEED.items():
            if name in self.presets.items:
                # 已有条目可能是用户修改过的，不能按角色表覆盖。
                continue
            self.presets.items[name] = {"content": content, "translated": ""}
            changed = True
        if changed:
            self.presets.save()

    def _migrate_legacy_preset_data(self) -> None:
        """把旧版 CivitAI 自动全局预设迁移到对应 LoRA，避免污染全局预设。

        旧版把 LoRA 的自动触发词写到了 presets.json。迁移只处理带有明确
        _auto_source 标记的条目；用户手动编辑过的条目没有该标记，因此不会
        被删除或覆盖。
        """
        changed_global = False
        changed_lora = False
        for name, item in list(self.presets.items.items()):
            if not isinstance(item, dict) or item.get("_auto_source") != "civitai_lora":
                continue
            source_key = str(item.get("_auto_source_key", "") or "").strip()
            content = str(item.get("content", "") or "").strip()
            if not source_key or not content:
                continue
            entries = self._normalize_lora_preset_entries(self.lora_presets.get(source_key, []))
            aliases = self._lora_command_aliases(source_key)
            if not aliases:
                aliases = [name]
            for alias in aliases:
                if any(
                    entry["tag"].casefold() == alias.casefold()
                    and entry["content"].casefold() == content.casefold()
                    for entry in entries
                ):
                    continue
                entries.append({"tag": alias, "content": content})
                changed_lora = True
            self.lora_presets[source_key] = entries
            # 该条目已经完整保存在对应 LoRA 下，不再让它出现在全局预设。
            del self.presets.items[name]
            changed_global = True

        if changed_lora:
            self._save_lora_presets()
        if changed_global:
            self.presets.save()

    def _remove_legacy_civitai_preset_fragments(self) -> bool:
        """移除旧版自动 CivitAI 内容，保留用户和全局预设内容。

        0.7.5 及更早版本可能把 CivitAI trainedWords 写入了 LoRA 专属
        预设的 ``content``，并用 ``_civitai_content`` 标记来源。现在这些
        词只在 LoRA 页面展示，因此升级时做一次精确清理，避免旧词继续
        充当简称或在后续同步时重新进入正面提示词。
        """
        changed = False
        for file_name in list(getattr(self, "lora_presets", {})):
            entries = self._lora_preset_entries_raw(file_name)
            next_entries: list[dict[str, Any]] = []
            file_changed = False
            for entry in entries:
                civitai = self._normalize_trigger_words(entry.get("_civitai_content", []))
                if not civitai:
                    next_entries.append(entry)
                    continue
                civitai_keys = {
                    item.casefold()
                    for raw in civitai
                    for item in self._prompt_fragments(raw)
                }
                remaining = [
                    item
                    for item in self._prompt_fragments(entry.get("content", ""))
                    if item.casefold() not in civitai_keys
                ]
                global_content = str(entry.get("_global_content", "") or "").strip()
                entry.pop("_civitai_content", None)
                entry["content"] = self._merge_prompt_fragments(remaining, global_content)
                file_changed = True
                # 纯自动条目清掉；同一个条目中有用户/全局内容则保留。
                if entry["content"] or not global_content and remaining:
                    next_entries.append(entry)
            if file_changed:
                if next_entries:
                    self.lora_presets[file_name] = next_entries
                else:
                    self.lora_presets.pop(file_name, None)
                changed = True
        if changed:
            self._save_lora_presets()
        return changed

    @filter.on_llm_request(priority=70)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """按文生图或图生图模式注入对应的工具调用规则。"""
        message = str(getattr(event, "message_str", "") or "").strip()
        feature_query = bool(re.search(
            r"(?:你|机器人|插件|这个插件).{0,12}(?:有什么(?:画图|绘图)?功能|能(?:够)?做什么|会(?:画|做)什么|支持(?:哪些)?(?:画图|绘图)?功能)|"
            r"(?:介绍|介绍一下|说明一下|列出).{0,12}(?:功能|用法)|怎么使用(?:这个)?(?:插件|绘图)|"
            r"(?:有哪些|能做哪些|可以做哪些)(?:画图|绘图)?功能|(?:画图|绘图)功能(?:有哪些|是什么)|"
            r"(?:功能|用法)是什么",
            message,
            flags=re.IGNORECASE,
        ))
        preset_query = bool(re.search(
            r"(?:有哪些|查看|查询|列出|显示).{0,8}(?:提示词)?预设|"
            r"预设(?:列表|有哪些|查询)|(?:鸣潮角色|画风).{0,12}预设",
            message,
            flags=re.IGNORECASE,
        ))
        if feature_query:
            current = str(getattr(req, "system_prompt", "") or "").strip()
            # 功能查询只注入一句话规则，避免人格模型把插件内部配置逐项展开。
            injection = (
                "【ComfyUI 绘图功能】\n"
                "用户询问本插件能做什么时，只回复这一句，不要改写、补充或逐项解释：支持文生图、图生图、高清放大、洗图、扩图、多角度处理，"
                "以及 LoRA 批量导入、免 API 下载 C 站模型和提示词预设管理。"
                "不要展开参数，不要调用绘图工具，也不要凭空开始生成图片。"
            )
            req.system_prompt = f"{injection}\n\n{current}".strip()
            return
        if preset_query:
            current = str(getattr(req, "system_prompt", "") or "").strip()
            injection = (
                "【提示词预设查询规则】\n"
                "用户是在查询本插件的提示词预设，不是在请求绘图。"
                "请调用 comfyui_ai_studio_presets 工具查询预设；如果用户提到分类，"
                "把分类名称填入 category，例如“鸣潮角色”。工具返回中文结果后直接整理回答，"
                "不要调用文生图或图生图工具。"
            )
            req.system_prompt = f"{injection}\n\n{current}".strip()
            return
        image_hint = self._event_has_image_hint(event)
        is_edit = image_hint or self._looks_like_edit_request(event, message)
        if not is_drawing_request(message) and not is_edit:
            return
        try:
            if image_hint:
                # AstrBot 核心会处理普通 Reply 图片，但某些 QQ 适配器只
                # 给主 LLM 留下 Forward(id) 占位符。这里把同一条引用中
                # 的实际图片加入 req，确保 LLM 看到的就是用户 @/回复的图。
                await self._inject_event_images_to_llm_request(event, req)
            if is_edit:
                instruction = DEFAULT_IMG2IMG_LLM_TOOL_PROMPT
                flux_hint = (
                    "当前 WebUI 若选择 Flux2，优先调用 comfyui_ai_studio_edit_flux2；"
                    "它使用当前消息中的 1 张参考图片，并使用独立的 Flux2 配置。"
                )
                req.system_prompt = f"【图生图编辑规则】\n{instruction}\n{flux_hint}\n\n{str(req.system_prompt or '').strip()}".strip()
                return
            context = str(self._get("plugin_ai_anima_context", "") or "").strip()
            if not context:
                context = build_anima_context(
                    self.plugin_dir, 2600, self._anima_template_override()
                )
            instruction = (
                "当前请求涉及绘图。请把自己当作 Anima3 提示词工程师：先理解用户真正想画什么，"
                "再按 Anima 规则生成提示词；调用本插件的绘图工具时，prompt 必须是适配 Anima 工作流的"
                "一行小写英文 Danbooru 标签。必须保留用户原话中的动作、姿势、正在进行的行为、表情、"
                "镜头、构图和场景。调用 "
                "comfyui_ai_studio_generate、comfyui_ai_studio_edit 或 "
                "comfyui_ai_studio_upscale 时，检查用户原话里的提示词预设名和 LoRA 指令简称，分别填入 "
                "preset 和 lora；两者可以同时填写，多个值用逗号分隔。"
                + self._llm_style_lora_hint()
                + "不要把明确的绘图请求改成只查询规则。"
            )
            injection = f"【内置 Anima 提示词工程师】\n{instruction}\n\n{context}"
            req.system_prompt = f"{injection}\n\n{str(req.system_prompt or '').strip()}".strip()
        except Exception as exc:
            logger.warning("[%s] 内置 Anima 规则注入失败：%s", PLUGIN_NAME, exc)

    def _register_web_api(self) -> None:
        if not hasattr(self.context, "register_web_api"):
            return
        routes = [
            ("upload_workflow", self.api_upload_workflow, ["POST"], "Upload API workflow JSON"),
            ("status", self.api_status, ["GET"], "读取 ComfyUI 状态和本地路径"),
            ("models", self.api_models, ["GET"], "读取核心模型、LoRA 和放大模型"),
            ("workflows", self.api_workflows, ["GET"], "读取和切换工作流"),
            ("config", self.api_config, ["GET", "POST"], "读取和保存绘画配置"),
            ("paths", self.api_path_settings, ["GET", "POST"], "读取并修改各类文件夹位置"),
            ("ai_models", self.api_ai_models, ["GET", "POST"], "获取 AI 模型并测试连接"),
            ("moderation_test", self.api_moderation_test, ["GET", "POST"], "测试图片安全审核链路"),
            ("presets", self.api_presets, ["GET", "POST"], "管理提示词预设"),
            ("artist_presets", self.api_artist_presets, ["GET", "POST"], "管理独立画师串预设"),
            ("lora_info", self.api_lora_info, ["GET", "POST"], "读取 LoRA 别名和 CivitAI 信息"),
            ("open_folder", self.api_open_folder, ["POST"], "打开本地模型文件夹"),
            ("open_lora", self.api_open_lora, ["POST"], "打开 LoRA 所在文件夹"),
            ("upload_lora", self.api_upload_lora, ["POST"], "上传 LoRA 文件"),
            ("upload_lora_batch", self.api_upload_lora_batch, ["POST"], "批量导入 LoRA 文件"),
            ("upload_style_lora", self.api_upload_style_lora, ["POST"], "上传并归类画风 LoRA 文件"),
            ("download_lora", self.api_download_lora, ["POST"], "从 CivitAI 下载 LoRA 文件"),
            ("download_lora_progress", self.api_download_lora_progress, ["POST"], "读取 LoRA 下载进度"),
            ("delete_lora", self.api_delete_lora, ["POST"], "删除 LoRA 文件"),
            ("recent", self.api_recent, ["GET"], "读取最近生成图片"),
        ]
        for name, handler, methods, description in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{name}", handler, methods, description
            )

    def _llm_style_lora_hint(self) -> str:
        """生成给 AstrBot LLM 的画风简称提示，不读取 ComfyUI。"""
        raw = self._get("style_lora_aliases", {})
        aliases: list[str] = []
        if isinstance(raw, dict):
            for value in raw.values():
                values = value if isinstance(value, (list, tuple)) else [value]
                for item in values:
                    alias = str(item or "").strip()
                    if alias and alias.casefold() not in {item.casefold() for item in aliases}:
                        aliases.append(alias)
        if not aliases:
            return (
                "如果用户提到画风 LoRA 的简称（例如画风1、画风2），必须把该简称原样填写到 lora 参数，"
                "不要把它写进 prompt，也不要把它改成英文标签。"
            )
        aliases.sort(key=str.casefold)
        return (
            "画风 LoRA 简称已由插件管理，当前可识别简称："
            + "、".join(aliases)
            + "。用户提到其中任何一个时，必须把简称原样填写到 lora 参数，"
            "不要把它写进 prompt，也不要把它改成英文标签。"
        )

    def _refresh_flux2_tool_workflows(self) -> bool:
        """Install the three supplied editor workflows as API workflows.

        The bundled/default files are refreshed from their configured source on
        startup. A user-selected custom file is left untouched, so changing a
        workflow in WebUI remains persistent across plugin restarts.
        """
        changed = False
        directory = self._workflow_dir()
        for mode in FLUX2_TOOL_SOURCE_DEFAULTS:
            source_key = f"{mode}_source_workflow"
            source_value = str(self._get(source_key, "") or "").strip()
            if source_value and source_value.casefold() == str(
                FLUX2_TOOL_SOURCE_DEFAULTS.get(mode, "")
            ).casefold():
                # 旧版本把作者本机路径写成了默认值，在别的机器上并不存在；
                # 清空后由插件内置的 workflows 目录接管。
                source_value = ""
                self._set(source_key, "")
                changed = True
            if not source_value:
                continue
            selected_key = f"workflow_{mode}"
            selected = str(self._get(selected_key, "") or "").strip()
            if not selected:
                self._set(selected_key, MODE_FILES[mode])
                selected = MODE_FILES[mode]
                changed = True
            selected_path = Path(selected)
            is_default_selection = selected.casefold() == MODE_FILES[mode].casefold()
            target = directory / MODE_FILES[mode]
            if not is_default_selection:
                continue
            source = Path(source_value)
            if not source.is_file():
                logger.warning("[%s] %s 源工作流不存在：%s", PLUGIN_NAME, MODE_NAMES[mode], _log_path(source))
                continue
            try:
                converted = load_workflow(source)
                target.write_text(
                    json.dumps(converted, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("[%s] 已安装%s工作流：%s -> %s", PLUGIN_NAME, MODE_NAMES[mode], _log_path(source), _log_path(target))
            except (OSError, WorkflowError, ValueError) as exc:
                logger.warning("[%s] %s工作流转换失败：%s", PLUGIN_NAME, MODE_NAMES[mode], exc)
        return changed

    def _refresh_img2img_flux2_workflow(self) -> bool:
        """Install the independent one-to-three-reference Flux2 Klein graph."""
        changed = False
        source_value = str(self._get("img2img_flux2_source_workflow", "") or "").strip()
        if source_value and source_value.casefold() in {
            FLUX2_IMG2IMG_SOURCE_DEFAULT.casefold(),
            *(item.casefold() for item in FLUX2_IMG2IMG_SOURCE_LEGACY),
        }:
            # 旧默认值是作者本机路径，清空后改用插件内置工作流副本。
            source_value = ""
            self._set("img2img_flux2_source_workflow", "")
            changed = True
        selected = str(
            self._get("workflow_img2img_flux2", FLUX2_IMG2IMG_WORKFLOW_FILE) or ""
        ).strip()
        if not selected or selected.casefold() in REMOVED_WORKFLOW_FILES:
            # 旧版本随插件附带过几个已废弃的 Flux2 图生图工作流；文件已从包内
            # 移除，配置指过去只会让 _workflow_path() 静默回退，不如直接改正。
            selected = FLUX2_IMG2IMG_WORKFLOW_FILE
            self._set("workflow_img2img_flux2", selected)
            changed = True
        if selected.casefold() != FLUX2_IMG2IMG_WORKFLOW_FILE.casefold():
            return changed
        if not source_value:
            # 未配置源工作流：直接使用插件内置副本，不输出警告。
            return changed
        source = Path(source_value)
        if not source.is_file():
            logger.warning("[%s] Flux2 Klein 图生图源工作流不存在：%s", PLUGIN_NAME, _log_path(source))
            return changed
        target = self._workflow_dir() / FLUX2_IMG2IMG_WORKFLOW_FILE
        try:
            # The selected source is the execution graph of record. Keep its
            # nodes and connections intact so the API submission matches the
            # graph used from the ComfyUI page.
            converted = load_workflow(source, preserve_disabled=True)
            target.write_text(
                json.dumps(converted, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("[%s] 已加载独立 Flux2 Klein 图生图工作流：%s", PLUGIN_NAME, _log_path(source))
        except (OSError, WorkflowError, ValueError) as exc:
            logger.warning("[%s] Flux2 Klein 图生图工作流转换失败：%s", PLUGIN_NAME, exc)
        return changed

    async def initialize(self) -> None:
        source = Path(str(self._get("source_workflow", "")))
        if source.is_file():
            try:
                # 原始工作流是唯一基准。每次加载都刷新默认副本，避免插件继续
                # 使用旧副本；用户在 WebUI 选择的自定义文件不会被覆盖。
                copy_and_repair_original(source, self._workflow_dir())
                logger.info("[%s] 已按原始工作流刷新三套默认工作流：%s", PLUGIN_NAME, _log_path(source))
            except Exception as exc:
                logger.warning("[%s] 从原始工作流刷新副本失败：%s", PLUGIN_NAME, exc)
        # 图生图使用独立的 Qwen Image Edit 工作流和模型链。
        # 这里迁移旧版配置，但不修改用户提供的源文件。
        configured_img2img_source = str(self._get("img2img_source_workflow", "") or "").strip()
        legacy_sources = {
            QWEN_IMG2IMG_SOURCE_DEFAULT.casefold(),
            *(item.casefold() for item in QWEN_IMG2IMG_SOURCE_LEGACY),
        }
        changed = False
        if configured_img2img_source and configured_img2img_source.casefold() in legacy_sources:
            # 旧默认值是作者本机路径，清空后改用插件内置工作流副本。
            configured_img2img_source = ""
            self._set("img2img_source_workflow", "")
            changed = True
        img2img_source = Path(configured_img2img_source) if configured_img2img_source else None
        img2img_target = self._workflow_dir() / MODE_FILES["img2img"]
        if img2img_source and img2img_source.is_file():
            try:
                img2img_workflow = load_workflow(img2img_source)
                img2img_target.write_text(
                    json.dumps(img2img_workflow, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("[%s] 已加载独立 Qwen Image Edit 图生图工作流：%s", PLUGIN_NAME, _log_path(img2img_source))
            except Exception as exc:
                logger.warning("[%s] Qwen 图生图工作流转换失败：%s", PLUGIN_NAME, exc)
        current_img2img = str(self._get("workflow_img2img", "") or "").strip()
        if not current_img2img or current_img2img.casefold() in {
            "图生图.json", "图生图_flux2.json", "img2img_qwen.json", "img2img_flux2.json",
        }:
            self._set("workflow_img2img", MODE_FILES["img2img"])
            changed = True
        # 只迁移旧版插件默认值，用户手动选择的 Qwen 模型和 LoRA 保留。
        model_migrations = {
            "img2img_unet_name": (
                "flux-2-klein-9b-kv-int8-convrot.safetensors",
                "flux-2-klein-9b-kv-fp8.safetensors",
                QWEN_IMG2IMG_UNET_DEFAULT,
            ),
            "img2img_clip_name": ("Qwen3-8B-Uncensor-v2.Q4_K_M.gguf", QWEN_IMG2IMG_CLIP_DEFAULT),
            "img2img_vae_name": ("flux2-vae.safetensors", QWEN_IMG2IMG_VAE_DEFAULT),
            "img2img_lora_name": (QWEN_IMG2IMG_LORA_DEFAULT,),
        }
        for key, migration in model_migrations.items():
            old_values = migration[:-1]
            new_value = migration[-1]
            current_value = str(self._get(key, "") or "").strip()
            if any(current_value.casefold() == str(old_value).casefold() for old_value in old_values):
                self._set(key, new_value)
                changed = True
        # The Flux2 source workflow ships with an example content LoRA. It is
        # not a safe default for image editing because it changes the subject
        # even when acceleration is disabled.
        current_flux2_lora = str(self._get("img2img_flux2_lora_name", "") or "").strip()
        if current_flux2_lora.casefold() == FLUX2_IMG2IMG_LEGACY_LORA_DEFAULT.casefold():
            self._set("img2img_flux2_lora_name", FLUX2_IMG2IMG_LORA_DEFAULT)
            changed = True
        if self._refresh_flux2_tool_workflows():
            changed = True
        if not str(self._get("img2img_engine", "") or "").strip():
            self._set("img2img_engine", "qwen")
            changed = True
        if self._refresh_img2img_flux2_workflow():
            changed = True
        # Flux2 Klein is restored to the original no-acceleration workflow.
        # Remove the old experiment's settings so they cannot reappear as
        # active controls after an upgrade.
        for key in (
            "img2img_flux2_accel_lora_enabled",
            "img2img_flux2_accel_lora_name",
            "img2img_flux2_accel_lora_strength",
        ):
            if key in self.config:
                self.config.pop(key, None)
                changed = True
        # Remove the old acceleration settings during rollback.  They are no
        # longer writable and must not affect execution after an upgrade.
        for key in (
            "img2img_accel_lora_enabled",
            "img2img_accel_lora_name",
            "img2img_accel_lora_strength",
            "img2img_accel_steps",
            "img2img_accel_cfg",
        ):
            if key in self.config:
                self.config.pop(key, None)
                changed = True
        # Restore the original Flux2 Klein sampling defaults. The previous
        # acceleration experiment used 12/2 and must not affect this module.
        try:
            configured_steps = int(self._get("img2img_flux2_steps", FLUX2_IMG2IMG_STEPS_DEFAULT) or 0)
        except (TypeError, ValueError):
            configured_steps = 0
        if configured_steps in {6, 12}:
            self._set("img2img_flux2_steps", FLUX2_IMG2IMG_STEPS_DEFAULT)
            changed = True
        try:
            configured_cfg = float(self._get("img2img_flux2_cfg", FLUX2_IMG2IMG_CFG_DEFAULT) or 0)
        except (TypeError, ValueError):
            configured_cfg = 0.0
        if abs(configured_cfg - 1.0) < 0.0001 or abs(configured_cfg - 2.0) < 0.0001:
            self._set("img2img_flux2_cfg", FLUX2_IMG2IMG_CFG_DEFAULT)
            changed = True
        if changed:
            self._save_config()
        root = detect_comfyui_root(str(self._get("comfyui_root", "")))
        logger.info(
            "[%s] 已加载。ComfyUI 根目录：%s；原始工作流：%s",
            PLUGIN_NAME,
            root or "未检测到",
            source,
        )

    async def terminate(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _get(self, key: str, default: Any = None) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            value = default
        return default if value is None else value

    @staticmethod
    def _load_map(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write_map(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _save_lora_aliases(self) -> None:
        self._write_map(self.lora_aliases_path, self.lora_aliases)

    def _save_lora_command_aliases(self) -> None:
        self._write_map(self.lora_command_aliases_path, self.lora_command_aliases)

    def _save_lora_categories(self) -> None:
        self._write_map(self.lora_categories_path, self.lora_categories)

    def _save_lora_category_entries(self) -> None:
        self._write_map(self.lora_category_entries_path, self.lora_category_entries)

    def _save_civitai_cache(self) -> None:
        self._write_map(self.civitai_cache_path, self.civitai_cache)

    def _save_civitai_links(self) -> None:
        self._write_map(self.civitai_links_path, self.civitai_links)

    def _save_civitai_overrides(self) -> None:
        self._write_map(self.civitai_overrides_path, self.civitai_overrides)

    def _save_lora_presets(self) -> None:
        if hasattr(self, "lora_presets_path"):
            self._write_map(self.lora_presets_path, self.lora_presets)

    def _save_lora_download_order(self) -> None:
        self._write_map(self.lora_download_order_path, self.lora_download_order)

    def _set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def _save_config(self) -> None:
        saver = getattr(self.config, "save_config", None)
        if callable(saver):
            saver()

    def _nsfw_group_ids(self) -> set[str]:
        """读取 WebUI 配置的群聊黑名单，兼容旧版字符串和新版列表。"""
        raw = self._get("nsfw_group_blacklist", [])
        values = [raw] if isinstance(raw, str) else raw if isinstance(raw, (list, tuple, set)) else []
        result: set[str] = set()
        for value in values:
            for group_id in re.split(r"[,，;；\s]+", str(value or "")):
                group_id = group_id.strip()
                if group_id:
                    result.add(group_id)
        return result

    @staticmethod
    def _nsfw_marker(texts: list[Any]) -> str:
        """返回命中的成人内容标记；负面提示词不应传入此方法。"""
        combined = " ".join(str(value or "") for value in texts).casefold()
        for marker in NSFW_PROMPT_MARKERS:
            if marker.casefold() in combined:
                return marker
        for marker in NSFW_ENGLISH_MARKERS:
            if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?:[a-z0-9]*)(?![a-z0-9])", combined):
                return marker
        return ""

    def _group_id(self, event: AstrMessageEvent | None) -> str:
        if event is None:
            return ""

        def clean(value: Any) -> str:
            return str(value or "").strip()

        getter = getattr(event, "get_group_id", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return clean(value)
            except Exception:
                pass
        message_obj = getattr(event, "message_obj", None)
        for obj in (message_obj, getattr(event, "message", None)):
            value = clean(getattr(obj, "group_id", "")) if obj is not None else ""
            if value:
                return value

        # 某些 LLM 工具事件没有完整的 message_obj，但仍保留群聊会话来源，
        # 例如 ``aiocqhttp:GroupMessage:912684101``。只在来源明确包含
        # group 时读取最后一段数字，避免把私聊用户 ID 当成群号。
        origin = clean(getattr(event, "unified_msg_origin", ""))
        if "group" in origin.casefold():
            matches = re.findall(r"\d+", origin)
            if matches:
                return matches[-1]

        message_type_getter = getattr(event, "get_message_type", None)
        message_type = ""
        if callable(message_type_getter):
            try:
                message_type = clean(message_type_getter())
            except Exception:
                message_type = ""
        if "group" in message_type.casefold():
            session_getter = getattr(event, "get_session_id", None)
            if callable(session_getter):
                try:
                    return clean(session_getter())
                except Exception:
                    pass
        return ""

    def _assert_nsfw_allowed(self, event: AstrMessageEvent | None, *texts: Any) -> None:
        """兼容旧调用方；群聊黑名单现在通过最终负面提示词生效。"""
        return

    def _nsfw_group_negative(self, event: AstrMessageEvent | None) -> str:
        """返回黑名单群聊必须加入的安全负面词。"""
        group_id = self._group_id(event)
        if not group_id or group_id not in self._nsfw_group_ids():
            return ""
        return ", ".join(NSFW_GROUP_NEGATIVE)

    # ------------------------------------------------------------------
    # 图片安全审核：输入 / 输出双向检测
    #
    # 与旧的 nsfw_group_blacklist（只追加安全负面词）完全独立。这一套会在
    # 输入端和输出端各调用一次视觉大模型，命中后直接回传公开警告，
    # 不把图片交给 ComfyUI，也不把生成结果发到群里。
    # ------------------------------------------------------------------

    def _moderation_config(self) -> ModerationConfig:
        return ModerationConfig.from_mapping(
            {key: self._get(key, "") for key in MODERATION_CONFIG_KEYS}
        )

    def _moderator(self) -> ImageModerator:
        """按配置指纹复用审核客户端，避免每次绘图都重建连接池。"""
        config = self._moderation_config()
        signature = (
            config.base_url,
            config.api_key,
            config.model,
            config.strictness,
            config.timeout,
            config.fail_open,
            config.max_side,
        )
        moderator = self._moderator_instance
        if moderator is None or self._moderator_signature != signature:
            moderator = ImageModerator(config)
            self._moderator_instance = moderator
            self._moderator_signature = signature
        return moderator

    @staticmethod
    def _moderation_targets(value: Any) -> set[str]:
        """把名单配置解析成小写标记集合，兼容列表和逗号/顿号分隔的字符串。"""
        values = value if isinstance(value, (list, tuple, set)) else [value]
        result: set[str] = set()
        for item in values:
            for token in re.split(r"[,，;；\s]+", str(item or "")):
                token = token.strip().casefold()
                if token:
                    result.add(token)
        return result

    def _moderation_aliases(self, event: AstrMessageEvent | None) -> set[str]:
        """返回可用于匹配名单的会话标记。

        群号是主要写法；同时接受完整会话来源（如
        ``aiocqhttp:GroupMessage:912684101``）以及最后一段（群号 / QQ 号），
        这样把某个 QQ 号写进名单就能只检测这个人。
        """
        aliases: set[str] = set()
        for value in (
            self._group_id(event),
            self._origin(event) if event is not None else "",
            self._sender_id(event) if event is not None else "",
        ):
            text = str(value or "").strip()
            if not text:
                continue
            aliases.add(text.casefold())
            tail = re.findall(r"[A-Za-z0-9_:-]+", text)
            if tail:
                aliases.add(tail[-1].casefold())
        return aliases

    def _moderation_enabled_for(self, event: AstrMessageEvent | None, direction: str) -> bool:
        """判断本次事件是否需要对指定方向做审核。direction 取 input / output。"""
        if event is None or not self._bool_config("moderation_enabled", False):
            return False
        targets = self._moderation_targets(self._get(f"moderation_{direction}_groups", ""))
        if not targets:
            return False
        if targets & MODERATION_ALL_MARKERS:
            return True
        return bool(targets & self._moderation_aliases(event))

    @staticmethod
    def _chain_image_paths(chain: list | None) -> list[str]:
        """取出结果消息链里的本地图片路径，转发节点同样会被展开。"""
        result: list[str] = []

        def collect(components: list | None) -> None:
            for component in components or []:
                if isinstance(component, Image):
                    path = str(
                        getattr(component, "path", "") or getattr(component, "file", "") or ""
                    ).strip()
                    # 只审核本地文件：URL / base64 不能直接交给审核模型读取。
                    if path and not path.startswith(("http://", "https://", "base64://")):
                        result.append(path)
                elif isinstance(component, Nodes):
                    for node in component.nodes or []:
                        collect(getattr(node, "content", None))
                elif isinstance(component, Node):
                    collect(getattr(component, "content", None))

        collect(chain)
        return result

    async def _moderation_verdict(
        self,
        event: AstrMessageEvent | None,
        direction: str,
        paths: list[str],
    ) -> ModerationVerdict | None:
        """执行一次审核；返回需要拦截的结果，None 表示放行。"""
        candidates = [str(path) for path in (paths or []) if str(path or "").strip()]
        if not candidates or not self._moderation_enabled_for(event, direction):
            return None
        moderator = self._moderator()
        if not moderator.available:
            logger.warning(
                "[%s] 已开启图片审核（%s），但审核模型的服务地址或模型名为空，本次跳过检测",
                PLUGIN_NAME,
                direction,
            )
            return None
        verdict = await moderator.check_many(candidates)
        if verdict.blocked:
            logger.warning(
                "[%s] 图片审核命中（%s）：%s；图片=%s",
                PLUGIN_NAME,
                direction,
                verdict.describe(),
                "、".join(candidates),
            )
            return verdict
        if verdict.failed:
            # fail_open 时放行但必须留下日志，方便排查审核服务异常。
            logger.warning("[%s] 图片审核失败（%s）：%s", PLUGIN_NAME, direction, verdict.describe())
        return None

    def _moderation_notice(
        self,
        event: AstrMessageEvent | None,
        verdict: ModerationVerdict,
        direction: str,
    ) -> list[Any]:
        """构造“公开 @发送者 + 警告文字”的回复链。"""
        template = (
            MODERATION_INPUT_WARNING if direction == "input" else MODERATION_OUTPUT_WARNING
        )
        category = MODERATION_CATEGORY_LABELS.get(
            verdict.category, verdict.category or "不安全内容"
        )
        details = [
            f"成人内容等级 {verdict.level}",
            f"猎奇等级 {verdict.guro}",
        ]
        if verdict.reason:
            details.append(f"模型说明：{verdict.reason}")
        text = template.format(category=category, detail="；".join(details))
        chain: list[Any] = []
        sender = str(self._sender_id(event) if event is not None else "").strip()
        # 平台没有发送者字段时 _sender_id 会返回“会话:xxx”，那种值不能用于 @。
        if sender and not sender.startswith("会话:"):
            chain.append(At(qq=sender))
            chain.append(Plain(" "))
        chain.append(Plain(text))
        return chain

    async def _moderation_input_chain(
        self,
        event: AstrMessageEvent,
        *paths: str,
    ) -> list[Any] | None:
        """输入端审核：命中返回警告消息链，未命中返回 None。"""
        verdict = await self._moderation_verdict(
            event, "input", [path for path in paths if path]
        )
        if verdict is None:
            return None
        return self._moderation_notice(event, verdict, "input")

    async def _moderation_output_chain(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        chain: list | None,
    ) -> list[Any] | None:
        """输出端审核：命中返回警告消息链，未命中返回 None。"""
        verdict = await self._moderation_verdict(
            event,
            "output",
            self._chain_image_paths(chain),
        )
        if verdict is None:
            return None
        # 标记本次任务已拦截，发送层据此跳过“生成完成”回复。
        params.moderation_blocked = True
        params.moderation_blocked_reason = "output"
        return self._moderation_notice(event, verdict, "output")

    def _apply_group_safety_negative(
        self,
        event: AstrMessageEvent | None,
        negative: str,
    ) -> str:
        """把群聊黑名单安全词强制合并到最终负面提示词。

        这一步必须在送入工作流前执行。提示词可能先后经过 AstrBot LLM、
        插件 AI、普通翻译和工作流适配，不能只依赖其中某一层保留黑名单词。

        黑名单的语义是“降低群聊中生成成人内容的概率”，不是拒绝整个绘图任务，
        因此这里只追加负面提示词，不拦截任务。
        """
        group_id = self._group_id(event)
        if not group_id or group_id not in self._nsfw_group_ids():
            return negative

        existing = {
            item.strip().casefold()
            for item in re.split(r"[,，]", str(negative or ""))
            if item.strip()
        }
        extra = [item for item in NSFW_GROUP_NEGATIVE if item.casefold() not in existing]
        if extra:
            negative = ", ".join(
                item.strip(" ,，")
                for item in [str(negative or ""), *extra]
                if item and str(item).strip(" ,，")
            )
        if extra:
            logger.info(
                "[%s] 群聊涩图黑名单命中：群号=%s；最终负面提示词已加入 %d 项安全限制",
                PLUGIN_NAME,
                group_id,
                len(extra),
            )
        return negative

    @staticmethod
    def _prompt_attachment(params: DrawParams) -> str:
        positive = str(getattr(params, "generated_positive", "") or "").strip()
        negative = str(getattr(params, "generated_negative", "") or "").strip()
        if not positive and not negative:
            return ""
        lines = ["本次绘图提示词："]
        if positive:
            lines.append(f"正面：{positive}")
        if negative:
            lines.append(f"负面：{negative}")
        return "\n".join(lines)

    @staticmethod
    async def _request_json(request_obj: Any, default: Any = None) -> Any:
        """兼容 AstrBot 新旧 Web API 的 JSON 请求读取方式。"""
        try:
            reader = getattr(request_obj, "json")
            if callable(reader):
                try:
                    value = reader(default=default)
                except TypeError:
                    # Some AstrBot/Starlette request versions expose json()
                    # without a default keyword. Do not silently discard the
                    # POST body in that case.
                    value = reader()
            else:
                value = reader
            if inspect.isawaitable(value):
                value = await value
            return default if value is None else value
        except Exception:
            return default

    @staticmethod
    async def _request_files(request_obj: Any) -> Any:
        """兼容 AstrBot 新旧 Web API 的 multipart 文件读取方式。"""
        try:
            reader = getattr(request_obj, "files")
            value = reader() if callable(reader) else reader
            if inspect.isawaitable(value):
                value = await value
            return value
        except Exception:
            return {}

    def _workflow_dir(self) -> Path:
        configured = str(self._get("workflow_dir", "") or "").strip()
        if configured:
            path = Path(configured)
            if path.is_dir():
                return path
        path = self.plugin_dir / "workflows"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _workflow_path(self, mode: str) -> Path:
        configured = str(self._get(f"workflow_{mode}", "") or "").strip()
        if configured:
            path = Path(configured)
            if path.is_file():
                return path
            candidate = self._workflow_dir() / configured
            if candidate.is_file():
                return candidate
        return self._workflow_dir() / MODE_FILES[mode]

    def _img2img_mode(self, engine: str | None = None) -> str:
        """Resolve the selected image-edit engine without mixing its config."""
        value = str(engine or self._get("img2img_engine", "qwen") or "qwen").strip().lower()
        if value in {"flux2", "flux", "flux-2", "img2img_flux2", "图生图flux2", "图生图2"}:
            return "img2img_flux2"
        return "img2img"

    def _client(self) -> ComfyClient:
        return ComfyClient(str(self._get("comfyui_url", "http://127.0.0.1:8188")))

    async def _environment(self, client: ComfyClient) -> dict[str, Any]:
        classes = set((await client.object_info()).keys())
        categories = await asyncio.gather(
            client.models("diffusion_models"),
            client.models("checkpoints"),
            client.models("loras"),
            client.models("upscale_models"),
            client.models("clip_gguf"),
            client.models("vae"),
            client.models("unet_gguf"),
            client.models("controlnet"),
            client.models("ipadapter"),
            client.models("clip_vision"),
            client.models("text_encoders"),
        )
        # ComfyUI 将 Flux2/Qwen3 的 safetensors 文本编码器放在
        # ``models/text_encoders``，而 Qwen Image 的 GGUF 编码器放在
        # ``models/clip`` 并通过 ``clip_gguf`` 暴露。旧代码误把
        # categories[4] (clip_gguf) 当成了所有文本编码器，导致 Flux2
        # 明明能在 ComfyUI 下拉框中看到的 qwen_3_8b_fp8mixed.safetensors
        # 被插件判定为不存在。
        clip_gguf_names = categories[4]
        text_encoder_names = categories[10]
        all_text_encoder_names = list(dict.fromkeys([*text_encoder_names, *clip_gguf_names]))
        ordered_loras = self._order_loras(categories[2])
        self._lora_names_cache = (time.monotonic(), list(ordered_loras))
        return {
            "diffusion_models": categories[0],
            "checkpoints": categories[1],
            "loras": ordered_loras,
            "upscale_models": categories[3],
            # ComfyUI-GGUF registers separate API folders for the GGUF
            # loaders. Keep these lists separate from txt2img checkpoints.
            "unet": categories[6],
            # 保留合并列表给历史 WebUI/配置读取者；绘图路径使用下面两个
            # 专用列表，避免不同工作流的模型目录再次混淆。
            "text_encoders": all_text_encoder_names,
            "flux2_text_encoders": text_encoder_names,
            "clip_gguf": clip_gguf_names,
            "vae": categories[5],
            "classes": classes,
            "controlnet": categories[7],
            "ipadapter": categories[8],
            "clip_vision": categories[9],
        }

    async def _available_loras(self, *, force: bool = False) -> list[str]:
        """短时间缓存 LoRA 列表，避免每条绘图指令都等待 ComfyUI 查询。"""
        now = time.monotonic()
        if not force and self._lora_names_cache is not None:
            cached_at, names = self._lora_names_cache
            if now - cached_at < 30:
                return list(names)
        names = self._order_loras(await self._client().models("loras"))
        self._lora_names_cache = (now, list(names))
        return list(names)

    def _order_loras(self, names: Any) -> list[str]:
        """按本插件记录的下载时间排列 LoRA，最新下载的放在最前。

        ComfyUI 返回的旧文件顺序不稳定，因此没有下载记录的文件保留
        当前来源顺序；只有本插件记录过下载时间的文件参与倒序排列。
        """
        # Keep the exact names returned by ComfyUI.  The separator is part of
        # the dropdown value used by /prompt; normalize only for metadata
        # lookup, never for the value sent to a loader node.
        values = [str(name).strip() for name in (names or []) if str(name).strip()]
        download_order = getattr(self, "lora_download_order", {})
        if not isinstance(download_order, dict):
            download_order = {}
        downloaded: list[tuple[float, int, str]] = []
        old: list[str] = []
        for index, name in enumerate(values):
            normalized_name = name.replace("\\", "/")
            raw_time = download_order.get(name)
            if raw_time is None:
                raw_time = download_order.get(normalized_name)
            if raw_time is None:
                raw_time = download_order.get(Path(name).name)
            try:
                download_time = float(raw_time)
            except (TypeError, ValueError):
                download_time = 0
            if download_time > 0:
                downloaded.append((download_time, index, name))
            else:
                old.append(name)
        downloaded.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, _, name in downloaded] + old

    async def _ensure_blank_image(self, client: ComfyClient, width: int, height: int, scale: float) -> str:
        upscale = max(1.0, float(scale or 1.0))
        input_width = max(64, int(round(width / upscale / 8) * 8))
        input_height = max(64, int(round(height / upscale / 8) * 8))
        content = _blank_png(input_width, input_height)
        path = self.data_dir / f"astrbot_blank_{input_width}x{input_height}.png"
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
        uploaded = await client.upload_image(str(path))
        return uploaded["name"]

    def _origin(self, event: AstrMessageEvent) -> str:
        value = getattr(event, "unified_msg_origin", "")
        if value:
            return str(value)
        return str(getattr(getattr(event, "session", None), "unified_msg_origin", "default"))

    def _draw_limit_admin_ids(self) -> set[str]:
        """读取插件管理员 ID。管理员不受绘图限额影响，也可控制 ComfyUI。"""
        raw = self._get("draw_limit_admin_ids", "")
        values = raw if isinstance(raw, list) else re.split(r"[\s,，;；]+", str(raw or ""))
        return {str(value).strip() for value in values if str(value).strip()}

    def _sender_id(self, event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_sender_id", None)
        try:
            value = getter() if callable(getter) else ""
        except Exception:
            value = ""
        if value not in (None, ""):
            return str(value)
        sender = getattr(getattr(event, "message_obj", None), "sender", None)
        value = getattr(sender, "user_id", "") if sender is not None else ""
        if value not in (None, ""):
            return str(value)
        # 少数平台没有发送者字段时按会话兜底，避免所有用户共用一个额度桶。
        return f"会话:{self._origin(event)}"

    def _is_draw_limit_admin(self, event: AstrMessageEvent) -> bool:
        return self._sender_id(event) in self._draw_limit_admin_ids()

    def _draw_limit_status(self, event: AstrMessageEvent | None = None) -> dict[str, Any]:
        """Return a small, non-sensitive view of the active draw limit."""
        count, window = self._draw_limit_settings()
        sender_id = self._sender_id(event) if event is not None else ""
        now = time.monotonic()
        entries = getattr(self, "_draw_limit_entries", {})
        active = [
            entry
            for entry in entries.get(sender_id, [])
            if now - entry[1] < window
        ] if sender_id else []
        used = sum(entry[2] for entry in active)
        return {
            "enabled": count > 0,
            "count": count,
            "window_seconds": int(window),
            "user_id": sender_id,
            "used": used,
            "remaining": max(0, count - used) if count > 0 else None,
            "admin_exempt": bool(event is not None and self._is_draw_limit_admin(event)),
        }

    def _comfyui_port(self) -> int:
        """只允许控制本机 ComfyUI，绝不通过插件关闭远程地址。"""
        try:
            parsed = urlparse(str(self._get("comfyui_url", "http://127.0.0.1:8188") or ""))
            host = (parsed.hostname or "").casefold()
            if host not in {"127.0.0.1", "localhost", "::1"}:
                return 0
            return int(parsed.port or (443 if parsed.scheme == "https" else 80))
        except (TypeError, ValueError):
            return 0

    def _comfyui_start_script(self) -> Path | None:
        configured = Path(str(self._get("comfyui_start_script", "") or "").strip())
        if str(configured) and configured.is_file():
            return configured
        root_text = detect_comfyui_root(str(self._get("comfyui_root", "") or ""))
        if not root_text:
            return None
        root = Path(root_text)
        candidates = [
            root / "run_nvidia_gpu.bat",
            root / "run.bat",
            root.parent / "run_nvidia_gpu.bat",
            root.parent / "run.bat",
        ]
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def _listening_pids_on_port(port: int) -> list[int]:
        if port <= 0:
            return []
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        pattern = re.compile(rf"^\s*TCP\s+\S+:{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.IGNORECASE)
        pids: list[int] = []
        for line in result.stdout.splitlines():
            match = pattern.match(line)
            if match:
                pid = int(match.group(1))
                if pid > 0 and pid not in pids:
                    pids.append(pid)
        return pids

    @staticmethod
    def _stop_pids(pids: list[int]) -> list[int]:
        stopped: list[int] = []
        for pid in pids:
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=12,
                )
                if result.returncode == 0:
                    stopped.append(pid)
            except (OSError, subprocess.SubprocessError):
                continue
        return stopped

    def _draw_limit_settings(self) -> tuple[int, float]:
        try:
            count = max(0, int(self._get("draw_limit_count", 0) or 0))
        except (TypeError, ValueError):
            count = 0
        try:
            window = max(1.0, float(self._get("draw_limit_window_seconds", 3600) or 3600))
        except (TypeError, ValueError):
            window = 3600.0
        return count, window

    def _bool_config(self, key: str, default: bool = False) -> bool:
        value = self._get(key, default)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "off", "no", "否", "关闭"}
        return bool(value)

    def _draw_queue_limit_settings(self) -> tuple[bool, int]:
        """读取提交前队列限制设置。"""
        enabled = self._bool_config("draw_queue_limit_enabled", False)
        try:
            limit = max(0, int(self._get("draw_queue_limit_count", 0) or 0))
        except (TypeError, ValueError):
            limit = 0
        return enabled and limit > 0, limit

    def _queue_requested_images(self, params: DrawParams, mode: str) -> int:
        """估算本次提交会增加的图片数，用于队列限制而不是工作流参数。"""
        try:
            requested = int(getattr(params, "batch", 1) or 1)
        except (TypeError, ValueError):
            requested = 1
        if requested <= 1:
            # 这些模式的批次只在各自页面配置中生效；显式 --batch=2
            # 则优先使用上面的参数值。
            config_keys = {
                "txt2img": "batch",
                "img2img_flux2": "img2img_flux2_batch",
                "wash": "wash_batch",
                "outpaint": "outpaint_batch",
                "multi_angle": "multi_angle_batch",
            }
            key = config_keys.get(mode)
            if key:
                try:
                    requested = int(self._get(key, 1) or 1)
                except (TypeError, ValueError):
                    requested = 1
        return max(1, min(16, requested))

    @staticmethod
    def _queue_item_prompt(item: Any) -> dict[str, Any] | None:
        """从不同 ComfyUI 版本的队列元素中取出 API 工作流图。"""
        if isinstance(item, dict):
            for key in ("prompt", "workflow"):
                value = item.get(key)
                if isinstance(value, dict):
                    return value
            # 某些代理直接返回 API 工作流对象。
            if any(isinstance(value, dict) and "inputs" in value for value in item.values()):
                return item
            return None
        if isinstance(item, (list, tuple)):
            # ComfyUI 原生格式通常为 [number, prompt_id, prompt, ...]。
            if len(item) > 2 and isinstance(item[2], dict):
                return item[2]
            for value in item:
                if isinstance(value, dict):
                    prompt = ComfyUIAIStudio._queue_item_prompt(value)
                    if prompt is not None:
                        return prompt
        return None

    @classmethod
    def _estimate_queue_item_images(cls, item: Any) -> int:
        """按队列工作流中的 batch_size 估算一个任务会输出几张图。"""
        prompt = cls._queue_item_prompt(item)
        if not prompt:
            return 1
        batches: list[int] = []
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            for key in ("batch_size", "batch"):
                value = inputs.get(key)
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= number <= 64:
                    batches.append(number)
        return max(batches, default=1)

    async def _queue_snapshot(self) -> dict[str, Any]:
        """读取并整理 ComfyUI 当前运行中和等待中的队列。"""
        data = await self._client().queue_info()
        running = list(data.get("queue_running", []) or [])
        pending = list(data.get("queue_pending", []) or [])
        running_images = sum(self._estimate_queue_item_images(item) for item in running)
        pending_images = sum(self._estimate_queue_item_images(item) for item in pending)
        return {
            "available": True,
            "running_tasks": len(running),
            "pending_tasks": len(pending),
            "running_images": running_images,
            "pending_images": pending_images,
            "queued_tasks": len(running) + len(pending),
            "queued_images": running_images + pending_images,
        }

    def _queue_notice_from_params(self, params: DrawParams) -> str:
        if not self._bool_config("draw_queue_notice_enabled", True):
            return ""
        if getattr(params, "queue_available", None) is None:
            return ""
        if getattr(params, "queue_available", None) is False:
            return str(
                getattr(params, "queue_notice_text", "")
                or "绘图前暂时无法读取 ComfyUI 排队数量，任务仍会提交。"
            )
        return (
            f"当前前面已有 {int(getattr(params, 'queue_images_before', 0) or 0)} 张图排队"
            f"（运行中 {int(getattr(params, 'queue_running_images', 0) or 0)} 张，"
            f"等待中 {int(getattr(params, 'queue_pending_images', 0) or 0)} 张，"
            f"共 {int(getattr(params, 'queue_tasks_before', 0) or 0)} 个任务）。"
        )

    async def _check_draw_queue(
        self,
        params: DrawParams,
        mode: str,
        event: AstrMessageEvent | None = None,
    ) -> None:
        """在开始回复前检查队列，并在需要时阻止过量提交。

        配置的插件管理员可以绕过队列上限，避免管理员无法在队列异常时
        处理或测试绘图；队列提示本身仍然会按 WebUI 开关显示。
        """
        notice_enabled = self._bool_config("draw_queue_notice_enabled", True)
        limit_enabled, limit = self._draw_queue_limit_settings()
        queue_limit_exempt = bool(event is not None and self._is_draw_limit_admin(event))
        if queue_limit_exempt:
            logger.info("[%s] 队列上限命中管理员豁免：用户=%s", PLUGIN_NAME, self._sender_id(event))
        enforce_limit = limit_enabled and not queue_limit_exempt
        if not notice_enabled and not enforce_limit:
            return

        params.queue_requested_images = self._queue_requested_images(params, mode)
        try:
            snapshot = await self._queue_snapshot()
        except Exception as exc:
            params.queue_available = False
            params.queue_notice_text = "绘图前暂时无法读取 ComfyUI 排队数量，任务仍会提交。"
            if enforce_limit:
                logger.warning("[%s] 队列限制已开启但读取队列失败：%s", PLUGIN_NAME, exc)
                raise UsageError(
                    "暂时无法读取 ComfyUI 排队数量，已阻止本次提交；"
                    "可关闭“队列过多时禁止继续提交”后重试"
                ) from exc
            logger.warning("[%s] 读取 ComfyUI 绘图队列失败，继续提交：%s", PLUGIN_NAME, exc)
            return

        params.queue_available = True
        params.queue_running_tasks = int(snapshot["running_tasks"])
        params.queue_pending_tasks = int(snapshot["pending_tasks"])
        params.queue_running_images = int(snapshot["running_images"])
        params.queue_pending_images = int(snapshot["pending_images"])
        params.queue_images_before = int(snapshot["queued_images"])
        params.queue_tasks_before = int(snapshot["queued_tasks"])
        params.queue_notice_text = self._queue_notice_from_params(params) if notice_enabled else ""

        if enforce_limit and params.queue_images_before + params.queue_requested_images > limit:
            raise UsageError(
                f"ComfyUI 当前前面已有 {params.queue_images_before} 张图排队，"
                f"本次预计增加 {params.queue_requested_images} 张，"
                f"超过允许的最多排队 {limit} 张"
            )

    def _img2img_output_size(
        self,
        image_path: str,
        width: int,
        height: int,
        *,
        use_input_size: bool = False,
        max_edge: int | None = None,
    ) -> tuple[int, int]:
        """Fit the output rectangle to the input image aspect ratio.

        The Qwen workflow receives an explicit size.  By default preserve the
        input dimensions when they are small enough, but shrink oversized
        inputs to ``max_edge``.  This keeps the source ratio without allowing
        a phone photo or an upscaled attachment to create an unexpectedly
        large latent on an 8 GB GPU.  When the user explicitly supplies a
        target size, treat it as a maximum box and derive both dimensions from
        the source ratio.
        """
        keep_ratio = self._bool_config("img2img_keep_aspect_ratio", True)
        if not keep_ratio or not image_path:
            return width, height
        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as image:
                source_width, source_height = image.size
            if source_width <= 0 or source_height <= 0:
                return width, height
            if use_input_size:
                # 只在输入图超过上限时缩小，避免小图被无意义放大。
                # 宽高从同一个比例因子计算，再分别对齐到 8，避免拉伸。
                input_limit = max(64, int(max_edge or max(width, height, 64)))
                scale = min(1.0, input_limit / max(source_width, source_height))
                fitted_width = max(64, int(round(source_width * scale / 8.0)) * 8)
                fitted_height = max(64, int(round(source_height * scale / 8.0)) * 8)
                logger.info(
                    "[%s] 图生图按原图比例处理：原图=%sx%s；最长边上限=%s；实际输出=%sx%s",
                    PLUGIN_NAME,
                    source_width,
                    source_height,
                    input_limit,
                    fitted_width,
                    fitted_height,
                )
                return fitted_width, fitted_height
            max_width = max(64, int(width or source_width))
            max_height = max(64, int(height or source_height))
            scale = min(float(max_width) / source_width, float(max_height) / source_height)

            # Round the width first and derive the height from the same ratio.
            # Rounding both independently can introduce avoidable distortion.
            fitted_width = max(64, int(round(source_width * scale / 8.0)) * 8)
            fitted_height = max(64, int(round(fitted_width * source_height / source_width / 8.0)) * 8)
            if fitted_width > max_width:
                fitted_width = max(64, (max_width // 8) * 8)
                fitted_height = max(64, int(round(fitted_width * source_height / source_width / 8.0)) * 8)
            if fitted_height > max_height:
                fitted_height = max(64, (max_height // 8) * 8)
                fitted_width = max(64, int(round(fitted_height * source_width / source_height / 8.0)) * 8)
            logger.info(
                "[%s] 图生图保持原图比例：原图=%sx%s；目标框=%sx%s；实际输出=%sx%s",
                PLUGIN_NAME,
                source_width,
                source_height,
                max_width,
                max_height,
                fitted_width,
                fitted_height,
            )
            return fitted_width, fitted_height
        except (OSError, ValueError, TypeError, ImportError) as exc:
            logger.warning("[%s] 读取图生图原图尺寸失败，使用配置尺寸：%s", PLUGIN_NAME, exc)
            return width, height

    async def _reserve_draw_limit(self, event: AstrMessageEvent, params: DrawParams) -> None:
        """在任务入队时预留图片额度，生成失败会由 _run 精确归还。"""
        count, window = self._draw_limit_settings()
        user_id = self._sender_id(event)
        if count <= 0:
            logger.info("[%s] 绘图限额未启用：count=%s；用户=%s", PLUGIN_NAME, count, user_id)
            return
        if self._is_draw_limit_admin(event):
            logger.info(
                "[%s] 绘图限额命中管理员豁免：用户=%s；配置管理员=%s",
                PLUGIN_NAME,
                user_id,
                ",".join(sorted(self._draw_limit_admin_ids())) or "无",
            )
            return
        cost = max(1, min(4, int(params.batch or 1)))
        lock = getattr(self, "_draw_limit_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._draw_limit_lock = lock
        if not hasattr(self, "_draw_limit_entries"):
            self._draw_limit_entries = {}
        now = time.monotonic()
        async with lock:
            active = [entry for entry in self._draw_limit_entries.get(user_id, []) if now - entry[1] < window]
            used = sum(entry[2] for entry in active)
            if used + cost > count:
                remaining = max(0.0, window - (now - min(entry[1] for entry in active))) if active else window
                raise UsageError(
                    f"绘图次数已达上限：{int(window)} 秒内最多 {count} 张，约 {int(remaining) + 1} 秒后可继续"
                )
            token = uuid.uuid4().hex
            active.append((token, now, cost))
            self._draw_limit_entries[user_id] = active
            logger.info(
                "[%s] 绘图限额已预留：用户=%s；本次=%s；已用=%s/%s；窗口=%ss",
                PLUGIN_NAME,
                user_id,
                cost,
                used + cost,
                count,
                int(window),
            )
        params.draw_limit_user_id = user_id
        params.draw_limit_token = token
        params.draw_limit_cost = cost

    async def _release_draw_limit(self, params: DrawParams) -> None:
        """只在提交后生成失败或任务取消时归还本次预留额度。"""
        user_id = str(getattr(params, "draw_limit_user_id", "") or "")
        token = str(getattr(params, "draw_limit_token", "") or "")
        if not user_id or not token:
            return
        lock = getattr(self, "_draw_limit_lock", None)
        if lock is None:
            return
        async with lock:
            entries = self._draw_limit_entries.get(user_id, [])
            entries = [entry for entry in entries if entry[0] != token]
            if entries:
                self._draw_limit_entries[user_id] = entries
            else:
                self._draw_limit_entries.pop(user_id, None)
        params.draw_limit_token = ""

    async def _astrbot_persona_context(self, event: AstrMessageEvent) -> tuple[list[Any] | None, str]:
        """获取当前 AstrBot 会话上下文和人格，不使用插件固定人格覆盖它。"""
        get_extra = getattr(event, "get_extra", None)
        provider_request = get_extra("provider_request") if callable(get_extra) else None
        if provider_request is not None:
            contexts = getattr(provider_request, "contexts", None)
            if isinstance(contexts, str):
                try:
                    contexts = json.loads(contexts)
                except (TypeError, ValueError):
                    contexts = None
            system_prompt = str(getattr(provider_request, "system_prompt", "") or "").strip()
            return contexts if isinstance(contexts, list) else None, system_prompt

        try:
            conversation_manager = getattr(self.context, "conversation_manager", None)
            if conversation_manager is None:
                return None, ""
            origin = self._origin(event)
            conversation_id = await conversation_manager.get_curr_conversation_id(origin)
            conversation = (
                await conversation_manager.get_conversation(origin, conversation_id)
                if conversation_id
                else None
            )
            if conversation is None:
                return None, ""
            contexts = json.loads(str(getattr(conversation, "history", "[]") or "[]"))
            system_prompt = ""
            config_getter = getattr(self.context, "get_config", None)
            persona_manager = getattr(self.context, "persona_manager", None)
            if callable(config_getter) and persona_manager is not None:
                config = config_getter(umo=origin) or {}
                _, persona, _, _ = await persona_manager.resolve_selected_persona(
                    umo=origin,
                    conversation_persona_id=getattr(conversation, "persona_id", None),
                    platform_name=str(getattr(event, "get_platform_name", lambda: "")() or ""),
                    provider_settings=config,
                )
                persona_prompt = str(
                    (persona.get("prompt", "") if isinstance(persona, dict) else getattr(persona, "prompt", ""))
                    or ""
                ).strip()
                if persona_prompt:
                    system_prompt = f"# Persona Instructions\n\n{persona_prompt}"
            return contexts if isinstance(contexts, list) else None, system_prompt
        except Exception as exc:
            logger.debug("[%s] 读取 AstrBot 会话人格失败：%s", PLUGIN_NAME, exc)
            return None, ""

    async def _astrbot_generate(
        self,
        event: AstrMessageEvent,
        prompt: str = "",
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        use_event_context: bool = False,
        preserve_newlines: bool = False,
    ) -> str:
        provider_getter = getattr(self.context, "get_current_chat_provider_id", None)
        llm_generate = getattr(self.context, "llm_generate", None)
        if not callable(provider_getter) or not callable(llm_generate):
            raise AIError("AstrBot 当前 AI 接口不可用")
        provider_id = await provider_getter(self._origin(event))
        if not provider_id:
            raise AIError("当前会话没有可用的 AstrBot AI")
        contexts = None
        effective_system_prompt = str(system_prompt or "").strip()
        if use_event_context:
            contexts, persona_prompt = await self._astrbot_persona_context(event)
            if persona_prompt and effective_system_prompt:
                effective_system_prompt = f"{persona_prompt}\n\n{effective_system_prompt}"
            elif persona_prompt:
                effective_system_prompt = persona_prompt
        kwargs: dict[str, Any] = {
            "chat_provider_id": provider_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if contexts:
            kwargs["contexts"] = contexts
        if effective_system_prompt:
            kwargs["system_prompt"] = effective_system_prompt
        response = await llm_generate(**kwargs)
        result = str(getattr(response, "completion_text", "") or "").strip()
        if not result:
            raise AIError("AstrBot AI 返回了空内容")
        if preserve_newlines:
            return result.strip(" `\n\t")
        return result.replace("\n", ", ").strip(" `,，。；;")

    async def _translate_prompt(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        force_enabled: bool = False,
        source_override: str = "",
    ) -> str:
        source = str(source_override or "astrbot").lower()
        is_flux2_img2img = source in {
            "plugin_img2img_flux2_llm",
            "astrbot_img2img_flux2_llm",
        }
        is_img2img_llm = source in {
            "plugin_img2img_llm",
            "astrbot_img2img_llm",
            "plugin_img2img_flux2_llm",
            "astrbot_img2img_flux2_llm",
        }
        img2img_config_prefix = "img2img_flux2" if is_flux2_img2img else "img2img"
        if is_img2img_llm:
            # 两个入口只使用同一套编辑规则。旧版分别配置两套系统提示词，
            # 很容易一边允许扩写、一边又把结果压缩成 change her pose。
            # 保留旧字段只为兼容配置，但不再让它覆盖统一规则。
            system_prompt = str(
                self._get(f"{img2img_config_prefix}_ai_system_prompt", "")
                or DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT
            ).strip()
            edit_knowledge = str(
                self._get(f"{img2img_config_prefix}_plugin_ai_knowledge", "") or ""
            ).strip()
            system_prompt = (
                f"{system_prompt}\n\n【图生图编辑知识】\n"
                f"{edit_knowledge or DEFAULT_IMG2IMG_PLUGIN_AI_KNOWLEDGE}"
            )
        elif source == "plugin_llm":
            system_prompt = str(
                self._get("plugin_ai_llm_system_prompt", "")
                or DEFAULT_PLUGIN_AI_LLM_SYSTEM_PROMPT
            ).strip()
        elif source == "plugin":
            system_prompt = str(
                self._get("plugin_ai_command_system_prompt", "")
                or DANBOORU_SYSTEM_PROMPT
            ).strip()
        else:
            system_prompt = DANBOORU_SYSTEM_PROMPT
        if not is_img2img_llm:
            # context.llm_generate() 不会再次经过当前插件的 on_llm_request
            # 过滤器，故在文生图提示词请求中主动复用内置 Anima 精简知识。
            anima_context = str(self._get("plugin_ai_anima_context", "") or "").strip()
            if not anima_context:
                anima_context = build_anima_context(
                    self.plugin_dir, 2600, self._anima_template_override()
                )
            try:
                system_prompt = (
                    f"{system_prompt}\n\n【Anima 提示词工程师知识】\n"
                    f"{anima_context}"
                )
            except Exception as exc:
                logger.debug("[%s] Anima 提示词知识加载失败，使用基础规则：%s", PLUGIN_NAME, exc)
        if source in {"astrbot_img2img_llm", "astrbot_img2img_flux2_llm"}:
            prompt_template = str(
                self._get(
                    f"{img2img_config_prefix}_prompt_template",
                    "",
                )
                or self._get("img2img_astrbot_user_prompt_template", "")
                or DEFAULT_IMG2IMG_ASTRBOT_USER_PROMPT_TEMPLATE
            )
            user_prompt = self._format_img2img_llm_template(prompt_template, text)
            output_format = str(
                self._get(f"{img2img_config_prefix}_output_format", "")
                or (self._get("img2img_plugin_ai_output_format", "") if not is_flux2_img2img else "")
                or DEFAULT_IMG2IMG_OUTPUT_FORMAT
            )
            system_prompt = f"{system_prompt}\n\n【输出格式要求】\n{output_format}".strip()
            result = await self._astrbot_generate(
                event,
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=256,
                preserve_newlines=is_img2img_llm,
            )
            return self._clean_img2img_edit_instruction(result)
        if source == "astrbot":
            result = await self._astrbot_generate(
                event,
                f"画面描述：{text}",
                system_prompt=system_prompt,
                max_tokens=512,
            )
            # AstrBot 当前模型也可能把分析过程混在 completion_text 中。
            # 所有文生图 AI 来源必须经过同一个 Anima 输出清洗器，不能
            # 让“分析用户请求”等文字进入 ComfyUI。
            return self._clean_txt2img_prompt(result)
        ai_config = self._config_dict()
        if source in {"plugin_img2img_llm", "plugin_img2img_flux2_llm"}:
            prompt_template = str(
                self._get(
                    f"{img2img_config_prefix}_prompt_template",
                    "",
                )
                or self._get("img2img_plugin_ai_user_prompt_template", "")
                or DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE
            )
            prompt = self._format_img2img_llm_template(prompt_template, text)
            output_format = str(
                self._get(f"{img2img_config_prefix}_output_format", "")
                or (self._get("img2img_plugin_ai_output_format", "") if not is_flux2_img2img else "")
                or DEFAULT_IMG2IMG_OUTPUT_FORMAT
            )
            system_prompt = f"{system_prompt}\n\n【输出格式要求】\n{output_format}".strip()
        else:
            prompt = text
        result = await AITranslator(ai_config).generate(
            prompt,
            system_prompt=system_prompt,
            max_tokens=256 if is_img2img_llm else 768 if source == "plugin_llm" else 512,
            preserve_newlines=is_img2img_llm or source == "plugin_llm",
        )
        if is_img2img_llm:
            return self._clean_img2img_edit_instruction(result)
        # 普通指令 AI 和 LLM 绘图 AI 都使用同一个最终标签截断点。
        # 这也覆盖了模型把 reasoning_content 错误回显到正文的情况。
        return self._clean_txt2img_prompt(result)

    @staticmethod
    def _format_img2img_llm_template(template: str, text: str) -> str:
        """Expand the editable image-edit user prompt without allowing format errors."""
        value = str(template or "").strip() or "{request}"
        request = str(text or "").strip()
        return (
            value.replace("{request}", request)
            .replace("{用户要求}", request)
            .replace("{用户原话}", request)
        )

    def _plain_translation_enabled(self, mode: str = "txt2img") -> bool:
        if mode == "img2img_flux2":
            key = "img2img_flux2_plain_translate_enabled"
            default = False
        elif mode == "img2img":
            key = "img2img_plain_translate_enabled"
            default = False
        else:
            key = "plain_translate_enabled"
            default = True
        enabled = self._get(key, default)
        if isinstance(enabled, str):
            return enabled.strip().lower() not in {"0", "false", "off", "no", "否", "关闭"}
        return bool(enabled)

    async def _plain_translate_prompt(
        self,
        text: str,
        mode: str = "txt2img",
    ) -> tuple[str, str]:
        """非 AI 模式下翻译用户中文；失败时保留原文，不中断绘图。"""
        if not self._plain_translation_enabled(mode) or not contains_chinese(text):
            return text, ""
        if mode == "img2img_flux2":
            url_key = "img2img_flux2_plain_translate_url"
        elif mode == "img2img":
            url_key = "img2img_plain_translate_url"
        else:
            url_key = "plain_translate_url"
        try:
            result = await PlainTranslator(
                str(
                    self._get(
                        url_key,
                        "https://translate.googleapis.com/translate_a/single",
                    )
                    or ""
                )
            ).translate(text)
            return result, "中文提示词已通过普通翻译网站转换为英文"
        except PlainTranslationError as exc:
            logger.warning("[%s] 普通翻译失败，保留原始中文继续绘图：%s", PLUGIN_NAME, exc)
            return text, "普通翻译网站暂时不可用，已保留原始中文提示词"

    @staticmethod
    def _clean_txt2img_prompt(value: str) -> str:
        """只提取文生图 AI 的最终标签，拒绝把推理过程送入 ComfyUI。

        文生图插件 AI 有时会遵守要求只返回标签，有时却会返回一整段
        “分析用户请求/确定标签/最终结果”的推理。这个清洗器只接受明确
        的最终字段、代码块，或分析段落之前的标签前缀；没有可靠结果时
        抛出 AIError，让上层回退到用户原始提示词。
        """
        raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            raise AIError("文生图 AI 返回了空内容")

        # 推理模型常把思考放进 <think>、<analysis> 或 ```thinking``` 块，
        # 最终标签可能在闭合标签后。先移除明确标记的推理块，再处理普通
        # “分析用户请求/最终提示词”格式，避免整段思考被当成标签。
        raw = re.sub(
            r"<(?P<tag>think|analysis|reasoning)>.*?</(?P=tag)>",
            "\n",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        raw = re.sub(
            r"```(?:thinking|analysis|reasoning)\s*.*?```",
            "\n",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        candidates: list[str] = []
        reasoning_marker = re.compile(
            r"(?:^|[,，;；\n])\s*(?:\d+\s*[.)、:]\s*)?"
            r"(?:\*\*|__)?(?:分析用户请求|用户要求(?:输出|包含)?|需要包含|分析|推理|思考|确定核心标签|确定标签|核心标签|分析结果|人数与性别|角色[与/]作品|外观|服装|姿势|动作|表情|镜头|场景|标签列表|最终组合|reasoning|analysis|thinking)"
            r"(?:\*\*|__)?\s*[:：]?",
            re.IGNORECASE,
        )
        final_marker = re.compile(
            r"(?:^|\n|[,，;；])\s*(?:\d+\s*[.)、:]\s*)?"
            r"(?:\*\*|__)?(?:final(?:\s+prompt)?|positive(?:\s+prompt)?|prompt|result|output|最终提示词|最终结果|最终标签|正面提示词|正面|提示词|结果)"
            r"(?:\*\*|__)?\s*[:：]\s*([^\n]+)",
            re.IGNORECASE,
        )

        # 优先读取“最终提示词：...”这一类明确字段。
        candidates.extend(match.group(1) for match in final_marker.finditer(raw))

        # 兼容 JSON、Markdown JSON，以及 ```...``` 中的 JSON/标签。
        for block in re.findall(r"\{.*?\}", raw, flags=re.DOTALL):
            try:
                data = json.loads(block)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict):
                for key in ("final", "prompt", "result", "output", "tags", "最终提示词", "最终结果"):
                    item = data.get(key)
                    if isinstance(item, str) and item.strip():
                        candidates.insert(0, item)
        code_blocks = re.findall(r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)```", raw, flags=re.DOTALL)
        candidates.extend(code_blocks)

        reasoning_matches = list(reasoning_marker.finditer(raw))
        if reasoning_matches:
            # 有推理时只允许使用推理开始前的前缀。这样可以处理模型返回
            # “youhu, loli, petite, 1. 分析用户请求：...”的情况，同时
            # 不会把分析中的某一行偶然误当成最终提示词。
            prefix = raw[: reasoning_matches[0].start()].strip(" `\n\t,，;；")
            if prefix:
                # 有些模型没有最终字段，但会在分析中把动作标签写成
                # `split`、"legs spread" 这类明确的英文标记。只把这类
                # ASCII 引号/反引号中的短标签拼回已有前缀，不读取普通
                # 英文句子，避免动作在截断推理时丢失。
                quoted_tags = [
                    item.strip()
                    for item in re.findall(
                        r"[`\"']([A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z][A-Za-z0-9_'-]*){0,4})[`\"']",
                        raw[reasoning_matches[0].start():],
                    )
                    if item.strip().casefold() not in {"user", "prompt", "result", "output"}
                ]
                combined = prefix
                if quoted_tags:
                    combined = f"{combined}, {', '.join(quoted_tags)}"
                candidates.insert(0, combined)
        elif not final_marker.search(raw) and not code_blocks:
            # 没有推理或字段包装时，完整返回值本身就是候选结果。
            candidates.append(raw)

        quality_terms = {
            "masterpiece", "best quality", "amazing quality", "very aesthetic",
            "extremely detailed", "highly detailed", "very detailed", "absurdres",
            "highres", "newest", "score_9", "score_8", "score_7", "year 2024",
        }
        rejected_markers = (
            "分析用户请求", "用户要求输出", "需要包含", "确定核心标签", "核心标签", "用户原话", "目标：",
            "we need", "user's original", "astrbot", "reasoning", "analysis",
            "thinking", "because", "therefore", "let's", "we should", "我将",
            "我们需要", "下面", "首先", "其次", "输出只保留", "标签列表",
        )

        def normalize(candidate: str) -> str:
            text = str(candidate or "").strip(" `\n\t")
            text = re.sub(r"^(?:[-*+]\s+|\d+\s*[.)、:]\s*)", "", text)
            text = re.sub(
                r"^(?:final(?:\s+prompt)?|positive(?:\s+prompt)?|prompt|result|output|最终提示词|最终结果|最终标签|正面提示词|正面|提示词|结果)\s*[:：]\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = text.replace("，", ",").replace("、", ",").replace("；", ",")
            text = text.replace("**", "").replace("__", "").replace("`", "")
            parts = re.split(r"\s*,\s*|\n+", text)
            kept: list[str] = []
            seen: set[str] = set()
            for part in parts:
                item = part.strip(" \t\r\n()[]{}<>")
                item = re.sub(r"^[-*+]\s+", "", item).strip()
                if not item:
                    continue
                if item.casefold() in quality_terms:
                    continue
                # 质量词有时和其它词被模型黏在同一段中，逐项移除。
                item = re.sub(
                    r"(?i)\b(?:masterpiece|best quality|amazing quality|very aesthetic|"
                    r"extremely detailed|highly detailed|very detailed|absurdres|highres|"
                    r"newest|score_[1-9]|year\s+2024)\b",
                    " ",
                    item,
                ).strip(" ,;:：()[]{}")
                key = item.casefold()
                if not item or key in seen:
                    continue
                seen.add(key)
                kept.append(item)
            return ", ".join(kept).strip(" ,;:：")

        for candidate in candidates:
            clean = normalize(candidate)
            lowered = clean.casefold()
            if not clean or len(clean) > 800:
                continue
            if any(marker.casefold() in lowered for marker in rejected_markers):
                continue
            if re.search(r"[\u4e00-\u9fff]", clean):
                continue
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.'()/_-]*", clean):
                continue
            if len(re.findall(r"[A-Za-z]+", clean)) > 80:
                continue
            return clean
        raise AIError("文生图 AI 返回了分析内容，未提取到有效最终提示词")

    @staticmethod
    def _looks_like_txt2img_reasoning(value: str) -> bool:
        """判断 LLM 工具传入的文生图文本是否混入了推理过程。"""
        text = str(value or "")
        if not text:
            return False
        return bool(
            re.search(
                r"<\/?(?:think|analysis|reasoning)>|"
                r"(?:分析用户请求|用户要求输出|需要包含|确定核心标签|核心标签|"
                r"最终提示词|最终结果|标签列表|人数与性别|角色[与/]作品|"
                r"reasoning|analysis|thinking)",
                text,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _sanitize_llm_txt2img_prompt(
        cls,
        value: str,
        fallback: str = "",
    ) -> str:
        """在 LLM 工具入口再次截断推理，防止未启用插件 AI 时漏过清洗。"""
        text = str(value or "").strip()
        if not cls._looks_like_txt2img_reasoning(text):
            return text
        try:
            return cls._clean_txt2img_prompt(text)
        except AIError:
            # 原始请求通常是中文自然语言，后续 _prompt_text 会按文生图
            # 的普通翻译规则处理；宁可回退原请求，也不能把分析过程送入模型。
            original = str(fallback or "").strip()
            if original and not cls._looks_like_txt2img_reasoning(original):
                return original
            raise

    @staticmethod
    def _clean_img2img_edit_instruction(value: str) -> str:
        """提取简短图生图编辑句，兼容旧版英文结果但优先接受中文。"""
        raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip(" `\n\t")
        if not raw:
            raise AIError("图生图插件 AI 返回了空内容")

        candidates: list[str] = []
        marker = re.compile(
            r"(?:^|[,，;；\n])\s*(?:edit|final(?:\s+edit)?|instruction|编辑指令|最终(?:编辑指令|结果)?)\s*[:：]\s*([^\n]+)",
            re.IGNORECASE,
        )
        candidates.extend(match.group(1) for match in marker.finditer(raw))

        # 兼容插件 AI 返回 JSON 或 Markdown JSON；只读取明确的编辑字段。
        for block in re.findall(r"\{.*?\}", raw, flags=re.DOTALL):
            try:
                data = json.loads(block)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict):
                for key in ("edit", "instruction", "edit_instruction", "prompt", "result", "最终编辑指令"):
                    item = data.get(key)
                    if isinstance(item, str) and item.strip():
                        candidates.insert(0, item)

        # 推理模型有时直接回显旧版“用户原话 + 画面描述”格式，只取用户原话。
        metadata_match = re.search(
            r"(?:user(?:'s)?\s+original\s+words|用户原话)\s*[:：]\s*(.*?)(?=\s+(?:astrbot\s+llm\s+picture\s+description|astrbot\s+llm\s+画面描述|astrbot\s+llm\s+description)\s*[:：]|$)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if metadata_match:
            candidates.insert(0, metadata_match.group(1))
        # 引号中的短句通常是模型真正给出的结果，放在逐行候选之前。
        candidates.extend(re.findall(r"[\"“'`]([^\"”'`\n]{3,240})[\"”'`]", raw))
        candidates.extend(line.strip() for line in raw.split("\n") if line.strip())
        if "\n" not in raw:
            candidates.append(raw)

        disallowed = (
            "we need", "user's original", "astrbot llm", "picture description",
            "system prompt", "analysis", "reasoning", "because", "therefore",
            "用户原话", "画面描述", "系统提示", "分析用户", "分析过程", "推理过程",
        )
        generic = {
            "change her pose", "change the pose", "change her expression", "change the expression",
            "change the outfit", "change her outfit", "改姿势", "改变姿势", "换个姿势",
            "改动作", "修改动作", "改衣服", "换衣服", "改表情", "换表情",
        }
        for candidate in candidates:
            clean = re.sub(
                r"^(?:edit|final(?:\s+edit)?|instruction|用户原话|画面描述|最终编辑指令)\s*[:：]\s*",
                "",
                str(candidate),
                flags=re.IGNORECASE,
            )
            clean = re.split(
                r"\s+(?:astrbot\s+llm\s+picture\s+description|astrbot\s+llm\s+画面描述|astrbot\s+llm\s+description)\s*[:：]",
                clean,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            clean = re.sub(r"[,，、;；]\s*(?:other\s+)?unchanged\s*$", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s+(?:其余|其它|其他)不变\s*$", "", clean)
            clean = re.sub(
                r"(?i)(?:^|[,，]\s*)(?:masterpiece|best quality|highly detailed|ultra[- ]detailed|8k|4k|anime style)(?:\s*,\s*|$)",
                " ",
                clean,
            )
            clean = re.sub(r"\s+", " ", clean).strip(" `\"'，,。.;:：")
            if not clean or len(clean) > 300:
                continue
            lowered = clean.casefold()
            if any(marker in lowered for marker in disallowed) or lowered in generic:
                continue
            if len(re.findall(r"[A-Za-z]+", clean)) > 45:
                continue
            if contains_chinese(clean):
                return clean
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.'()/_-]*", clean):
                return clean
        raise AIError("图生图 AI 返回了分析内容，未提取到有效编辑指令")

    @staticmethod
    def _repair_img2img_edit_instruction(
        value: str,
        source_text: str,
        *,
        preserve_english: bool = False,
    ) -> str:
        """修正已知的中文动作直译，并防止空泛结果覆盖用户具体要求。"""
        result = str(value or "").strip()
        source = str(source_text or "").strip()
        if re.search(r"一字马|劈叉", source) and (
            re.search(r"\bhorse\b", result, re.IGNORECASE)
            or re.search(r"\bstand(?:ing)?\s+(?:on|onto)\b", result, re.IGNORECASE)
        ):
            direction = ""
            if "侧劈叉" in source or "横劈叉" in source:
                direction = "侧"
            elif "前劈叉" in source:
                direction = "前"
            return f"让人物站立完成{direction}一字马（劈叉）动作，其他内容保持原图不变"
        # 图生图 AI 的目标是输出中文编辑句，而不是把中文原话重新直译成
        # 一条可能改变语义的英文 prompt。中文原话存在且 AI 返回纯英文时，
        # 保留用户原意；Qwen Image Edit 可以直接理解这条中文短指令。
        if contains_chinese(source) and result and not contains_chinese(result) and not preserve_english:
            return f"{source}，其他内容保持原图不变"
        if result.casefold() in {
            "change her pose", "change the pose", "change her expression", "change the expression",
            "change the outfit", "change her outfit",
        } and source:
            return f"{source}，其他内容保持原图不变"
        return result

    @staticmethod
    def _finalize_img2img_instruction(value: str) -> str:
        """给 Qwen 的最终编辑指令只保留改动，并固定原图保护声明。"""
        text = str(value or "").strip()
        if not text:
            return ""
        # 普通 /图生图 指令可能没有经过 LLM 清洗，也要拦截上游误传的
        # 元信息和质量词，避免它们进入 TextEncodeQwenImageEditPlus。
        text = re.sub(
            r"^(?:edit|final(?:\s+edit)?|instruction)\s*[:：]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?:user(?:'s)?\s+original\s+words|用户原话)\s*[:：]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.split(
            r"\s+(?:astrbot\s+llm\s+picture\s+description|astrbot\s+llm\s+画面描述|astrbot\s+llm\s+description)\s*[:：]",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        text = re.sub(r"[,，、;；]\s*(?:other\s+)?unchanged\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(?:其余|其它|其他)不变\s*$", "", text)
        text = re.sub(
            r"(?i)(?:^|[,，]\s*)(?:masterpiece|best quality|highly detailed|ultra[- ]detailed|8k|4k|anime style)(?:\s*,\s*|$)",
            " ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip(" `\"'，,。.;:：")
        if not text:
            return ""
        if (
            IMG2IMG_PRESERVE_TEXT.casefold() not in text.casefold()
            and IMG2IMG_PRESERVE_TEXT_EN.casefold() not in text.casefold()
        ):
            text = (
                f"{text}，{IMG2IMG_PRESERVE_TEXT}"
                if contains_chinese(text)
                else f"{text}; {IMG2IMG_PRESERVE_TEXT_EN}"
            )
        return text

    async def _prompt_text(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        lora_trigger_words: list[str] | None = None,
        lora_prompt_values: list[str] | None = None,
        mode: str = "txt2img",
    ) -> tuple[str, str, str]:
        # 预设必须在任何翻译或扩写前识别，避免 AI 把预设名称翻译成普通描述。
        self._extract_inline_presets(params)
        original_img2img_text = str(params.prompt or "").strip()
        text = self.presets.expand(params.prompt)
        preset_values: list[str] = []
        for name in params.presets:
            value = self.presets.effective(name)
            if not value:
                raise UsageError(f"预设不存在：{name}")
            preset_values.append(value)
        # 结构化 LLM 工具的 prompt 已经是上游 LLM 或 WebUI 选择的插件 AI
        # 处理结果。这里再次按 ``ai`` 进入文生图翻译，会导致二次调用、
        # 推理文本污染和“LLM 触发却回退普通绘图”。
        use_ai = (
            not getattr(params, "llm_invocation", False)
            and (params.auto_ai or params.ai is True)
        )
        is_img2img = mode in {"img2img", "img2img_flux2"}
        plain_img2img_command = (
            is_img2img
            and params.command_invocation
            and not use_ai
            and not params.img2img_edit_instruction_ready
        )
        img2img_config_prefix = "img2img_flux2" if mode == "img2img_flux2" else "img2img"
        if is_img2img and params.img2img_edit_instruction_ready:
            # LLM 图生图的编辑结果是独立字段。params.prompt 可能仍是
            # AstrBot 工具传入的自然语言或旧版长描述，不能让它覆盖最终编辑指令。
            text = str(params.img2img_edit_instruction or text or "").strip()
        source = str(params.ai_source or ("astrbot" if params.auto_ai else "plugin")).lower()
        if is_img2img and use_ai:
            # 图生图不适用 Anima 文生图扩写；即使是 /图生图 ai，也使用
            # 专用编辑知识把需求收敛成“只改什么”。
            configured_source = str(
                self._get(
                    "img2img_flux2_llm_prompt_source"
                    if mode == "img2img_flux2"
                    else "img2img_llm_prompt_source",
                    "plugin",
                )
                or "plugin"
            ).lower()
            source = (
                "astrbot_img2img_flux2_llm"
                if mode == "img2img_flux2"
                and configured_source in {"astrbot", "astrbot_ai", "astrbot llm", "astrbot_llm"}
                else "plugin_img2img_flux2_llm"
                if mode == "img2img_flux2"
                else "astrbot_img2img_llm"
                if configured_source in {"astrbot", "astrbot_ai", "astrbot llm", "astrbot_llm"}
                else "plugin_img2img_llm"
            )
        note = ""
        # 预设或 LoRA 控制项已经消耗掉角色名时，不再对空文本调用 AI，
        # 避免模型凭空补出 Danbooru 身份词。
        if (
            text.strip()
            and not plain_img2img_command
            and not (is_img2img and params.img2img_edit_instruction_ready)
            and use_ai
            and (contains_chinese(text) or params.ai is True)
        ):
            try:
                text = await self._translate_prompt(
                    event,
                    text,
                    force_enabled=params.ai is True or params.auto_ai,
                    source_override=source,
                )
                if is_img2img:
                    text = self._repair_img2img_edit_instruction(
                        text,
                        self._remove_locked_control_terms(original_img2img_text, params),
                    )
                note = (
                    "图生图编辑要求已由插件 AI 整理为精简编辑指令"
                    if source == "plugin_img2img_llm"
                    else
                    "提示词已由插件 AI 根据 LLM 绘图要求整理为 Danbooru 标签"
                    if source == "plugin_llm"
                    else "提示词已由插件 AI 优化为 Danbooru 标签并修正角色词条"
                    if source == "plugin"
                    else "提示词已由 AstrBot 当前 AI 优化为 Danbooru 标签并修正角色词条"
                )
            except AIError as exc:
                # AI/noai 是可选增强功能，AI 服务异常不应阻断普通绘图。
                # 图生图只在 WebUI 明确开启独立翻译时才走翻译网站；默认把
                # 用户短句原样交给 Qwen，不能把长的 AstrBot 上下文送去翻译。
                logger.warning("[%s] AI 提示词处理失败，回退普通绘图：%s", PLUGIN_NAME, exc)
                note = f"AI 不可用，已按普通方式继续出图（{exc}）"
                text, plain_note = await self._plain_translate_prompt(text, mode=mode)
                if is_img2img:
                    text = self._repair_img2img_edit_instruction(
                        text,
                        self._remove_locked_control_terms(original_img2img_text, params),
                        preserve_english=True,
                    )
                if plain_note:
                    note = f"{note}；{plain_note}"
        elif (
            contains_chinese(text)
            and not plain_img2img_command
            and not (is_img2img and params.img2img_edit_instruction_ready)
            and not getattr(params, "llm_invocation", False)
        ):
            text, plain_note = await self._plain_translate_prompt(text, mode=mode)
            note = plain_note
        if is_img2img:
            # 最终提交前再做一次防御性收敛。这样即使普通指令、旧版
            # AstrBot 工具参数或自定义 AI 返回了整段描述，也不会把它
            # 当成文生图 prompt 送入 Qwen。
            if use_ai and not params.img2img_edit_instruction_ready:
                text = self._repair_img2img_edit_instruction(
                    text,
                    self._remove_locked_control_terms(original_img2img_text, params),
                )
            text = self._finalize_img2img_instruction(text)
        artist_name = "" if is_img2img else str(
            params.artist_preset
            or self._get("artist_preset", DEFAULT_ARTIST_PRESET_NAME)
            or ""
        ).strip()
        artist_text = ""
        if artist_name.casefold() not in {"无", "关闭", "禁用", "none", "off", "disable"}:
            if artist_name:
                artist_text = self.artist_presets.effective(artist_name)
                if not artist_text:
                    raise UsageError(f"画师串预设不存在：{artist_name}")
        # default_positive 是新的可编辑字段；quality_prefix 保留给旧配置兼容。
        if is_img2img:
            quality = str(self._get(f"{img2img_config_prefix}_default_positive", "") or "").strip()
        else:
            quality = str(
                self._get("default_positive", "")
                or self._get("quality_prefix", "")
                or ""
            ).strip()
        # 预设和 LoRA 专属预设是插件控制项，必须在翻译完成后重新放入最终
        # 提示词最前面。它们不会被 AI/普通翻译改写；CivitAI 触发词只展示，
        # 不参与绘图提示词合成。
        fragments = [*preset_values, *(lora_prompt_values or [])]
        # 图生图只保留用户显式命中的预设/LoRA、独立基础正面词和编辑要求。
        # 全局画师串属于文生图风格，不应擅自改变参考图。
        fragments.extend([quality, text] if is_img2img else [quality, artist_text, text])
        # 预设、LoRA 专属内容、默认正面词和 AI 结果可能来自不同入口。
        # 按逗号拆分后统一去重，避免“预设已展开一次、又作为控制项注入一次”。
        positive = self._merge_prompt_fragments(
            *(value.strip(" ,，") for value in fragments if value.strip(" ,，"))
        )
        negative = params.negative or str(
            self._get(f"{img2img_config_prefix}_default_negative", "")
            if is_img2img
            else self._get("default_negative", "")
            or ""
        )
        # 黑名单群不依赖中文关键词命中，直接把安全约束写入最终负面提示词。
        # 这样“低胸、露背”等边界描述也会按群聊策略处理，同时普通图片仍可生成。
        negative = self._apply_group_safety_negative(event, negative)
        return positive, negative, note

    def _config_dict(self) -> dict[str, Any]:
        keys = {"ai_base_url", "ai_api_key", "ai_model"}
        return {key: self._get(key, "") for key in keys}

    def _lora_alias(self, file_name: str) -> str:
        alias = str(self._lora_map_value(self.lora_aliases, file_name, "") or "").strip()
        return alias or Path(file_name).stem

    @staticmethod
    def _lora_path_match(value: str, available: list[str]) -> str | None:
        """按 ComfyUI 的实际 LoRA 名称匹配配置中的文件名。"""
        target = str(value or "").strip().replace("\\", "/").casefold()
        if not target:
            return None
        exact = {
            str(name).replace("\\", "/").casefold(): str(name)
            for name in available
            if str(name).strip()
        }
        if target in exact:
            return exact[target]
        basename = Path(target).name
        matches = [
            str(name) for name in available
            if Path(str(name).replace("\\", "/")).name.casefold() == basename
        ]
        return matches[0] if len(matches) == 1 else None

    def _style_lora_entries(
        self,
        available: list[str] | None = None,
        *,
        selected_only: bool = True,
    ) -> list[dict[str, Any]]:
        """读取画风 LoRA。

        ``selected_only=True`` 返回参与随机/全部自动选择的候选；
        ``selected_only=False`` 返回所有已经分类为“画风”的文件，供简称、
        专属预设和用户临时触发解析。分类边界与随机候选边界必须分开，
        否则未勾入随机列表的画风 LoRA 无法通过简称触发。
        """
        values = list(available or [])
        raw_list = self._get("style_lora_list", [])
        if isinstance(raw_list, str):
            raw_list = re.split(r"[,，\n]+", raw_list)
        if not isinstance(raw_list, (list, tuple)):
            raw_list = []
        configured_names = [str(item or "").strip() for item in raw_list if str(item or "").strip()]
        style_aliases = self._get("style_lora_aliases", {})
        style_aliases = style_aliases if isinstance(style_aliases, dict) else {}

        if selected_only:
            source_names = configured_names
        elif values:
            source_names = list(values) + configured_names
        else:
            source_names = configured_names + list(style_aliases.keys())
            source_names.extend(
                name
                for name, category in getattr(self, "lora_categories", {}).items()
                if str(category or "").strip() == "画风"
            )

        weights = self._get("style_lora_weights", {})
        weights = weights if isinstance(weights, dict) else {}
        # 画风简称是独立的稳定映射。旧版本曾把“画风2”等名称写进
        # lora_command_aliases 或 LoRA 专属预设，导致同一个简称同时命中
        # 两个文件。显式画风映射即使尚未同步到分类文件，也应保留为可触发
        # 的画风条目；随机候选仍然继续受“画风”分类和 style_lora_list 控制。
        style_alias_names: set[str] = set()
        for raw_name in style_aliases:
            configured_name = str(raw_name or "").strip()
            if not configured_name:
                continue
            actual_name = self._lora_path_match(configured_name, values) if values else configured_name
            if actual_name:
                style_alias_names.add(actual_name.replace("\\", "/").casefold())
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in source_names:
            item = str(raw or "").strip()
            if not item:
                continue
            name, separator, suffix = item.rpartition(":")
            configured_weight = suffix if separator and re.fullmatch(r"\d+(?:\.\d+)?", suffix) else ""
            configured_name = name if configured_weight else item
            actual = self._lora_path_match(configured_name, values) if values else configured_name
            if not actual:
                continue
            is_style_category = self._lora_category(actual) == "画风"
            is_explicit_style_alias = actual.replace("\\", "/").casefold() in style_alias_names
            if selected_only and not is_style_category:
                continue
            if not selected_only and not (is_style_category or is_explicit_style_alias):
                continue
            key = actual.replace("\\", "/").casefold()
            if key in seen:
                continue
            seen.add(key)
            weight_value = self._lora_map_value(weights, actual, configured_weight or 0.8)
            try:
                weight = max(0.0, min(2.0, float(weight_value)))
            except (TypeError, ValueError):
                weight = 0.8
            result.append({"file_name": actual, "weight": weight})
        return result

    def _style_lora_alias_map(self, available: list[str] | None = None) -> dict[str, str]:
        """返回稳定画风简称到实际文件名的唯一映射。

        画风简称曾经被同时保存到旧的手动简称/专属预设文件中。绘图时
        必须以 WebUI 的画风简称为准，否则一个输入可能加载多个 LoRA。
        键使用 casefold，值保留 ComfyUI 返回的原始文件名。
        """
        values = [str(item).strip() for item in (available or []) if str(item).strip()]
        result: dict[str, str] = {}
        for item in self._style_lora_entries(values, selected_only=False):
            actual = str(item.get("file_name", "") or "").strip()
            if not actual:
                continue
            for alias in self._style_lora_aliases_for(actual, values):
                key = str(alias or "").strip().casefold()
                if key:
                    result.setdefault(key, actual)
        return result

    def _lora_alias_pairs(self, available: list[str]) -> list[tuple[str, str]]:
        """构建简称映射，并让画风简称覆盖旧的重复简称。"""
        values = [str(item).strip() for item in available if str(item).strip()]
        style_alias_map = self._style_lora_alias_map(values)
        style_names = {
            str(item["file_name"]).replace("\\", "/").casefold()
            for item in self._style_lora_entries(values, selected_only=False)
        }
        ordered_names = [
            *[name for name in values if name.replace("\\", "/").casefold() in style_names],
            *[name for name in values if name.replace("\\", "/").casefold() not in style_names],
        ]
        result: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for name in ordered_names:
            for alias in self._lora_command_aliases_for(name, values):
                alias_text = str(alias or "").strip()
                alias_key = alias_text.casefold()
                if not alias_key:
                    continue
                preferred = style_alias_map.get(alias_key)
                if preferred and preferred.replace("\\", "/").casefold() != name.replace("\\", "/").casefold():
                    # 旧手动简称/专属预设与当前画风简称冲突时，保留 WebUI
                    # 画风页的映射，避免一次输入加载两个不同 LoRA。
                    continue
                pair = (alias_key, name.replace("\\", "/").casefold())
                if pair in seen:
                    continue
                seen.add(pair)
                result.append((alias_text, name))
        return result

    def _ensure_style_lora_aliases(
        self,
        available: list[str] | None = None,
        *,
        persist: bool = True,
    ) -> bool:
        """为没有自定义简称的画风 LoRA 分配稳定的“画风数字”。

        旧版本按列表下标动态生成简称，新增 LoRA 排到前面后会导致旧简称
        整体变化。这里只给没有映射的文件分配新编号，已有映射和自定义简称
        永远保留；编号按下载时间从旧到新分配，新增文件自然使用最大编号加一。
        """
        aliases = self._get("style_lora_aliases", {})
        aliases = dict(aliases) if isinstance(aliases, dict) else {}
        values = [str(item or "").strip() for item in (available or []) if str(item or "").strip()]
        raw_list = self._get("style_lora_list", [])
        if isinstance(raw_list, str):
            raw_list = re.split(r"[,，\n]+", raw_list)
        configured = list(raw_list) if isinstance(raw_list, (list, tuple)) else []
        source_names: list[str] = []
        seen: set[str] = set()

        def add_name(raw: Any) -> None:
            value = str(raw or "").strip()
            if not value:
                return
            name, separator, suffix = value.rpartition(":")
            if separator and re.fullmatch(r"\d+(?:\.\d+)?", suffix):
                value = name.strip()
            actual = self._lora_path_match(value, values) if values else value
            if not actual:
                return
            key = actual.replace("\\", "/").casefold()
            if key not in seen:
                seen.add(key)
                source_names.append(actual)

        for item in configured:
            add_name(item)
        for name, category in getattr(self, "lora_categories", {}).items():
            if str(category or "").strip() == "画风":
                add_name(name)
        for name in aliases:
            add_name(name)
        if not source_names:
            return False

        def mapped_value(name: str) -> Any:
            return self._lora_map_value(aliases, name, "")

        def has_mapping(name: str) -> bool:
            value = mapped_value(name)
            if isinstance(value, (list, tuple)):
                return any(str(item or "").strip() for item in value)
            return bool(str(value or "").strip())

        used_numbers: set[int] = set()
        for value in aliases.values():
            values_to_check = value if isinstance(value, (list, tuple)) else [value]
            for alias in values_to_check:
                match = re.fullmatch(r"画风(\d+)", str(alias or "").strip())
                if match:
                    used_numbers.add(int(match.group(1)))

        order = {name.replace("\\", "/").casefold(): index for index, name in enumerate(source_names)}

        def download_time(name: str) -> float:
            candidates = (name, name.replace("\\", "/"), Path(name).name)
            for key in candidates:
                try:
                    value = float(getattr(self, "lora_download_order", {}).get(key, 0) or 0)
                except (TypeError, ValueError):
                    value = 0
                if value > 0:
                    return value
            return 0.0

        unassigned = [name for name in source_names if not has_mapping(name)]
        unassigned.sort(
            key=lambda name: (
                0 if download_time(name) <= 0 else 1,
                download_time(name) if download_time(name) > 0 else order.get(name.replace("\\", "/").casefold(), 0),
            )
        )
        next_number = max(used_numbers, default=0) + 1
        changed = False
        for name in unassigned:
            while next_number in used_numbers:
                next_number += 1
            aliases[name] = f"画风{next_number}"
            used_numbers.add(next_number)
            next_number += 1
            changed = True
        if changed:
            self._set("style_lora_aliases", aliases)
            if persist:
                self._save_config()
        return changed

    def _style_lora_random_count(self, candidate_count: int = 0) -> int:
        """读取随机画风 LoRA 数量，并限制在实际候选数量内。

        数量为 1 是默认且已验证的路径；数量大于 1 时由调用方使用
        ``random.sample``，保证一次任务不会重复套用同一个文件。
        """
        try:
            value = int(self._get("style_lora_random_count", 1) or 1)
        except (TypeError, ValueError):
            value = 1
        value = max(1, min(16, value))
        if candidate_count > 0:
            value = min(value, candidate_count)
        return value

    def _style_lora_alias(self, file_name: str, index: int, available: list[str] | None = None) -> str:
        self._ensure_style_lora_aliases(available, persist=True)
        aliases = self._get("style_lora_aliases", {})
        aliases = aliases if isinstance(aliases, dict) else {}
        value = self._lora_map_value(aliases, file_name, "")
        if isinstance(value, (list, tuple)):
            value = next((item for item in value if str(item or "").strip()), "")
        return str(value or "").strip() or f"画风{index + 1}"

    def _style_lora_aliases_for(self, file_name: str, available: list[str] | None = None) -> list[str]:
        self._ensure_style_lora_aliases(available, persist=True)
        entries = self._style_lora_entries(available, selected_only=False)
        target = str(file_name or "").replace("\\", "/").casefold()
        for index, item in enumerate(entries):
            if str(item["file_name"]).replace("\\", "/").casefold() == target:
                aliases = self._get("style_lora_aliases", {})
                aliases = aliases if isinstance(aliases, dict) else {}
                value = self._lora_map_value(aliases, item["file_name"], "")
                if isinstance(value, str):
                    values = [value]
                elif isinstance(value, (list, tuple)):
                    values = list(value)
                else:
                    values = []
                result: list[str] = []
                for alias in values:
                    alias = str(alias or "").strip()
                    if alias and alias.casefold() not in {item.casefold() for item in result}:
                        result.append(alias)
                return result or [f"画风{index + 1}"]
        return []

    def _style_lora_mode(self) -> str:
        value = str(self._get("style_lora_mode", "random") or "random").strip().casefold()
        return value if value in {"off", "random", "all"} else "random"

    @staticmethod
    def _style_lora_disable_patterns() -> tuple[re.Pattern[str], ...]:
        return (
            re.compile(
                r"(?:不用|不使用|不要(?:使用|用|加)?|不加|关闭|禁用|取消|去掉|移除|禁止)"
                r"\s*(?:随机\s*)?画风(?:\s*(?:LoRA|lora))?",
                re.IGNORECASE,
            ),
            re.compile(r"(?:无|没有)\s*画风(?:\s*(?:LoRA|lora))?", re.IGNORECASE),
            re.compile(
                r"(?:no|without|disable|disabled|turn\s*off)\s*(?:the\s*)?style(?:\s*LoRA|\s*lora)?",
                re.IGNORECASE,
            ),
        )

    @classmethod
    def _strip_style_lora_disable_terms(cls, text: str) -> str:
        result = str(text or "")
        for pattern in cls._style_lora_disable_patterns():
            result = pattern.sub(" ", result)
        return re.sub(r"\s+", " ", result).strip(" ,，。；;")

    def _extract_style_lora_disable(self, params: DrawParams, text: str = "") -> bool:
        """读取本次请求的“不要画风”控制项，不改变长期 WebUI 配置。"""
        combined = " ".join(
            value for value in (str(getattr(params, "prompt", "") or ""), str(text or "")) if value.strip()
        )
        patterns = self._style_lora_disable_patterns()
        matched = any(pattern.search(combined) for pattern in patterns)
        if not matched:
            return bool(getattr(params, "style_lora_disabled", False))
        params.style_lora_disabled = True
        # 控制词不应被继续交给提示词翻译或工作流文本编码器。
        params.prompt = self._strip_style_lora_disable_terms(params.prompt)
        return True

    def _select_style_loras(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按配置决定本次实际使用的画风 LoRA，随机模式不重复抽取。"""
        mode = self._style_lora_mode()
        if mode == "off" or not entries:
            return []
        if mode == "all":
            return list(entries)
        return random.sample(entries, k=self._style_lora_random_count(len(entries)))

    def _style_lora_candidates_without_persistent(
        self,
        entries: list[dict[str, Any]],
        configured_loras: list[str],
        available: list[str],
        style_catalog: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """从随机候选中排除已经在 WebUI 中常态启用的画风 LoRA。"""
        style_names = {
            str(item["file_name"]).replace("\\", "/").casefold()
            for item in style_catalog
        }
        persistent_names: set[str] = set()
        for item in configured_loras:
            raw_name = str(item).rsplit(":", 1)[0].strip()
            actual = self._resolve_lora(raw_name, available)
            if actual and actual.replace("\\", "/").casefold() in style_names:
                persistent_names.add(actual.replace("\\", "/").casefold())
        return [
            item
            for item in entries
            if str(item["file_name"]).replace("\\", "/").casefold() not in persistent_names
        ]

    @staticmethod
    def _lora_map_value(mapping: dict[str, Any], file_name: str, default: Any = None) -> Any:
        """按 ComfyUI 返回的文件名读取旧版或不同斜杠格式的 LoRA 配置。"""
        if file_name in mapping:
            return mapping[file_name]
        target = str(file_name or "").replace("\\", "/").casefold()
        target_name = Path(target).name
        for key, value in mapping.items():
            normalized = str(key or "").replace("\\", "/").casefold()
            if normalized == target or Path(normalized).name == target_name:
                return value
        return default

    def _lora_command_aliases(self, file_name: str) -> list[str]:
        """返回一个 LoRA 的全部自定义指令简称，并迁移旧版单字符串配置。"""
        raw = self._lora_map_value(getattr(self, "lora_command_aliases", {}), file_name, [])
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple)):
            values = list(raw)
        else:
            values = []
        result: list[str] = []
        for value in values:
            alias = str(value or "").strip()
            if alias and alias.casefold() not in {item.casefold() for item in result}:
                result.append(alias)
        if raw != result and hasattr(self, "lora_command_aliases"):
            self.lora_command_aliases[file_name] = result
        return result

    def _lora_command_aliases_for(self, file_name: str, available: list[str] | None = None) -> list[str]:
        # LoRA 专属预设的左侧名称就是自动简称。它不写入手动简称文件，
        # 因此用户改名或删除专属预设后，旧简称会立即失效，不会残留。
        aliases = self._lora_command_aliases(file_name)
        for preset_tag in self._lora_preset_tags(file_name):
            if preset_tag.casefold() not in {item.casefold() for item in aliases}:
                aliases.append(preset_tag)
        for style_alias in self._style_lora_aliases_for(file_name, available):
            if style_alias.casefold() not in {item.casefold() for item in aliases}:
                aliases.append(style_alias)
        if aliases:
            return aliases
        names = self._order_loras(available or [file_name])
        try:
            index = names.index(file_name) + 1
        except ValueError:
            index = 1
        return [f"{index}号lora"]

    def _strip_lora_preset_aliases_from_manual(self) -> bool:
        """从手动简称文件剥离专属预设左侧名称，并保留其它简称。"""
        changed = False
        for file_name in list(self.lora_command_aliases):
            raw = self._lora_map_value(self.lora_command_aliases, file_name, [])
            values = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else []
            preset_keys = {tag.casefold() for tag in self._lora_preset_tags(file_name)}
            cleaned: list[str] = []
            for value in values:
                alias = str(value or "").strip()
                if not alias or alias.casefold() in preset_keys:
                    continue
                if alias.casefold() not in {item.casefold() for item in cleaned}:
                    cleaned.append(alias)
            if cleaned != values:
                self.lora_command_aliases[file_name] = cleaned
                changed = True
        return changed

    def _lora_command_alias(self, file_name: str, available: list[str] | None = None) -> str:
        return self._lora_command_aliases_for(file_name, available)[0]

    def _lora_category(self, file_name: str) -> str:
        categories = getattr(self, "lora_categories", {})
        categories = categories if isinstance(categories, dict) else {}
        category_entries = getattr(self, "lora_category_entries", {})
        category_entries = category_entries if isinstance(category_entries, dict) else {}
        category = str(self._lora_map_value(categories, file_name, "") or "").strip()
        return category if category in category_entries else "未分类"

    def _lora_categories_list(self) -> list[str]:
        values = {
            str(value).strip()
            for value in self.lora_category_entries
            if str(value).strip() and str(value).strip() != "未分类"
        }
        values.update(
            str(value).strip()
            for value in self.lora_categories.values()
            if str(value).strip() and str(value).strip() != "未分类"
        )
        return ["未分类", *sorted(values, key=str.casefold)]

    def _resolve_lora(self, value: str, available: list[str]) -> str | None:
        """Resolve a LoRA name, alias, or command alias.

        ComfyUI may return nested model paths with either slash direction,
        while older plugin defaults and saved settings commonly use the other
        one. Compare normalized paths first and return the exact spelling from
        the live ComfyUI list so the workflow receives a valid model name.
        """
        value = str(value or "").strip()
        if not value:
            return None
        normalized_value = value.replace("\\", "/").casefold()
        exact = {
            str(name).replace("\\", "/").casefold(): str(name)
            for name in available
            if str(name).strip()
        }
        if normalized_value in exact:
            return exact[normalized_value]
        # A saved setting may contain only the filename while ComfyUI returns
        # a nested path. Only use basename matching when it is unambiguous.
        basename = Path(normalized_value).name
        basename_matches = [
            str(name)
            for name in available
            if Path(str(name).replace("\\", "/")).name.casefold() == basename
        ]
        if len(basename_matches) == 1:
            return basename_matches[0]
        aliases = {self._lora_alias(name).lower(): name for name in available}
        resolved = aliases.get(value.lower())
        if resolved:
            return resolved
        # WebUI 的画风简称优先于旧版手动简称，避免“画风2”同时解析
        # 到多个文件。其余简称继续使用统一映射表。
        style_match = self._style_lora_alias_map(available).get(value.casefold())
        if style_match:
            return style_match
        command_aliases: dict[str, str] = {}
        for alias, name in self._lora_alias_pairs(available):
            command_aliases.setdefault(alias.casefold(), name)
        return command_aliases.get(value.lower())

    def _resolve_lora_command_alias_only(self, value: str, available: list[str]) -> str | None:
        """只按 LoRA 指令简称解析，避免误删普通提示词中的同名单词。"""
        target = str(value or "").strip().casefold()
        if not target:
            return None
        style_match = self._style_lora_alias_map(available).get(target)
        if style_match:
            return style_match
        for alias, name in self._lora_alias_pairs(available):
            if alias.casefold() == target:
                return name
        return None

    def _lora_trigger_words(self, file_name: str) -> list[str]:
        """读取仍未并入专属预设的 CivitAI 触发词，避免重复加入。"""
        preset_fragments = {
            item.casefold()
            for entry in self._lora_preset_entries_raw(file_name)
            for item in self._prompt_fragments(entry.get("content", ""))
        }
        candidates = [file_name, Path(file_name).name]
        for key in candidates:
            override = self.civitai_overrides.get(key)
            if isinstance(override, dict) and "trigger_words" in override:
                return [
                    item for item in self._normalize_trigger_words(override.get("trigger_words", []))
                    if item.casefold() not in preset_fragments
                ]
            cached = self.civitai_cache.get(key)
            if not isinstance(cached, dict):
                continue
            data = cached.get("data", cached)
            if isinstance(data, dict) and "trigger_words" in data:
                return [
                    item for item in self._normalize_trigger_words(data.get("trigger_words", []))
                    if item.casefold() not in preset_fragments
                ]
        return []

    def _lora_civitai_tags(self, file_name: str) -> list[str]:
        """读取 CivitAI 模型 tags；旧缓存没有 tags 时自然返回空列表。"""
        candidates = [file_name, Path(file_name).name]
        for key in candidates:
            override = getattr(self, "civitai_overrides", {}).get(key)
            if isinstance(override, dict) and "tags" in override:
                return self._normalize_civitai_tags(override.get("tags", []))
            cached = getattr(self, "civitai_cache", {}).get(key)
            if not isinstance(cached, dict):
                continue
            data = cached.get("data", cached)
            if isinstance(data, dict) and "tags" in data:
                return self._normalize_civitai_tags(data.get("tags", []))
        return []

    def _lora_preset_entries_raw(self, file_name: str) -> list[dict[str, Any]]:
        """读取单个 LoRA 的预设原始条目，保留内部同步字段。"""
        mapping = getattr(self, "lora_presets", {})
        raw = self._lora_map_value(mapping, file_name, []) if isinstance(mapping, dict) else []
        if isinstance(raw, dict):
            raw = raw.get("entries", raw.get("presets", raw))
        if isinstance(raw, dict):
            raw = [{"tag": key, "content": value} for key, value in raw.items()]
        if not isinstance(raw, list):
            return []
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            source_item = item if isinstance(item, dict) else {}
            if isinstance(item, dict):
                tag = str(item.get("tag", item.get("civitai_tag", "")) or "").strip()
                content = str(item.get("content", item.get("value", "")) or "").strip()
            elif isinstance(item, str):
                tag, content = item.strip(), item.strip()
            else:
                continue
            if not tag and not content:
                continue
            if len(tag) > 160 or len(content) > 4000:
                continue
            key = (tag.casefold(), content.casefold())
            if key in seen:
                continue
            seen.add(key)
            entry: dict[str, Any] = {"tag": tag, "content": content}
            # 这些字段不在 WebUI 显示，但用来区分“全局预设同步内容”和
            # “CivitAI 触发词同步内容”，因此修改或删除时不会误删另一部分。
            if isinstance(source_item.get("_global_content"), str):
                entry["_global_content"] = source_item["_global_content"]
            if isinstance(source_item.get("_civitai_content"), list):
                entry["_civitai_content"] = self._normalize_trigger_words(source_item["_civitai_content"])
            result.append(entry)
        return result

    def _lora_preset_visible_content(self, entry: dict[str, Any]) -> str:
        """隐藏旧版自动同步的 CivitAI 片段，保留用户和全局预设内容。"""
        content = self._prompt_fragments(entry.get("content", ""))
        civitai_content = {
            item.casefold()
            for raw in self._normalize_trigger_words(entry.get("_civitai_content", []))
            for item in self._prompt_fragments(raw)
        }
        return self._merge_prompt_fragments(
            [item for item in content if item.casefold() not in civitai_content]
        )

    def _lora_preset_entries(self, file_name: str) -> list[dict[str, str]]:
        """读取单个 LoRA 的可见简称和预设内容。

        旧版本可能已经把 CivitAI trainedWords 写进 lora_presets.json；这些
        片段仍保留在磁盘上以便回退，但在界面和最终提示词中只显示用户内容。
        """
        result: list[dict[str, str]] = []
        for item in self._lora_preset_entries_raw(file_name):
            tag = str(item.get("tag", "")).strip()
            raw_content = str(item.get("content", "") or "").strip()
            content = self._lora_preset_visible_content(item)
            # 纯旧版 CivitAI 自动条目不再冒充用户专属预设；CivitAI 内容会
            # 在 LoRA 卡片的“CivitAI 提示词（仅显示）”区域单独展示。
            if not content and raw_content and item.get("_civitai_content"):
                continue
            if tag or content:
                result.append({"tag": tag, "content": content})
        return result

    @staticmethod
    def _prompt_fragments(value: Any) -> list[str]:
        """按提示词逗号/换行拆分并去重，保留带空格的 Danbooru tag。"""
        if isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = re.split(r"[,，\n]+", str(value or ""))
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = str(raw or "").strip(" ,，\n")
            key = item.casefold()
            if item and key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @classmethod
    def _merge_prompt_fragments(cls, *values: Any) -> str:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            for item in cls._prompt_fragments(value):
                if item.casefold() not in seen:
                    seen.add(item.casefold())
                    result.append(item)
        return ", ".join(result)

    def _known_lora_preset_keys(self) -> list[str]:
        keys: set[str] = set()
        for mapping in (
            getattr(self, "lora_presets", {}),
            getattr(self, "lora_command_aliases", {}),
            getattr(self, "lora_aliases", {}),
            getattr(self, "civitai_cache", {}),
            getattr(self, "civitai_overrides", {}),
        ):
            if isinstance(mapping, dict):
                keys.update(str(key) for key in mapping if str(key).strip())
        style_list = self._get("style_lora_list", [])
        if isinstance(style_list, str):
            style_list = re.split(r"[,，\n]+", style_list)
        if isinstance(style_list, (list, tuple)):
            for item in style_list:
                name = str(item or "").rsplit(":", 1)[0].strip()
                if name:
                    keys.add(name)
        style_aliases = self._get("style_lora_aliases", {})
        if isinstance(style_aliases, dict):
            keys.update(str(key) for key in style_aliases if str(key).strip())
        for name, category in getattr(self, "lora_categories", {}).items():
            if str(category or "").strip() == "画风" and str(name).strip():
                keys.add(str(name))
        return sorted(keys, key=str.casefold)

    def _lora_preset_tags(self, file_name: str) -> list[str]:
        return [
            str(item.get("tag", "")).strip()
            for item in self._lora_preset_entries(file_name)
            if str(item.get("tag", "")).strip()
        ]

    def _lora_preset_matches_name(self, file_name: str, name: str) -> bool:
        target = str(name or "").strip().casefold()
        if not target:
            return False
        aliases = self._lora_command_aliases(file_name)
        return any(value.casefold() == target for value in [*aliases, *self._lora_preset_tags(file_name)])

    def _lora_entry_for_tag(self, entries: list[dict[str, Any]], tag: str) -> dict[str, Any] | None:
        target = str(tag or "").strip().casefold()
        for entry in entries:
            if str(entry.get("tag", "")).strip().casefold() == target:
                return entry
        return None

    def _sync_lora_global_entry(self, file_name: str, tag: str, content: str) -> bool:
        """把一个全局预设同步进对应 LoRA 专属预设。"""
        tag = str(tag or "").strip()
        content = str(content or "").strip()
        if not tag or not content:
            return False
        entries = self._lora_preset_entries_raw(file_name)
        entry = self._lora_entry_for_tag(entries, tag)
        if entry is None:
            entry = {"tag": tag, "content": ""}
            entries.append(entry)
        old_global = str(entry.get("_global_content", "") or "")
        current = self._prompt_fragments(entry.get("content", ""))
        if old_global:
            old_keys = {item.casefold() for item in self._prompt_fragments(old_global)}
            current = [item for item in current if item.casefold() not in old_keys]
        entry["_global_content"] = content
        # CivitAI trainedWords 只展示，不再并入专属预设或最终提示词。
        entry.pop("_civitai_content", None)
        entry["content"] = self._merge_prompt_fragments(current, content)
        if not hasattr(self, "lora_presets"):
            self.lora_presets = {}
        self.lora_presets[file_name] = entries
        return True

    def _sync_lora_civitai_entry(
        self,
        file_name: str,
        trigger_words: list[str],
        aliases: list[str] | None = None,
    ) -> bool:
        """把 CivitAI trainedWords 合并到 LoRA 专属预设，不写全局预设。"""
        trigger_words = self._normalize_trigger_words(trigger_words)
        if not trigger_words:
            return False
        aliases = list(aliases or self._lora_command_aliases(file_name))
        entries = self._lora_preset_entries_raw(file_name)
        # 已存在的同名专属预设优先；否则以第一个指令简称作为左侧名称。
        tag = next((item for item in aliases if item.strip()), "")
        if not tag:
            tag = next((str(item.get("tag", "")).strip() for item in entries if str(item.get("tag", "")).strip()), "")
        if not tag:
            tag = self._lora_alias(file_name)
        entry = self._lora_entry_for_tag(entries, tag)
        if entry is None:
            entry = {"tag": tag, "content": ""}
            entries.append(entry)
        old_civitai = self._normalize_trigger_words(entry.get("_civitai_content", []))
        entry["_civitai_content"] = trigger_words
        current = self._prompt_fragments(entry.get("content", ""))
        old_keys = {item.casefold() for item in old_civitai}
        current = [item for item in current if item.casefold() not in old_keys]
        entry["content"] = self._merge_prompt_fragments(current, trigger_words)
        if not hasattr(self, "lora_presets"):
            self.lora_presets = {}
        self.lora_presets[file_name] = entries
        return True

    def _remove_lora_global_link(self, name: str) -> bool:
        """删除全局预设对应的 LoRA 同名内容。"""
        changed = False
        target = str(name or "").strip().casefold()
        if not target:
            return False
        for file_name in self._known_lora_preset_keys():
            entries = self._lora_preset_entries_raw(file_name)
            next_entries: list[dict[str, Any]] = []
            file_changed = False
            for entry in entries:
                if str(entry.get("tag", "")).strip().casefold() != target:
                    next_entries.append(entry)
                    continue
                old_global = str(entry.get("_global_content", "") or "")
                if old_global:
                    current = [
                        item for item in self._prompt_fragments(entry.get("content", ""))
                        if item.casefold() not in {part.casefold() for part in self._prompt_fragments(old_global)}
                    ]
                    entry.pop("_global_content", None)
                    entry.pop("_civitai_content", None)
                    entry["content"] = self._merge_prompt_fragments(current)
                    if entry["content"]:
                        next_entries.append(entry)
                else:
                    file_changed = True
                file_changed = True
            if file_changed:
                if next_entries:
                    self.lora_presets[file_name] = next_entries
                else:
                    self.lora_presets.pop(file_name, None)
                changed = True
        return changed

    def _sync_global_preset_links(self, name: str, content: str) -> bool:
        """把新增/修改的全局预设同步到所有同名 LoRA 专属预设。"""
        changed = False
        for file_name in self._known_lora_preset_keys():
            if self._lora_preset_matches_name(file_name, name):
                changed = self._sync_lora_global_entry(file_name, name, content) or changed
        return changed

    def _sync_all_preset_links(self) -> bool:
        """启动时迁移旧数据，并建立全局预设与 LoRA 专属预设的关联。"""
        changed = False
        # 先把全局预设写入已有同名 LoRA 条目。
        for name in self.presets.items:
            content = self.presets.effective(name)
            changed = self._sync_global_preset_links(name, content) or changed
        if changed:
            self._save_lora_presets()
        return changed

    def _sync_globals_from_lora_entries(
        self,
        file_name: str,
        old_entries: list[dict[str, Any]],
        new_entries: list[dict[str, Any]],
    ) -> bool:
        """LoRA 专属预设编辑后，反向同步已经存在的同名全局预设。"""
        changed = False
        old_tags = {
            str(item.get("tag", "")).strip().casefold()
            for item in old_entries
            if str(item.get("tag", "")).strip()
        }
        new_by_tag = {
            str(item.get("tag", "")).strip().casefold(): item
            for item in new_entries
            if str(item.get("tag", "")).strip()
        }
        # 删除 LoRA 专属条目时，同名全局预设也删除；CivitAI 独立触发词
        # 后续会由同步函数重新保留为专属条目。
        for old_tag in old_tags - set(new_by_tag):
            name = next(
                (str(item.get("tag", "")).strip() for item in old_entries
                 if str(item.get("tag", "")).strip().casefold() == old_tag),
                "",
            )
            if name in self.presets.items:
                del self.presets.items[name]
                changed = True
        for key, entry in new_by_tag.items():
            name = str(entry.get("tag", "")).strip()
            if name not in self.presets.items:
                # 指令简称本身不强制创建全局预设；只有用户原来已经有
                # 同名全局预设时，编辑 LoRA 专属内容才反向同步它。
                continue
            content = str(entry.get("content", "") or "").strip()
            current = self.presets.items.get(name)
            if not content or not isinstance(current, dict):
                continue
            if current.get("content") != content or current.get("translated"):
                self.presets.items[name] = {"content": content, "translated": ""}
                changed = True
        if changed:
            self.presets.save()
        return changed

    def _save_linked_preset_change(
        self,
        *,
        old_name: str = "",
        name: str,
        content: str,
    ) -> None:
        """保存全局预设后同步所有同名 LoRA 专属预设。"""
        old_name = str(old_name or "").strip()
        name = str(name or "").strip()
        if old_name and old_name.casefold() != name.casefold():
            self._remove_lora_global_link(old_name)
        linked_content = self.presets.effective(name) or content
        self._sync_global_preset_links(name, linked_content)
        self._save_lora_presets()

    @staticmethod
    def _normalize_lora_preset_entries(value: Any) -> list[dict[str, str]]:
        if isinstance(value, dict):
            value = value.get("entries", value.get("presets", value))
        if isinstance(value, dict):
            value = [{"tag": key, "content": item} for key, item in value.items()]
        if not isinstance(value, list):
            raise UsageError("LoRA 预设必须是列表")
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in value:
            if not isinstance(item, dict):
                raise UsageError("LoRA 预设每一项必须包含 tag 和 content")
            tag = str(item.get("tag", item.get("civitai_tag", "")) or "").strip()
            content = str(item.get("content", item.get("value", "")) or "").strip()
            if not tag and not content:
                continue
            if len(tag) > 160 or len(content) > 4000:
                raise UsageError("LoRA tag 不能超过 160 个字符，预设内容不能超过 4000 个字符")
            key = (tag.casefold(), content.casefold())
            if key in seen:
                continue
            seen.add(key)
            result.append({"tag": tag, "content": content})
        return result

    def _lora_prompt_values(self, file_name: str) -> list[str]:
        """生成当前 LoRA 的专属预设内容，不包含 CivitAI 自动触发词。"""
        values: list[str] = []
        for entry in self._lora_preset_entries(file_name):
            # 左侧简称只是这条预设的分类/触发标识，右侧才是实际加入的内容。
            value = entry["content"]
            if value:
                values.append(value)
        return self._normalize_civitai_tags(values)

    def _lora_prompt_values_for_tags(self, file_name: str, tags: list[str]) -> list[str]:
        """只生成当前任务命中的 LoRA 专属预设内容。"""
        wanted = {str(tag or "").strip().casefold() for tag in tags if str(tag or "").strip()}
        if not wanted:
            return []
        values: list[str] = []
        for entry in self._lora_preset_entries(file_name):
            tag = str(entry.get("tag", "") or "").strip()
            if tag.casefold() not in wanted:
                continue
            content = str(entry.get("content", "") or "").strip()
            if content:
                values.append(content)
        return self._normalize_civitai_tags(values)

    def _lora_preset_tag_for_alias(self, file_name: str, alias: str) -> str | None:
        """返回简称对应的专属预设左侧名称，未命中则返回 None。"""
        target = str(alias or "").strip().casefold()
        if not target:
            return None
        for tag in self._lora_preset_tags(file_name):
            if tag.casefold() == target:
                return tag
        return None

    def _remember_lora_preset_match(
        self,
        params: DrawParams,
        file_name: str,
        alias: str,
    ) -> None:
        """记录本次任务命中的 LoRA 控制项和专属预设。"""
        mapping = getattr(params, "lora_preset_tags", None)
        if not isinstance(mapping, dict):
            mapping = {}
            params.lora_preset_tags = mapping
        key = str(file_name or "").replace("\\", "/").casefold()
        values = mapping.setdefault(key, [])
        if not isinstance(values, list):
            values = []
            mapping[key] = values
        tag = self._lora_preset_tag_for_alias(file_name, alias)
        if not tag:
            return
        if tag.casefold() not in {str(value).casefold() for value in values}:
            values.append(tag)

    def _lora_prompt_values_for_task(self, params: DrawParams, file_name: str) -> list[str]:
        """按任务命中的简称读取专属预设；无命中记录时保留旧行为。"""
        mapping = getattr(params, "lora_preset_tags", None)
        if isinstance(mapping, dict):
            target = str(file_name or "").replace("\\", "/").casefold()
            for key, tags in mapping.items():
                if str(key or "").replace("\\", "/").casefold() == target:
                    return self._lora_prompt_values_for_tags(file_name, tags if isinstance(tags, list) else [])
        return self._lora_prompt_values(file_name)

    @staticmethod
    def _fuzzy_alias_normalize(value: str) -> str:
        """把简称和用户原话归一化，忽略全半角、空格和标点差异。"""
        normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
        return "".join(
            char
            for char in normalized
            if char.isalnum() or "\u3400" <= char <= "\u9fff"
        )

    @staticmethod
    def _fuzzy_edit_distance(left: str, right: str) -> int:
        """计算很短文本的编辑距离；简称数量很小，不需要第三方库。"""
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, 1):
            current = [left_index]
            for right_index, right_char in enumerate(right, 1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1]

    @classmethod
    def _fuzzy_alias_similarity(
        cls,
        alias: str,
        candidate: str,
        *,
        allow_short_substitution: bool = False,
    ) -> float | None:
        """返回简称与候选片段的相似度，不满足保守阈值时返回 None。"""
        alias_key = cls._fuzzy_alias_normalize(alias)
        candidate_key = cls._fuzzy_alias_normalize(candidate)
        if not alias_key or not candidate_key:
            return None
        alias_length = len(alias_key)
        # 单字简称无法可靠判断；两字简称只允许增删一个字符，避免把
        # 普通描述中的同音/近形词误判成 LoRA。
        if alias_length < 2:
            return None
        maximum_distance = 1 if alias_length <= 6 else 2
        distance = cls._fuzzy_edit_distance(alias_key, candidate_key)
        if distance > maximum_distance:
            return None
        ratio = difflib.SequenceMatcher(
            None,
            alias_key,
            candidate_key,
            autojunk=False,
        ).ratio()
        if alias_length == 2:
            # 两字简称的单字符替换只有 0.5 相似度，直接拒绝；保留
            # “小爱”/“小小爱”这类自然语言增字变化。只有原话明确处于
            # 绘图/调用语境时，才允许两字简称的一字错写。
            if ratio < 0.74 and not (
                allow_short_substitution
                and len(alias_key) == len(candidate_key)
                and distance == 1
                and ratio >= 0.5
            ):
                return None
            if abs(len(alias_key) - len(candidate_key)) > 1:
                return None
        elif alias_length <= 3:
            if ratio < 0.64:
                return None
        elif alias_length <= 6:
            if ratio < 0.72:
                return None
        elif ratio < 0.78:
            return None
        return ratio

    @classmethod
    def _fuzzy_alias_windows(
        cls,
        text: str,
        alias: str,
    ) -> list[tuple[str, int, int]]:
        """生成简称附近的有限候选窗口，避免把整句拿来比较。"""
        normalized_text = unicodedata.normalize("NFKC", str(text or "")).casefold()
        alias_key = cls._fuzzy_alias_normalize(alias)
        if not alias_key:
            return []
        maximum_distance = 1 if len(alias_key) <= 6 else 2
        windows: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int, int]] = set()
        if re.search(r"[\u3400-\u9fff]", alias):
            compact_text = cls._fuzzy_alias_normalize(normalized_text)
            minimum_length = max(1, len(alias_key) - maximum_distance)
            maximum_length = len(alias_key) + maximum_distance
            for length in range(minimum_length, maximum_length + 1):
                for start in range(0, max(0, len(compact_text) - length + 1)):
                    end = start + length
                    item = (compact_text[start:end], start, end)
                    if item not in seen:
                        seen.add(item)
                        windows.append(item)
            return windows

        # 纯英文/数字简称按完整单词组合比较，防止 alias=art 匹配到 artist。
        tokens = [match.group(0) for match in re.finditer(r"[a-z0-9]+", normalized_text)]
        alias_tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", alias).casefold())
        alias_token_count = max(1, len(alias_tokens))
        minimum_tokens = max(1, alias_token_count - 1)
        maximum_tokens = alias_token_count + 1
        for start in range(len(tokens)):
            for count in range(minimum_tokens, maximum_tokens + 1):
                end = start + count
                if end > len(tokens):
                    continue
                candidate = "".join(tokens[start:end])
                item = (candidate, start, end)
                if item not in seen:
                    seen.add(item)
                    windows.append(item)
        return windows

    @classmethod
    def _fuzzy_nearby_draw_context(cls, text: str, start: int, end: int) -> bool:
        """判断短简称附近是否确实出现了绘图或 LoRA 调用语境。"""
        compact_text = cls._fuzzy_alias_normalize(text)
        nearby = compact_text[max(0, start - 8): min(len(compact_text), end + 8)]
        return bool(
            re.search(
                r"(?:画|绘|图|生成|图片|图像|使用|用上|加载|开启|打开|启用|调用|指令|简称|lora)",
                nearby,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _fuzzy_lora_alias_matches(
        cls,
        text: str,
        alias_pairs: list[tuple[str, str]],
    ) -> list[tuple[str, str, float, int, int]]:
        """返回非重叠的最佳简称候选，结果按置信度从高到低排列。"""
        matches: list[tuple[str, str, float, int, int, int]] = []
        seen_aliases: set[str] = set()
        for order, (alias, actual) in enumerate(alias_pairs):
            alias = str(alias or "").strip()
            actual = str(actual or "").strip()
            alias_key = cls._fuzzy_alias_normalize(alias)
            if not alias_key or alias_key in seen_aliases:
                continue
            seen_aliases.add(alias_key)
            # 拉丁字母简称至少三个字符；中文/混合简称至少两个字符。
            has_cjk = bool(re.search(r"[\u3400-\u9fff]", alias))
            if len(alias_key) < (2 if has_cjk else 3):
                continue
            best: tuple[float, int, int, str] | None = None
            for candidate, start, end in cls._fuzzy_alias_windows(text, alias):
                score = cls._fuzzy_alias_similarity(
                    alias,
                    candidate,
                    allow_short_substitution=cls._fuzzy_nearby_draw_context(text, start, end),
                )
                if score is None:
                    continue
                candidate_key = cls._fuzzy_alias_normalize(candidate)
                item = (score, -abs(len(alias_key) - len(candidate_key)), -start, candidate)
                if best is None or item > best:
                    best = item
            if best is None:
                continue
            score, _, _, candidate = best
            windows = [
                (item_start, item_end)
                for item_candidate, item_start, item_end in cls._fuzzy_alias_windows(text, alias)
                if item_candidate == candidate
            ]
            if not windows:
                continue
            start, end = windows[0]
            matches.append((alias, actual, score, start, end, order))

        # 先保留最像、最长的候选；同一片文本只允许一个 LoRA，
        # 但用户在一句话中明确写了两个相近简称时，非重叠候选仍可同时生效。
        matches.sort(key=lambda item: (-item[2], -len(cls._fuzzy_alias_normalize(item[0])), item[5]))
        accepted: list[tuple[str, str, float, int, int]] = []
        for alias, actual, score, start, end, _ in matches:
            if any(start < used_end and end > used_start for _, _, _, used_start, used_end in accepted):
                continue
            accepted.append((alias, actual, score, start, end))
        return accepted

    async def _extract_inline_loras(
        self,
        params: DrawParams,
        extra_text: str = "",
        *,
        allow_fuzzy: bool = False,
    ) -> None:
        """提取提示词和用户原话中的简称，并只对当前任务临时加载。

        LLM 工具经常只把英文提示词放进 prompt，而把用户原话里的 LoRA 简称
        丢在工具参数之外。因此这里同时检查当前消息原文，保证第二简称也能命中。
        `allow_fuzzy` 只由 LLM 自然语言路径开启，普通斜杠指令保持精确匹配。
        """
        self._extract_style_lora_disable(params, extra_text)
        # 没有任何 LoRA 参数或已知简称时，不查询 ComfyUI。这个快速路径是
        # LLM 自然语言绘图的首响应关键：普通角色请求不应先等待模型列表接口。
        # 预设先于 LoRA 解析时，LLM 可能已经把简称从 prompt 中移除，
        # 但正式预设名仍保留在 params.presets。把它也纳入控制项输入，
        # 让“同名预设 + LoRA 简称”在 LLM 路径中同时生效。
        preset_text = " ".join(str(value or "") for value in params.presets)
        source_text = f"{params.prompt} {extra_text} {preset_text}".casefold()
        known_aliases = [
            alias
            for name in self._known_lora_preset_keys()
            for alias in self._lora_command_aliases_for(name)
            if str(alias or "").strip()
        ]
        fuzzy_source_text = str(extra_text or "").strip()
        if allow_fuzzy and not fuzzy_source_text:
            fuzzy_source_text = str(params.prompt or "").strip()
        known_alias_pairs = [
            (alias, name)
            for name in self._known_lora_preset_keys()
            for alias in self._lora_command_aliases_for(name)
            if str(alias or "").strip()
        ]
        has_fuzzy_candidate = bool(
            allow_fuzzy
            and fuzzy_source_text
            and self._fuzzy_lora_alias_matches(fuzzy_source_text, known_alias_pairs)
        )
        if not params.loras and not any(str(alias).casefold() in source_text for alias in known_aliases) and not has_fuzzy_candidate:
            return

        available = await self._available_loras()
        extracted: list[str] = []
        remaining: list[str] = []

        # 结构化 LLM 工具可能已经把简称放入 params.loras，先把对应的
        # 专属预设记下来，后面的提示词生成仍然只使用这条简称。
        for item in list(params.loras):
            raw_name = str(item or "").rsplit(":", 1)[0].strip()
            actual = self._resolve_lora(raw_name, available)
            if actual:
                self._remember_lora_preset_match(params, actual, raw_name)

        def existing_names() -> set[str]:
            result: set[str] = set()
            for value in params.loras:
                raw_name = str(value).rsplit(":", 1)[0].strip()
                resolved = self._resolve_lora(raw_name, available)
                result.add((resolved or raw_name).casefold())
            return result

        def append_lora(actual: str, weight: str) -> None:
            if actual.casefold() not in existing_names():
                item = f"{actual}:{weight}"
                params.loras.append(item)
                extracted.append(item)

        def resolve_piece(piece: str) -> tuple[str | None, str | None, str | None]:
            value = piece.strip(" \t\r\n,，")
            name, separator, possible_weight = value.rpartition(":")
            weight = "0.8"
            if separator and re.fullmatch(r"\d+(?:\.\d+)?", possible_weight):
                value, weight = name.strip(), possible_weight
            actual = self._resolve_lora_command_alias_only(value, available)
            if not actual:
                return None, None, None
            # 同名普通提示词预设要保留给 PresetStore 展开，这样一个词可以同时
            # 触发临时 LoRA 和提示词预设；带权重时保留不带权重的预设名称。
            keep_for_prompt_preset = value if value in self.presets.items else None
            preset_tag = self._lora_preset_tag_for_alias(actual, value)
            return f"{actual}:{weight}", keep_for_prompt_preset, preset_tag

        for token in params.prompt.split():
            pieces = [piece for piece in re.split(r"[,，]", token) if piece.strip()]
            resolved = [resolve_piece(piece) for piece in pieces]
            if pieces and all(item[0] is not None for item in resolved):
                for item in resolved:
                    if item[0] is not None:
                        actual_name, _, actual_weight = item[0].rpartition(":")
                        append_lora(actual_name, actual_weight)
                        if item[2] is not None:
                            self._remember_lora_preset_match(params, actual_name, item[2])
                kept = [item[1] for item in resolved if item[1] is not None]
                if kept:
                    remaining.append(", ".join(kept))
            else:
                remaining.append(token)
        if extracted:
            params.prompt = " ".join(remaining).strip()

        # LLM 传入的自然语言经常没有空格，例如“使用1号lora画夏空”。
        # 对指令简称做一次安全的独立匹配，使它和 /文生图 末尾写简称的行为一致。
        style_names = {
            str(item["file_name"]).replace("\\", "/").casefold()
            for item in self._style_lora_entries(available, selected_only=False)
        }
        alias_pairs = sorted(
            self._lora_alias_pairs(available),
            key=lambda pair: (
                0 if str(pair[1]).replace("\\", "/").casefold() in style_names else 1,
                -len(pair[0]),
            ),
        )
        exact_prompt_aliases: set[str] = set()
        for alias, actual in alias_pairs:
            if not alias:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?::(?P<weight>\d+(?:\.\d+)?))?(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )

            def replace(match: re.Match[str]) -> str:
                weight = match.group("weight") or "0.8"
                append_lora(actual, weight)
                self._remember_lora_preset_match(params, actual, alias)
                exact_prompt_aliases.add(alias.casefold())
                return alias if alias in self.presets.items else " "

            params.prompt = pattern.sub(replace, params.prompt)

        # extra_text 是当前用户原话，只用于识别控制项，绝不把它混入最终提示词。
        # 这样 LLM 即使只传英文 prompt，也不会丢掉用户原话中的第二简称。
        source_text = str(extra_text or "").strip()
        if source_text:
            for alias, actual in alias_pairs:
                if not alias:
                    continue
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?::(?P<weight>\d+(?:\.\d+)?))?(?![A-Za-z0-9_])",
                    re.IGNORECASE,
                )
                for match in pattern.finditer(source_text):
                    append_lora(actual, match.group("weight") or "0.8")
                    self._remember_lora_preset_match(params, actual, alias)
                    exact_prompt_aliases.add(alias.casefold())

        # 只对 LLM 原话进行模糊匹配。精确简称已经在上面处理过，
        # 因此这里不会用一个近似词覆盖已有的精确命中。
        if allow_fuzzy and fuzzy_source_text:
            fuzzy_pairs = [
                (alias, actual)
                for alias, actual in alias_pairs
                if alias.casefold() not in exact_prompt_aliases
                and not any(
                    self._fuzzy_alias_similarity(
                        alias,
                        exact_alias,
                        allow_short_substitution=False,
                    ) is not None
                    for exact_alias in exact_prompt_aliases
                    if alias.casefold() != exact_alias
                )
            ]
            for alias, actual, _, _, _ in self._fuzzy_lora_alias_matches(fuzzy_source_text, fuzzy_pairs):
                if actual.casefold() in existing_names():
                    continue
                append_lora(actual, "0.8")
                self._remember_lora_preset_match(params, actual, alias)

        # 预设名称和 LoRA 指令简称相同时，预设不能吞掉 LoRA 控制项。
        # 这里只按完整简称匹配，不会因为普通提示词中的英文标签误加载 LoRA。
        # 如果这个名称没有对应的全局提示词预设，它只是用户通过
        # `--预设=简称` 或 LLM preset 参数传入的 LoRA 简称，不应在后续
        # _prompt_text 中被误判为“预设不存在”。同名的全局预设仍保留，
        # 这样“同名简称 + 普通预设”继续同时生效。
        lora_only_presets: set[str] = set()
        for preset_name in list(params.presets):
            target = str(preset_name or "").strip().casefold()
            if not target:
                continue
            for alias, actual in alias_pairs:
                if alias.casefold() == target:
                    append_lora(actual, "0.8")
                    self._remember_lora_preset_match(params, actual, alias)
                    if self._resolve_preset_name(str(preset_name)) is None:
                        lora_only_presets.add(target)
                    break
        if lora_only_presets:
            params.presets = [
                value
                for value in params.presets
                if str(value or "").strip().casefold() not in lora_only_presets
            ]
        params.prompt = re.sub(r"\s+", " ", params.prompt).strip(" ,，。；;")

    @staticmethod
    def _preset_alias_found(text: str, alias: str) -> bool:
        """按中文子串或英文标签边界判断预设是否出现。"""
        text = str(text or "")
        alias = str(alias or "").strip()
        if not text or not alias:
            return False
        if re.search(r"[\u4e00-\u9fff]", alias):
            return alias.casefold() in text.casefold()
        normalized = alias.replace(r"\(", "(").replace(r"\)", ")").strip()
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])", text, re.IGNORECASE):
            return True
        # 不拆分完整 Danbooru 标签。预设必须使用 WebUI 中保存的名称或完整内容，
        # 避免把 verina_(wuthering_waves) 隐式映射成用户没有设置的 verina。
        return False

    def _preset_aliases(self, name: str) -> list[str]:
        """返回预设名以及其英文内容中的稳定识别词。"""
        aliases = [str(name or "").strip()]
        effective = self.presets.effective(name)
        if effective:
            aliases.append(effective)
            # 只取第一个逗号前的稳定角色/主题 tag。后面的服装、颜色、
            # 发型等通用 tag 不能成为整个预设的触发词。
            first_fragment = re.split(r"[,，\n]+", effective, maxsplit=1)[0].strip()
            if first_fragment:
                aliases.append(first_fragment)
        result: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            key = alias.casefold()
            if alias and key not in seen:
                seen.add(key)
                result.append(alias)
        return result

    def _resolve_preset_name(self, value: str) -> str | None:
        """把 LLM 传入的中文名或英文预设内容解析为正式预设名。"""
        target = str(value or "").strip()
        if not target:
            return None
        for name in sorted(self.presets.items, key=len, reverse=True):
            if name.casefold() == target.casefold():
                return name
            if any(alias.casefold() == target.casefold() for alias in self._preset_aliases(name)[1:]):
                return name
        return None

    def _extract_inline_presets(self, params: DrawParams, extra_text: str = "") -> None:
        """从用户原话、LLM 英文提示词和结构化参数提取正式预设。"""
        normalized_presets: list[str] = []
        for value in params.presets:
            resolved = self._resolve_preset_name(value) or str(value or "").strip()
            if resolved and resolved not in normalized_presets:
                normalized_presets.append(resolved)
        params.presets = normalized_presets

        text = params.prompt
        original = str(extra_text or "")
        for name in sorted(self.presets.items, key=len, reverse=True):
            aliases = self._preset_aliases(name)
            prompt_aliases = [alias for alias in aliases if self._preset_alias_found(text, alias)]
            original_hit = any(self._preset_alias_found(original, alias) for alias in aliases)
            explicit_hit = name in params.presets
            if not prompt_aliases and not original_hit and not explicit_hit:
                continue
            if name not in params.presets:
                params.presets.append(name)
            # 只有命中 LLM prompt 的别名才从画面描述中移除；用户原话只负责
            # 锁定预设，不能把动作、服装或场景文字删掉。
            for alias in prompt_aliases:
                if re.search(r"[\u4e00-\u9fff]", alias):
                    pattern = re.compile(re.escape(alias), re.IGNORECASE)
                else:
                    normalized = alias.replace(r"\(", "(").replace(r"\)", ")").strip()
                    pattern = re.compile(
                        rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
                        re.IGNORECASE,
                    )
                text = pattern.sub(" ", text)
            # 已命中预设时，清掉 LLM 可能重新生成的同角色 Danbooru 身份词。
            # 预设内容由 WebUI 决定，不能被 AI 的角色词条替换或叠加。
            if (original_hit or explicit_hit) and not prompt_aliases:
                effective = self.presets.effective(name)
                first_fragment = re.split(r"[,，\n]+", effective, maxsplit=1)[0].strip()
                first_word = re.split(r"[\s,(，、_\-]+", first_fragment, maxsplit=1)[0].strip("\\()[]{}")
                if len(first_word) >= 3 and re.fullmatch(r"[A-Za-z0-9_ -]+", first_word):
                    generated_identity = re.compile(
                        rf"(?<![A-Za-z0-9]){re.escape(first_word)}"
                        rf"(?:_[A-Za-z0-9]+|_\([^)]*\)|\s*\([^)]*\))*"
                        rf"(?![A-Za-z0-9_])",
                        re.IGNORECASE,
                    )
                    text = generated_identity.sub(" ", text)
        params.prompt = re.sub(r"\s+", " ", text).strip(" ,，。；;")

    @staticmethod
    def _civitai_normalize(value: str) -> str:
        value = Path(value).stem.lower()
        value = re.sub(r"\.(safetensors|ckpt|pt|bin)$", "", value)
        value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _civitai_known_hosts() -> set[str]:
        """返回已知的 CivitAI 及兼容站域名。

        ``civitai.red`` 是可用的 CivitAI API 兼容站；``civital.red``
        是用户容易输入的近似域名，目前可能只是停放页，也保留在这里，
        以便给出明确的中文错误提示，而不是误报成普通网络异常。
        """
        return {
            "civitai.com", "www.civitai.com",
            "civitai.ai", "www.civitai.ai",
            "civitai.red", "www.civitai.red",
            "civital.red", "www.civital.red",
        }

    def _civitai_base_url(self, value: Any = None) -> str:
        """规范化 CivitAI 兼容站基地址，不保留 API 路径和查询参数。"""
        raw = str(
            (self._get("civitai_base_url", "https://civitai.com") if value is None else value)
            or ""
        ).strip()
        if not raw:
            raw = "https://civitai.com"
        parsed = urlparse(raw)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise UsageError("CivitAI 兼容站地址必须是完整的 http:// 或 https:// 地址")
        if parsed.username or parsed.password:
            raise UsageError("CivitAI 兼容站地址不能包含账号或密码")
        try:
            parsed.port
        except ValueError as exc:
            raise UsageError("CivitAI 兼容站地址中的端口无效") from exc
        path = parsed.path.rstrip("/")
        path = re.sub(r"/api/v1$", "", path, flags=re.IGNORECASE)
        path = re.sub(r"/api$", "", path, flags=re.IGNORECASE)
        return parsed._replace(
            scheme=parsed.scheme.lower(),
            path=path.rstrip("/"),
            params="",
            query="",
            fragment="",
        ).geturl().rstrip("/")

    @staticmethod
    def _civitai_origin(value: str) -> str:
        parsed = urlparse(str(value or "").strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""
        return parsed._replace(path="", params="", query="", fragment="").geturl().rstrip("/")

    def _civitai_link_api_base(self, link: str) -> str:
        """获取模型链接对应的 API 地址；自定义链接优先使用链接自身站点。"""
        parsed = urlparse(str(link or "").strip())
        host = (parsed.hostname or "").lower()
        if not host:
            return ""
        configured_host = ""
        configured_base = ""
        try:
            configured_base = self._civitai_base_url()
            configured_host = (urlparse(configured_base).hostname or "").lower()
        except UsageError:
            pass
        if host not in self._civitai_known_hosts() and host != configured_host:
            return ""
        if configured_base and host == configured_host:
            configured_path = urlparse(configured_base).path.rstrip("/")
            if configured_path and (
                parsed.path.rstrip("/") == configured_path
                or parsed.path.startswith(f"{configured_path}/")
            ):
                return configured_base
        return self._civitai_origin(link)

    def _civitai_non_api_message(self, base_url: str, detail: str = "") -> str:
        """将停放页、反向代理错误页等转换成中文提示。"""
        host = (urlparse(base_url).hostname or "").lower()
        if host in {"civital.red", "www.civital.red"}:
            return (
                "你填写的是 civital.red；该域名当前返回停放/售卖页面，不是 CivitAI API。"
                "请确认是否要改用 https://civitai.red。"
            )
        suffix = f"：{detail}" if detail else ""
        return f"CivitAI 兼容站 {base_url} 返回的不是 API JSON，请检查站点地址或接口兼容性{suffix}"

    def _civitai_search_url(self, file_name: str) -> str:
        from urllib.parse import quote_plus

        return f"{self._civitai_base_url()}/search/models?query=" + quote_plus(Path(file_name).stem)

    @staticmethod
    def _validate_civitai_link(value: Any) -> str:
        from urllib.parse import urlparse

        link = str(value or "").strip()
        if not link:
            return ""
        parsed = urlparse(link)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise UsageError("CivitAI 链接必须是完整的 http:// 或 https:// 地址")
        return link

    def _civitai_headers(
        self,
        accept: str,
        *,
        base_url: str | None = None,
        use_token: bool = True,
    ) -> dict[str, str]:
        """构造 CivitAI API 和 CDN 下载都能使用的请求头。

        ``use_token=False`` 时完全不发送 Authorization，用于「免 API」下载：
        公共 LoRA 的直链/CDN 不需要任何密钥即可下载，携带无效密钥反而
        会被 CDN 判定为鉴权失败。
        """
        base = self._civitai_base_url(base_url) if base_url else self._civitai_base_url()
        headers = {
            "Accept": accept,
            "User-Agent": f"AstrBot-ComfyUI-AI-Studio/{PLUGIN_VERSION}",
            "Referer": f"{base}/",
        }
        if not use_token:
            return headers
        token = str(self._get("civitai_token", "") or "").strip()
        if token:
            headers["Authorization"] = (
                token if token.lower().startswith("bearer ") else f"Bearer {token}"
            )
        return headers

    def _civitai_download_mode(self) -> str:
        """返回 C 站下载方式：auto（默认）/ nokey（免 API）/ token（API Key）。"""
        value = str(self._get("civitai_download_mode", "auto") or "auto").strip().lower()
        if value in {"nokey", "no_key", "免api", "免密钥", "无api", "public"}:
            return "nokey"
        if value in {"token", "api", "apikey", "api_key", "密钥"}:
            return "token"
        return "auto"

    @staticmethod
    def _civitai_direct_download_id(link: str) -> str:
        """识别 CivitAI 直链下载地址，返回版本 ID（无需 API Key）。"""
        match = re.search(
            r"/(?:api/)?download/models/(\d+)(?:/|$)",
            urlparse(str(link or "").strip()).path,
            re.IGNORECASE,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _civitai_filename_from_headers(headers: Any, fallback: str) -> str:
        """从下载响应的 Content-Disposition 中解析真实文件名。"""
        raw = ""
        try:
            raw = str(headers.get("content-disposition", "") or "")
        except Exception:
            raw = ""
        if not raw:
            return fallback
        match = re.search(
            r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;]+)",
            raw,
            re.IGNORECASE,
        ) or re.search(r"filename\s*=\s*\"?([^\";]+)\"?", raw, re.IGNORECASE)
        if not match:
            return fallback
        name = unquote(match.group(1).strip().strip('"'))
        name = Path(name.replace("\\", "/")).name
        return name or fallback

    @staticmethod
    def _extract_civitai_download_from_html(html: str) -> tuple[str, str]:
        """从模型页 HTML 内嵌 JSON 中提取下载地址和文件名（不调用 API）。

        CivitAI 的模型页会把完整的模型信息以 ``__NEXT_DATA__`` JSON 形式
        内嵌在页面里；直接解析它即可拿到 ``downloadUrl``，无需任何 API Key，
        也不需要访问 ``/api/v1`` 接口。
        """
        text = str(html or "")
        if not text:
            return "", ""
        payloads: list[str] = []
        for match in re.finditer(
            r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            payloads.append(match.group(1))
        # 兼容部分镜像站点的其它内嵌 JSON 形式。
        for match in re.finditer(
            r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            payloads.append(match.group(1))

        download_url = ""
        file_name = ""
        for payload in payloads:
            raw = payload.strip()
            if not raw or raw[0] not in "[{":
                continue
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue
            download_url, file_name = CivitAIDownloadResolver.find_download(data)
            if download_url:
                break

        if not download_url:
            # 最后的兜底：直接在 HTML 中寻找下载直链。
            match = re.search(
                r"https?://[^\s\"'<>\\]+?/api/download/models/\d+[^\s\"'<>\\]*",
                text,
                re.IGNORECASE,
            )
            if match:
                download_url = match.group(0).replace("\\u0026", "&").replace("\\/", "/")
        if not download_url:
            match = re.search(
                r"[\"'\\]+(/api/download/models/\d+)[\"'\\]*",
                text,
                re.IGNORECASE,
            )
            if match:
                download_url = match.group(1)
        return download_url.strip(), file_name.strip()

    async def _civitai_resolve_download_nokey(self, link: str) -> tuple[str, str, int]:
        """免 API 解析 C 站下载地址：只接受直链或模型页，全程不使用 API Key。

        返回 ``(下载地址, 文件名, 版本 ID)``。支持的链接形式：
        1. 直链下载地址，例如 ``https://civitai.com/api/download/models/123456``；
        2. 模型页或版本页地址，通过解析页面内嵌 JSON 获得下载地址。
        """
        link = self._validate_civitai_link(link)
        if not link:
            raise UsageError("请输入 CivitAI 模型链接或直链下载地址")

        direct_version = self._civitai_direct_download_id(link)
        if direct_version:
            # 直链本身就带版本 ID，直接下载最稳，不需要任何站点 API。
            return link, f"civitai_{direct_version}.safetensors", int(direct_version)

        api_base = self._civitai_link_api_base(link) or self._civitai_base_url()
        headers = self._civitai_headers(
            "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            base_url=api_base,
            use_token=False,
        )
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
                response = await client.get(link)
                if response.is_error:
                    detail = self._civitai_error_detail(response.text)
                    raise UsageError(
                        self._civitai_http_error(response.status_code, detail, "模型页请求")
                    )
                html = response.text
        except UsageError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise UsageError(f"读取 CivitAI 模型页失败：{exc}") from exc

        download_url, file_name = self._extract_civitai_download_from_html(html)
        if not download_url:
            raise UsageError(
                "没有从 CivitAI 页面解析到下载地址。请改用 CivitAI 模型链接，"
                "或直接复制模型页的下载直链（形如 .../api/download/models/版本ID）后重试"
            )
        if download_url.startswith("/"):
            download_url = f"{api_base}{download_url}"
        version_id = self._civitai_direct_download_id(download_url)
        filename = self._safe_lora_filename(
            file_name,
            f"civitai_{version_id or 'unknown'}.safetensors",
        )
        if Path(filename).suffix.lower() not in {".safetensors", ".pt", ".ckpt", ".bin"}:
            filename = f"civitai_{version_id or 'unknown'}.safetensors"
        return download_url, filename, int(version_id or 0)


    @staticmethod
    def _civitai_error_detail(payload: bytes | str, limit: int = 360) -> str:
        """从 CivitAI 的 JSON/HTML 错误响应中提取短中文可读信息。"""
        if isinstance(payload, bytes):
            text = payload[:4096].decode("utf-8", errors="replace")
        else:
            text = str(payload or "")[:4096]
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "远端没有返回错误详情"
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            for key in ("message", "error", "detail", "title"):
                value = parsed.get(key)
                if isinstance(value, dict):
                    value = value.get("message", value.get("detail", ""))
                if value:
                    text = str(value).strip()
                    break
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit] + ("..." if len(text) > limit else "")

    def _civitai_http_error(self, status_code: int, detail: str, operation: str) -> str:
        """把 CivitAI 常见授权错误转换成可执行的中文处理提示。"""
        if status_code == 401:
            if self._get("civitai_token", ""):
                return (
                    f"CivitAI {operation}被拒绝（HTTP 401）：当前 API Key 无效、已过期，"
                    "或没有该模型的下载权限，请重新生成有下载权限的 CivitAI API Key"
                )
            return (
                f"CivitAI {operation}被拒绝（HTTP 401）：该模型下载需要 CivitAI API Key。"
                "请打开 AstrBot WebUI -> AI 与提示词，在“CivitAI API Key”中填写有下载权限的密钥后重试"
            )
        if status_code == 403:
            return (
                f"CivitAI {operation}被拒绝（HTTP 403）：当前 API Key 没有下载权限，"
                "或该模型限制下载；请检查 CivitAI API Key 和模型页面权限"
            )
        return f"CivitAI {operation}失败（HTTP {status_code}）：{detail}"

    @staticmethod
    def _safe_lora_filename(value: Any, fallback: str) -> str:
        """把 CivitAI 文件名转换为 Windows 和 Linux 都可写入的文件名。"""
        raw = unquote(str(value or "")).replace("\\", "/")
        name = Path(raw).name
        name = re.sub(r'[<>:"/|?*\x00-\x1f]', "_", name).rstrip(" .")
        if not name or name in {".", ".."}:
            name = fallback
        if name.upper().split(".", 1)[0] in {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }:
            name = f"_{name}"
        suffix = Path(name).suffix
        if len(name) > 180:
            name = name[: 180 - len(suffix)] + suffix
        return name

    def _civitai_link_override(self, file_name: str) -> str:
        try:
            override = self.civitai_overrides.get(file_name, {})
            if isinstance(override, dict) and override.get("url"):
                return self._validate_civitai_link(override.get("url"))
            return self._validate_civitai_link(self.civitai_links.get(file_name, ""))
        except UsageError:
            return ""

    @staticmethod
    def _normalize_lora_preview_data(value: Any, *, strict: bool = False) -> str:
        """校验 WebUI 拖入的本地预览图，只接受小型 PNG/JPEG/WEBP data URL。"""
        raw = str(value or "").strip()
        if not raw:
            return ""
        match = re.fullmatch(
            r"data:(image/(?:png|jpeg|jpg|webp));base64,([A-Za-z0-9+/=\s]+)",
            raw,
            re.IGNORECASE,
        )
        if not match:
            if strict:
                raise UsageError("预览图必须是 PNG、JPEG 或 WEBP 图片")
            return ""
        mime = match.group(1).lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
        encoded = re.sub(r"\s+", "", match.group(2))
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            if strict:
                raise UsageError("预览图数据无效，请重新拖入图片")
            return ""
        if len(decoded) > 8 * 1024 * 1024:
            if strict:
                raise UsageError("预览图不能超过 8 MB")
            return ""
        valid_header = (
            (mime == "image/png" and decoded.startswith(b"\x89PNG\r\n\x1a\n"))
            or (mime == "image/jpeg" and decoded.startswith(b"\xff\xd8\xff"))
            or (
                mime == "image/webp"
                and len(decoded) >= 12
                and decoded[:4] == b"RIFF"
                and decoded[8:12] == b"WEBP"
            )
        )
        if not valid_header:
            if strict:
                raise UsageError("预览图文件内容与图片格式不匹配")
            return ""
        return f"data:{mime};base64,{encoded}"

    def _apply_civitai_link_override(self, file_name: str, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        override = self.civitai_overrides.get(file_name, {})
        if not isinstance(override, dict):
            override = {}
        custom_url = self._civitai_link_override(file_name)
        custom_name = str(override.get("name", "") or "").strip()
        custom_images = override.get("images", [])
        if not isinstance(custom_images, list):
            custom_images = []
        custom_images = [
            {"url": str(url).strip(), "nsfw": 0}
            for url in custom_images
            if isinstance(url, str) and url.strip()
        ]
        custom_images = custom_images[:1]
        custom_image_data = self._normalize_lora_preview_data(
            override.get("custom_image_data", ""),
            strict=False,
        )
        if custom_url and "trigger_words" in override:
            result["trigger_words"] = self._normalize_trigger_words(override.get("trigger_words", []))
        if custom_url and "tags" in override:
            result["tags"] = self._normalize_civitai_tags(override.get("tags", []))
        if custom_name:
            result["model_name"] = custom_name
        if custom_image_data:
            result["images"] = [{"url": custom_image_data, "nsfw": 0}]
        elif custom_images:
            result["images"] = custom_images[:8]
        elif custom_url:
            # 使用自定义 CivitAI 链接时，不能继续显示旧文件名匹配到的图片。
            result["images"] = []
        else:
            result["images"] = list(result.get("images", []) or [])[:1]
        result["custom_url"] = custom_url
        result["custom_name"] = custom_name
        result["custom_images"] = [item["url"] for item in custom_images]
        result["custom_image_data"] = custom_image_data
        result["civitai_tags"] = self._normalize_civitai_tags(result.get("tags", []))
        result["custom_link"] = bool(custom_url)
        result["custom_info"] = bool(custom_url or custom_name or custom_images or custom_image_data)
        result["show_images"] = override.get("show_images", True) is not False
        if custom_url:
            result["model_url"] = custom_url
            result["found"] = True
            if not result.get("model_name"):
                result["model_name"] = "我的自定义链接"
        elif custom_name or custom_images or custom_image_data:
            result["found"] = True
        return result

    async def _fetch_civitai_lora(self, file_name: str, *, force: bool = False) -> dict[str, Any]:
        base_url = self._civitai_base_url()
        cached = self.civitai_cache.get(file_name)
        if (
            not force
            and isinstance(cached, dict)
            and time.time() - float(cached.get("fetched_at", 0) or 0) < 86400
            and cached.get("url_format") == "versioned-model-url-v2"
            and str(cached.get("base_url", "https://civitai.com")).rstrip("/") == base_url
            and isinstance(cached.get("data"), dict)
        ):
            return self._apply_civitai_link_override(file_name, cached["data"])

        query = Path(file_name).stem
        base = self._civitai_normalize(file_name)
        result: dict[str, Any] = {
            "found": False,
            "model_name": "",
            "model_url": self._civitai_search_url(file_name),
            "images": [],
            "trigger_words": [],
            "tags": [],
            "error": "",
        }
        try:
            headers = self._civitai_headers("application/json", base_url=base_url)
            async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
                response = await client.get(
                    f"{base_url}/api/v1/models",
                    params={"query": query, "types": "LORA", "limit": 10},
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise UsageError(self._civitai_non_api_message(base_url)) from exc
                if not isinstance(payload, dict):
                    raise UsageError(self._civitai_non_api_message(base_url))
                items = payload.get("items", [])
            if not isinstance(items, list):
                items = []
            scored: list[tuple[int, dict[str, Any]]] = []
            base_tokens = set(base.split())
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_name = str(item.get("name", ""))
                normalized = self._civitai_normalize(item_name)
                item_tokens = set(normalized.split())
                score = len(base_tokens & item_tokens)
                if normalized == base:
                    score += 100
                elif base and (base in normalized or normalized in base):
                    score += 30
                scored.append((score, item))
            if scored:
                best_score, item = max(scored, key=lambda pair: pair[0])
                if best_score <= 0:
                    item = None
            else:
                item = None
            if item is not None:
                model_id = item.get("id")
                if model_id:
                    result["found"] = True
                    result["model_name"] = str(item.get("name", ""))
                    versions = item.get("modelVersions", []) or []
                    selected_version = next(
                        (version for version in versions if isinstance(version, dict) and version.get("id")),
                        {},
                    )
                    version_id = selected_version.get("id")
                    result["model_url"] = f"{base_url}/models/{model_id}"
                    if version_id:
                        result["model_url"] += f"?modelVersionId={version_id}"
                    result["trigger_words"] = self._normalize_trigger_words(
                        selected_version.get("trainedWords", [])
                    )
                    result["tags"] = self._normalize_civitai_tags(item.get("tags", []))
                    images: list[dict[str, Any]] = []
                    for version in item.get("modelVersions", []) or []:
                        for image in version.get("images", []) or []:
                            image_url = str(image.get("url", "") or "")
                            if image_url.startswith(("http://", "https://")):
                                images.append({"url": image_url, "nsfw": image.get("nsfwLevel", 0)})
                    result["images"] = images[:8]
        except UsageError as exc:
            result["error"] = str(exc)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            result["error"] = f"CivitAI 查询失败：{exc}"
        async with self._civitai_cache_lock:
            self.civitai_cache[file_name] = {
                "fetched_at": time.time(),
                "url_format": "versioned-model-url-v2",
                "base_url": base_url,
                "data": result,
            }
            self._save_civitai_cache()
        return self._apply_civitai_link_override(file_name, result)

    async def _fetch_civitai_link_info(self, link: str) -> dict[str, Any]:
        """按用户填写的 CivitAI 模型链接立即读取名称和首张预览图。"""
        from urllib.parse import urlparse

        parsed = urlparse(link)
        base_url = self._civitai_link_api_base(link)
        if not base_url:
            return {}
        model_match = re.search(r"/models/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        version_match = re.search(r"/model-versions/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        requested_version = parse_qs(parsed.query).get("modelVersionId", [""])[0]
        version_id = requested_version or (version_match.group(1) if version_match else "")
        if not model_match and not version_id:
            return {}
        try:
            headers = self._civitai_headers("application/json", base_url=base_url)
            async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
                if model_match:
                    response = await client.get(f"{base_url}/api/v1/models/{model_match.group(1)}")
                    response.raise_for_status()
                    try:
                        item = response.json()
                    except ValueError as exc:
                        return {"error": self._civitai_non_api_message(base_url)}
                else:
                    response = await client.get(f"{base_url}/api/v1/model-versions/{version_id}")
                    response.raise_for_status()
                    try:
                        version_item = response.json()
                    except ValueError:
                        return {"error": self._civitai_non_api_message(base_url)}
                    if not isinstance(version_item, dict):
                        return {}
                    model_id = version_item.get("modelId")
                    item = {
                        "name": str(
                            (version_item.get("model") or {}).get("name", "")
                            if isinstance(version_item.get("model"), dict)
                            else version_item.get("modelName", "")
                        ),
                        "modelVersions": [version_item],
                    }
                    if model_id:
                        try:
                            model_response = await client.get(f"{base_url}/api/v1/models/{model_id}")
                            model_response.raise_for_status()
                            model_item = model_response.json()
                            if isinstance(model_item, dict):
                                item = model_item
                        except (httpx.HTTPError, ValueError, TypeError):
                            pass
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.debug("[%s] 读取自定义 CivitAI 链接失败：%s", PLUGIN_NAME, exc)
            return {"error": f"读取 CivitAI 兼容站模型信息失败：{exc}"}
        if not isinstance(item, dict):
            return {}
        versions = [version for version in item.get("modelVersions", []) or [] if isinstance(version, dict)]
        selected_version = next(
            (version for version in versions if str(version.get("id", "")) == str(requested_version)),
            versions[0] if versions else {},
        )
        images: list[dict[str, Any]] = []
        for version in versions:
            if not isinstance(version, dict):
                continue
            for image in version.get("images", []) or []:
                if not isinstance(image, dict):
                    continue
                image_url = str(image.get("url", "") or "")
                if image_url.startswith(("http://", "https://")):
                    images.append({"url": image_url, "nsfw": image.get("nsfwLevel", 0)})
        return {
            "model_name": str(item.get("name", "") or "").strip(),
            "images": images[:1],
            "trigger_words": self._normalize_trigger_words(selected_version.get("trainedWords", [])),
            "tags": self._normalize_civitai_tags(item.get("tags", [])),
        }

    @staticmethod
    def _normalize_trigger_words(value: Any) -> list[str]:
        """规范化 CivitAI trainedWords，保留触发词中的空格。"""
        if isinstance(value, str):
            values = re.split(r"[,，\n]+", value)
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = []
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            word = str(raw or "").strip()
            key = word.casefold()
            if word and key not in seen:
                seen.add(key)
                result.append(word)
        return result

    def _civitai_trigger_words(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        return self._normalize_trigger_words(data.get("trigger_words", []))

    @staticmethod
    def _normalize_civitai_tags(value: Any) -> list[str]:
        """规范化 CivitAI tags，保留 tag 中的空格并去重。"""
        if isinstance(value, str):
            values = re.split(r"[,，\n]+", value)
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = []
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            tag = str(raw or "").strip()
            key = tag.casefold()
            if tag and key not in seen:
                seen.add(key)
                result.append(tag)
        return result

    def _civitai_tags(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        return self._normalize_civitai_tags(data.get("tags", []))

    async def _sync_lora_trigger_presets(
        self,
        file_name: str,
        aliases: list[str] | None = None,
        trigger_words: list[str] | None = None,
    ) -> None:
        """保留旧接口，但不再把 CivitAI 触发词写入绘图预设。

        0.7.7 起 CivitAI trainedWords 只用于 LoRA 页面展示；保留这个空方法
        是为了兼容旧版本可能调用它的运行环境，同时避免旧调用再次污染用户预设。
        """
        return None

    @staticmethod
    def _civitai_reference(link: str, allowed_hosts: set[str] | None = None) -> tuple[str, str]:
        """从 CivitAI 模型页提取模型 ID 和可选版本 ID。"""
        parsed = urlparse(link)
        host = (parsed.hostname or "").lower()
        hosts = set(allowed_hosts or ()) or ComfyUIAIStudio._civitai_known_hosts()
        if parsed.scheme.lower() not in {"http", "https"} or host not in hosts:
            raise UsageError("下载地址必须是 CivitAI 或其兼容站的模型链接")
        model_match = re.search(r"/models/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        version_match = re.search(r"/model-versions/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        direct_version_match = re.search(r"/(?:api/)?download/models/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        api_version_match = re.search(r"/api/v1/model-versions/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        # /api/download/models/<version> contains a numeric segment but it is
        # a version id, not a model id.
        model_id = model_match.group(1) if model_match and not direct_version_match else ""
        version_id = (
            version_match.group(1)
            if version_match
            else direct_version_match.group(1)
            if direct_version_match
            else api_version_match.group(1)
            if api_version_match
            else ""
        )
        query = parse_qs(parsed.query)
        for key in ("modelVersionId", "model_version_id", "versionId", "version_id"):
            query_version = query.get(key, [""])[0]
            if str(query_version).isdigit():
                version_id = str(query_version)
                break
        if not model_id and not version_id:
            raise UsageError("链接中没有找到 CivitAI 模型 ID")
        return model_id, version_id

    async def _civitai_download_file(self, link: str) -> tuple[str, str, int]:
        """读取 CivitAI 模型页，返回下载地址、文件名和版本 ID。"""
        configured_host = (urlparse(self._civitai_base_url()).hostname or "").lower()
        model_id, version_id = self._civitai_reference(
            link,
            self._civitai_known_hosts() | ({configured_host} if configured_host else set()),
        )
        parsed_link = urlparse(link)
        api_base = self._civitai_link_api_base(link) or self._civitai_base_url()
        is_direct_download = bool(re.search(r"/(?:api/)?download/models/\d+(?:/|$)", parsed_link.path, re.IGNORECASE))
        headers = self._civitai_headers("application/json", base_url=api_base)

        async def get_json(client: httpx.AsyncClient, url: str) -> Any:
            for attempt in range(3):
                response = await client.get(url, headers=headers)
                if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    if response.is_error:
                        detail = self._civitai_error_detail(response.text)
                        raise UsageError(
                            self._civitai_http_error(response.status_code, detail, "模型信息请求")
                        )
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise UsageError(self._civitai_non_api_message(api_base)) from exc
                retry_after = response.headers.get("retry-after", "")
                try:
                    delay = max(0.5, min(3.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = 0.8 * (attempt + 1)
                await asyncio.sleep(delay)
            raise UsageError("CivitAI 模型信息请求失败")

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
                if not version_id:
                    model = await get_json(client, f"{api_base}/api/v1/models/{model_id}")
                    versions = model.get("modelVersions", []) if isinstance(model, dict) else []
                    version_id = str(
                        next(
                            (
                                item.get("id") for item in versions
                                if isinstance(item, dict) and item.get("id")
                            ),
                            "",
                        )
                    )
                if not version_id or not version_id.isdigit():
                    raise UsageError("CivitAI 模型没有可用版本")
                version = await get_json(client, f"{api_base}/api/v1/model-versions/{version_id}")
        except UsageError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise UsageError(f"读取 CivitAI 模型信息失败：{exc}") from exc

        files = version.get("files", []) if isinstance(version, dict) else []
        if not isinstance(files, list):
            files = []
        allowed = {".safetensors", ".pt", ".ckpt", ".bin"}
        candidates = [
            item for item in files
            if isinstance(item, dict)
            and Path(str(item.get("name", "") or "")).suffix.lower() in allowed
            and (str(item.get("type", "Model") or "Model").lower() in {"model", "lora", "checkpoint"})
        ]
        if not candidates:
            raise UsageError("CivitAI 版本中没有可下载的 LoRA 文件")
        selected = next((item for item in candidates if item.get("primary") is True), candidates[0])
        download_url = link if is_direct_download else str(selected.get("downloadUrl", selected.get("download_url", "")) or "").strip()
        if not download_url:
            download_url = f"{api_base}/api/download/models/{version_id}"
        filename = self._safe_lora_filename(
            selected.get("name", ""),
            f"civitai_{version_id}.safetensors",
        )
        if Path(filename).suffix.lower() not in allowed:
            filename = f"civitai_{version_id}.safetensors"
        return download_url, filename, int(version_id)

    async def _lora_details(self, *, force: bool = False) -> list[dict[str, Any]]:
        names = await self._local_lora_names()
        if not names:
            try:
                names = await self._client().models("loras")
            except ComfyError:
                names = []
        self._ensure_style_lora_aliases(names, persist=True)
        limiter = asyncio.Semaphore(4)
        style_entries = self._style_lora_entries(names, selected_only=False)
        style_weights = self._get("style_lora_weights", {})
        style_weights = style_weights if isinstance(style_weights, dict) else {}

        async def fetch(name: str) -> dict[str, Any]:
            async with limiter:
                info = await self._fetch_civitai_lora(name, force=force)
            # CivitAI 触发词只作为资料展示，不自动写入专属预设或最终提示词。
            visible_info = dict(info)
            category = self._lora_category(name)
            style_index = next(
                (
                    index for index, item in enumerate(style_entries)
                    if str(item["file_name"]).replace("\\", "/").casefold()
                    == str(name).replace("\\", "/").casefold()
                ),
                0,
            )
            return {
                "file_name": name,
                "alias": self._lora_alias(name),
                "command_aliases": self._lora_command_aliases_for(name, names),
                "manual_command_aliases": self._lora_command_aliases(name),
                "command_alias": self._lora_command_alias(name, names),
                "category": category,
                "style_alias": self._style_lora_alias(name, style_index, names) if category == "画风" else "",
                "style_weight": self._lora_map_value(style_weights, name, 0.8),
                "lora_presets": self._lora_preset_entries(name),
                "civitai_trigger_words": self._civitai_trigger_words(info),
                **visible_info,
            }

        return await asyncio.gather(*(fetch(name) for name in names))

    async def _lora_payload(self, *, force: bool = False) -> dict[str, Any]:
        return {
            "items": await self._lora_details(force=force),
            "categories": self._lora_categories_list(),
            # 只返回可见的 tag/content；同步用的内部字段不暴露给 WebUI，
            # 避免简洁模式把来源标记误显示成用户可编辑内容。
            "lora_presets": {
                file_name: self._lora_preset_entries(file_name)
                for file_name in getattr(self, "lora_presets", {})
            },
        }

    def _extra_model_paths_sections(self) -> tuple[str, dict[str, dict[str, list[str]]]]:
        """返回（实际使用的配置文件路径，解析结果）。

        未配置时自动在 ComfyUI 根目录及其上一层寻找 extra_model_paths.yaml。
        结果按「路径 + 修改时间」缓存，避免在循环里反复读盘解析。
        """
        path = str(self._get("extra_model_paths_file", "") or "").strip()
        if not path:
            root = detect_comfyui_root(str(self._get("comfyui_root", "") or ""))
            path = find_extra_model_paths_file(root)
        stamp: tuple[str, int] | None = None
        if path:
            try:
                stamp = (path, Path(path).stat().st_mtime_ns)
            except OSError:
                stamp = None
        cache = getattr(self, "_extra_model_paths_cache", None)
        if cache is not None and cache[0] == path and cache[1] == stamp and stamp is not None:
            return path, cache[2]
        sections = read_extra_model_paths(path)
        self._extra_model_paths_cache = (path, stamp, sections)
        return path, sections

    def _model_dir_overrides(self) -> dict[str, str]:
        """WebUI 的模型目录手动覆盖，兼容 dict 与 JSON 文本两种写法。"""
        raw = self._get("model_dir_overrides", {})
        if isinstance(raw, dict):
            items = list(raw.items())
        else:
            text = str(raw or "").strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                logger.warning("[%s] 模型目录手动覆盖不是合法 JSON，已忽略", PLUGIN_NAME)
                return {}
            items = list(parsed.items()) if isinstance(parsed, dict) else []
        overrides: dict[str, str] = {}
        for key, value in items:
            name = str(key or "").strip()
            target = str(value or "").strip()
            if name and target:
                overrides[name] = target
        return overrides

    def _model_dirs(self, category: str) -> list[ModelDir]:
        """按「手动覆盖 > extra_model_paths.yaml > 默认位置」解析模型目录。"""
        root = detect_comfyui_root(str(self._get("comfyui_root", "") or ""))
        _, sections = self._extra_model_paths_sections()
        overrides = self._model_dir_overrides()
        override = ""
        for name in category_aliases(category):
            if overrides.get(name):
                override = overrides[name]
                break
        return resolve_model_dirs(root, category, override=override, sections=sections)

    def _lora_dirs(self) -> list[Path]:
        return [entry.path for entry in self._model_dirs("loras")]

    def _lora_relative_target(self, file_name: str) -> Path | None:
        """在全部 LoRA 目录中定位文件，顺带兼容大小写与斜杠差异。"""
        relative = str(file_name or "").replace("\\", "/").strip(" /")
        if not relative:
            return None
        parts = [part for part in relative.split("/") if part not in {"", ".", ".."}]
        if not parts:
            return None
        directories = self._lora_dirs()
        for directory in directories:
            candidate = directory.joinpath(*parts)
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        # Windows 通常不区分大小写，但 ComfyUI 的 API 列表可能来自额外模型
        # 路径，做一次文件名回退，兼容旧配置写入的大小写或斜杠差异。
        target_name = Path(relative).name.casefold()
        for directory in directories:
            try:
                for path in directory.rglob(Path(relative).name):
                    if path.is_file() and path.name.casefold() == target_name:
                        return path
            except OSError:
                continue
        return None

    def _path_sources(self) -> dict[str, str]:
        """各类别当前生效目录的来源，供 WebUI 标注「默认 / 额外 / 手动」。"""
        sources: dict[str, str] = {}
        for category in MODEL_CATEGORIES:
            entries = self._model_dirs(category)
            if entries:
                sources[category] = entries[0].source
        return sources

    def _describe_model_dirs(self, category: str, title: str) -> list[str]:
        """把某类别的全部目录格式化成多行文本，标注非默认来源。"""
        entries = self._model_dirs(category)
        if not entries:
            return [f"{title}：未检测到"]
        lines = [f"{title}：{entries[0].path}（{entries[0].label}）"]
        for entry in entries[1:]:
            lines.append(f"{title}（{entry.label}）：{entry.path}")
        return lines


    # ------------------------------------------------------- 路径设置面板 --
    def _path_setting_targets(self) -> dict[str, tuple[str, str]]:
        """把 UI 条目 id 映射到写入目标：("config", 配置键) 或 ("model", 类别)。"""
        targets: dict[str, tuple[str, str]] = {}
        for group in PATH_SETTING_GROUPS:
            for entry in group["entries"]:
                if entry.get("readonly"):
                    continue
                if entry["kind"] == "model":
                    targets[str(entry["id"])] = ("model", str(entry["category"]))
                else:
                    targets[str(entry["id"])] = ("config", str(entry["config_key"]))
        return targets

    def _path_entry_value(self, entry: dict[str, Any]) -> str:
        """读取某个条目当前「手动填写」的值。"""
        if entry.get("kind") == "model":
            return str(self._model_dir_overrides().get(str(entry.get("category")), "") or "")
        return str(self._get(str(entry.get("config_key")), "") or "")

    def _resolve_path_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """把一个条目解析成 UI 需要的完整信息：当前路径、来源、检测结果。"""
        entry_id = str(entry["id"])
        kind = str(entry["kind"])
        configured = self._path_entry_value(entry)
        resolved = ""
        source = "none"
        fallbacks: list[str] = []

        if kind == "model":
            entries = self._model_dirs(str(entry["category"]))
            if entries:
                resolved = str(entries[0].path)
                source = entries[0].source
                fallbacks = [str(item.path) for item in entries[1:]]
        elif entry_id == "comfyui_root":
            detected = detect_comfyui_root(configured)
            resolved = detected or (configured if Path(configured).is_dir() else "")
            source = "override" if (configured and resolved) else ("auto" if resolved else "none")
        elif entry_id == "extra_model_paths_file":
            found, _ = self._extra_model_paths_sections()
            resolved = found
            source = "override" if (configured and found) else ("auto" if found else "none")
        elif entry_id == "workflow_dir":
            resolved = str(self._workflow_dir())
            source = "override" if configured else "builtin"
        elif entry_id == "source_workflow":
            resolved = configured
            source = "override" if configured else "builtin"
        elif entry_id in {"image_font_regular", "image_font_bold"}:
            regular, bold = image_font_candidates(
                str(self._get("image_font_regular", "") or ""),
                str(self._get("image_font_bold", "") or ""),
            )
            candidates = regular if entry_id == "image_font_regular" else bold
            found = next((item for item in candidates if Path(item).is_file()), "")
            resolved = configured or found
            source = "override" if configured else ("auto" if found else "none")
        elif entry_id == "anima_template_path":
            template = find_template_path(self.plugin_dir, configured)
            resolved = configured or (str(template) if template else "")
            source = "override" if configured else ("builtin" if template else "none")
        elif entry_id == "comfyui_start_script":
            script = self._comfyui_start_script()
            resolved = configured or (str(script) if script else "")
            source = "override" if configured else ("auto" if script else "none")
        elif entry_id == "output_dir":
            resolved = str(self.output_dir)
            source = "runtime"
        elif entry_id == "data_dir":
            resolved = str(self.data_dir)
            source = "runtime"

        exists, files = scan_path(
            resolved,
            kind="file" if kind == "file" else "dir",
            suffixes=tuple(entry.get("suffixes") or ()),
        )
        return {
            "id": entry_id,
            "label": str(entry["label"]),
            "kind": kind,
            "readonly": bool(entry.get("readonly")),
            "configured": configured,
            "resolved": resolved,
            "source": source,
            "source_label": SOURCE_LABELS.get(source, source),
            "exists": exists,
            "files": files,
            "fallbacks": fallbacks,
            "suffixes": list(entry.get("suffixes") or ()),
        }

    def path_settings_payload(self) -> dict[str, Any]:
        """路径设置面板的完整数据。"""
        groups = [
            {
                "group": str(group["group"]),
                "note": str(group.get("note", "")),
                "entries": [self._resolve_path_entry(entry) for entry in group["entries"]],
            }
            for group in PATH_SETTING_GROUPS
        ]
        return {
            "ok": True,
            "groups": groups,
            "writable_ids": sorted(self._path_setting_targets()),
            "extra_model_paths_file": self._extra_model_paths_sections()[0],
            "comfyui_root": detect_comfyui_root(str(self._get("comfyui_root", "") or "")),
        }

    @staticmethod
    def normalize_path_value(value: Any) -> str:
        """把用户填的路径归一化成字符串（Windows 反斜杠原样保留）。"""
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value).strip().strip('"').strip("'")

    def save_path_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        """保存路径设置；空值表示清除覆盖、恢复自动解析。"""
        targets = self._path_setting_targets()
        overrides = dict(self._model_dir_overrides())
        changed: list[str] = []
        skipped: list[str] = []
        for raw_id, raw_value in (values or {}).items():
            entry_id = str(raw_id)
            target = targets.get(entry_id)
            if target is None:
                skipped.append(entry_id)
                continue
            scope, name = target
            value = self.normalize_path_value(raw_value)
            if scope == "model":
                if value:
                    overrides[name] = value
                else:
                    overrides.pop(name, None)
            else:
                self._set(name, value)
            changed.append(entry_id)
        if any(targets[item][0] == "model" for item in changed):
            self._set("model_dir_overrides", overrides)
        # 配置文件或覆盖变了，缓存必须失效
        self._extra_model_paths_cache = None
        self._save_config()
        payload = {"ok": True, "changed": changed, "skipped": skipped}
        payload.update(self.path_settings_payload())
        return payload

    def reset_path_settings(self) -> dict[str, Any]:
        """把所有可写的路径设置恢复为「自动」。"""
        targets = self._path_setting_targets()
        cleared: list[str] = []
        for entry_id, (scope, name) in targets.items():
            if scope == "config":
                self._set(name, "")
            cleared.append(entry_id)
        self._set("model_dir_overrides", {})
        self._extra_model_paths_cache = None
        self._save_config()
        payload = {"ok": True, "cleared": sorted(cleared)}
        payload.update(self.path_settings_payload())
        return payload

    def check_path_settings(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        """检测路径：给了值就检查填的值，没给就检查当前解析结果。"""
        values = values or {}
        results: dict[str, Any] = {}
        for group in PATH_SETTING_GROUPS:
            for entry in group["entries"]:
                entry_id = str(entry["id"])
                if values and entry_id not in values:
                    continue
                text = self.normalize_path_value(values.get(entry_id))
                if not text:
                    text = self._resolve_path_entry(entry)["resolved"]
                kind = "file" if entry["kind"] == "file" else "dir"
                suffixes = tuple(entry.get("suffixes") or ())
                exists, files = scan_path(text, kind=kind, suffixes=suffixes)
                results[entry_id] = {
                    "path": text,
                    "exists": exists,
                    "files": files,
                    "kind": kind,
                    "suffixes": list(suffixes),
                }
        return {"ok": True, "results": results}

    @staticmethod
    def browse_path(raw_path: str = "") -> dict[str, Any]:
        """路径面板用的极简目录浏览：列出子目录、上级与可用根。"""
        text = str(raw_path or "").strip().strip('"').strip("'")
        roots: list[str] = []
        if os.name == "nt":
            for letter in "CDEFGH":
                drive = f"{letter}:\\"
                if Path(drive).is_dir():
                    roots.append(drive)
        else:
            roots.append("/")

        current = Path(text).expanduser() if text else None
        if current is None or not current.is_dir():
            current = Path(roots[0]) if roots else Path.home()
        try:
            children = sorted(
                (
                    {"name": item.name, "path": str(item)}
                    for item in current.iterdir()
                    if item.is_dir() and not item.name.startswith(".")
                ),
                key=lambda item: item["name"].casefold(),
            )
        except OSError:
            children = []
        return {
            "ok": True,
            "path": str(current),
            "parent": str(current.parent) if str(current) not in roots else "",
            "roots": roots,
            "entries": children,
        }

    async def api_path_settings(self):
        from astrbot.api.web import json_response, request

        if request.method == "GET":
            return json_response(self.path_settings_payload())
        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        action = str(data.get("action", "save") or "save").strip().lower()
        values = data.get("values")
        if not isinstance(values, dict):
            values = {}
        try:
            if action == "save":
                payload = self.save_path_settings(values)
            elif action == "reset":
                payload = self.reset_path_settings()
            elif action == "check":
                payload = self.check_path_settings(values)
            elif action == "browse":
                payload = self.browse_path(str(data.get("path", "") or ""))
            else:
                return json_response({"error": f"未知操作：{action}"}, status_code=400)
        except (OSError, UsageError, ValueError) as exc:
            return json_response({"error": str(exc)}, status_code=400)
        payload.update(self.path_settings_payload())
        return json_response(payload)

    def _model_folder_path(self, kind: str) -> Path:
        kind = str(kind or "").strip()
        if kind in MODEL_CATEGORIES:
            entries = self._model_dirs(kind)
            if not entries:
                raise UsageError("未检测到 ComfyUI 根目录，也没有配置额外模型路径")
            return entries[0].path
        if kind == "workflows":
            return self._workflow_dir()
        if kind == "source_workflow":
            configured_source = str(self._get("source_workflow", "") or "").strip()
            if not configured_source:
                return self._workflow_dir()
            return Path(configured_source).parent
        raise UsageError("不支持的文件夹类型")

    async def api_lora_info(self):
        from astrbot.api.web import json_response, request

        if request.method == "GET":
            try:
                return json_response(await self._lora_payload())
            except (ComfyError, OSError, UsageError) as exc:
                return json_response({"error": str(exc)}, status_code=500)
        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        action = str(data.get("action", "") or "")
        try:
            available = await self._local_lora_names()
            file_name = str(data.get("file_name", "") or "").strip()
            if action in {"set_preview_image", "set_lora_preview"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                try:
                    preview = self._normalize_lora_preview_data(
                        data.get("image_data", data.get("preview_image", "")),
                        strict=True,
                    )
                except UsageError as exc:
                    return json_response({"error": str(exc)}, status_code=400)
                override = self.civitai_overrides.get(file_name, {})
                override = dict(override) if isinstance(override, dict) else {}
                override["custom_image_data"] = preview
                self.civitai_overrides[file_name] = override
                self._save_civitai_overrides()
                return json_response({"ok": True, **(await self._lora_payload())})
            if action in {"clear_preview_image", "clear_lora_preview"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                override = self.civitai_overrides.get(file_name, {})
                if isinstance(override, dict):
                    override = dict(override)
                    override.pop("custom_image_data", None)
                    self.civitai_overrides[file_name] = override
                    self._save_civitai_overrides()
                return json_response({"ok": True, **(await self._lora_payload())})
            if action == "category_add":
                category = str(data.get("category", "") or "").strip()
                if not category or category == "未分类":
                    return json_response({"error": "请输入有效的自定义分类名称"}, status_code=400)
                if len(category) > 40:
                    return json_response({"error": "分类名称不能超过 40 个字符"}, status_code=400)
                self.lora_category_entries[category] = True
                self._save_lora_category_entries()
                return json_response({"ok": True, "categories": self._lora_categories_list()})
            if action == "category_delete":
                category = str(data.get("category", "") or "").strip()
                if not category or category == "未分类":
                    return json_response({"error": "不能删除未分类"}, status_code=400)
                self.lora_category_entries.pop(category, None)
                for name, current in list(self.lora_categories.items()):
                    if str(current).strip() == category:
                        self.lora_categories.pop(name, None)
                self._save_lora_category_entries()
                self._save_lora_categories()
                return json_response({"ok": True, "categories": self._lora_categories_list()})
            if action in {"category", "set_category"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                category = str(data.get("category", "") or "").strip()
                if category != "未分类" and category not in self._lora_categories_list():
                    return json_response({"error": "分类不存在，请先建立分类"}, status_code=400)
                if category == "未分类" or not category:
                    self.lora_categories.pop(file_name, None)
                else:
                    self.lora_categories[file_name] = category
                self._save_lora_categories()
                if category == "画风":
                    self._ensure_style_lora_aliases(available, persist=True)
                return json_response({"ok": True, "items": (await self._lora_payload())["items"], "categories": self._lora_categories_list()})
            if action in {"category_batch", "set_category_batch"}:
                # 批量导入后统一归类：一次请求处理多个文件，减少 WebUI 往返。
                raw_names = data.get("file_names", data.get("files", []))
                if isinstance(raw_names, str):
                    raw_names = re.split(r"[,，\n]+", raw_names)
                if not isinstance(raw_names, (list, tuple)):
                    return json_response({"error": "file_names 必须是数组"}, status_code=400)
                category = str(data.get("category", "") or "").strip()
                if category != "未分类" and category not in self._lora_categories_list():
                    return json_response({"error": "分类不存在，请先建立分类"}, status_code=400)
                changed: list[str] = []
                missing: list[str] = []
                for raw_name in raw_names:
                    name = str(raw_name or "").strip().replace("\\", "/")
                    if not name:
                        continue
                    if name not in available:
                        missing.append(name)
                        continue
                    if category == "未分类" or not category:
                        self.lora_categories.pop(name, None)
                    else:
                        self.lora_categories[name] = category
                    changed.append(name)
                self._save_lora_categories()
                if category == "画风":
                    self._ensure_style_lora_aliases(available, persist=True)
                return json_response({
                    "ok": True,
                    "changed": changed,
                    "missing": missing,
                    "items": (await self._lora_payload())["items"],
                    "categories": self._lora_categories_list(),
                })
            if action in {"civitai_override", "set_civitai_override"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                try:
                    custom_url = self._validate_civitai_link(data.get("url", data.get("civitai_url", "")))
                except UsageError as exc:
                    return json_response({"error": f"CivitAI 信息无效：{exc}"}, status_code=400)
                custom_name = str(data.get("name", data.get("model_name", "")) or "").strip()[:120]
                link_info = await self._fetch_civitai_link_info(custom_url) if custom_url else {}
                if link_info.get("error"):
                    return json_response(
                        {"error": f"CivitAI 信息查询失败：{link_info['error']}"},
                        status_code=400,
                    )
                if not custom_name:
                    custom_name = str(link_info.get("model_name", "") or "")[:120]
                custom_images = [
                    str(item.get("url", "") or "")
                    for item in (link_info.get("images", []) if isinstance(link_info, dict) else [])
                    if isinstance(item, dict) and str(item.get("url", "") or "").strip()
                ][:1]
                previous_override = self.civitai_overrides.get(file_name, {})
                previous_preview = self._normalize_lora_preview_data(
                    previous_override.get("custom_image_data", "")
                    if isinstance(previous_override, dict) else "",
                    strict=False,
                )
                override = {
                    "url": custom_url,
                    "name": custom_name,
                    "images": custom_images,
                    "trigger_words": self._normalize_trigger_words(link_info.get("trigger_words", [])),
                    "tags": self._normalize_civitai_tags(link_info.get("tags", [])),
                    "show_images": bool(data.get("show_images", True)),
                }
                if previous_preview:
                    override["custom_image_data"] = previous_preview
                self.civitai_overrides[file_name] = override
                self.civitai_links.pop(file_name, None)
                self._save_civitai_overrides()
                self._save_civitai_links()
                return json_response({
                    "ok": True,
                    "lora_list": list(self._get("lora_list", []) or []),
                    "style_lora_list": list(self._get("style_lora_list", []) or []),
                    "style_lora_aliases": dict(self._get("style_lora_aliases", {}) or {}),
                    "style_lora_weights": dict(self._get("style_lora_weights", {}) or {}),
                    **(await self._lora_payload()),
                })
            if action in {"save_lora", "save"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                alias = str(data.get("alias", "") or "").strip()
                if alias:
                    self.lora_aliases[file_name] = alias[:80]
                else:
                    self.lora_aliases.pop(file_name, None)

                raw_command_aliases = data.get("command_aliases", data.get("command_alias", data.get("alias2", [])))
                if isinstance(raw_command_aliases, str):
                    raw_command_aliases = re.split(r"[,，\n]+", raw_command_aliases)
                if not isinstance(raw_command_aliases, list):
                    raw_command_aliases = []
                command_aliases: list[str] = []
                for raw_alias in raw_command_aliases:
                    command_alias = str(raw_alias or "").strip()
                    if not command_alias:
                        continue
                    if len(command_alias) > 40:
                        return json_response({"error": "每个指令简称不能超过 40 个字符"}, status_code=400)
                    if re.search(r"[\s,:，：]", command_alias):
                        return json_response({"error": "指令简称不能包含空格、逗号或冒号"}, status_code=400)
                    if command_alias.casefold() not in {item.casefold() for item in command_aliases}:
                        command_aliases.append(command_alias)
                for command_alias in command_aliases:
                    duplicate = next(
                        (
                            name for name in available
                            if name != file_name
                            and any(
                                command_alias.casefold() == existing.casefold()
                                for existing in self._lora_command_aliases_for(name, available)
                            )
                        ),
                        None,
                    )
                    if duplicate:
                        return json_response({"error": f"指令简称已被占用：{command_alias}"}, status_code=409)
                # WebUI 展示的 command_aliases 同时包含自动简称（专属预设左侧）
                # 和手动简称；只把手动部分写回文件，自动部分始终从预设实时生成。
                # 专属预设左侧名称是动态简称，不能写入手动简称文件。
                # 同时过滤旧的左侧名称，确保改名后旧简称立即失效。
                old_preset_tags = {
                    str(item.get("tag", "")).strip().casefold()
                    for item in self._lora_preset_entries_raw(file_name)
                    if str(item.get("tag", "")).strip()
                }
                incoming_manual_aliases = [
                    item for item in command_aliases
                    if item.casefold() not in old_preset_tags
                ]

                category = str(data.get("category", "未分类") or "未分类").strip()
                if category != "未分类" and category not in self.lora_category_entries:
                    return json_response({"error": "请选择已有的自定义分类"}, status_code=400)
                if category == "未分类":
                    self.lora_categories.pop(file_name, None)
                else:
                    self.lora_categories[file_name] = category

                try:
                    custom_url = self._validate_civitai_link(data.get("civitai_url", data.get("url", "")))
                except UsageError as exc:
                    return json_response({"error": str(exc)}, status_code=400)
                custom_name = str(data.get("civitai_name", data.get("name", "")) or "").strip()[:120]
                link_info = await self._fetch_civitai_link_info(custom_url) if custom_url else {}
                if not custom_name:
                    custom_name = str(link_info.get("model_name", "") or "")[:120]
                custom_images = [
                    str(item.get("url", "") or "")
                    for item in (link_info.get("images", []) if isinstance(link_info, dict) else [])
                    if isinstance(item, dict) and str(item.get("url", "") or "").strip()
                ][:1]
                previous_override = self.civitai_overrides.get(file_name, {})
                previous_preview = self._normalize_lora_preview_data(
                    previous_override.get("custom_image_data", "")
                    if isinstance(previous_override, dict) else "",
                    strict=False,
                )
                override = {
                    "url": custom_url,
                    "name": custom_name,
                    "images": custom_images,
                    "trigger_words": self._normalize_trigger_words(link_info.get("trigger_words", [])),
                    "tags": self._normalize_civitai_tags(link_info.get("tags", [])),
                    "show_images": bool(data.get("show_images", True)),
                }
                if previous_preview:
                    override["custom_image_data"] = previous_preview
                self.civitai_overrides[file_name] = override

                old_lora_entries = self._lora_preset_entries_raw(file_name)
                if "lora_presets" in data or "presets" in data:
                    try:
                        new_lora_entries = self._normalize_lora_preset_entries(
                            data.get("lora_presets", data.get("presets", []))
                        )
                    except UsageError as exc:
                        return json_response({"error": str(exc)}, status_code=400)
                    old_by_tag = {
                        str(item.get("tag", "")).strip().casefold(): item
                        for item in old_lora_entries
                        if str(item.get("tag", "")).strip()
                    }
                    for entry in new_lora_entries:
                        previous = old_by_tag.get(str(entry.get("tag", "")).strip().casefold())
                        if not previous:
                            continue
                        for internal_key in ("_global_content",):
                            if internal_key in previous:
                                entry[internal_key] = copy.deepcopy(previous[internal_key])
                    self.lora_presets[file_name] = new_lora_entries
                    self._sync_globals_from_lora_entries(
                        file_name,
                        old_lora_entries,
                        new_lora_entries,
                    )
                    new_preset_tags = {
                        str(item.get("tag", "")).strip().casefold()
                        for item in new_lora_entries
                        if str(item.get("tag", "")).strip()
                    }
                    incoming_manual_aliases = [
                        item for item in incoming_manual_aliases
                        if item.casefold() not in new_preset_tags
                    ]
                self.lora_command_aliases[file_name] = incoming_manual_aliases
                self.civitai_links.pop(file_name, None)

                try:
                    weight = float(data.get("weight", 0.8) or 0.8)
                except (TypeError, ValueError):
                    return json_response({"error": "LoRA 权重必须是数字"}, status_code=400)
                if not 0 <= weight <= 2:
                    return json_response({"error": "LoRA 权重范围必须是 0 到 2"}, status_code=400)
                enabled = bool(data.get("enabled", False))
                current = list(self._get("lora_list", []) or [])
                kept = [
                    item for item in current
                    if str(item).rsplit(":", 1)[0].casefold() != file_name.casefold()
                ]
                if enabled:
                    kept.append(f"{file_name}:{weight:g}")
                self._set("lora_list", kept)
                if "style_selected" in data:
                    raw_style_list = self._get("style_lora_list", [])
                    if isinstance(raw_style_list, str):
                        raw_style_list = re.split(r"[,，\n]+", raw_style_list)
                    style_list = list(raw_style_list) if isinstance(raw_style_list, (list, tuple)) else []
                    normalized_file = file_name.replace("\\", "/").casefold()
                    style_list = [
                        item for item in style_list
                        if str(item).rsplit(":", 1)[0].replace("\\", "/").casefold() != normalized_file
                    ]
                    if bool(data.get("style_selected", False)):
                        if category != "画风":
                            return json_response({"error": "画风候选必须归类为“画风”"}, status_code=400)
                        style_list.append(file_name)
                    self._set("style_lora_list", style_list)

                    style_aliases = self._get("style_lora_aliases", {})
                    style_aliases = dict(style_aliases) if isinstance(style_aliases, dict) else {}
                    style_alias = str(data.get("style_alias", "") or "").strip()
                    if style_alias:
                        if len(style_alias) > 40 or re.search(r"[\s,:，：]", style_alias):
                            return json_response({"error": "画风指令简称不能包含空格、逗号或冒号，且不能超过 40 个字符"}, status_code=400)
                        style_aliases[file_name] = style_alias
                    else:
                        style_aliases.pop(file_name, None)

                    style_weights = self._get("style_lora_weights", {})
                    style_weights = dict(style_weights) if isinstance(style_weights, dict) else {}
                    if data.get("style_weight", None) not in (None, ""):
                        try:
                            style_weight = float(data.get("style_weight", 0.8) or 0.8)
                        except (TypeError, ValueError):
                            return json_response({"error": "画风 LoRA 权重必须是数字"}, status_code=400)
                        if not 0 <= style_weight <= 2:
                            return json_response({"error": "画风 LoRA 权重范围必须是 0 到 2"}, status_code=400)
                        style_weights[file_name] = style_weight
                    self._set("style_lora_aliases", style_aliases)
                    self._set("style_lora_weights", style_weights)
                if category == "画风":
                    self._ensure_style_lora_aliases(available, persist=True)
                self._save_lora_aliases()
                self._save_lora_command_aliases()
                self._save_lora_categories()
                self._save_civitai_overrides()
                self._save_lora_presets()
                self._save_civitai_links()
                self._save_config()
                self._sync_all_preset_links()
                self._save_lora_presets()
                return json_response({"ok": True, "lora_list": list(self._get("lora_list", []) or []), **(await self._lora_payload())})
            if action in {"command_alias", "set_command_alias"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                command_alias = str(data.get("alias", data.get("command_alias", "")) or "").strip()
                if len(command_alias) > 40:
                    return json_response({"error": "指令简称不能超过 40 个字符"}, status_code=400)
                if re.search(r"[\s,:，：]", command_alias):
                    return json_response({"error": "指令简称不能包含空格、逗号或冒号"}, status_code=400)
                aliases = self._lora_command_aliases(file_name)
                if command_alias:
                    duplicate = next(
                        (
                            name for name in available
                            if name != file_name
                            and any(command_alias.casefold() == existing.casefold() for existing in self._lora_command_aliases_for(name, available))
                        ),
                        None,
                    )
                    if duplicate:
                        return json_response({"error": f"指令简称已被占用：{command_alias}"}, status_code=409)
                    if command_alias.casefold() not in {item.casefold() for item in aliases}:
                        aliases.append(command_alias)
                else:
                    aliases = []
                self.lora_command_aliases[file_name] = aliases
                self._save_lora_command_aliases()
                self._sync_all_preset_links()
                self._save_lora_presets()
                return json_response({"ok": True, **(await self._lora_payload())})
            if action in {"alias", "translate_alias"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                if action == "alias":
                    alias = str(data.get("alias", "") or "").strip()
                    if alias:
                        self.lora_aliases[file_name] = alias[:80]
                    else:
                        self.lora_aliases.pop(file_name, None)
                    self._save_lora_aliases()
                else:
                    alias = await AITranslator(self._config_dict()).generate(
                        f"LoRA 文件名：{file_name}",
                        system_prompt="你是二次元模型管理助手。把 LoRA 文件名改成简短、易记、自然的中文昵称。只输出昵称，不要解释，不要标点。",
                        max_tokens=40,
                    )
                    self.lora_aliases[file_name] = alias[:80]
                    self._save_lora_aliases()
                return json_response({"ok": True, **(await self._lora_payload())})
            if action in {"link", "set_link"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                try:
                    custom_url = self._validate_civitai_link(data.get("url", data.get("civitai_url", "")))
                except UsageError as exc:
                    return json_response({"error": str(exc)}, status_code=400)
                if custom_url:
                    self.civitai_links[file_name] = custom_url
                else:
                    self.civitai_links.pop(file_name, None)
                existing_override = self.civitai_overrides.get(file_name)
                if not isinstance(existing_override, dict):
                    existing_override = {}
                if custom_url:
                    link_info = await self._fetch_civitai_link_info(custom_url)
                    existing_override["url"] = custom_url
                    existing_override["trigger_words"] = self._normalize_trigger_words(
                        link_info.get("trigger_words", [])
                    )
                    existing_override["tags"] = self._normalize_civitai_tags(
                        link_info.get("tags", [])
                    )
                else:
                    existing_override.pop("url", None)
                    existing_override.pop("trigger_words", None)
                    existing_override.pop("tags", None)
                self.civitai_overrides[file_name] = existing_override
                self._save_civitai_overrides()
                self._save_civitai_links()
                return json_response({"ok": True, "custom_url": custom_url, **(await self._lora_payload())})
            if action == "refresh":
                return json_response(await self._lora_payload(force=True))
            return json_response({"error": "未知操作"}, status_code=400)
        except (ComfyError, AIError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    async def api_open_folder(self):
        """打开文件夹：既支持按类别（kind），也支持直接传本地路径（path）。"""
        from astrbot.api.web import json_response, request

        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        try:
            raw_path = str(data.get("path", "") or "").strip()
            if raw_path:
                path = Path(raw_path).expanduser()
                if path.is_file():
                    path = path.parent
                if not path.is_dir():
                    raise UsageError(f"目录不存在：{path}")
            else:
                path = self._model_folder_path(str(data.get("kind", "")))
                path.mkdir(parents=True, exist_ok=True)
            startfile = getattr(os, "startfile", None)
            if callable(startfile):
                startfile(str(path))
            else:
                raise UsageError("当前系统不支持打开本地文件夹")
            return json_response({"ok": True, "path": str(path)})
        except (OSError, UsageError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    async def api_open_lora(self):
        """Open the folder containing a selected LoRA in Windows Explorer."""
        from astrbot.api.web import json_response, request

        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        try:
            raw_filename = str(data.get("file_name", data.get("filename", "")) or "").strip()
            normalized = raw_filename.replace("\\", "/")
            relative = Path(normalized)
            if not normalized or relative.is_absolute() or ".." in relative.parts:
                return json_response({"error": "请选择有效的 LoRA 文件"}, status_code=400)
            available = await self._local_lora_names()
            filename = relative.as_posix()
            if filename not in available:
                return json_response({"error": "LoRA 文件不存在"}, status_code=404)
            target = self._lora_relative_target(filename)
            if target is None:
                return json_response({"error": "LoRA 路径无效"}, status_code=400)
            startfile = getattr(os, "startfile", None)
            if not callable(startfile):
                raise UsageError("当前系统不支持打开本地文件夹")
            await asyncio.to_thread(startfile, str(target.parent))
            return json_response({"ok": True, "path": str(target.parent), "file_name": filename})
        except (OSError, UsageError, ComfyError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    # 允许上传/导入的 LoRA 文件扩展名。
    LORA_FILE_SUFFIXES = {".safetensors", ".pt", ".ckpt", ".bin"}
    # 批量上传时兼容的表单字段名。
    LORA_UPLOAD_FIELDS = ("files", "file", "lora_files", "upload")

    @staticmethod
    def _collect_upload_items(files: Any, keys: tuple[str, ...]) -> list[Any]:
        """从 multipart 表单中收集全部上传项，兼容单文件与多文件两种提交方式。"""
        uploads: list[Any] = []
        if not files:
            return uploads
        getlist = getattr(files, "getlist", None)
        if callable(getlist):
            for key in keys:
                try:
                    items = getlist(key) or []
                except Exception:
                    continue
                for item in items:
                    if item is not None and item not in uploads:
                        uploads.append(item)
        if uploads:
            return uploads
        getter = getattr(files, "get", None)
        if callable(getter):
            for key in keys:
                try:
                    item = getter(key)
                except Exception:
                    item = None
                if item is not None and item not in uploads:
                    uploads.append(item)
        if uploads:
            return uploads
        values = getattr(files, "values", None)
        if callable(values):
            try:
                uploads = [item for item in values() if item is not None]
            except Exception:
                uploads = []
        return uploads

    def _apply_lora_category(self, filenames: list[str], category: str) -> None:
        """把一批已入库的 LoRA 归入指定分类；分类为“画风”时同步画风候选。"""
        category = str(category or "").strip()
        if not category or not filenames:
            return
        is_style = category == "画风"
        raw_style_list = self._get("style_lora_list", [])
        if isinstance(raw_style_list, str):
            raw_style_list = re.split(r"[,，\n]+", raw_style_list)
        style_list = list(raw_style_list) if isinstance(raw_style_list, (list, tuple)) else []
        known_style_files = {
            str(item).rsplit(":", 1)[0].replace("\\", "/").casefold() for item in style_list
        }
        for filename in filenames:
            self.lora_category_entries[category] = True
            self.lora_categories[filename] = category
            if is_style and filename.replace("\\", "/").casefold() not in known_style_files:
                style_list.append(filename)
        if is_style:
            self._set("style_lora_list", style_list)
            self._ensure_style_lora_aliases(list(filenames), persist=True)
        self._save_lora_category_entries()
        self._save_lora_categories()
        self._save_config()

    async def _save_single_lora_upload(self, upload: Any, directory: Path) -> str:
        """保存一个上传的 LoRA 文件，返回最终文件名；失败时抛出 UsageError。"""
        filename = Path(str(getattr(upload, "filename", "") or "")).name
        if not filename:
            raise UsageError("上传项缺少文件名")
        if Path(filename).suffix.lower() not in self.LORA_FILE_SUFFIXES:
            raise UsageError(f"{filename}：只支持 safetensors、pt、ckpt、bin 文件")
        target = directory / filename
        if target.exists():
            raise UsageError(f"文件已存在：{filename}，请先处理原文件")
        await upload.save(str(target))
        return filename

    async def _upload_lora_file(self, *, category: str = "", batch: bool = False):
        """上传 LoRA；``batch=True`` 时支持一次提交多个文件（批量导入）。"""
        from astrbot.api.web import json_response, request

        try:
            directory = self._model_folder_path("loras")
            directory.mkdir(parents=True, exist_ok=True)
            files = await self._request_files(request)
            uploads = self._collect_upload_items(files, self.LORA_UPLOAD_FIELDS)
            if not uploads:
                return json_response({"error": "请选择 LoRA 文件"}, status_code=400)
            if not batch:
                filename = await self._save_single_lora_upload(uploads[0], directory)
                self._record_lora_upload([filename], category)
                return json_response(
                    {"ok": True, "file_name": filename, "path": str(directory / filename)}
                )

            saved: list[str] = []
            failed: list[dict[str, str]] = []
            for upload in uploads:
                try:
                    saved.append(await self._save_single_lora_upload(upload, directory))
                except UsageError as exc:
                    failed.append(
                        {
                            "file_name": Path(str(getattr(upload, "filename", "") or "")).name,
                            "error": str(exc),
                        }
                    )
            self._record_lora_upload(saved, category)
            if not saved:
                return json_response(
                    {
                        "error": "全部文件都导入失败",
                        "ok": False,
                        "saved": [],
                        "failed": failed,
                    },
                    status_code=400,
                )
            return json_response(
                {
                    "ok": True,
                    "batch": True,
                    "count": len(saved),
                    "saved": saved,
                    "failed": failed,
                    "file_name": saved[0],
                }
            )
        except (OSError, UsageError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    def _record_lora_upload(self, filenames: list[str], category: str = "") -> None:
        """登记新上传的 LoRA：刷新缓存、记录导入顺序并可选归类。"""
        if not filenames:
            return
        self._lora_names_cache = None
        now = time.time()
        for filename in filenames:
            self.lora_download_order[filename] = now
        self._save_lora_download_order()
        self._apply_lora_category(filenames, category)

    async def api_upload_lora(self):
        return await self._upload_lora_file()

    async def api_upload_lora_batch(self):
        """批量导入 LoRA：一次请求提交多个文件，逐个保存并返回明细。"""
        return await self._upload_lora_file(batch=True)

    async def api_upload_style_lora(self):
        """上传入口专门用于画风页，完成后自动归入“画风”分类。"""
        return await self._upload_lora_file(category="画风")

    async def api_download_lora(self):
        """创建后台下载任务，WebUI 通过 download_lora_progress 读取进度。"""
        from astrbot.api.web import error_response, json_response, request

        try:
            data = await self._request_json(request, {})
            if not isinstance(data, dict):
                return error_response("请求体必须是对象", status_code=400)
            link = self._validate_civitai_link(data.get("url", data.get("civitai_url", "")))
            if not link:
                return error_response("请输入 CivitAI 模型链接", status_code=400)
            job_id = uuid.uuid4().hex
            self._download_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "stage": "等待开始",
                "progress": 0,
                "downloaded": 0,
                "total": 0,
                "file_name": "",
                "download_mode": self._civitai_download_mode(),
                "error": "",
            }
            task = asyncio.create_task(
                self._run_lora_download_job(
                    job_id,
                    link,
                    bool(data.get("overwrite", False)),
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return json_response({"ok": True, "job_id": job_id, "status": "queued"})
        except (UsageError, ValueError, TypeError) as exc:
            return error_response(str(exc), status_code=400)
        except Exception as exc:
            logger.exception("[%s] 创建 LoRA 下载任务失败", PLUGIN_NAME)
            return error_response(f"创建 LoRA 下载任务失败：{exc}", status_code=500)

    def _update_download_job(self, job_id: str, **changes: Any) -> None:
        job = self._download_jobs.get(job_id)
        if job is not None:
            job.update(changes)

    async def _run_lora_download_job(self, job_id: str, link: str, overwrite: bool) -> None:
        """后台下载 LoRA 并持续记录字节数，避免 WebUI 请求被大文件阻塞。

        下载方式由 ``civitai_download_mode`` 决定：
        ``nokey`` 全程免 API Key；``token`` 始终走 CivitAI API；
        ``auto`` 先尝试免 API 解析，失败后再退回 API Key。
        """
        temp_path: Path | None = None
        mode = self._civitai_download_mode()

        async def resolve(*, retry: bool = False) -> tuple[str, str, int, bool]:
            """返回 (下载地址, 文件名, 版本 ID, 是否使用了 API Key)。"""
            if mode == "nokey":
                url, name, version = await self._civitai_resolve_download_nokey(link)
                return url, name, version, False
            if mode == "token":
                url, name, version = await self._civitai_download_file(link)
                return url, name, version, True
            # auto：首次优先免 API；重试时直接走 API，避免在同一解析分支反复失败。
            if not retry:
                try:
                    self._update_download_job(job_id, stage="正在免 API 解析 C 站下载地址")
                    url, name, version = await self._civitai_resolve_download_nokey(link)
                    if url:
                        return url, name, version, False
                except UsageError as exc:
                    logger.info("[%s] 免 API 解析未成功，改用 CivitAI API：%s", PLUGIN_NAME, exc)
                    self._update_download_job(
                        job_id, stage="免 API 解析未成功，改用 CivitAI API"
                    )
            url, name, version = await self._civitai_download_file(link)
            return url, name, version, True

        try:
            self._update_download_job(job_id, status="running", stage="正在读取 CivitAI 模型信息")
            download_url, filename, version_id, used_token = await resolve()
            directory = self._model_folder_path("loras")
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / filename
            if target.exists() and not overwrite:
                raise UsageError(f"文件已存在：{filename}，请先删除原文件或确认覆盖")
            temp_path = directory / f".astrbot_download_{uuid.uuid4().hex}.tmp"
            self._update_download_job(
                job_id,
                stage="正在下载模型文件",
                file_name=filename,
                version_id=version_id,
                download_mode="免 API 直链下载" if not used_token else "CivitAI API 下载",
            )
            headers = self._civitai_headers(
                "application/octet-stream, */*;q=0.8",
                base_url=self._civitai_link_api_base(link) or self._civitai_base_url(),
                use_token=used_token,
            )
            retry_statuses = {408, 429, 500, 502, 503, 504}
            header_filename = ""
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60, read=180),
                follow_redirects=True,
            ) as client:
                for attempt in range(3):
                    current_url = download_url
                    if attempt:
                        # CivitAI 的 CDN 地址带短时签名；重试时重新获取，避免复用过期地址。
                        current_url, _, _, used_token = await resolve(retry=True)
                        headers = self._civitai_headers(
                            "application/octet-stream, */*;q=0.8",
                            base_url=self._civitai_link_api_base(link) or self._civitai_base_url(),
                            use_token=used_token,
                        )
                    try:
                        async with client.stream("GET", current_url, headers=headers) as response:
                            if response.status_code in retry_statuses:
                                payload = await response.aread()
                                detail = self._civitai_error_detail(payload)
                                if attempt < 2:
                                    self._update_download_job(
                                        stage=f"CivitAI 暂时不可用，正在重试（{attempt + 1}/2）"
                                    )
                                    await asyncio.sleep(1.2 * (attempt + 1))
                                    continue
                                raise UsageError(
                                    self._civitai_http_error(response.status_code, detail, "下载")
                                )
                            if response.is_error:
                                payload = await response.aread()
                                detail = self._civitai_error_detail(payload)
                                raise UsageError(
                                    self._civitai_http_error(response.status_code, detail, "下载")
                                )

                            content_type = str(response.headers.get("content-type", "")).lower()
                            header_filename = self._civitai_filename_from_headers(
                                response.headers, header_filename
                            )
                            stream = response.aiter_bytes(1024 * 1024)
                            try:
                                first_chunk = await stream.__anext__()
                            except StopAsyncIteration:
                                first_chunk = b""
                            sample = first_chunk[:4096].lstrip()
                            looks_like_error = (
                                sample.startswith((b"{", b"[", b"<"))
                                and (
                                    "json" in content_type
                                    or "text" in content_type
                                    or "xml" in content_type
                                    or sample.startswith((b"<html", b"<!doctype"))
                                )
                            )
                            if looks_like_error:
                                detail = self._civitai_error_detail(first_chunk)
                                raise UsageError(
                                    f"CivitAI 返回了错误内容而不是模型文件：{detail}"
                                )
                            try:
                                total = max(0, int(response.headers.get("content-length", "0") or 0))
                            except (TypeError, ValueError):
                                total = 0
                            downloaded = 0
                            self._update_download_job(job_id, total=total, downloaded=0, progress=0)
                            with temp_path.open("wb") as output:
                                if first_chunk:
                                    output.write(first_chunk)
                                    downloaded = len(first_chunk)
                                    progress = round(downloaded * 100 / total, 1) if total else 0
                                    self._update_download_job(
                                        job_id,
                                        downloaded=downloaded,
                                        progress=min(99.9, progress) if total else 0,
                                    )
                                async for chunk in stream:
                                    if not chunk:
                                        continue
                                    output.write(chunk)
                                    downloaded += len(chunk)
                                    progress = round(downloaded * 100 / total, 1) if total else 0
                                    self._update_download_job(
                                        job_id,
                                        downloaded=downloaded,
                                        progress=min(99.9, progress) if total else 0,
                                    )
                        break
                    except (httpx.TransportError, asyncio.TimeoutError) as exc:
                        if attempt >= 2:
                            raise UsageError(f"CivitAI 下载连接中断：{exc}") from exc
                        self._update_download_job(
                            job_id,
                            stage=f"下载连接中断，正在重试（{attempt + 1}/2）",
                        )
                        await asyncio.sleep(1.2 * (attempt + 1))
            if not temp_path.exists() or temp_path.stat().st_size < 1024:
                raise UsageError("下载结果为空或文件不完整")
            # 免 API 解析时只能拿到「civitai_版本ID.safetensors」这类占位名；
            # 下载完成后用 CDN 响应头里的真实文件名修正，避免用户看到无意义的名字。
            if header_filename and filename.startswith("civitai_"):
                safe_name = self._safe_lora_filename(header_filename, filename)
                if Path(safe_name).suffix.lower() in {".safetensors", ".pt", ".ckpt", ".bin"}:
                    candidate = directory / safe_name
                    if safe_name != filename and (overwrite or not candidate.exists()):
                        filename = safe_name
                        target = candidate
            # 若真实文件名与其它的同名文件冲突且未允许覆盖，追加短哈希避免覆盖既有文件。
            if target.exists() and not overwrite:
                target = directory / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
                filename = target.name
            temp_path.replace(target)
            self.lora_download_order[filename] = time.time()
            self._save_lora_download_order()
            self._lora_names_cache = None
            self._update_download_job(
                job_id,
                status="done",
                stage="下载完成，等待 WebUI 刷新",
                progress=100,
                file_name=filename,
                downloaded=target.stat().st_size,
                total=target.stat().st_size,
                path=str(target),
            )
        except asyncio.CancelledError:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        except (UsageError, httpx.HTTPError, OSError, ValueError) as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            message = str(exc)
            if not isinstance(exc, UsageError):
                message = f"下载 LoRA 失败：{message}"
            self._update_download_job(job_id, status="error", stage="下载失败", error=message)
        except Exception as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            logger.exception("[%s] LoRA 后台下载任务异常", PLUGIN_NAME)
            self._update_download_job(job_id, status="error", stage="下载失败", error=f"下载 LoRA 失败：{exc}")

    async def api_download_lora_progress(self):
        from astrbot.api.web import error_response, json_response, request

        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return error_response("请求体必须是对象", status_code=400)
        job_id = str(data.get("job_id", "") or "").strip()
        job = self._download_jobs.get(job_id)
        if not job:
            return error_response("下载任务不存在或已过期", status_code=404)
        return json_response(dict(job))

    async def api_delete_lora(self):
        """删除本地 LoRA，并清理插件中该文件的所有管理信息。"""
        from astrbot.api.web import json_response, request

        try:
            data = await self._request_json(request, {})
            if not isinstance(data, dict):
                return json_response({"error": "请求体必须是对象"}, status_code=400)
            raw_filename = str(data.get("file_name", data.get("filename", "")) or "").strip()
            # WebUI 传的是相对路径；统一分隔符，兼容旧页面传来的 Windows 路径。
            normalized_filename = raw_filename.replace("\\", "/")
            relative = Path(normalized_filename)
            if not raw_filename or relative.is_absolute() or ".." in relative.parts:
                return json_response({"error": "请选择要删除的 LoRA"}, status_code=400)
            available = await self._local_lora_names()
            filename = relative.as_posix()
            if filename not in available:
                return json_response({"error": "LoRA 文件不存在"}, status_code=404)
            # 文件可能位于额外模型路径，因此按相对路径在所有 LoRA 目录里查找。
            target = self._lora_relative_target(filename)
            if target is None:
                return json_response({"error": "本地 LoRA 文件不存在"}, status_code=404)
            target.unlink()
            self._lora_names_cache = None
            metadata_keys = {filename, relative.name}
            for key in metadata_keys:
                self.lora_aliases.pop(key, None)
                self.lora_command_aliases.pop(key, None)
                self.lora_categories.pop(key, None)
                self.lora_download_order.pop(key, None)
                self.civitai_cache.pop(key, None)
                self.civitai_links.pop(key, None)
                self.civitai_overrides.pop(key, None)
                self.lora_presets.pop(key, None)
                self.presets.sync_auto(
                    source="civitai_lora",
                    source_key=key,
                    names=[],
                    content="",
                )
            enabled = list(self._get("lora_list", []) or [])
            self._set(
                "lora_list",
                [
                    item for item in enabled
                    if str(item).rsplit(":", 1)[0].casefold() not in {value.casefold() for value in metadata_keys}
                ],
            )
            raw_style_list = self._get("style_lora_list", [])
            if isinstance(raw_style_list, str):
                raw_style_list = re.split(r"[,，\n]+", raw_style_list)
            if isinstance(raw_style_list, (list, tuple)):
                self._set(
                    "style_lora_list",
                    [
                        item for item in raw_style_list
                        if str(item).rsplit(":", 1)[0].replace("\\", "/").casefold()
                        not in {value.replace("\\", "/").casefold() for value in metadata_keys}
                    ],
                )
            for style_key in ("style_lora_aliases", "style_lora_weights"):
                mapping = self._get(style_key, {})
                if isinstance(mapping, dict):
                    self._set(
                        style_key,
                        {
                            key: value
                            for key, value in mapping.items()
                            if str(key).replace("\\", "/").casefold()
                            not in {item.replace("\\", "/").casefold() for item in metadata_keys}
                        },
                    )
            self._save_lora_aliases()
            self._save_lora_command_aliases()
            self._save_lora_categories()
            self._save_lora_download_order()
            self._save_civitai_cache()
            self._save_civitai_links()
            self._save_civitai_overrides()
            self._save_lora_presets()
            self._save_config()
            return json_response({
                "ok": True,
                "file_name": filename,
                "lora_list": list(self._get("lora_list", []) or []),
            })
        except (ComfyError, OSError, UsageError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    async def _local_lora_names(self) -> list[str]:
        """读取本地 LoRA 文件名，管理操作不依赖 ComfyUI 的缓存列表。

        模型可能分布在默认位置和 extra_model_paths.yaml 声明的多个目录里，
        因此逐个扫描后按相对路径合并去重。
        """
        allowed = {suffix.lower() for suffix in self.LORA_FILE_SUFFIXES}
        names: list[str] = []
        seen: set[str] = set()
        for directory in self._lora_dirs():
            try:
                if not directory.is_dir():
                    continue
                for path in directory.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in allowed:
                        continue
                    name = path.relative_to(directory).as_posix()
                    if name in seen:
                        continue
                    seen.add(name)
                    names.append(name)
            except OSError:
                continue
        return self._order_loras(names)

    async def api_upload_workflow(self):
        from astrbot.api.web import json_response, request

        directory = self._workflow_dir()
        try:
            files = await self._request_files(request)
            upload = files.get("file")
            filename = Path(str(getattr(upload, "filename", "") or "")).name
            if not upload or not filename:
                return json_response({"error": "请选择工作流 JSON 文件"}, status_code=400)
            if Path(filename).suffix.lower() != ".json":
                return json_response({"error": "工作流必须是 .json 文件"}, status_code=400)
            target = directory / filename
            if target.exists():
                return json_response({"error": f"工作流已存在：{filename}，请改名后再上传"}, status_code=409)
            await upload.save(str(target))
            try:
                parsed = load_workflow(target)
                # Store editor workflows as API workflows so subsequent
                # requests do not need to repeat conversion or retain UI-only
                # metadata in the queue payload.
                target.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            except WorkflowError:
                target.unlink(missing_ok=True)
                return json_response({"error": "工作流格式无效，请上传 ComfyUI 工作流 JSON"}, status_code=400)
            return json_response({"ok": True, "file_name": filename, "path": str(target)})
        except (OSError, UsageError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    def _compatible_models(self, env: dict[str, Any], mode: str = "txt2img") -> list[str]:
        candidates = list(dict.fromkeys(env["diffusion_models"] + env["checkpoints"]))
        try:
            workflow = load_api_workflow(self._workflow_path(mode))
        except WorkflowError:
            return candidates
        if any(
            isinstance(node, dict) and node.get("class_type") == "AnimaBoosterLoader"
            for node in workflow.values()
        ):
            # AnimaBoosterLoader 的 model_name 输入只接受 diffusion_models。
            return list(env["diffusion_models"])
        return candidates

    def _lora_local_path(self, file_name: str) -> Path | None:
        """按 ComfyUI 的 LoRA 相对路径找到本地文件（含额外模型路径）。"""
        return self._lora_relative_target(file_name)

    def _lora_architecture_profile(self, file_name: str) -> str:
        """识别 LoRA 适用的架构：anima_base、anima_29b、flux2 或 unknown。

        Anima Base 是 28 层（块编号 0-27），Anima-2.9B 是扩展后的 40 层。
        维里奈 LoRA 的元数据为 ``networks.lora_anima`` 且最大块编号为 27，
        因此只能安全地用于 Base，不应悄悄塞进 2.9B 或 Flux2。
        """
        key = str(file_name or "").replace("\\", "/").casefold()
        if key in self._lora_profile_cache:
            return self._lora_profile_cache[key]
        path = self._lora_local_path(file_name)
        profile = "unknown"
        if path is not None:
            try:
                with path.open("rb") as stream:
                    header_size_raw = stream.read(8)
                    if len(header_size_raw) == 8:
                        header_size = struct.unpack("<Q", header_size_raw)[0]
                        # safetensors 头部只包含 JSON 元数据和张量索引，正常远小于
                        # 64 MiB；超过上限时放弃检查，避免异常文件占用内存。
                        if 0 < header_size <= 64 * 1024 * 1024:
                            header = stream.read(header_size)
                            data = json.loads(header.decode("utf-8"))
                            metadata = data.get("__metadata__", {}) if isinstance(data, dict) else {}
                            metadata_text = json.dumps(metadata, ensure_ascii=False).casefold()
                            tensor_text = header.decode("utf-8", errors="ignore")
                            if "flux" in metadata_text or "flux" in tensor_text.casefold():
                                profile = "flux2"
                            elif "lora_anima" in metadata_text or "ss_base_model_version" in metadata_text:
                                blocks = [
                                    int(match.group(1))
                                    for match in re.finditer(r"lora_unet_blocks_(\d+)_", tensor_text)
                                ]
                                if blocks:
                                    profile = "anima_29b" if max(blocks) >= 28 else "anima_base"
                                else:
                                    profile = "anima_base"
            except (OSError, ValueError, UnicodeError, struct.error):
                profile = "unknown"
        self._lora_profile_cache[key] = profile
        return profile

    def _assert_flux2_loras_compatible(self, loras: list[str]) -> None:
        incompatible = []
        for item in loras:
            name = item.rsplit(":", 1)[0].strip()
            profile = self._lora_architecture_profile(name)
            if profile in {"anima_base", "anima_29b"}:
                incompatible.append(name)
        if incompatible:
            raise UsageError(
                "当前图生图工作流使用 Flux2 Klein，不能加载 Anima/SD LoRA："
                + "、".join(incompatible)
                + "。请使用 /文生图 维里奈；如果要在图生图使用 LoRA，需准备 Flux2 版本的该 LoRA。"
            )

    def _assert_qwen_loras_compatible(self, loras: list[str]) -> None:
        """允许 Qwen Image Edit LoRA，并拒绝已明确识别为 Flux2 的 LoRA。"""
        incompatible = []
        for item in loras:
            name = item.rsplit(":", 1)[0].strip()
            if self._lora_architecture_profile(name) == "flux2":
                incompatible.append(name)
        if incompatible:
            raise UsageError(
                "当前 Qwen 图生图工作流不能加载 Flux2 LoRA："
                + "、".join(incompatible)
                + "。请在图生图设置中选择 Qwen Image Edit 兼容的 LoRA。"
            )

    def _assert_anima_loras_compatible(self, model: str, loras: list[str]) -> None:
        """只提示 Anima LoRA 与核心模型不匹配，不替用户切换模型。"""
        profiles = {
            self._lora_architecture_profile(item.rsplit(":", 1)[0].strip())
            for item in loras
        }
        profiles.intersection_update({"anima_base", "anima_29b"})
        if len(profiles) > 1:
            raise UsageError(
                "本次同时使用了 Anima Base LoRA 和 Anima-2.9B LoRA；两种架构不能混用，"
                "请分开绘制。"
            )
        if not profiles:
            return
        profile = next(iter(profiles))
        if profile == "anima_base" and is_anima_29b_model(model):
            raise UsageError(
                "维里奈 LoRA 是 Anima Base LoRA，当前核心模型是 Anima-2.9B，二者不兼容。"
                "插件不会自动切换模型，请在 WebUI 的工作流/模型界面手动选择"
                "anima-base-v1.0.safetensors 后再绘制。"
            )
        if profile == "anima_29b" and not is_anima_29b_model(model):
            raise UsageError(
                "当前 LoRA 是 Anima-2.9B LoRA，当前核心模型不是 2.9B。"
                "插件不会自动切换模型，请在 WebUI 手动选择对应的 Anima-2.9B 核心模型。"
            )

    def _anima_29b_patch_installed(self) -> bool:
        """Check whether the CivitAI Anima-2.9B ComfyUI patch is present."""
        root = detect_comfyui_root(str(self._get("comfyui_root", "")))
        if not root:
            return False
        candidates = (
            Path(root) / "custom_nodes" / "ComfyUI-Anima-2.9B" / "__init__.py",
            Path(root) / "custom_nodes" / "comfyui-anima-2.9b" / "__init__.py",
        )
        return any(path.is_file() for path in candidates)

    @staticmethod
    def _model_profile(model_name: str) -> dict[str, Any]:
        if is_anima_29b_model(model_name):
            return {
                "family": "Anima-2.9B",
                "description": "需要 ComfyUI-Anima-2.9B 动态层数补丁；推荐 28-50 步、CFG 3.5-5",
                "recommended": anima_sampling_defaults(model_name),
            }
        return {
            "family": "Anima Base 1.0",
            "description": "使用标准 Anima 28 层模型配置；推荐 30 步、CFG 5",
            "recommended": anima_sampling_defaults(model_name),
        }

    def _model_name(self, env: dict[str, Any], requested: str = "", mode: str = "txt2img") -> str:
        candidates = self._compatible_models(env, mode)
        value = requested or str(self._get("model_name", "") or "")
        if not value and candidates:
            value = next((x for x in candidates if "anima" in x.lower()), candidates[0])
        if value not in candidates:
            raise UsageError(f"当前工作流不支持核心模型：{value}；请先使用 /模型 列表")
        return value

    def _build_semaphore(self) -> asyncio.Semaphore:
        limit = max(1, min(4, int(self._get("max_concurrent", 1) or 1)))
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(limit)
        return self.semaphore

    async def _persist_input_image(self, path: str | os.PathLike[str]) -> str:
        """把 AstrBot 临时媒体复制到插件目录，供后台绘图任务使用。"""
        source = Path(path)
        if not source.is_file():
            return ""
        suffix = source.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            suffix = ".img"
        target = self.input_cache_dir / f"input_{uuid.uuid4().hex}{suffix}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                # copy2 能在 Windows 上完成一次打开、读取、关闭，随后后台任务
                # 只使用 target，不再依赖 AstrBot 的临时文件生命周期。
                await asyncio.to_thread(shutil.copy2, source, target)
                if target.is_file() and target.stat().st_size > 0:
                    return str(target)
            except (OSError, shutil.Error) as exc:
                last_error = exc
                target.unlink(missing_ok=True)
            if attempt < 2:
                await asyncio.sleep(0.12 * (attempt + 1))
        target.unlink(missing_ok=True)
        if last_error:
            logger.warning("[%s] 缓存输入图片失败：%s -> %s：%s", PLUGIN_NAME, source, target, last_error)
        return ""

    @staticmethod
    def _image_ref_key(image_ref: str) -> str:
        """为图片引用生成稳定键，兼容本地路径、file URI 和远程 URL。"""
        value = str(image_ref or "").strip()
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.scheme.lower() in {"http", "https"}:
            # URL 的主机名和协议不区分大小写，但路径、查询参数可能区分，
            # 因此只规范协议/主机，不删除签名参数。
            host = (parsed.hostname or "").lower()
            port = parsed.port
            netloc = host
            if port is not None:
                netloc = f"{host}:{port}"
            if parsed.username:
                auth = parsed.username
                if parsed.password:
                    auth += f":{parsed.password}"
                netloc = f"{auth}@{netloc}"
            # 使用 ParseResult 自带的序列化方法，避免旧版 AstrBot 进程
            # 载入不完整导入时再次出现 ``urlunparse 未定义``。
            return parsed._replace(
                scheme=parsed.scheme.lower(),
                netloc=netloc,
            ).geturl()
        if parsed.scheme.lower() == "file":
            value = unquote(parsed.path or "")
            if re.match(r"^/[A-Za-z]:", value):
                value = value[1:]
            elif parsed.netloc:
                value = f"//{parsed.netloc}{value}"
        try:
            return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))
        except (OSError, RuntimeError, ValueError):
            return unicodedata.normalize("NFKC", value).casefold()

    @staticmethod
    def _image_content_key(path: str | os.PathLike[str]) -> str:
        """用文件大小和 SHA-256 识别不同引用指向的同一张图片。"""
        try:
            source = Path(path)
            stat = source.stat()
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return f"{stat.st_size}:{digest.hexdigest()}"
        except (OSError, ValueError):
            return ""

    async def _persist_image_ref(self, image_ref: str) -> str:
        """将本地路径、file URI 或远程图片引用缓存到插件目录。"""
        value = str(image_ref or "").strip()
        if not value:
            return ""
        cache = getattr(self, "_image_ref_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._image_ref_cache = cache

        def cached_path(key: str) -> str:
            if not key:
                return ""
            cached = str(cache.get(key, "") or "")
            if cached and Path(cached).is_file():
                return cached
            if cached:
                cache.pop(key, None)
            return ""

        ref_key = self._image_ref_key(value)
        cached = cached_path(ref_key)
        if cached:
            return cached
        if os.path.isfile(value):
            path = value
        else:
            try:
                path = await Image(file=value).convert_to_file_path()
            except Exception as exc:
                logger.debug("[%s] 转换图片引用失败：%s：%s", PLUGIN_NAME, value[:160], exc)
                return ""
        resolved_key = self._image_ref_key(str(path))
        cached = cached_path(resolved_key)
        if cached:
            cache[ref_key] = cached
            return cached
        content_key = await asyncio.to_thread(self._image_content_key, str(path))
        cached = cached_path(f"content:{content_key}") if content_key else ""
        if cached:
            cache[ref_key] = cached
            if resolved_key:
                cache[resolved_key] = cached
            return cached
        cached = await self._persist_input_image(path)
        if not cached:
            return ""
        if ref_key:
            cache[ref_key] = cached
        if resolved_key:
            cache[resolved_key] = cached
        if content_key:
            cache[f"content:{content_key}"] = cached
        return cached

    @staticmethod
    def _forward_component_id(component: Any) -> str:
        """读取 AstrBot Forward 组件或原始 OneBot forward 段的消息 ID。"""
        if isinstance(component, Forward):
            for attr in ("id", "message_id", "resid", "res_id"):
                value = getattr(component, attr, None)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return ""
        if isinstance(component, Json):
            data = getattr(component, "data", None)
            if isinstance(data, Mapping):
                for key in ("resid", "res_id", "id", "message_id"):
                    value = data.get(key)
                    if value is not None and str(value).strip():
                        return str(value).strip()
                raw = data.get("data") or data.get("json")
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, Mapping):
                        for key in ("resid", "res_id", "id", "message_id"):
                            value = parsed.get(key)
                            if value is not None and str(value).strip():
                                return str(value).strip()
                        config = parsed.get("config")
                        if isinstance(config, Mapping) and config.get("resid"):
                            return str(config["resid"]).strip()
        if isinstance(component, Mapping):
            component_type = str(component.get("type", "")).strip().lower()
            if component_type == "json":
                data = component.get("data")
                data = data if isinstance(data, Mapping) else {}
                for key in ("resid", "res_id", "id", "message_id"):
                    value = data.get(key)
                    if value is not None and str(value).strip():
                        return str(value).strip()
                raw = data.get("data") or data.get("json")
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, Mapping):
                        for key in ("resid", "res_id", "id", "message_id"):
                            value = parsed.get(key)
                            if value is not None and str(value).strip():
                                return str(value).strip()
                return ""
            if component_type not in {"forward", "forward_msg", "nodes"}:
                return ""
            data = component.get("data")
            data = data if isinstance(data, Mapping) else component
            for key in ("id", "message_id", "resid", "res_id"):
                value = data.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    @staticmethod
    def _reply_component_id(component: Any) -> str:
        """读取 AstrBot Reply 或原始 OneBot reply 段的消息 ID。"""
        if isinstance(component, Reply):
            value = getattr(component, "id", None)
            value = str(value or "").strip()
            return value if value and value != "0" else ""
        if isinstance(component, Mapping):
            component_type = str(component.get("type", "")).strip().lower()
            if component_type != "reply":
                return ""
            data = component.get("data")
            data = data if isinstance(data, Mapping) else component
            for key in ("id", "message_id"):
                value = data.get(key)
                if value is not None and str(value).strip() and str(value).strip() != "0":
                    return str(value).strip()
        return ""

    @staticmethod
    def _raw_event_components(event: AstrMessageEvent) -> list[Any]:
        """读取适配器保留的原始 OneBot 消息段，兼容解析器丢失 Forward 的情况。"""
        owners = [getattr(event, "message_obj", None), event]
        all_components: list[Any] = []
        for owner in owners:
            raw = getattr(owner, "raw_message", None)
            if isinstance(raw, list):
                all_components.extend(raw)
                continue
            if not isinstance(raw, Mapping):
                continue
            segments = raw.get("message") or raw.get("messages")
            if isinstance(segments, list):
                all_components.extend(segments)
            # 少数适配器把回复目标放在事件顶层，而不是放进 message
            # 数组。统一转换为 OneBot reply 段，交给后面的 get_msg 逻辑。
            for key in ("reply", "quoted_message", "quote"):
                value = raw.get(key)
                if value is None or value == "":
                    continue
                if isinstance(value, Mapping):
                    value_type = str(value.get("type", "") or "").strip().lower()
                    if value_type == "reply":
                        all_components.append(value)
                    else:
                        all_components.append({"type": "reply", "data": dict(value)})
                else:
                    all_components.append({"type": "reply", "data": {"id": str(value)}})
            for key in ("reply_id", "quoted_message_id", "quote_id"):
                value = raw.get(key)
                if value is not None and str(value).strip() and str(value).strip() != "0":
                    all_components.append({"type": "reply", "data": {"id": str(value).strip()}})
        return all_components

    @staticmethod
    def _event_level_reference_components(event: AstrMessageEvent) -> list[Any]:
        """读取适配器挂在事件对象上的回复目标。

        AstrBot 的标准消息链通常包含 Reply，但部分平台/旧版适配器会把
        回复目标单独暴露为 ``event.reply`` 或 ``event.reply_id``。这些字段
        仍然是用户明确指定的输入图片来源，不能被当作普通聊天历史处理。
        """
        components: list[Any] = []
        owners = [event, getattr(event, "message_obj", None)]
        for owner in owners:
            if owner is None:
                continue
            for key in ("reply", "quoted_message", "quote"):
                value = getattr(owner, key, None)
                if callable(value) or value is None or value == "":
                    continue
                if isinstance(value, Mapping):
                    value_type = str(value.get("type", "") or "").strip().lower()
                    if value_type == "reply":
                        components.append(value)
                    else:
                        components.append({"type": "reply", "data": dict(value)})
                elif isinstance(value, (Reply, Forward, Node, Nodes, Image)):
                    components.append(value)
                else:
                    text = str(value).strip()
                    if text and text != "0":
                        components.append({"type": "reply", "data": {"id": text}})
            for key in ("reply_id", "quoted_message_id", "quote_id"):
                value = getattr(owner, key, None)
                if value is None or str(value).strip() in {"", "0"}:
                    continue
                components.append({"type": "reply", "data": {"id": str(value).strip()}})
        return components

    @classmethod
    def _component_has_explicit_reference(cls, component: Any, depth: int = 0) -> bool:
        """判断消息段是否明确指向用户引用/转发的消息。"""
        if component is None or depth > 8:
            return False
        if cls._forward_component_id(component) or cls._reply_component_id(component):
            return True
        if isinstance(component, Reply):
            return any(
                cls._component_has_explicit_reference(nested, depth + 1)
                for nested in (component.chain or [])
            )
        if isinstance(component, Node):
            return any(
                cls._component_has_explicit_reference(nested, depth + 1)
                for nested in (component.content or [])
            )
        if isinstance(component, Nodes):
            return any(
                cls._component_has_explicit_reference(nested, depth + 1)
                for node in (component.nodes or [])
                for nested in (node.content or [])
            )
        if isinstance(component, Mapping):
            data = component.get("data")
            data = data if isinstance(data, Mapping) else component
            nested_values: list[Any] = []
            for key in ("content", "message", "messages", "nodes", "chain"):
                value = data.get(key)
                if isinstance(value, list):
                    nested_values.extend(value)
            return any(
                cls._component_has_explicit_reference(nested, depth + 1)
                for nested in nested_values
            )
        return False

    @classmethod
    def _event_has_explicit_image_reference(cls, event: AstrMessageEvent) -> bool:
        """判断本次消息是否明确引用了图片/合并转发。"""
        components = cls._event_components(event)
        components.extend(cls._raw_event_components(event))
        components.extend(cls._event_level_reference_components(event))
        return any(cls._component_has_explicit_reference(component) for component in components)

    @staticmethod
    def _event_components(event: AstrMessageEvent) -> list[Any]:
        try:
            return list(event.get_messages() or [])
        except Exception:
            return []

    @classmethod
    def _event_may_have_remote_images(cls, event: AstrMessageEvent) -> bool:
        """判断是否值得查询消息 ID 对应的远程引用/转发内容。"""
        def contains_reference(component: Any, depth: int = 0) -> bool:
            if component is None or depth > 8:
                return False
            if cls._forward_component_id(component):
                return True
            if isinstance(component, Reply):
                # Reply 可能只有 id，正文和合并转发节点要通过 get_msg 获取。
                return True
            if isinstance(component, Node):
                return any(contains_reference(item, depth + 1) for item in (component.content or []))
            if isinstance(component, Nodes):
                return any(
                    contains_reference(item, depth + 1)
                    for node in (component.nodes or [])
                    for item in (node.content or [])
                )
            return False

        components = cls._event_components(event)
        components.extend(cls._raw_event_components(event))
        components.extend(cls._event_level_reference_components(event))
        return any(contains_reference(component) for component in components)

    def _cached_event_images(self, event: AstrMessageEvent) -> list[str]:
        values = getattr(event, "_comfyui_input_images", None)
        if not isinstance(values, list):
            return []
        return [
            str(value)
            for value in values
            if str(value or "").strip() and Path(str(value)).is_file()
        ]

    @staticmethod
    def _remember_event_images(event: AstrMessageEvent, images: list[str]) -> None:
        if not images:
            return
        try:
            setattr(event, "_comfyui_input_images", list(dict.fromkeys(images)))
        except Exception:
            pass

    async def _ensure_event_images(self, event: AstrMessageEvent) -> list[str]:
        """提前解析只有引用 ID 的消息，让 LLM 工具路由也能识别图生图。"""
        cached = self._cached_event_images(event)
        if cached:
            return cached
        images = await self._extract_images(event)
        self._remember_event_images(event, images)
        return images

    async def _inject_event_images_to_llm_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> list[str]:
        """把引用/合并转发中的实际图片加入 AstrBot 当前多模态请求。"""
        if not self._event_has_image_hint(event):
            return []
        images = await self._ensure_event_images(event)
        if not images:
            return []
        existing = {str(value).strip() for value in (req.image_urls or []) if value}
        added: list[str] = []
        for image_path in images:
            normalized = str(image_path).strip()
            if not normalized or normalized in existing:
                continue
            req.image_urls.append(normalized)
            existing.add(normalized)
            added.append(normalized)
        if added:
            logger.info(
                "[%s] 已将引用/合并转发图片注入 AstrBot LLM：数量=%d",
                PLUGIN_NAME,
                len(added),
            )
        return images

    @staticmethod
    def _normalize_forward_payload(payload: Any) -> dict[str, Any] | None:
        """Normalize the different get_forward_msg response envelopes.

        NapCat, go-cqhttp and a few OneBot adapters do not use the same
        envelope.  The useful part is always a list of forward nodes, but it
        may be named ``messages``, ``message``, ``nodes`` or ``nodeList`` and
        may be wrapped in one or more ``data`` objects.
        """
        if isinstance(payload, list):
            return {"messages": payload}
        if not isinstance(payload, Mapping):
            return None

        status = str(payload.get("status", "") or "").strip().lower()
        if status in {"failed", "error"}:
            return None
        retcode = payload.get("retcode")
        if retcode not in (None, 0, "0"):
            return None

        for key in ("messages", "message", "nodes", "nodeList"):
            value = payload.get(key)
            if isinstance(value, list):
                return {"messages": value}
            if isinstance(value, Mapping):
                return {"messages": [dict(value)]}

        data = payload.get("data")
        if isinstance(data, (Mapping, list)):
            normalized = ComfyUIAIStudio._normalize_forward_payload(data)
            if normalized:
                return normalized

        # Some adapters return a single node without a messages wrapper.
        if any(key in payload for key in ("sender", "content", "message")):
            return {"messages": [dict(payload)]}
        return None

    @classmethod
    def _scan_forward_payload(
        cls,
        payload: Any,
        depth: int = 0,
    ) -> tuple[list[str], list[str]]:
        """Find image references and nested forward IDs without trusting one parser.

        The scanner is deliberately limited to typed OneBot segments.  It
        does not treat arbitrary ``file`` or ``id`` fields such as sender IDs
        as images/forward messages.
        """
        if payload is None or depth > 12:
            return [], []
        image_refs: list[str] = []
        forward_ids: list[str] = []

        def visit(value: Any, level: int) -> None:
            if value is None or level > 12:
                return
            if isinstance(value, str):
                text = value.strip()
                if text[:1] in {"{", "["} and len(text) <= 2_000_000:
                    try:
                        visit(json.loads(text), level + 1)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                return
            if isinstance(value, list):
                for item in value:
                    visit(item, level + 1)
                return
            if not isinstance(value, Mapping):
                return

            segment_type = str(value.get("type", "") or "").strip().lower()
            segment_data = value.get("data")
            if not isinstance(segment_data, Mapping):
                segment_data = value

            if segment_type in {"image", "image_url"}:
                for key in ("url", "file", "path", "file_id"):
                    candidate = segment_data.get(key)
                    if isinstance(candidate, (str, int)) and str(candidate).strip():
                        image_refs.append(str(candidate).strip())
                        break
            elif segment_type in {"forward", "forward_msg", "nodes"}:
                for key in ("id", "message_id", "resid", "res_id"):
                    candidate = segment_data.get(key)
                    if isinstance(candidate, (str, int)) and str(candidate).strip():
                        forward_ids.append(str(candidate).strip())
                        break

            # Forward payloads may use ``message`` as a JSON string and may
            # nest another forward segment in node content.
            for key, child in value.items():
                if key in {"type", "data"}:
                    continue
                if key in {"message", "messages", "nodes", "nodeList", "content", "chain", "origin"}:
                    visit(child, level + 1)
            if isinstance(value.get("data"), (Mapping, list)):
                visit(value["data"], level + 1)

        visit(payload, depth)
        return (
            list(dict.fromkeys(image_refs)),
            list(dict.fromkeys(forward_ids)),
        )

    async def _get_forward_payload(
        self,
        event: AstrMessageEvent,
        forward_id: str,
    ) -> dict[str, Any] | None:
        """通过当前平台 API 读取合并转发的实际节点内容。"""
        # Prefer AstrBot's shared OneBot client so action routing follows the
        # adapter's own compatibility rules.  Keep the local fallback below
        # for adapters that expose a non-standard ``resid`` parameter.
        try:
            from astrbot.core.utils.quoted_message.onebot_client import OneBotClient

            payload = await OneBotClient(event).get_forward_msg(forward_id)
            normalized = self._normalize_forward_payload(payload)
            if normalized is not None:
                logger.debug(
                    "[%s] 已读取合并转发：id=%s；来源=OneBotClient",
                    PLUGIN_NAME,
                    str(forward_id)[:96],
                )
                return normalized
        except Exception as exc:
            logger.debug(
                "[%s] AstrBot OneBotClient 读取合并转发失败：id=%s：%s",
                PLUGIN_NAME,
                str(forward_id)[:96],
                exc,
            )

        bot = getattr(event, "bot", None)
        owners = (getattr(bot, "api", None), bot)
        callers: list[Any] = []
        for owner in owners:
            caller = getattr(owner, "call_action", None)
            if callable(caller) and all(caller is not item for item in callers):
                callers.append(caller)
        if not callers:
            return None

        params_list: list[dict[str, Any]] = [
            {"message_id": forward_id},
            {"id": forward_id},
            {"resid": forward_id},
            {"res_id": forward_id},
        ]
        if forward_id.isdigit():
            numeric_id = int(forward_id)
            params_list.extend(
                [
                    {"message_id": numeric_id},
                    {"id": numeric_id},
                    {"resid": numeric_id},
                    {"res_id": numeric_id},
                ]
            )
        for caller in callers:
            for params in params_list:
                try:
                    payload = caller("get_forward_msg", **params)
                except TypeError:
                    try:
                        payload = caller(action="get_forward_msg", **params)
                    except Exception:
                        continue
                except Exception:
                    continue
                if inspect.isawaitable(payload):
                    try:
                        payload = await payload
                    except Exception:
                        continue
                normalized = self._normalize_forward_payload(payload)
                if normalized is not None:
                    logger.debug(
                        "[%s] 已读取合并转发：id=%s；参数=%s",
                        PLUGIN_NAME,
                        str(forward_id)[:96],
                        ",".join(params.keys()),
                    )
                    return normalized
        return None

    async def _extract_forward_component_images(
        self,
        event: AstrMessageEvent,
        forward_id: str,
        depth: int = 0,
        seen_forward_ids: set[str] | None = None,
    ) -> list[str]:
        """拉取 Forward(id) 并递归提取其中的图片，支持嵌套转发。"""
        if not forward_id or depth > 8:
            return []
        seen = seen_forward_ids if seen_forward_ids is not None else set()
        normalized_id = str(forward_id).strip()
        if not normalized_id or normalized_id in seen:
            return []
        seen.add(normalized_id)
        payload = await self._get_forward_payload(event, normalized_id)
        if not payload:
            logger.debug("[%s] 读取合并转发失败：id=%s", PLUGIN_NAME, normalized_id)
            return []

        try:
            from astrbot.core.utils.quoted_message.chain_parser import OneBotPayloadParser

            parsed = OneBotPayloadParser().parse_get_forward_payload(payload)
        except Exception as exc:
            logger.warning(
                "[%s] 解析合并转发内容失败：id=%s：%s",
                PLUGIN_NAME,
                normalized_id,
                exc,
            )
            parsed = {"image_refs": [], "forward_ids": []}

        scanned_refs, scanned_forward_ids = self._scan_forward_payload(payload)
        raw_refs = list(
            dict.fromkeys(
                [str(value).strip() for value in parsed.get("image_refs", []) if value]
                + scanned_refs
            )
        )
        nested_forward_ids = list(
            dict.fromkeys(
                [str(value).strip() for value in parsed.get("forward_ids", []) if value]
                + scanned_forward_ids
            )
        )
        logger.info(
            "[%s] 合并转发内容已解析：id=%s；图片引用=%d；嵌套转发=%d",
            PLUGIN_NAME,
            normalized_id,
            len(raw_refs),
            len(nested_forward_ids),
        )
        resolved_refs = raw_refs
        if raw_refs:
            try:
                from astrbot.core.utils.quoted_message.image_resolver import ImageResolver

                resolved_refs = await ImageResolver(event).resolve_for_llm(raw_refs)
            except Exception as exc:
                logger.debug(
                    "[%s] 解析转发图片引用失败，改用原始引用：%s",
                    PLUGIN_NAME,
                    exc,
                )
            if not resolved_refs:
                resolved_refs = raw_refs

        result: list[str] = []
        seen_paths: set[str] = set()
        for image_ref in resolved_refs:
            cached = await self._persist_image_ref(image_ref)
            if cached and cached not in seen_paths:
                result.append(cached)
                seen_paths.add(cached)

        for nested_id in nested_forward_ids:
            for cached in await self._extract_forward_component_images(
                event,
                str(nested_id),
                depth + 1,
                seen,
            ):
                if cached not in seen_paths:
                    result.append(cached)
                    seen_paths.add(cached)
        if result:
            logger.info(
                "[%s] 已从合并转发提取图生图输入图片：forward=%s；数量=%d",
                PLUGIN_NAME,
                normalized_id,
                len(result),
            )
        return result

    async def _extract_reply_id_images(
        self,
        event: AstrMessageEvent,
        reply_id: str,
        depth: int = 0,
        seen_forward_ids: set[str] | None = None,
    ) -> list[str]:
        """直接拉取指定引用消息，兼容 Reply 只保留 ID 的适配器。"""
        if depth > 8 or not reply_id or reply_id == "0":
            return []
        payload: dict[str, Any] | None = None
        try:
            from astrbot.core.utils.quoted_message.chain_parser import OneBotPayloadParser
            from astrbot.core.utils.quoted_message.image_resolver import ImageResolver
            from astrbot.core.utils.quoted_message.onebot_client import OneBotClient

            payload = await OneBotClient(event).get_msg(reply_id)
            if not payload:
                return []
            parsed = OneBotPayloadParser().parse_get_msg_payload(payload)
            resolved_refs = await ImageResolver(event).resolve_for_llm(
                [str(value).strip() for value in parsed.get("image_refs", []) if value]
            )
        except Exception as exc:
            logger.debug(
                "[%s] 读取引用消息失败：id=%s：%s",
                PLUGIN_NAME,
                reply_id,
                exc,
            )
            parsed = {"image_refs": [], "forward_ids": []}

        if not payload:
            return []

        scanned_refs, scanned_forward_ids = self._scan_forward_payload(payload)
        raw_refs = list(
            dict.fromkeys(
                [str(value).strip() for value in parsed.get("image_refs", []) if value]
                + scanned_refs
            )
        )
        resolved_refs = raw_refs
        if raw_refs:
            try:
                from astrbot.core.utils.quoted_message.image_resolver import ImageResolver

                resolved = await ImageResolver(event).resolve_for_llm(raw_refs)
            except Exception as exc:
                logger.debug(
                    "[%s] 解析引用消息图片引用失败，改用原始引用：%s",
                    PLUGIN_NAME,
                    exc,
                )
                resolved = []
            if resolved:
                resolved_refs = resolved

        result: list[str] = []
        seen_paths: set[str] = set()
        for image_ref in resolved_refs:
            cached = await self._persist_image_ref(image_ref)
            if cached and cached not in seen_paths:
                result.append(cached)
                seen_paths.add(cached)
        nested_forward_ids = list(
            dict.fromkeys(
                [str(value).strip() for value in parsed.get("forward_ids", []) if value]
                + scanned_forward_ids
            )
        )
        for nested_id in nested_forward_ids:
            for cached in await self._extract_forward_component_images(
                event,
                str(nested_id),
                depth + 1,
                seen_forward_ids,
            ):
                if cached not in seen_paths:
                    result.append(cached)
                seen_paths.add(cached)
        if result:
            logger.info(
                "[%s] 已从引用消息提取图生图输入图片：reply=%s；数量=%d",
                PLUGIN_NAME,
                reply_id,
                len(result),
            )
        return result

    async def _extract_reply_component_images(
        self,
        event: AstrMessageEvent,
        reply: Reply,
        depth: int = 0,
        seen_forward_ids: set[str] | None = None,
    ) -> list[str]:
        """直接拉取 Reply(id) 的原消息，兼容引用链未被适配器展开的情况。"""
        return await self._extract_reply_id_images(
            event,
            self._reply_component_id(reply),
            depth,
            seen_forward_ids,
        )

    async def _extract_component_image(
        self,
        component: Any,
        depth: int = 0,
        event: AstrMessageEvent | None = None,
    ) -> str:
        """递归读取直接图片、引用链和合并转发节点中的第一张图片。"""
        if component is None or depth > 8:
            return ""
        forward_id = self._forward_component_id(component)
        if forward_id and event is not None:
            images = await self._extract_forward_component_images(event, forward_id, depth + 1)
            return images[0] if images else ""
        if isinstance(component, Image):
            try:
                path = await component.convert_to_file_path()
            except Exception:
                return ""
            return await self._persist_image_ref(path)
        if isinstance(component, Mapping):
            images = await self._extract_component_images(
                component,
                depth=depth,
                event=event,
            )
            return images[0] if images else ""
        if isinstance(component, Reply):
            for nested in component.chain or []:
                cached = await self._extract_component_image(nested, depth + 1, event)
                if cached:
                    return cached
            if event is not None:
                images = await self._extract_reply_component_images(event, component, depth + 1)
                if images:
                    return images[0]
            return ""
        if isinstance(component, Node):
            for nested in component.content or []:
                cached = await self._extract_component_image(nested, depth + 1, event)
                if cached:
                    return cached
            return ""
        if isinstance(component, Nodes):
            for node in component.nodes or []:
                cached = await self._extract_component_image(node, depth + 1, event)
                if cached:
                    return cached
        return ""

    async def _extract_component_images(
        self,
        component: Any,
        depth: int = 0,
        event: AstrMessageEvent | None = None,
        seen_forward_ids: set[str] | None = None,
    ) -> list[str]:
        """递归提取一条消息中的全部图片，并立即复制到插件缓存目录。"""
        if component is None or depth > 8:
            return []
        seen = seen_forward_ids if seen_forward_ids is not None else set()
        forward_id = self._forward_component_id(component)
        if forward_id and event is not None:
            return await self._extract_forward_component_images(event, forward_id, depth + 1, seen)
        if isinstance(component, Image):
            try:
                path = await component.convert_to_file_path()
            except Exception:
                return []
            cached = await self._persist_image_ref(path)
            return [cached] if cached else []
        nested_items: list[Any] = []
        if isinstance(component, Mapping):
            component_type = str(component.get("type", "")).strip().lower()
            data = component.get("data")
            data = data if isinstance(data, Mapping) else component
            if component_type == "image":
                for key in ("url", "file", "path"):
                    value = data.get(key)
                    if value is None or not str(value).strip():
                        continue
                    cached = await self._persist_image_ref(str(value).strip())
                    if cached:
                        return [cached]
                return []
            if component_type == "reply":
                for key in ("chain", "message", "content", "origin"):
                    value = data.get(key)
                    if isinstance(value, list):
                        nested_items.extend(value)
                embedded: list[str] = []
                for nested in nested_items:
                    embedded.extend(
                        await self._extract_component_images(
                            nested,
                            depth + 1,
                            event,
                            seen,
                        )
                    )
                if embedded:
                    return list(dict.fromkeys(embedded))
                if event is not None:
                    return await self._extract_reply_id_images(
                        event,
                        self._reply_component_id(component),
                        depth + 1,
                        seen,
                    )
                return []
            if component_type in {"node", "nodes", "forward", "forward_msg"}:
                for key in ("content", "message", "messages", "nodes"):
                    value = data.get(key)
                    if isinstance(value, list):
                        nested_items.extend(value)
        elif isinstance(component, Reply):
            nested_items = list(component.chain or [])
            if event is not None:
                embedded: list[str] = []
                for nested in nested_items:
                    embedded.extend(
                        await self._extract_component_images(
                            nested,
                            depth + 1,
                            event,
                            seen,
                        )
                    )
                if embedded:
                    return embedded
                return await self._extract_reply_component_images(
                    event,
                    component,
                    depth + 1,
                    seen,
                )
        elif isinstance(component, Node):
            nested_items = list(component.content or [])
        elif isinstance(component, Nodes):
            nested_items = [item for node in (component.nodes or []) for item in (node.content or [])]
        result: list[str] = []
        for nested in nested_items:
            result.extend(
                await self._extract_component_images(
                    nested,
                    depth + 1,
                    event,
                    seen,
                )
            )
        return result

    async def _extract_images(self, event: AstrMessageEvent) -> list[str]:
        """提取当前消息/引用/转发中的图片，保持消息内原有顺序。"""
        cached_images = self._cached_event_images(event)
        if cached_images:
            return cached_images
        components = self._event_components(event)
        raw_components = self._raw_event_components(event)
        event_level_components = self._event_level_reference_components(event)
        has_explicit_reference = self._event_has_explicit_image_reference(event)
        result: list[str] = []
        seen: set[str] = set()
        seen_forward_ids: set[str] = set()
        # OneBot 适配器通常会把 Reply 展开成链，但部分版本会把合并转发
        # 压缩成 Reply(id)。优先使用结构化链，再用原始消息段补充没有被
        # AstrBot 组件模型保留下来的 forward/reply ID。
        candidate_components = [*components, *raw_components, *event_level_components]
        if has_explicit_reference:
            candidate_components = [
                component
                for component in candidate_components
                if self._component_has_explicit_reference(component)
            ]
        for component in candidate_components:
            for cached in await self._extract_component_images(
                component,
                event=event,
                seen_forward_ids=seen_forward_ids,
            ):
                if cached and cached not in seen:
                    result.append(cached)
                    seen.add(cached)
        if not has_explicit_reference:
            direct_refs: list[str] = []
            direct_image = getattr(event, "image", None)
            if direct_image:
                direct_refs.append(str(direct_image))
            direct_refs.extend(
                str(value)
                for value in (getattr(event, "image_list", None) or [])
                if value
            )
            for image_ref in direct_refs:
                cached = await self._persist_image_ref(image_ref)
                if cached and cached not in seen:
                    result.append(cached)
                    seen.add(cached)
        if result:
            self._remember_event_images(event, result)
            return result
        if has_explicit_reference:
            # 消息明确引用了图片/合并转发但远程接口暂时取不到时，不能
            # 回退到会话最近出图，否则会把另一张图误当成用户引用的图。
            logger.warning(
                "[%s] 已检测到明确的引用/合并转发，但未提取到其中图片；禁止使用最近出图回退",
                PLUGIN_NAME,
            )
        try:
            from astrbot.core.utils.quoted_message import extract_quoted_message_images

            for image_ref in await extract_quoted_message_images(event):
                cached = await self._persist_image_ref(image_ref)
                if cached and cached not in seen:
                    result.append(cached)
                    seen.add(cached)
            self._remember_event_images(event, result)
        except Exception as exc:
            logger.warning("[%s] 解析引用/转发图片失败：%s", PLUGIN_NAME, exc)
        return result

    async def _extract_image(self, event: AstrMessageEvent) -> str:
        """提取消息或转发消息中的图片并立即持久化。"""
        images = await self._extract_images(event)
        if images:
            logger.info("[%s] 已缓存图生图输入图片：%s", PLUGIN_NAME, ",".join(images))
            return images[0]
        logger.warning("[%s] 未找到可用的图生图输入图片", PLUGIN_NAME)
        return ""

    @staticmethod
    def _resolve_flux2_model(value: str, values: list[str], label: str) -> str:
        """Resolve a WebUI model selection while preserving ComfyUI's spelling.

        Model names are identifiers, not filesystem paths.  ComfyUI may return
        nested names with either ``/`` or ``\\`` depending on the endpoint and
        platform.  Normalize only while comparing; the value submitted to
        ``/prompt`` must be the exact string from the current model list or
        ComfyUI rejects it with ``value_not_in_list``.
        """
        configured = str(value or "").strip().replace("\\", "/")
        if not configured:
            return ""
        normalized = configured.casefold()
        for candidate in values:
            candidate_original = str(candidate).strip()
            candidate_text = candidate_original.replace("\\", "/")
            if candidate_text.casefold() == normalized:
                return candidate_original
        basename = Path(configured).name.casefold()
        for candidate in values:
            candidate_original = str(candidate).strip()
            candidate_text = candidate_original.replace("\\", "/")
            if Path(candidate_text).name.casefold() == basename:
                return candidate_original
        raise UsageError(
            f"Flux2 {label}未检测到：{value}；请确认文件已放入 ComfyUI 对应模型目录并刷新模型列表"
        )

    async def _generate_flux2_tool(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str,
    ) -> list[Path]:
        """Run one of the standalone Flux.2 image tools.

        This path deliberately does not call prompt expansion, translation,
        inline LoRA extraction, or the global LoRA list. The editor workflow
        remains the source of truth for its own enabled LoRA nodes.
        """
        if not image_path:
            raise UsageError(f"{MODE_NAMES[mode]}需要附一张图片或回复一张图片")
        client = self._client()
        env = await self._environment(client)

        def configured(name: str, default: Any) -> Any:
            value = self._get(f"{mode}_{name}", default)
            return default if value is None else value

        model_value = str(params.model or configured("model_name", "") or "").strip()
        clip_value = str(configured("clip_name", "") or "").strip()
        vae_value = str(configured("vae_name", "") or "").strip()
        model_name = (
            self._resolve_flux2_model(
                model_value,
                list(dict.fromkeys(env["diffusion_models"] + env["unet"] + env["checkpoints"])),
                "核心模型",
            )
            if model_value
            else ""
        )
        clip_name = (
            self._resolve_flux2_model(
                clip_value,
                env.get("flux2_text_encoders") or env["text_encoders"],
                "文本编码器",
            )
            if clip_value
            else ""
        )
        vae_name = (
            self._resolve_flux2_model(vae_value, env["vae"], "VAE")
            if vae_value
            else ""
        )
        uploaded = await client.upload_image(image_path)
        image_name = str(uploaded.get("name", "") or "")
        if not image_name:
            raise ComfyError(f"{MODE_NAMES[mode]}上传输入图片后没有返回图片名称")
        logger.info(
            "[%s] %s只使用第一张输入图：name=%s；忽略其它图片由命令层传入的参考图",
            PLUGIN_NAME,
            MODE_NAMES[mode],
            image_name,
        )

        default_positive = str(
            configured("default_positive", FLUX2_TOOL_DEFAULT_POSITIVE.get(mode, ""))
            or FLUX2_TOOL_DEFAULT_POSITIVE.get(mode, "")
        ).strip()
        user_prompt = str(params.prompt or "").strip()
        positive = ", ".join(value for value in (default_positive, user_prompt) if value)
        if not positive:
            positive = FLUX2_TOOL_DEFAULT_POSITIVE.get(mode, "请保持主体清晰，画面自然")
        negative = str(params.negative or configured("default_negative", "") or "").strip()
        negative = self._apply_group_safety_negative(event, negative)
        params.generated_positive = positive
        params.generated_negative = negative

        def as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        configured_seed = as_int(configured("seed", -1), -1)
        seed = params.seed if params.seed >= 0 else configured_seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        batch = params.batch if params.batch != 1 else as_int(configured("batch", 1), 1)
        workflow = load_workflow(self._workflow_path(mode))
        patched, report = adapt_flux2_tool(
            workflow,
            tool=mode,
            positive=positive,
            negative=negative,
            image_name=image_name,
            model_name=model_name,
            clip_name=clip_name,
            vae_name=vae_name,
            size=as_int(configured("size", 0), 0),
            steps=params.steps or as_int(configured("steps", 6), 6),
            cfg=params.cfg or as_float(configured("cfg", 1.0), 1.0),
            seed=seed,
            denoise=params.denoise or as_float(configured("denoise", 0.6), 0.6),
            sampler_name=params.sampler or str(configured("sampler_name", "euler") or "euler"),
            scheduler=params.scheduler or str(configured("scheduler", "simple") or "simple"),
            filename_prefix=str(configured("filename_prefix", f"astrbot/{mode}") or f"astrbot/{mode}"),
            left=as_int(configured("left", 0), 0),
            top=as_int(configured("top", 0), 0),
            right=as_int(configured("right", 0), 0),
            bottom=as_int(configured("bottom", 0), 0),
            feathering=as_int(configured("feathering", 0), 0),
            horizontal_angle=as_int(configured("horizontal_angle", 71), 71),
            vertical_angle=as_int(configured("vertical_angle", 42), 42),
            zoom=as_float(configured("zoom", 4.5), 4.5),
            default_prompts=self._bool_config(f"{mode}_default_prompts", True),
            camera_view=self._bool_config(f"{mode}_camera_view", False),
            caption_tokens=as_int(configured("caption_tokens", 1024), 1024),
            batch=max(1, min(4, as_int(batch, 1))),
        )
        logger.info("[%s] %s兼容提示：%s", PLUGIN_NAME, MODE_NAMES[mode], "；".join(report) or "无")
        prompt_id = await client.queue(patched)
        history = await client.wait(prompt_id, timeout=900)
        images = await client.download_outputs(history, self.output_dir)
        if not images:
            raise ComfyError(f"{MODE_NAMES[mode]}完成，但没有找到 SaveImage 输出")
        self.last_images[self._origin(event)] = str(images[0])
        return images

    async def _generate_img2img_flux2(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        image_paths: list[str],
    ) -> list[Path]:
        """Run the independent Flux2 Klein one-to-three-reference editor."""
        image_paths = [str(value or "").strip() for value in image_paths if str(value or "").strip()]
        if not image_paths:
            raise UsageError("图生图 Flux2 需要一张参考图片")
        if len(image_paths) > 3:
            raise UsageError("图生图 Flux2 最多支持 3 张参考图片")

        # 普通命令的 ``ai`` 开关也必须使用 Flux2 自己的中文编辑规则。
        # LLM 工具已经在 _prepare_llm_prompt() 中处理过，不能再次调用 AI，
        # 否则会把一条短编辑句重复改写或变成长描述。
        if (
            params.ai is True
            and not getattr(params, "llm_invocation", False)
            and not params.img2img_edit_instruction_ready
        ):
            source = (
                "plugin_img2img_flux2_llm"
                if self._img2img_flux2_llm_plugin_ai_enabled()
                else "astrbot_img2img_flux2_llm"
            )
            source_text = str(params.prompt or "").strip()
            if source_text:
                params.plugin_ai_debug_input = source_text
                try:
                    params.prompt = await self._translate_prompt(
                        event,
                        source_text,
                        force_enabled=True,
                        source_override=source,
                    )
                    params.prompt = self._repair_img2img_edit_instruction(
                        params.prompt,
                        source_text,
                    )
                    params.img2img_edit_instruction = params.prompt
                    params.img2img_edit_instruction_ready = True
                    params.plugin_ai_debug_prompt = params.prompt
                except AIError as exc:
                    params.prompt = self._repair_img2img_edit_instruction(
                        source_text,
                        source_text,
                    )
                    params.img2img_edit_instruction = params.prompt
                    params.img2img_edit_instruction_ready = True
                    params.plugin_ai_fallback_note = f"Flux2 图生图 AI 不可用，已按原中文要求继续：{exc}"
                    logger.warning("[%s] Flux2 图生图命令 AI 失败，回退原中文要求：%s", PLUGIN_NAME, exc)

        client = self._client()
        env = await self._environment(client)

        def configured(name: str, default: Any) -> Any:
            value = self._get(f"img2img_flux2_{name}", default)
            return default if value is None else value

        def as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        model_value = str(
            params.model or configured("unet_name", FLUX2_IMG2IMG_UNET_DEFAULT) or ""
        ).strip()
        clip_value = str(configured("clip_name", FLUX2_IMG2IMG_CLIP_DEFAULT) or "").strip()
        vae_value = str(configured("vae_name", FLUX2_IMG2IMG_VAE_DEFAULT) or "").strip()
        model_name = self._resolve_flux2_model(
            model_value,
            list(dict.fromkeys(env["diffusion_models"] + env["unet"] + env["checkpoints"])),
            "核心模型",
        )
        clip_name = self._resolve_flux2_model(
            clip_value,
            env.get("flux2_text_encoders") or env["text_encoders"],
            "文本编码器",
        )
        vae_name = self._resolve_flux2_model(vae_value, env["vae"], "VAE")
        lora_value = str(configured("lora_name", FLUX2_IMG2IMG_LORA_DEFAULT) or "").strip()
        lora_name = (
            self._resolve_lora(lora_value, list(env["loras"]))
            if lora_value
            else ""
        )
        if lora_value and not lora_name:
            raise UsageError(
                f"Flux2 图生图独立 LoRA 未检测到：{lora_value}；"
                "请确认文件已放入 ComfyUI 的 loras 文件夹"
            )
        # 【特别标注】Flux2 图生图只受本页的独立 LoRA 影响：不读取文生图的
        # lora_list，也不参与画风 LoRA 的随机/全部自动选择。这里显式记录，
        # 便于日志核对，也用于开始回复中的特别标注。
        params.lora_isolation_notice = IMG2IMG_LORA_ISOLATION_NOTICE
        logger.info(
            "[%s] 图生图 Flux2 独立 LoRA 隔离生效：独立 LoRA=%s；文生图 LoRA 列表与画风 LoRA 均不参与",
            PLUGIN_NAME,
            lora_name or "未启用",
        )
        # Flux2 Klein is deliberately restored to the pre-acceleration graph.
        # Keep legacy settings readable, but never resolve or inject them.
        uploaded_names: list[str] = []
        for index, image_path in enumerate(image_paths, 1):
            uploaded = await client.upload_image(image_path)
            image_name = str(uploaded.get("name", "") or "").strip()
            if not image_name:
                raise ComfyError(f"Flux2 图生图第 {index} 张参考图上传后没有返回图片名称")
            uploaded_names.append(image_name)
            logger.info(
                "[%s] Flux2 图生图参考图 %d 已上传：name=%s；subfolder=%s；type=%s",
                PLUGIN_NAME,
                index,
                image_name,
                uploaded.get("subfolder", ""),
                uploaded.get("type", ""),
            )

        default_positive = str(configured("default_positive", FLUX2_IMG2IMG_DEFAULT_POSITIVE) or "").strip()
        user_prompt = str(params.prompt or "").strip()
        positive = ", ".join(value for value in (default_positive, user_prompt) if value)
        if not positive:
            positive = "保持人物主体和构图基本不变，根据编辑要求修改画面"
        negative = str(params.negative or configured("default_negative", FLUX2_IMG2IMG_DEFAULT_NEGATIVE) or "").strip()
        negative = self._apply_group_safety_negative(event, negative)
        params.generated_positive = positive
        params.generated_negative = negative

        configured_seed = as_int(configured("seed", -1), -1)
        seed = params.seed if params.seed >= 0 else configured_seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        # ``size`` is the maximum edge passed to EasySizeSimpleImage.  That
        # node scales the longest edge and keeps the source aspect ratio. Do
        # not replace it with the raw input dimensions: a portrait image can
        # otherwise submit millions of pixels to an 8 GB GPU and appear stuck.
        size = as_int(configured("size", FLUX2_IMG2IMG_SIZE_DEFAULT), FLUX2_IMG2IMG_SIZE_DEFAULT)
        # Keep 0 as "use the source workflow default".  The adapter only
        # writes node 90 when a positive size is configured.
        size = max(0, size)
        if params.width or params.height:
            requested_size = max(int(params.width or 0), int(params.height or 0), 64)
            size = requested_size if size <= 0 else min(size, requested_size)
        batch = params.batch if params.batch != 1 else as_int(configured("batch", 1), 1)
        workflow = load_workflow(self._workflow_path("img2img_flux2"))

        # 源工作流自带的 LoRA（节点 38）属于它「模型链」的一部分，独立 LoRA 留空时
        # 必须沿用，否则采样器会退化成没有内容 LoRA 的裸基模——这正是此前
        # Flux2 图生图看起来「无效」的原因。只有该文件在 ComfyUI 中确实不可用，
        # 才自动关闭并给出提示，避免直接提交失败。
        keep_source_lora = True
        source_lora_name = ""
        source_lora_node = workflow.get("38")
        if isinstance(source_lora_node, dict):
            source_lora_name = str(
                (source_lora_node.get("inputs") or {}).get("lora_name") or ""
            ).strip()
        if not str(lora_name or "").strip() and source_lora_name:
            try:
                available_loras = {
                    name.replace("\\", "/").casefold()
                    for name in await self._available_loras()
                }
            except Exception as exc:  # noqa: BLE001 - 查询失败时保守沿用，交给 ComfyUI 报错
                logger.warning(
                    "[%s] 查询 ComfyUI LoRA 列表失败，暂按源工作流 LoRA 处理：%s",
                    PLUGIN_NAME,
                    exc,
                )
                available_loras = set()
            if available_loras and source_lora_name.replace("\\", "/").casefold() not in available_loras:
                keep_source_lora = False
                logger.warning(
                    "[%s] 源工作流 LoRA 在 ComfyUI 中不可用，将自动关闭：%s",
                    PLUGIN_NAME,
                    source_lora_name,
                )

        patched, report = adapt_flux2_klein_img2img(
            workflow,
            positive=positive,
            negative=negative,
            image_names=uploaded_names,
            unet_name=model_name,
            clip_name=clip_name,
            vae_name=vae_name,
            lora_name=lora_name,
            lora_strength=as_float(configured("lora_strength", 0.9), 0.9),
            keep_source_lora=keep_source_lora,
            size=size,
            steps=params.steps or as_int(configured("steps", FLUX2_IMG2IMG_STEPS_DEFAULT), FLUX2_IMG2IMG_STEPS_DEFAULT),
            cfg=params.cfg or as_float(configured("cfg", FLUX2_IMG2IMG_CFG_DEFAULT), FLUX2_IMG2IMG_CFG_DEFAULT),
            seed=seed,
            sampler_name=params.sampler or str(configured("sampler_name", "euler") or "euler"),
            scheduler=params.scheduler or str(configured("scheduler", "simple") or "simple"),
            denoise=params.denoise or as_float(configured("denoise", FLUX2_IMG2IMG_DENOISE_DEFAULT), FLUX2_IMG2IMG_DENOISE_DEFAULT),
            batch=max(1, min(4, as_int(batch, 1))),
            filename_prefix=str(
                configured("filename_prefix", "astrbot/img2img_flux2_klein")
                or "astrbot/img2img_flux2_klein"
            ),
        )
        logger.info(
            "[%s] Flux2 Klein 图生图已准备：参考图=%d；核心模型=%s；LoRA=%s；正面=%s；兼容提示=%s",
            PLUGIN_NAME,
            len(uploaded_names),
            model_name,
            lora_name or (source_lora_name if keep_source_lora and source_lora_name else "关闭"),
            positive[:300],
            "；".join(report) or "无",
        )
        prompt_id = await client.queue(patched)
        logger.info("[%s] Flux2 Klein 图生图已提交：prompt_id=%s", PLUGIN_NAME, prompt_id)
        history = await client.wait(prompt_id, timeout=900)
        images = await client.download_outputs(history, self.output_dir)
        if not images:
            raise ComfyError("图生图 Flux2 完成，但没有找到 SaveImage 输出")
        self.last_images[self._origin(event)] = str(images[0])
        return images

    async def _generate(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str = "",
        second_image_path: str = "",
        third_image_path: str = "",
    ) -> list[Path]:
        logger.info(
            "[%s] 进入生成流程：模式=%s；输入图片=%s；提示词=%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            image_path or "无",
            str(params.prompt or "")[:160],
        )
        if mode in {"wash", "outpaint", "multi_angle"}:
            # These workflows are intentionally independent tools: only the
            # first image is used and all prompt/LoRA automation is bypassed.
            return await self._generate_flux2_tool(event, params, mode, image_path)
        if mode == "img2img_flux2":
            return await self._generate_img2img_flux2(
                event,
                params,
                [path for path in (image_path, second_image_path, third_image_path) if path],
            )
        if mode == "img2img" and not image_path:
            raise UsageError(
                f"{MODE_NAMES.get(mode, mode)}没有拿到输入图片，已阻止提交空白图任务；"
                "请在同一条消息附图，或回复一张图片/含图合并转发"
            )
        client = self._client()
        env = await self._environment(client)
        model = ""
        if mode != "img2img":
            model = self._model_name(env, params.model, mode)
        # 临时简称 LoRA 与 WebUI 中长期启用的 LoRA 合并；同一文件以临时权重为准。
        # 【特别标注】图生图只受 img2img_lora_name 这一个「独立 LoRA」影响：
        # 文生图的 lora_list 与画风 LoRA 都不会串入 Qwen 图生图工作流；只有
        # 用户在本次指令里显式写出的 LoRA 才允许临时叠加。
        configured_img2img_loras: list[str] = []
        if mode == "img2img":
            configured_img2img_lora = str(self._get("img2img_lora_name", "") or "").strip()
            if configured_img2img_lora:
                configured_img2img_loras.append(
                    f"{configured_img2img_lora}:{float(self._get('img2img_lora_strength', 0.8) or 0.0):g}"
                )
            params.lora_isolation_notice = IMG2IMG_LORA_ISOLATION_NOTICE
            logger.info(
                "[%s] 图生图独立 LoRA 隔离生效：独立 LoRA=%s；文生图 LoRA 列表与画风 LoRA 均不参与",
                PLUGIN_NAME,
                configured_img2img_lora or "未启用",
            )
        configured_loras = (
            configured_img2img_loras
            if mode == "img2img"
            else list(self._get("lora_list", []) or [])
        )
        temporary_loras = list(params.loras)
        style_catalog = (
            self._style_lora_entries(list(env["loras"]), selected_only=False)
            if mode == "txt2img"
            else []
        )
        style_names = {
            str(item["file_name"]).replace("\\", "/").casefold()
            for item in style_catalog
        }
        # 显式触发画风 LoRA 时，它的架构优先级高于 WebUI 中长期启用的
        # 默认 LoRA。只移除“配置列表”里的相反架构，用户在本次命令中
        # 明确写出的相反架构仍交给最终兼容性检查并报错，避免静默改写
        # 用户的明确要求。
        explicit_style_profiles: set[str] = set()
        if mode == "txt2img":
            for item in temporary_loras:
                raw_name = str(item or "").rsplit(":", 1)[0].strip()
                actual = self._resolve_lora(raw_name, list(env["loras"]))
                if not actual or actual.replace("\\", "/").casefold() not in style_names:
                    continue
                profile = self._lora_architecture_profile(actual)
                if profile in {"anima_base", "anima_29b"}:
                    explicit_style_profiles.add(profile)
        if len(explicit_style_profiles) == 1 and not params.model:
            requested_profile = next(iter(explicit_style_profiles))
            model_matches_request = (
                is_anima_29b_model(model)
                if requested_profile == "anima_29b"
                else not is_anima_29b_model(model)
            )
            if not model_matches_request:
                compatible_models = self._compatible_models(env, mode)
                replacement = next(
                    (
                        candidate for candidate in compatible_models
                        if (
                            is_anima_29b_model(candidate)
                            if requested_profile == "anima_29b"
                            else not is_anima_29b_model(candidate)
                        )
                    ),
                    "",
                )
                if replacement:
                    logger.info(
                        "[%s] 显式画风 LoRA=%s，临时切换到兼容核心模型=%s",
                        PLUGIN_NAME,
                        requested_profile,
                        replacement,
                    )
                    model = replacement
        if len(explicit_style_profiles) == 1:
            preferred_profile = next(iter(explicit_style_profiles))
            filtered_configured: list[str] = []
            for item in configured_loras:
                raw_name = str(item or "").rsplit(":", 1)[0].strip()
                actual = self._resolve_lora(raw_name, list(env["loras"]))
                profile = self._lora_architecture_profile(actual or raw_name)
                if profile in {"anima_base", "anima_29b"} and profile != preferred_profile:
                    logger.info(
                        "[%s] 显式画风 LoRA=%s，跳过默认启用的相反架构 LoRA=%s",
                        PLUGIN_NAME,
                        preferred_profile,
                        actual or raw_name,
                    )
                    continue
                filtered_configured.append(item)
            configured_loras = filtered_configured
        style_entries = (
            self._style_lora_candidates_without_persistent(
                (
                    self._style_lora_entries(list(env["loras"]), selected_only=True)
                    or self._style_lora_entries(list(env["loras"]), selected_only=False)
                ),
                configured_loras,
                list(env["loras"]),
                style_catalog,
            )
            if mode == "txt2img" and not params.style_lora_disabled
            else []
        )
        if mode == "txt2img" and style_entries and not explicit_style_profiles:
            # 画风列表为空时，随机/全部模式默认使用所有“画风”分类条目。
            # 这样首次安装或旧配置没有 style_lora_list 时，自动画风仍能
            # 工作；选择“关闭”或写“不用画风”仍然会跳过整个链路。
            model_profile = "anima_29b" if is_anima_29b_model(model) else "anima_base"
            compatible_style_entries = []
            for item in style_entries:
                profile = self._lora_architecture_profile(str(item["file_name"]))
                if profile in {"anima_base", "anima_29b"} and profile != model_profile:
                    continue
                compatible_style_entries.append(item)
            if len(compatible_style_entries) != len(style_entries):
                logger.info(
                    "[%s] 按核心模型架构过滤随机画风 LoRA：模型=%s；保留=%s/%s",
                    PLUGIN_NAME,
                    model_profile,
                    len(compatible_style_entries),
                    len(style_entries),
                )
            style_entries = compatible_style_entries
        explicit_style_lora = False
        loras: list[str] = []
        resolved_loras: dict[str, str] = {}
        params.style_loras_used = []

        def remember_style_lora(
            file_name: str,
            weight: float | str,
            source: str = "",
        ) -> None:
            normalized = str(file_name or "").replace("\\", "/").casefold()
            if normalized not in style_names:
                return
            try:
                numeric_weight = max(0.0, min(2.0, float(weight)))
            except (TypeError, ValueError):
                numeric_weight = 0.8
            entry = next(
                (
                    item for item in style_catalog
                    if str(item["file_name"]).replace("\\", "/").casefold() == normalized
                ),
                None,
            )
            if entry is None:
                return
            existing = next(
                (
                    item for item in params.style_loras_used
                    if str(item.get("file_name", "")).replace("\\", "/").casefold() == normalized
                ),
                None,
            )
            if existing is not None:
                # 同一画风可能同时出现在“常态启用”和临时触发列表中。
                # 保留一条实际 LoRA 记录，但把所有来源合并，避免用户误以为
                # 随机选择没有生效。
                sources = [
                    str(value or "").strip()
                    for value in str(existing.get("source", "")).split("、")
                    if str(value or "").strip()
                ]
                source = str(source or "").strip()
                if source and source not in sources:
                    sources.append(source)
                existing["source"] = "、".join(sources)
                return
            style_index = next(
                index for index, item in enumerate(style_catalog)
                if str(item["file_name"]).replace("\\", "/").casefold() == normalized
            )
            params.style_loras_used.append(
                {
                    "file_name": str(entry["file_name"]),
                    "alias": self._style_lora_alias(str(entry["file_name"]), style_index, list(env["loras"])),
                    "display_name": self._lora_alias(str(entry["file_name"])),
                    "weight": numeric_weight,
                    "source": str(source or "").strip(),
                }
            )

        for source_loras, user_triggered in (
            (configured_loras, False),
            (temporary_loras, True),
        ):
            for item in source_loras:
                name, separator, weight = str(item).rpartition(":")
                if not separator or not re.fullmatch(r"\d+(?:\.\d+)?", weight):
                    name, weight = str(item), "0.8"
                actual = self._resolve_lora(name, list(env["loras"]))
                if not actual:
                    raise UsageError(f"LoRA 不存在或昵称未设置：{name}；请先使用 /lora 列表")
                if params.style_lora_disabled and actual.replace("\\", "/").casefold() in style_names:
                    continue
                resolved_loras[actual.casefold()] = f"{actual}:{weight}"
                if actual.replace("\\", "/").casefold() in style_names:
                    explicit_style_lora = explicit_style_lora or user_triggered
                    remember_style_lora(
                        actual,
                        weight,
                        "常态启用" if not user_triggered else "指令触发",
                    )
        # 画风 LoRA 是文生图专属自动链路。只有用户本次临时明确写出
        # 画风简称/专属预设时才关闭随机；WebUI 的长期启用 LoRA 不会
        # 阻止其它候选参与随机。mode=random 是默认行为；没有候选时跳过。
        if mode == "txt2img" and style_entries and not explicit_style_lora:
            style_mode = self._style_lora_mode()
            selected_style = self._select_style_loras(style_entries)
            for item in selected_style:
                actual = str(item["file_name"])
                resolved_loras[actual.casefold()] = f"{actual}:{float(item['weight']):g}"
                remember_style_lora(
                    actual,
                    item["weight"],
                    "随机选择" if self._style_lora_mode() == "random" else "自动全部",
                )
            if selected_style:
                logger.info(
                    "[%s] 文生图自动选择画风 LoRA：模式=%s；数量=%s；文件=%s",
                    PLUGIN_NAME,
                    style_mode,
                    len(selected_style),
                    ",".join(str(item["file_name"]) for item in selected_style),
                )
        loras = list(resolved_loras.values())
        lora_prompt_values: list[str] = []
        # The resolved list contains both the independent WebUI LoRA and any
        # temporary command LoRA. Duplicate files are already collapsed above,
        # with the later command value taking precedence.
        prompt_loras = list(loras)
        for item in prompt_loras:
            actual = item.rsplit(":", 1)[0]
            lora_prompt_values.extend(self._lora_prompt_values_for_task(params, actual))
        lora_prompt_values = self._normalize_civitai_tags(lora_prompt_values)
        if mode == "img2img":
            # 图生图使用 Qwen Image Edit 的独立模型链。兼容性检查不能再
            # 按 Flux2 规则执行，否则会错误拒绝 Qwen LoRA。
            self._assert_qwen_loras_compatible(prompt_loras)
        else:
            self._assert_anima_loras_compatible(model, loras)
        logger.info(
            "[%s] %s 参数：预设=%s；临时/启用 LoRA=%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            ",".join(params.presets) or "无",
            ",".join(loras) or "无",
        )
        positive, negative, note = await self._prompt_text(
            event,
            params,
            None,
            lora_prompt_values,
            mode=mode,
        )
        # 最后一道兜底：无论未来新增哪种 AI/翻译/工作流分支，实际提交给
        # ComfyUI 的正负面提示词都必须遵守群聊黑名单配置。
        negative = self._apply_group_safety_negative(event, negative)
        params.generated_positive = positive
        params.generated_negative = negative
        if mode == "img2img":
            logger.info(
                "[%s] 图生图最终提示词：编辑=%s；正面=%s；负面=%s；denoise=%s",
                PLUGIN_NAME,
                str(getattr(params, "img2img_edit_instruction", "") or "无")[:300],
                positive[:500],
                negative[:300],
                params.denoise or self._get("img2img_denoise", DEFAULT_IMG2IMG_DENOISE),
            )
        configured_seed = (
            self._get("img2img_seed", -1)
            if mode == "img2img"
            else self._get("seed", -1)
        )
        try:
            configured_seed = int(configured_seed)
        except (TypeError, ValueError):
            configured_seed = -1
        seed = params.seed if params.seed >= 0 else configured_seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        width = params.width or int(
            self._get("img2img_width", 512)
            if mode == "img2img"
            else self._get("width", 832) or 832
        )
        height = params.height or int(
            self._get("img2img_height", 960)
            if mode == "img2img"
            else self._get("height", 1216) or 1216
        )
        if mode == "img2img" and image_path:
            use_input_size = not params.width and not params.height and self._bool_config(
                "img2img_match_input_size", True
            )
            try:
                img2img_input_max_edge = max(
                    64,
                    int(
                        self._get(
                            "img2img_largest_size",
                            max(width, height, 64),
                        )
                        or max(width, height, 64)
                    ),
                )
            except (TypeError, ValueError):
                img2img_input_max_edge = max(width, height, 64)
            width, height = self._img2img_output_size(
                image_path,
                width,
                height,
                use_input_size=use_input_size,
                max_edge=img2img_input_max_edge,
            )
        scale = params.scale or float(self._get("hires_scale", 2.0) or 2.0)
        workflow = load_workflow(self._workflow_path(mode))
        is_qwen_img2img = mode == "img2img" and all(
            node_id in workflow
            for node_id in ("1", "10", "11", "13", "14", "30", "39", "119", "152", "158", "160", "161")
        )
        if mode == "img2img" and not is_qwen_img2img:
            raise WorkflowError(
                "当前图生图工作流不是 Qwen Image Edit 工作流；"
                "请在 WebUI 的「Qwen Image Edit 图生图源工作流位置」填入正确的文件后重启插件，"
                "或清空该项以使用插件内置工作流"
        )
        image_name = ""
        second_image_name = ""
        if image_path:
            uploaded = await client.upload_image(image_path)
            image_name = str(uploaded.get("name", "") or "")
            logger.info(
                "[%s] 输入图片已上传到 ComfyUI：name=%s；subfolder=%s；type=%s",
                PLUGIN_NAME,
                image_name or "空",
                uploaded.get("subfolder", ""),
                uploaded.get("type", ""),
            )
            if not image_name:
                raise ComfyError("ComfyUI 上传输入图片后没有返回图片名称")
        if mode == "img2img" and second_image_path:
            uploaded_second = await client.upload_image(second_image_path)
            second_image_name = str(uploaded_second.get("name", "") or "")
            logger.info(
                "[%s] 第二张参考图已上传到 ComfyUI：name=%s；subfolder=%s；type=%s",
                PLUGIN_NAME,
                second_image_name or "空",
                uploaded_second.get("subfolder", ""),
                uploaded_second.get("type", ""),
            )
            if not second_image_name:
                raise ComfyError("ComfyUI 上传第二张参考图后没有返回图片名称")
        if not image_path:
            image_name = await self._ensure_blank_image(client, width, height, scale)
        upscale = params.upscale_model or str(self._get("hires_upscale_model", "") or "")
        if not upscale and env["upscale_models"]:
            upscale = env["upscale_models"][0]
        if is_qwen_img2img:
            def resolve_model(value: str, values: list[str], label: str) -> str:
                configured = str(value or "").strip().replace("\\", "/")
                normalized = configured.casefold()
                match = next(
                    (candidate for candidate in values if str(candidate).replace("\\", "/").casefold() == normalized),
                    "",
                )
                if match:
                    return match
                base = Path(configured).name.casefold()
                match = next(
                    (candidate for candidate in values if Path(str(candidate).replace("\\", "/")).name.casefold() == base),
                    "",
                )
                if match:
                    return match
                raise UsageError(
                    f"Qwen 图生图未检测到可用的{label}：{value or '未配置'}；"
                    "请确认模型已放入 ComfyUI 对应目录，然后刷新模型列表"
                )

            qwen_unet = resolve_model(
                str(self._get("img2img_unet_name", "") or QWEN_IMG2IMG_UNET_DEFAULT),
                env["unet"],
                "UNet",
            )
            qwen_clip = resolve_model(
                str(self._get("img2img_clip_name", "") or QWEN_IMG2IMG_CLIP_DEFAULT),
                # Qwen Image 的文本编码器是 GGUF，来自 models/clip；
                # Flux2 才使用 models/text_encoders 下的 safetensors。
                env.get("clip_gguf") or env["text_encoders"],
                "文本编码器",
            )
            qwen_vae = resolve_model(
                str(self._get("img2img_vae_name", "") or QWEN_IMG2IMG_VAE_DEFAULT),
                env["vae"],
                "VAE",
            )
            configured_img2img_lora = str(self._get("img2img_lora_name", "") or "").strip()
            if configured_img2img_lora:
                configured_img2img_lora = self._resolve_lora(configured_img2img_lora, list(env["loras"])) or configured_img2img_lora
            normal_steps = int(
                self._get("img2img_steps", QWEN_IMG2IMG_STEPS_DEFAULT)
                or QWEN_IMG2IMG_STEPS_DEFAULT
            )
            normal_cfg = float(
                self._get("img2img_cfg", QWEN_IMG2IMG_CFG_DEFAULT)
                or QWEN_IMG2IMG_CFG_DEFAULT
            )
            configured_steps = normal_steps
            configured_cfg = normal_cfg
            configured_largest_size = img2img_input_max_edge
            if use_input_size:
                # ImageScaleToMaxDimension 位于 Qwen 编码前。不要把小图先
                # 放大到 1152 再缩回去；这会增加显存和时间，却不会增加
                # 原图信息。超大图则使用上面相同的最长边上限。
                configured_largest_size = min(configured_largest_size, max(width, height))
            patched, report = adapt_qwen_img2img(
                workflow,
                positive=positive,
                negative=negative,
                image_name=image_name,
                second_image_name=second_image_name,
                unet_name=qwen_unet,
                clip_name=qwen_clip,
                vae_name=qwen_vae,
                lora_name=configured_img2img_lora,
                lora_strength=float(self._get("img2img_lora_strength", 0.8) or 0.0),
                loras=prompt_loras,
                width=width,
                height=height,
                steps=params.steps or configured_steps,
                cfg=params.cfg or configured_cfg,
                seed=seed,
                sampler_name=params.sampler or str(self._get("img2img_sampler_name", "euler") or "euler"),
                scheduler=params.scheduler or str(self._get("img2img_scheduler", "simple") or "simple"),
                denoise=params.denoise or float(
                    self._get("img2img_denoise", QWEN_IMG2IMG_DENOISE_DEFAULT)
                    or QWEN_IMG2IMG_DENOISE_DEFAULT
                ),
                scale_method=str(self._get("img2img_scale_method", "lanczos") or "lanczos"),
                largest_size=configured_largest_size,
                # disabled/none 表示保持原图完整画布；只有用户明确选择
                # center 等裁剪模式时才允许工作流裁掉输入图边缘。
                crop=str(self._get("img2img_crop", "disabled") or "disabled"),
                filename_prefix=str(self._get("img2img_filename_prefix", "astrbot/img2img_qwen") or "astrbot/img2img_qwen"),
            )
            logger.info(
                "[%s] Qwen 节点1 prompt 已写入：%s；节点39 negative=%s；节点152 steps=%s cfg=%s denoise=%s；加速LoRA=关闭",
                PLUGIN_NAME,
                str(patched.get("1", {}).get("inputs", {}).get("prompt", ""))[:500],
                str(patched.get("39", {}).get("inputs", {}).get("prompt", ""))[:300],
                patched.get("152", {}).get("inputs", {}).get("steps", ""),
                patched.get("152", {}).get("inputs", {}).get("cfg", ""),
                patched.get("152", {}).get("inputs", {}).get("denoise", ""),
            )
        else:
            model = self._model_name(env, params.model, mode)
            if is_anima_29b_model(model) and not self._anima_29b_patch_installed():
                raise UsageError(
                    "当前选择的是 Anima-2.9B，但 ComfyUI 未安装 "
                    "ComfyUI-Anima-2.9B 节点。请安装该节点并重启 ComfyUI 后再生成，"
                    "否则会得到噪点图。"
                )
            sampling = anima_sampling_defaults(model)
            configured_sampler = str(self._get("sampler_name", "") or "").strip()
            configured_scheduler = str(self._get("scheduler", "") or "").strip()
            # ``normal`` was written by older plugin versions and is not an
            # Anima default. Preserve an explicit user choice, but repair this
            # stale value according to the selected model family.
            if not configured_sampler:
                configured_sampler = str(sampling["sampler_name"])
            if not configured_scheduler or configured_scheduler.lower() == "normal":
                configured_scheduler = str(sampling["scheduler"])
            patched, report = adapt_original(
                workflow,
                mode=mode,
                model_name=model,
                loras=loras,
                positive=positive,
                negative=negative,
                image_name=image_name,
                width=width,
                height=height,
                steps=(
                    params.steps or int(self._get("steps", 30) or 30)
                    if mode != "hires"
                    else params.steps or int(self._get("hires_steps", 20) or 20)
                ),
                cfg=params.cfg or float(self._get("cfg", 5.0) or 5.0) or 5.0,
                seed=seed,
                denoise=(
                    params.denoise or float(self._get("denoise", 0.6) or 0.6)
                    if mode != "hires"
                    else params.denoise or float(self._get("hires_denoise", 0.25) or 0.25)
                ),
                scale=scale,
                upscale_model=upscale,
                sampler_name=params.sampler or configured_sampler,
                scheduler=configured_scheduler,
                batch=params.batch,
                filename_prefix=f"astrbot/{mode}",
            )
        logger.info(
            "[%s] 正在提交%s工作流到 ComfyUI：节点=%s；输入图片=%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            ",".join(str(node_id) for node_id in patched.keys()),
            image_name or "无",
        )
        prompt_id = await client.queue(patched)
        logger.info(
            "[%s] %s已提交到 ComfyUI：prompt_id=%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            prompt_id,
        )
        history = await client.wait(prompt_id, timeout=900)
        logger.info("[%s] ComfyUI 任务完成：prompt_id=%s", PLUGIN_NAME, prompt_id)
        images = await client.download_outputs(history, self.output_dir)
        if not images:
            raise ComfyError("ComfyUI 任务完成，但没有找到 SaveImage 输出")
        logger.info(
            "[%s] 已下载%s输出图片：%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            ",".join(str(path) for path in images),
        )
        origin = self._origin(event)
        self.last_images[origin] = str(images[0])
        if report:
            logger.warning("[%s] 工作流兼容提示：%s", PLUGIN_NAME, "；".join(report))
        if note:
            logger.info("[%s] %s", PLUGIN_NAME, note)
        return images

    @staticmethod
    def _format_reply_template(template: str, values: dict[str, Any], fallback: str) -> str:
        try:
            return str(template or "").format(**values).strip() or fallback
        except (KeyError, ValueError):
            return fallback

    def _draw_start_reply(
        self,
        mode: str,
        prompt: str,
        params: DrawParams | None = None,
    ) -> str:
        fallback = f"{MODE_NAMES[mode]}任务已提交，生成期间可以继续聊天，完成后会发送结果。"
        template = str(self._get("draw_start_reply", "") or "")
        lora_notice = str(getattr(params, "lora_isolation_notice", "") or "") if params else ""
        reply = self._format_reply_template(
            template,
            {
                "mode": MODE_NAMES[mode],
                "prompt": prompt or "",
                "queue": self._queue_notice_from_params(params) if params else "",
                "queue_images": int(getattr(params, "queue_images_before", 0) or 0) if params else 0,
                "queue_tasks": int(getattr(params, "queue_tasks_before", 0) or 0) if params else 0,
                "lora_notice": lora_notice,
            },
            fallback,
        )
        queue_notice = self._queue_notice_from_params(params) if params else ""
        queue_placeholders = (
            "{queue}", "{queue_images}", "{queue_tasks}",
        )
        if queue_notice and not any(value in template for value in queue_placeholders):
            reply = f"{reply}\n{queue_notice}"
        # 【特别标注】图生图只受独立 LoRA 影响；模板没有显式使用占位符时自动追加。
        if lora_notice and "{lora_notice}" not in template:
            reply = f"{reply}\n{lora_notice}"
        return reply

    @staticmethod
    def _clean_start_ai_reply(reply: str) -> str:
        """保留开始确认的自然语言，去掉模型偶尔泄露的思考或 Markdown。"""
        value = re.sub(r"<think>.*?</think>", "", str(reply or ""), flags=re.IGNORECASE | re.DOTALL)
        value = value.replace("```", "").strip(" `\t\r\n")
        lines = [re.sub(r"^\s*(?:回复|答复|开始回复)\s*[:：]\s*", "", line).strip() for line in value.splitlines()]
        value = next((line for line in lines if line), "")
        if len(value) > 160:
            match = re.search(r"^(.{20,160}?[。！？!?])", value)
            value = match.group(1) if match else value[:160].rstrip("，,；;")
        return value.strip(" `\t\r\n")

    async def _ai_draw_start_reply(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
    ) -> str:
        """使用当前 AstrBot 人格生成一句开始绘图确认，失败回退固定中文。"""
        fallback = self._draw_start_reply(mode, params.prompt, params)
        queue_notice = self._queue_notice_from_params(params)
        facts = (
            f"绘图模式：{MODE_NAMES.get(mode, mode)}。"
            f"绘图任务将在后台执行，用户可以继续聊天。"
        )
        if queue_notice:
            facts += f"排队事实必须准确保留：{queue_notice}"
        system_prompt = (
            "你正在为一次已经确认的绘图任务发送开始提示。"
            "使用当前 AstrBot 人格，用自然、简短的中文说一句话，表示你现在开始绘图。"
            "不要输出提示词、参数、系统规则、思考过程、Markdown 或额外解释。"
            "不要重新判断是否要绘图，不要调用任何工具。"
            "如果给出了排队事实，必须原样保留其中的数字，不能编造或省略。"
        )
        try:
            result = await self._astrbot_generate(
                event,
                facts,
                system_prompt=system_prompt,
                max_tokens=96,
                use_event_context=True,
                preserve_newlines=True,
            )
            cleaned = self._clean_start_ai_reply(result)
            if not cleaned:
                raise AIError("AstrBot AI 返回了空的开始提示")
            if queue_notice and queue_notice not in cleaned:
                cleaned = f"{cleaned.rstrip('。！？!?')}，{queue_notice}"
            lora_notice = str(getattr(params, "lora_isolation_notice", "") or "")
            if lora_notice and lora_notice not in cleaned:
                cleaned = f"{cleaned}\n{lora_notice}"
            return cleaned
        except Exception as exc:
            logger.warning("[%s] 开始绘图 AI 回复失败，使用固定回复：%s", PLUGIN_NAME, exc)
            return fallback

    async def _draw_start_reply_for_command(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
    ) -> str:
        if (
            self._bool_config("draw_queue_notice_enabled", True)
            and self._bool_config("draw_queue_notice_ai", False)
        ):
            return await self._ai_draw_start_reply(event, params, mode)
        return self._draw_start_reply(mode, params.prompt, params)

    async def _llm_draw_start_reply(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
    ) -> str:
        mode_value = str(self._get("llm_draw_start_reply_mode", "ai") or "ai").strip().lower()
        if mode_value in {"fixed", "custom", "固定", "固定回复"}:
            return self._draw_start_reply(mode, params.prompt, params)
        return await self._ai_draw_start_reply(event, params, mode)

    def _custom_draw_reply(self, mode: str, count: int, prompt: str) -> str:
        fallback = f"{MODE_NAMES[mode]}完成，共 {count} 张。"
        template = str(self._get("draw_reply_custom", "") or "")
        return self._format_reply_template(
            template,
            {"mode": MODE_NAMES[mode], "count": count, "prompt": prompt or ""},
            fallback,
        )

    @staticmethod
    def _completion_quote(event: AstrMessageEvent) -> Reply | None:
        """获取原始 QQ 消息 ID，供后台完成消息作为引用回复。"""
        message_obj = getattr(event, "message_obj", None)
        message_id = getattr(message_obj, "message_id", None)
        if message_id in (None, ""):
            return None
        return Reply(id=str(message_id))

    async def _draw_reply(self, event: AstrMessageEvent, params: DrawParams, mode: str, count: int) -> str:
        """返回固定完成事实，不再为完成通知额外调用 AI。

        绘图工具已经由 LLM 决定是否调用；任务完成后的状态不需要再次交给
        LLM 判断。固定消息也会让 AstrBot 和用户明确知道图片已经发送完成，
        避免人格模型超时、改写状态或重复复述画图要求。
        """
        return self._custom_draw_reply(mode, count, params.prompt)

    @staticmethod
    def _normalize_draw_reply(reply: str, mode: str, count: int) -> str:
        """清理完成回复，拒绝明显错误状态并补齐可验证事实。"""
        value = str(reply or "").strip().strip("` ")
        value = re.sub(r"^(?:完成回复|回复|答复)\s*[:：]\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip(" ，,。；;\n")
        if not value:
            return ""
        if re.search(r"(?:正在生成|生成中|等待生成|尚未完成|生成失败|没有生成)", value):
            return ""
        # 完成通知不承载绘图请求或实际提示词；这些内容既会让回复变长，
        # 也容易让 AstrBot 人格重复用户原话。
        if re.search(r"(?:用户(?:原始)?需求|画图要求|绘图要求|提示词|正面提示词|负面提示词|prompt)", value, re.IGNORECASE):
            return ""
        # LLM 可以自由保持人格口吻，但事实锚点由插件补齐，避免把一张图
        # 说成多张，或把文生图误称为其它模式。
        fact = f"{MODE_NAMES.get(mode, mode)}已完成，共 {max(0, int(count))} 张。"
        if not re.search(rf"{re.escape(str(max(0, int(count))))}\s*张", value):
            value = f"{value}。{fact}"
        return value

    async def _run(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str = "",
        second_image_path: str = "",
        third_image_path: str = "",
    ) -> list:
        # 输入端图片审核。所有绘图路径（斜杠指令和 LLM 工具）最终都会经过
        # 这里，指令入口会再提前拦一次，这里作为兜底，保证任何新入口都不会
        # 绕过审核。命中时在占用并发名额之前直接返回警告消息链。
        blocked_chain = await self._moderation_input_chain(
            event, image_path, second_image_path, third_image_path
        )
        if blocked_chain is not None:
            params.moderation_blocked = True
            params.moderation_blocked_reason = "input"
            return blocked_chain
        semaphore = self._build_semaphore()
        async with semaphore:
            try:
                images = await self._generate(
                    event,
                    params,
                    mode,
                    image_path,
                    second_image_path,
                    third_image_path,
                )
            except (UsageError, WorkflowError, ComfyError) as exc:
                await self._release_draw_limit(params)
                message = str(exc)
                if isinstance(exc, ComfyError):
                    message = (
                        f"{message}\nComfyUI 未启动或未连接时，请让插件管理员发送 /comfy打开 后再试。"
                    )
                return [Plain(f"生成失败：{message}")]
            except asyncio.CancelledError:
                await self._release_draw_limit(params)
                raise
            except Exception as exc:
                await self._release_draw_limit(params)
                logger.exception("[%s] 生成异常", PLUGIN_NAME)
                return [Plain(f"生成失败：{exc}")]
            finally:
                # 只有 _extract_image() 创建的 input_cache 文件会命中这里；
                # 最近生成图片和用户配置的其它路径不会被误删。
                for cleanup_path in {image_path, second_image_path, third_image_path}:
                    if not cleanup_path:
                        continue
                    try:
                        cached_path = Path(cleanup_path).resolve()
                        cache_root = self.input_cache_dir.resolve()
                        if cached_path.parent == cache_root:
                            cached_path.unlink(missing_ok=True)
                    except (OSError, RuntimeError):
                        logger.debug("[%s] 清理输入图片缓存失败：%s", PLUGIN_NAME, cleanup_path)
            # 合并转发模式下，提示词作为最后一个节点放回图片转发记录；
            # 普通模式仍由完成回复发送层附带提示词。
            if str(self._get("draw_delivery_mode", "normal") or "normal").lower() == "forward":
                forward_text = ""
                if self._get("draw_attach_prompt", False):
                    forward_text = self._prompt_attachment(params)
                return self._forward_result_chain(event, forward_text, images, params=params)
            return [Image.fromFileSystem(str(path)) for path in images]

    def _style_lora_usage_text(self, params: DrawParams | None) -> str:
        """返回本次任务实际加载的画风 LoRA 清单。"""
        style_loras = list(getattr(params, "style_loras_used", []) or []) if params else []
        if not style_loras:
            return "本次使用画风 LoRA：未使用"
        lines = ["本次使用画风 LoRA："]
        for item in style_loras:
            try:
                weight = float(item.get("weight", 0.8) or 0.0)
            except (TypeError, ValueError):
                weight = 0.8
            alias = str(item.get("alias", "画风") or "画风").strip()
            display_name = str(
                item.get("display_name", "") or item.get("file_name", "")
            ).strip()
            source = str(item.get("source", "") or "已加载").strip()
            file_name = str(item.get("file_name", "") or "").strip()
            lines.append(
                f"{alias}｜{display_name}｜{source}｜文件：{file_name}｜权重：{weight:g}"
            )
        return "\n".join(lines)

    def _forward_result_chain(
        self,
        event: AstrMessageEvent,
        reply: str,
        images: list[Path],
        params: DrawParams | None = None,
    ) -> list:
        """把图片和可选提示词组织成群聊常见的合并转发消息。"""
        get_self_id = getattr(event, "get_self_id", None)
        uin = str(get_self_id() if callable(get_self_id) else "0") or "0"
        forward_images = [self._prepare_forward_image(path) for path in images]
        nodes = [
            Node(
                uin=uin,
                name="ComfyUI 绘图",
                content=[Image.fromFileSystem(str(path))],
            )
            for path in forward_images
        ]
        if params is not None:
            nodes.append(
                Node(
                    uin=uin,
                    name="ComfyUI 绘图",
                    content=[Plain(self._style_lora_usage_text(params))],
                )
            )
        if str(reply or "").strip():
            nodes.append(Node(uin=uin, name="ComfyUI 绘图", content=[Plain(reply)]))
        chain = [Nodes(nodes)]
        # ``reply`` 在当前发送流程中是提示词附带内容；完成后的自然人格
        # 回复仍由 _send_completion_reply 单独发送，避免等待 AI 阻塞图片。
        if str(reply or "").strip() or params is not None:
            quote = self._completion_quote(event)
            return [quote, *chain] if quote else chain
        return chain

    def _prepare_forward_image(self, path: Path) -> Path:
        """为合并转发准备较小的图片副本，避免 OneBot 请求体过大。

        只处理实际存在且超过限制的文件。原始输出不修改，临时副本会在
        发送完成后由 `_cleanup_forward_chain` 删除。
        """
        cache_dir = getattr(self, "forward_cache_dir", None)
        if not cache_dir:
            return path
        try:
            source = Path(path)
            if not source.is_file():
                return source
            # Nodes.to_dict() 会把图片再次编码成 base64。4 MB 的文件会
            # 变成约 5.3 MB 的 WebSocket 请求，在 NapCat/OneBot 上很容易
            # 先发送成功、但等不到 API 响应，最终表现为 API call timeout。
            # 转发只需要可读预览，不应让原始输出承担这个传输限制。
            max_bytes = 1_500_000
            if source.stat().st_size <= max_bytes:
                return source

            from PIL import Image as PILImage

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / f"forward_{uuid.uuid4().hex}.jpg"
            with PILImage.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((1600, 1600), PILImage.Resampling.LANCZOS)
                quality = 78
                while True:
                    image.save(target, format="JPEG", quality=quality, optimize=True)
                    if target.stat().st_size <= max_bytes or quality <= 48:
                        break
                    quality -= 6
            return target
        except Exception as exc:
            logger.warning(
                "[%s] 转发图片压缩失败，将使用原图：%s：%s",
                PLUGIN_NAME,
                path,
                exc,
            )
            return path

    def _cleanup_forward_chain(self, chain: list) -> None:
        """删除本次合并转发创建的临时图片。"""
        cache_dir = getattr(self, "forward_cache_dir", None)
        if not cache_dir:
            return
        try:
            cache_root = Path(cache_dir).resolve()
        except (OSError, RuntimeError):
            return
        for component in chain:
            if not isinstance(component, Nodes):
                continue
            for node in component.nodes:
                for content in node.content:
                    if not isinstance(content, Image):
                        continue
                    candidate = getattr(content, "path", None)
                    if not candidate:
                        continue
                    try:
                        resolved = Path(candidate).resolve()
                        if resolved.parent == cache_root:
                            resolved.unlink(missing_ok=True)
                    except (OSError, RuntimeError):
                        logger.debug("[%s] 清理转发临时图片失败：%s", PLUGIN_NAME, candidate)

    async def _send_forward_component(
        self,
        event: AstrMessageEvent,
        component: Nodes,
    ) -> None:
        """发送合并转发节点，并对瞬时 OneBot 错误重试一次。"""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                await event.send(event.chain_result([component]))
                return
            except Exception as exc:
                last_error = exc
                # WebSocket action timeout 只代表客户端没有等到响应，
                # 请求本身可能已经被 QQ/NapCat 接收。此时重试会重复发出
                # 同一个合并转发，正是“图片重复发送”的主要来源；把它视为
                # 已提交，不再把生成任务判定为失败。
                if self._is_websocket_api_timeout(exc):
                    logger.warning(
                        "[%s] 合并转发等待 WebSocket 响应超时，停止重试以避免重复发送：%r",
                        PLUGIN_NAME,
                        exc,
                    )
                    return
                if attempt == 0:
                    logger.warning(
                        "[%s] 合并转发第 1 次发送失败，准备重试：%r",
                        PLUGIN_NAME,
                        exc,
                    )
                    await asyncio.sleep(0.8)
                else:
                    logger.exception(
                        "[%s] 合并转发发送失败（重试后仍失败）：%r",
                        PLUGIN_NAME,
                        exc,
                    )
        raise RuntimeError(f"合并转发发送失败：{last_error}") from last_error

    @staticmethod
    def _is_websocket_api_timeout(exc: BaseException) -> bool:
        """判断 OneBot WebSocket 是否只是在等待动作响应时超时。"""
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return True
        text = str(exc or "").casefold()
        return "websocket api call timeout" in text or "api call timeout" in text

    async def _send_forward_chain(self, event: AstrMessageEvent, chain: list) -> None:
        """发送合并转发结果，兼容 QQ 不支持 Reply+Nodes 混合链的情况。

        aiocqhttp 会把一条 ``[Reply, Nodes]`` 链拆成两个独立的 API 调用，
        于是 Reply 会先变成一条没有正文的空引用消息；随后较大的
        ``send_group_forward_msg`` 还可能等待超时。结果消息只发送 Nodes，
        原消息的引用由后面的完成正文统一发送，确保引用带正文且不会重复
        重试空 Reply。
        """
        if len(chain) == 2 and isinstance(chain[0], Reply) and isinstance(chain[1], Nodes):
            try:
                # 不把 Reply 和 Nodes 混在同一消息链交给平台适配器。完成
                # 回复会在图片发送后使用 [Reply, Plain(...)] 单独发送，
                # 那条引用有正文，不会产生空消息。
                await self._send_forward_component(event, chain[1])
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[%s] 合并转发发送失败，准备逐条发送结果图片：%r",
                    PLUGIN_NAME,
                    exc,
                )
            # Forward API 超时或平台不支持合并转发时，逐张发送。单张发送
            # 失败只记录，不再把已经完成的生成任务包装成“生成失败”；后面
            # 的完成正文仍会发送，用户可以据此看到真实的发送故障。
            flattened = self._flatten_forward_chain([chain[1]])
            sent_any = False
            for component in flattened:
                if not isinstance(component, (Image, Plain)):
                    continue
                try:
                    await event.send(event.chain_result([component]))
                    sent_any = True
                except Exception as fallback_error:
                    logger.warning(
                        "[%s] 合并转发降级发送单个结果失败：%r",
                        PLUGIN_NAME,
                        fallback_error,
                    )
            if sent_any:
                return
            # WebSocket 超时没有可靠的“服务端未执行”语义；可能已经发出，
            # 此处不再抛出二次异常，避免后台任务误报“生成失败”。
            logger.error(
                "[%s] 合并转发及逐条发送都未收到成功响应，原始错误：%r",
                PLUGIN_NAME,
                last_error,
            )
            return
        for component in chain:
            if isinstance(component, Nodes):
                await self._send_forward_component(event, component)
            elif isinstance(component, Reply):
                # 单独的 Reply 没有正文时会在 QQ 中显示成空消息。引用由
                # _send_completion_reply 以 Reply+Plain 的形式发送。
                continue
            else:
                await event.send(event.chain_result([component]))

    @staticmethod
    def _flatten_forward_chain(chain: list) -> list:
        """平台不支持合并转发时，恢复为普通消息链。"""
        flattened = []
        for component in chain:
            if isinstance(component, Nodes):
                for node in component.nodes:
                    flattened.extend(node.content)
            else:
                flattened.append(component)
        return flattened

    @staticmethod
    def _result_image_count(chain: list | None) -> int:
        """统计结果消息中的图片数量，错误文字不触发完成回复。"""
        count = 0
        for component in chain or []:
            if isinstance(component, Image):
                count += 1
            elif isinstance(component, Nodes):
                count += sum(
                    1
                    for node in (component.nodes or [])
                    for content in (node.content or [])
                    if isinstance(content, Image)
                )
        return count

    async def _send_completion_reply(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        count: int,
        *,
        extra: str = "",
    ) -> None:
        """在图片发送完成后，单独发送自然的 AstrBot 人格回复。"""
        if count <= 0:
            return
        reply = await self._draw_reply(event, params, mode, count)
        delivery_mode = str(self._get("draw_delivery_mode", "normal") or "normal").lower()
        prompt_in_forward = delivery_mode == "forward"
        if self._get("draw_attach_prompt", False) and not prompt_in_forward:
            prompt_attachment = self._prompt_attachment(params)
            if prompt_attachment:
                reply = f"{reply}\n\n{prompt_attachment}"
        if delivery_mode != "forward":
            style_usage = self._style_lora_usage_text(params)
            if style_usage:
                reply = f"{reply}\n\n{style_usage}" if reply else style_usage
        if extra:
            reply = f"{reply}\n\n{extra}" if reply else extra
        if str(reply or "").strip():
            quote = self._completion_quote(event)
            content = [quote, Plain(reply)] if quote else [Plain(reply)]
            await event.send(event.chain_result(content))

    async def _background_draw(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str = "",
        second_image_path: str = "",
        third_image_path: str = "",
    ) -> None:
        chain = None
        try:
            chain = await self._run(
                event,
                params,
                mode,
                image_path,
                second_image_path,
                third_image_path,
            )
            blocked_chain = await self._moderation_output_chain(event, params, chain)
            if blocked_chain is not None:
                # 命中输出审核时图片已经生成完毕，但绝不对外发送，只回传警告。
                await event.send(event.chain_result(blocked_chain))
                return
            if any(isinstance(component, Nodes) for component in chain):
                await self._send_forward_chain(event, chain)
            else:
                await event.send(event.chain_result(chain))
            await self._send_completion_reply(event, params, mode, self._result_image_count(chain))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[%s] 后台绘图发送结果失败", PLUGIN_NAME)
            try:
                await event.send(event.plain_result(f"生成失败：{exc}"))
            except Exception:
                logger.exception("[%s] 无法发送后台绘图错误", PLUGIN_NAME)
        finally:
            if chain:
                self._cleanup_forward_chain(chain)

    async def _llm_draw(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str = "",
        second_image_path: str = "",
        third_image_path: str = "",
    ):
        """为 LLM 工具提交绘图任务，并由插件后台可靠发送最终图片。

        AstrBot 的本地 LLM 工具结果会经过 ``tool_direct_result`` 包装。部分
        平台适配器会把其中的 Nodes/图片当成工具结果处理，导致文本能看到但
        图片丢失。因此 LLM 工具只返回“已提交”状态，实际结果统一走插件的
        普通发送路径；这样也保留了“画图期间可以继续聊天”。
        """
        # 开始提示直接发送，不再写入 event.result。LLM 工具执行器会把
        # event.result 当成 tool_direct_result 再发送一次，旧做法容易在
        # 工具结束后留下空 AT/空回复消息。
        start_reply = await self._llm_draw_start_reply(event, params, mode)
        if start_reply:
            try:
                start_result = event.plain_result(start_reply)
                sender = getattr(event, "send", None)
                if callable(sender):
                    await sender(start_result)
                    clear_result = getattr(event, "clear_result", None)
                    if callable(clear_result):
                        # 防止工具执行器在看到本轮事件结果时再次发送一条
                        # 空的 At/Reply；开始提示已经由 send() 发出。
                        clear_result()
                else:
                    # 兼容没有 send() 的最小测试事件；真实 AstrBot 事件
                    # 都会走上面的直接发送路径。
                    set_result = getattr(event, "set_result", None)
                    if callable(set_result):
                        set_result(start_result)
            except Exception as exc:
                logger.warning("[%s] LLM 绘图开始提示发送失败，继续执行任务：%s", PLUGIN_NAME, exc)
        task = asyncio.create_task(
            self._run(
                event,
                params,
                mode,
                image_path,
                second_image_path,
                third_image_path,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        send_task = asyncio.create_task(self._send_llm_task_result(event, task, params))
        self._tasks.add(send_task)
        send_task.add_done_callback(self._tasks.discard)
        return None

    async def _send_llm_task_result(
        self,
        event: AstrMessageEvent,
        task: asyncio.Task,
        params: DrawParams | None = None,
    ) -> None:
        """发送超时后完成的 LLM 绘图结果，并清理合并转发临时文件。"""
        chain = None
        try:
            chain = await task
            if params is not None:
                blocked_chain = await self._moderation_output_chain(event, params, chain)
                if blocked_chain is not None:
                    # 与 _background_draw 一致：命中输出审核只发警告，不发图。
                    await event.send(event.chain_result(blocked_chain))
                    return
            if any(isinstance(component, Nodes) for component in chain):
                await self._send_forward_chain(event, chain)
            else:
                await event.send(event.chain_result(chain))
            if params is not None:
                debug_reply = ""
                if (
                    self._img2img_plugin_ai_debug_enabled()
                    if params.mode == "img2img"
                    else self._img2img_flux2_plugin_ai_debug_enabled()
                    if params.mode == "img2img_flux2"
                    else self._plugin_ai_debug_enabled()
                ) and getattr(params, "plugin_ai_debug_prompt", ""):
                    debug_reply = self._llm_debug_reply(
                        "",
                        str(getattr(params, "plugin_ai_debug_prompt", "") or ""),
                        str(getattr(params, "plugin_ai_debug_input", "") or ""),
                    )
                await self._send_completion_reply(
                    event,
                    params,
                    params.mode,
                    self._result_image_count(chain),
                    extra=debug_reply,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[%s] LLM 后台绘图发送结果失败", PLUGIN_NAME)
            try:
                await event.send(event.plain_result(f"生成失败：{exc}"))
            except Exception:
                logger.exception("[%s] 无法发送 LLM 后台绘图错误", PLUGIN_NAME)
        finally:
            if chain:
                self._cleanup_forward_chain(chain)

    def _schedule_draw(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str = "",
        second_image_path: str = "",
        third_image_path: str = "",
    ) -> None:
        task = asyncio.create_task(
            self._background_draw(
                event,
                params,
                mode,
                image_path,
                second_image_path,
                third_image_path,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _help_text(self) -> str:
        """帮助图片不可用时的文字版兜底内容。"""
        return (
            "Anima 全能绘画台指令\n"
            "/文生图 描述\n"
            "/图生图 描述（同一条消息附图，或回复一张图片/含图合并转发）\n"
            "/图生图Flux2 描述（支持 1 到 3 张参考图，第一张为主参考图）\n"
            "/高清放大（附图片、回复图片，或放大最近出图）\n"
            "/洗图 中文要求（可附多张图片，只使用第一张，其余忽略；不调用 AI 或翻译）\n"
            "/扩图 中文要求（可附多张图片，只使用第一张，其余忽略；扩展边距在 WebUI 设置）\n"
            "/多角度 中文要求（可附多张图片，只使用第一张，其余忽略；角度参数在 WebUI 设置）\n"
            "在画图指令中单独写 ai 可开启本次 AI 提示词优化，例如：/文生图 ai 夏空；ai 不会进入最终提示词。也可写 noai 或继续使用 --ai=开、--noai。\n"
            "画图指令末尾可直接写 LoRA 指令简称临时加载，例如：/文生图 夏空海边 1号lora；也支持 1号lora:0.6，任务结束后不会保存为默认 LoRA\n"
            "本次不想使用任何画风 LoRA 时，可在指令或自然语言中写“不用画风”；它会同时跳过常态画风和随机画风，不影响普通 LoRA。\n"
            "如果 LoRA 指令简称与普通提示词预设同名，直接写名称会同时启用 LoRA 和预设；只使用预设可写 --预设=名称\n"
            "/模型 列表 或 /模型 名称\n"
            "/lora 或 /.lora 查看全部 LoRA 总览图片（含每个 LoRA 一张预览图、分类、简称和开启状态）；/lora 1号lora 查看单个 LoRA；/loraon 1号lora 开启 LoRA；/lora 添加或删除 指令简称\n"
            "/预设 列表 查看按 LoRA 分类的指令简称与预设图片；/预设 添加 夏空=ciaccona；/预设 修改 夏空=新内容；/预设 翻译 夏空；/预设 删除 夏空\n"
            "/画师串 列表；/画师串 使用 画风001；/画师串 添加 名称=画师 tags；/画师串 关闭\n"
            "/工作流 列表；/工作流 文生图 文件名；/工作流 图生图 文件名；/工作流 高清放大 文件名；/工作流 洗图 文件名；/工作流 扩图 文件名；/工作流 多角度 文件名\n"
            "/画图配置 查询模型位置、工作流位置和当前配置\n"
            "/comfy状态 查询 ComfyUI 状态；管理员可用 /comfy打开 启动本机 ComfyUI、/comfy关闭 关闭服务\n"
            "直接用自然语言要求 AstrBot 画图时，会自动识别提示词预设和 LoRA 指令简称；预设与 LoRA 可同时生效，其余内容作为画面描述。\n"
            "生成结果的发送方式可在 WebUI 回复设置中选择普通消息或群聊合并转发；不支持合并转发的平台会自动退回普通消息。\n"
            "鸣潮角色知识和 Anima 提示词工程师由独立插件提供；本插件只负责绘图执行，并保留临时 LoRA 和提示词预设的最高优先级。\n"
            "绘画选项：--模型=名称 --lora=a:0.8,b:0.5 --预设=夏空 --画师=画风001 --负面=内容 --宽=832 --高=1216 --步数=24 --种子=-1 --ai=开 --noai --强度=0.6 --放大=2"
        )

    def _anima_template_override(self) -> str:
        """WebUI 配置的额外提示词模板路径（文件或目录），未配置时为空串。"""
        return str(self._get("anima_template_path", "") or "").strip()

    def _list_image_font_paths(self, bold: bool = False) -> list[str]:
        """图片字体候选路径：优先使用 WebUI 配置，未配置时自动探测系统字体。"""
        regular_paths, bold_paths = image_font_candidates(
            str(self._get("image_font_regular", "") or ""),
            str(self._get("image_font_bold", "") or ""),
        )
        return bold_paths if bold else regular_paths

    def _list_image_font(self, size: int, bold: bool = False):
        from PIL import ImageFont

        for font_path in self._list_image_font_paths(bold=bold):
            try:
                if Path(font_path).is_file():
                    return ImageFont.truetype(font_path, size=size)
            except (OSError, ValueError):
                continue
        return ImageFont.load_default()

    @staticmethod
    def _list_image_wrap(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
        result: list[str] = []
        current = ""
        for char in str(text or ""):
            candidate = current + char
            box = draw.textbbox((0, 0), candidate, font=font)
            if current and box[2] - box[0] > max_width:
                result.append(current)
                current = char
            else:
                current = candidate
        if current:
            result.append(current)
        return result or [""]

    def _save_list_image(self, image: Any, name: str) -> Path:
        path = self.data_dir / name
        temp_path = self.data_dir / f".{name}.tmp"
        image.save(temp_path, format="JPEG", quality=88, optimize=True)
        temp_path.replace(path)
        return path

    async def _download_lora_preview(self, url: str) -> Any | None:
        """下载并缓存一个 CivitAI 预览图，失败时返回 None。"""
        from PIL import Image as PILImage

        url = str(url or "").strip()
        if not url.startswith(("http://", "https://")):
            return None
        cache_dir = self.data_dir / "list_preview_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.img"
        try:
            if cache_path.is_file() and cache_path.stat().st_size > 0:
                with PILImage.open(cache_path) as cached:
                    return cached.convert("RGB").copy()
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(12, read=30),
                follow_redirects=True,
                headers={"User-Agent": "AstrBot-ComfyUI-AI-Studio"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.content
            if not payload or len(payload) > 12 * 1024 * 1024:
                return None
            with PILImage.open(BytesIO(payload)) as downloaded:
                image = downloaded.convert("RGB").copy()
            image.save(cache_path, format="JPEG", quality=86, optimize=True)
            return image
        except (httpx.HTTPError, OSError, ValueError, TypeError):
            return None

    async def _preset_list_image_path(self, category: str = "") -> Path:
        """生成按 LoRA 分组的简称与专属预设长图，可按分类筛选。"""
        from PIL import Image as PILImage
        from PIL import ImageDraw

        available = await self._local_lora_names()
        names = self._order_loras(available or self._known_lora_preset_keys())
        category = str(category or "").strip()
        if category:
            names = [name for name in names if self._lora_category(name) == category]
        groups: list[tuple[str, str, list[tuple[str, str]]]] = []
        linked_names: set[str] = set()
        for file_name in names:
            entries = self._lora_preset_entries(file_name)
            if not entries:
                continue
            linked_names.update(str(entry.get("tag", "")).strip() for entry in entries)
            groups.append(
                (
                    self._lora_alias(file_name),
                    file_name,
                    [
                        (
                            str(entry.get("tag", "")).strip() or "未命名简称",
                            str(entry.get("content", "")).strip() or "（空内容）",
                        )
                        for entry in entries
                    ],
                )
            )
        global_entries: list[tuple[str, str]] = []
        for name, item in self.presets.items.items():
            # 兼容早期配置中的异常条目，不能让一个坏预设阻断整张列表图片。
            if not isinstance(item, dict):
                continue
            preset_name = str(name).strip()
            preset_content = str(item.get("translated") or item.get("content", "")).strip()
            if preset_name and preset_name not in linked_names and preset_content:
                global_entries.append((preset_name, preset_content))
        if global_entries:
            groups.append(("未关联 LoRA 的全局预设", "这些预设不会自动加载 LoRA", global_entries))

        title_font = self._list_image_font(48, bold=True)
        section_font = self._list_image_font(28, bold=True)
        label_font = self._list_image_font(22, bold=True)
        body_font = self._list_image_font(20)
        footer_font = self._list_image_font(18)
        measure = ImageDraw.Draw(PILImage.new("RGB", (1, 1)))
        width = 1500
        margin = 58
        card_width = width - margin * 2
        card_heights: list[tuple[int, list[list[str]]]] = []
        for alias, file_name, entries in groups:
            lines: list[list[str]] = []
            for tag, content in entries:
                lines.append(self._list_image_wrap(measure, f"{tag}  =  {content}", body_font, card_width - 68))
            height = 78 + sum(max(1, len(item)) * 31 + 8 for item in lines) + 20
            card_heights.append((height, lines))
        header = 142
        footer = 54
        height = header + footer + sum(item[0] + 22 for item in card_heights) + margin
        height = max(height, 260)
        image = PILImage.new("RGB", (width, height), (15, 23, 29))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 10), fill=(104, 211, 181))
        draw.text((margin, 40), "提示词预设与 LoRA 简称", font=title_font, fill=(238, 248, 246))
        scope = category or "全部分类"
        draw.text((margin, 100), f"按 LoRA 分类：{scope} · 共 {sum(len(item[2]) for item in groups)} 条 · 插件 v{PLUGIN_VERSION}", font=footer_font, fill=(164, 187, 190))
        y = header
        for index, ((alias, file_name, entries), (card_height, lines)) in enumerate(zip(groups, card_heights), 1):
            draw.rounded_rectangle((margin, y, margin + card_width, y + card_height), radius=16, fill=(25, 37, 44), outline=(49, 69, 76), width=2)
            draw.rounded_rectangle((margin, y, margin + 9, y + card_height), radius=5, fill=(104, 211, 181))
            draw.text((margin + 28, y + 18), f"{index}. {alias}", font=section_font, fill=(255, 210, 122))
            draw.text((margin + 28, y + 53), f"文件：{file_name}", font=footer_font, fill=(146, 170, 174))
            line_y = y + 88
            for (tag, content), wrapped in zip(entries, lines):
                draw.text((margin + 34, line_y), f"{tag}", font=label_font, fill=(104, 211, 181))
                line_y += 29
                for line in self._list_image_wrap(draw, content, body_font, card_width - 100):
                    draw.text((margin + 62, line_y), line, font=body_font, fill=(226, 237, 237))
                    line_y += 31
                line_y += 8
            y += card_height + 22
        draw.text((margin, height - footer + 8), "指令简称可直接写在 /文生图 或自然语言绘图中；同一 LoRA 可按需使用不同简称。", font=footer_font, fill=(146, 170, 174))
        return self._save_list_image(image, "preset_list.jpg")

    async def _lora_list_image_path(self, details: list[dict[str, Any]], enabled: list[str]) -> Path:
        """生成包含 CivitAI 预览图的 LoRA 总览长图。"""
        from PIL import Image as PILImage
        from PIL import ImageDraw

        enabled_map: dict[str, str] = {}
        for value in enabled:
            file_name, separator, weight = str(value).rpartition(":")
            enabled_map[(file_name if separator else value).replace("\\", "/").casefold()] = weight if separator else "0.8"
        preview_urls = [
            str((item.get("images") or [{}])[0].get("url", ""))
            if item.get("show_images", True) is not False
            and isinstance((item.get("images") or [{}])[0], dict)
            else ""
            for item in details
        ]
        previews = await asyncio.gather(*(self._download_lora_preview(url) for url in preview_urls))
        width = 1600
        margin = 54
        gap = 24
        card_width = (width - margin * 2 - gap) // 2
        card_height = 286
        title_font = self._list_image_font(48, bold=True)
        card_font = self._list_image_font(26, bold=True)
        body_font = self._list_image_font(19)
        small_font = self._list_image_font(16)
        measure = ImageDraw.Draw(PILImage.new("RGB", (1, 1)))
        rows = (len(details) + 1) // 2
        header = 148
        footer = 56
        height = max(260, header + footer + rows * (card_height + gap) + margin)
        image = PILImage.new("RGB", (width, height), (15, 23, 29))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 10), fill=(104, 211, 181))
        draw.text((margin, 40), "LoRA 模型总览", font=title_font, fill=(238, 248, 246))
        draw.text((margin, 102), f"共 {len(details)} 个 · 按下载时间从新到旧 · 每个 LoRA 展示一张 CivitAI 预览图", font=small_font, fill=(164, 187, 190))
        for index, (item, preview) in enumerate(zip(details, previews)):
            row, column = divmod(index, 2)
            x = margin + column * (card_width + gap)
            y = header + row * (card_height + gap)
            draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=16, fill=(25, 37, 44), outline=(49, 69, 76), width=2)
            thumb_box = (x + 18, y + 18, x + 238, y + 238)
            draw.rounded_rectangle(thumb_box, radius=10, fill=(12, 18, 23), outline=(49, 69, 76), width=1)
            if preview is not None:
                thumb = preview.copy()
                thumb.thumbnail((204, 204), PILImage.Resampling.LANCZOS)
                tx = thumb_box[0] + (220 - thumb.width) // 2
                ty = thumb_box[1] + (220 - thumb.height) // 2
                image.paste(thumb, (tx, ty))
            else:
                placeholder = self._list_image_wrap(draw, "暂无 CivitAI 预览图", body_font, 190)
                for line_index, line in enumerate(placeholder):
                    draw.text((x + 34, y + 105 + line_index * 27), line, font=body_font, fill=(146, 170, 174))
            text_x = x + 264
            text_width = card_width - 292
            alias = str(item.get("alias", "未命名 LoRA"))
            draw.text((text_x, y + 22), alias, font=card_font, fill=(255, 210, 122))
            file_name = str(item.get("file_name", ""))
            state = "已开启" if file_name.replace("\\", "/").casefold() in enabled_map else "未开启"
            weight = enabled_map.get(file_name.replace("\\", "/").casefold(), "")
            lines = [
                f"状态：{state}" + (f"（权重 {weight}）" if weight else ""),
                f"分类：{item.get('category', '未分类')}",
                f"简称：{'、'.join(item.get('command_aliases') or [item.get('command_alias', '无')])}",
                f"文件：{file_name}",
                f"C站：{item.get('model_url', '暂无链接')}",
            ]
            line_y = y + 66
            for line in lines:
                for wrapped in self._list_image_wrap(measure, line, body_font, text_width):
                    draw.text((text_x, line_y), wrapped, font=body_font, fill=(226, 237, 237))
                    line_y += 27
                line_y += 3
        draw.text((margin, height - footer + 8), "输入 /lora 或 /.lora 查看此图；输入 /lora 指令简称查看单个 LoRA 的 CivitAI 信息。", font=small_font, fill=(146, 170, 174))
        return self._save_list_image(image, "lora_list.jpg")

    def _help_image_path(self) -> Path:
        """生成适合聊天发送的中文帮助长图。"""
        from PIL import Image as PILImage
        from PIL import ImageDraw, ImageFont

        font_paths, bold_font_paths = image_font_candidates(
            str(self._get("image_font_regular", "") or ""),
            str(self._get("image_font_bold", "") or ""),
        )

        def load_font(size: int, bold: bool = False):
            for font_path in (bold_font_paths if bold else font_paths):
                try:
                    if Path(font_path).is_file():
                        return ImageFont.truetype(font_path, size=size)
                except (OSError, ValueError):
                    continue
            return ImageFont.load_default()

        small = load_font(22)
        card_title = load_font(29, bold=True)
        title_font = load_font(52, bold=True)
        subtitle_font = load_font(24)
        footer_font = load_font(20)

        width = 1500
        margin = 66
        column_gap = 26
        card_width = (width - margin * 2 - column_gap) // 2
        sections = [
            (
                "基础绘图",
                [
                    "/文生图 内容：文生图",
                    "/图生图 内容：附图后修改画面",
                    "/图生图Flux2 内容：强制 Flux2，使用 1 张参考图",
                    "普通 /图生图 使用 WebUI 选择的 Qwen 或 Flux2 引擎",
                    "/高清放大 附图、回复图片或放大最近出图",
                    "画图时可以继续聊天，任务完成后会自动发送结果",
                ],
            ),
            (
                "LoRA 与预设",
                [
                    "/lora 或 /.lora：发送包含预览图的全部 LoRA 总览图片",
                    "/lora 简称：查看单个 LoRA 的 C站信息",
                    "/loraon 简称 开启或关闭 LoRA",
                    "画图内容后写 LoRA 简称，可临时加载且任务结束自动关闭",
                    "预设和 LoRA 简称同名时会同时生效",
                ],
            ),
            (
                "工作流与配置",
                [
                    "/洗图 中文要求：可附多张图，仅使用第一张，其余忽略",
                    "/扩图 中文要求：可附多张图，仅使用第一张，其余忽略",
                    "/多角度 中文要求：可附多张图，仅使用第一张，其余忽略",
                    "/模型 列表 或 /模型 名称",
                    "/工作流 列表；/工作流 文生图 文件名；图生图、高清放大、洗图、扩图、多角度均可切换",
                    "/画图配置 查看模型、LoRA 和工作流位置",
                    "宽高、步数、种子、CFG、采样器等可在 WebUI 调整",
                ],
            ),
            (
                "自然语言绘图",
                [
                    "直接说：帮我画一张夏空在海边的图片",
                    "AstrBot 会识别绘图模式、动作、预设和 LoRA 简称",
                    "可在 WebUI 选择 AstrBot AI 或插件 AI 生成提示词",
                    "提示词优化失败时会提示原因，不会静默改变用户要求",
                ],
            ),
            (
                "预设与画师串",
                [
                    "/预设 列表：发送按 LoRA 分类的简称预设图片",
                    "/预设 添加 名称=内容；修改或删除名称",
                    "/预设 修改 名称=新内容；/预设 删除 名称",
                    "/画师串 列表；/画师串 使用 画风001",
                    "普通提示词预设和 LoRA 独立 tag 预设均可在 WebUI 管理",
                ],
            ),
            (
                "WebUI 与状态",
                [
                    "/comfy状态 查看 ComfyUI 连接状态",
                    "WebUI 可切换工作流、模型、明暗主题和回复方式",
                    "支持批量导入 LoRA，可一次选择多个文件并统一归类",
                    "支持 CivitAI 链接下载，免 API Key 也能下载公共 LoRA",
                    "下载完成后会自动刷新模型、LoRA 和预设信息",
                ],
            ),
        ]

        def wrap(text: str, font, max_width: int) -> list[str]:
            result: list[str] = []
            current = ""
            for char in text:
                candidate = current + char
                box = measure_draw.textbbox((0, 0), candidate, font=font)
                if current and box[2] - box[0] > max_width:
                    result.append(current)
                    current = char
                else:
                    current = candidate
            if current:
                result.append(current)
            return result or [""]

        measure_draw = ImageDraw.Draw(PILImage.new("RGB", (1, 1)))
        card_data = []
        for title, lines in sections:
            wrapped_lines = []
            for line in lines:
                wrapped_lines.extend(wrap(line, small, card_width - 54))
            card_height = 72 + len(wrapped_lines) * 34 + 24
            card_data.append((title, wrapped_lines, card_height))

        header_height = 176
        footer_height = 66
        row_heights = [
            max(card_data[index][2], card_data[index + 1][2])
            for index in range(0, len(card_data), 2)
        ]
        height = header_height + sum(row_heights) + 28 * len(row_heights) + footer_height + margin
        image = PILImage.new("RGB", (width, height), (15, 23, 29))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 10), fill=(104, 211, 181))
        draw.text((margin, 48), "Anima 全能绘画台", font=title_font, fill=(238, 248, 246))
        draw.text(
            (margin, 116),
            f"本地 ComfyUI 绘画工作台 · AstrBot 绘图指令速查 · 插件版本 {PLUGIN_VERSION}",
            font=subtitle_font,
            fill=(164, 187, 190),
        )

        y = header_height
        for row_index, row_height in enumerate(row_heights):
            for column in range(2):
                item_index = row_index * 2 + column
                title, lines, _ = card_data[item_index]
                x = margin + column * (card_width + column_gap)
                draw.rounded_rectangle(
                    (x, y, x + card_width, y + row_height),
                    radius=18,
                    fill=(25, 37, 44),
                    outline=(49, 69, 76),
                    width=2,
                )
                draw.rounded_rectangle((x, y, x + 9, y + row_height), radius=5, fill=(104, 211, 181))
                draw.text((x + 28, y + 20), title, font=card_title, fill=(255, 210, 122))
                line_y = y + 76
                for line in lines:
                    draw.ellipse((x + 30, line_y + 9, x + 39, line_y + 18), fill=(104, 211, 181))
                    draw.text((x + 52, line_y), line, font=small, fill=(226, 237, 237))
                    line_y += 34
            y += row_height + 28

        draw.text(
            (margin, height - footer_height + 7),
            "输入 /helpd 可再次查看帮助 · 详细参数和模型管理请打开 AstrBot WebUI",
            font=footer_font,
            fill=(146, 170, 174),
        )
        path = self.data_dir / "helpd.png"
        temp_path = self.data_dir / "helpd.tmp.png"
        image.save(temp_path, format="PNG", optimize=True)
        temp_path.replace(path)
        return path

    @filter.command("helpd", desc="查询 Anima 全能绘画台绘图指令")
    async def command_help(self, event: AstrMessageEvent):
        try:
            image_path = self._help_image_path()
            yield event.chain_result([Image.fromFileSystem(str(image_path))])
        except Exception as exc:
            logger.warning("[%s] 生成帮助图片失败，回退文字帮助：%s", PLUGIN_NAME, exc)
            yield event.plain_result(self._help_text())

    @filter.command("文生图", alias=["txt2img"], desc="使用原始工作流进行文生图")
    async def command_txt2img(self, event: AstrMessageEvent, prompt: GreedyStr):
        try:
            params = fill_params(str(prompt), "txt2img")
            await self._extract_inline_loras(params)
            self._extract_inline_presets(params)
            if not params.prompt and not params.presets and not params.loras:
                raise UsageError("请提供画面描述、提示词预设或 LoRA 指令简称")
        except (UsageError, ComfyError) as exc:
            yield event.plain_result(f"用法错误：{exc}")
            return
        try:
            await self._check_draw_queue(params, "txt2img", event)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        start_reply = await self._draw_start_reply_for_command(event, params, "txt2img")
        if start_reply:
            yield event.plain_result(start_reply)
        self._schedule_draw(event, params, "txt2img")

    async def _command_img2img_mode(
        self,
        event: AstrMessageEvent,
        prompt: str,
        mode: str,
    ):
        logger.info(
            "[%s] %s命令已触发：原文=%s；消息图片字段=%s；图片数量=%s；回复=%s",
            PLUGIN_NAME,
            MODE_NAMES[mode],
            str(getattr(event, "message_str", "") or "")[:160],
            bool(getattr(event, "image", None)),
            len(getattr(event, "image_list", None) or []),
            getattr(event, "reply", None) or getattr(event, "reply_id", None) or "无",
        )
        try:
            params = fill_params(str(prompt), mode)
            # 这是用户直接发送的斜杠命令。普通图生图必须把用户原话
            # 原样交给工作流；只有显式 ``ai`` 才允许提示词 AI 介入。
            params.command_invocation = True
            if mode == "img2img":
                await self._extract_inline_loras(params)
                self._extract_inline_presets(params)
        except (UsageError, ComfyError) as exc:
            yield event.plain_result(f"用法错误：{exc}")
            return
        image_paths = await self._extract_images(event)
        image_path = image_paths[0] if image_paths else ""
        second_image_path = image_paths[1] if len(image_paths) > 1 else ""
        third_image_path = image_paths[2] if len(image_paths) > 2 else ""
        if not image_path:
            logger.warning("[%s] %s命令已触发，但未提取到输入图片", PLUGIN_NAME, MODE_NAMES[mode])
            yield event.plain_result(f"{MODE_NAMES[mode]}需要在同一条消息附图，或回复一张图片/含图合并转发后发送指令")
            return
        # 输入端审核提前到开始提示之前：命中时不占用队列、不回复“正在生成”。
        blocked_chain = await self._moderation_input_chain(
            event, image_path, second_image_path, third_image_path
        )
        if blocked_chain is not None:
            yield event.chain_result(blocked_chain)
            return
        if mode == "img2img" and third_image_path:
            yield event.plain_result("原版 Qwen 图生图最多使用两张参考图，第三张及之后的图片已忽略。")
        elif mode == "img2img_flux2" and len(image_paths) > 3:
            logger.info(
                "[%s] Flux2 图生图收到 %s 张图片，只使用前三张",
                PLUGIN_NAME,
                len(image_paths),
            )
            yield event.plain_result("Flux2 图生图最多支持 3 张参考图，第 4 张及之后的图片已忽略。")
            image_paths = image_paths[:3]
            image_path = image_paths[0]
            second_image_path = image_paths[1] if len(image_paths) > 1 else ""
            third_image_path = image_paths[2] if len(image_paths) > 2 else ""
        try:
            await self._check_draw_queue(params, mode, event)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        logger.info(
            "[%s] %s命令已拿到输入图片，开始后台提交：参考图=%s",
            PLUGIN_NAME,
            MODE_NAMES[mode],
            image_path,
        )
        # 指令触发必须立即给出固定中文状态，避免本次命令因为后台任务
        # 没有即时结果而继续落入 AstrBot 人格回复。LLM 工具走 _llm_draw，
        # 不经过这里，仍由它自己的开始提示处理。
        start_reply = await self._draw_start_reply_for_command(event, params, mode)
        if start_reply:
            yield event.plain_result(start_reply)
        self._schedule_draw(
            event,
            params,
            mode,
            image_path,
            second_image_path,
            third_image_path if mode == "img2img_flux2" else "",
        )

    @filter.command("图生图", alias=["img2img", "局部重绘", "inpaint"], desc="使用当前选择的图生图工作流进行参考图编辑")
    async def command_img2img(self, event: AstrMessageEvent, prompt: GreedyStr):
        async for result in self._command_img2img_mode(
            event,
            str(prompt),
            self._img2img_mode(),
        ):
            yield result

    @filter.command("图生图Flux2", alias=["图生图2", "Flux2图生图"], desc="使用 Flux2 Klein 工作流进行 1 到 3 张参考图编辑")
    async def command_img2img_flux2(self, event: AstrMessageEvent, prompt: GreedyStr):
        async for result in self._command_img2img_mode(event, str(prompt), "img2img_flux2"):
            yield result

    async def _command_flux2_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        mode: str,
    ):
        try:
            params = fill_params(str(prompt), mode)
        except UsageError as exc:
            yield event.plain_result(f"用法错误：{exc}")
            return
        image_paths = await self._extract_images(event)
        image_path = image_paths[0] if image_paths else ""
        if not image_path:
            yield event.plain_result(f"{MODE_NAMES[mode]}需要在同一条消息附一张图片，或回复一张图片/含图合并转发")
            return
        # 只审核实际会使用的第一张图片，其余图片本来就会被忽略。
        blocked_chain = await self._moderation_input_chain(event, image_path)
        if blocked_chain is not None:
            yield event.chain_result(blocked_chain)
            return
        if len(image_paths) > 1:
            logger.info(
                "[%s] %s收到 %s 张图片，只使用第一张：%s",
                PLUGIN_NAME,
                MODE_NAMES[mode],
                len(image_paths),
                image_path,
            )
        # Flux2 工具模式不会读取全局/临时 LoRA、预设、AI 或普通翻译。
        params.loras = []
        params.presets = []
        try:
            await self._check_draw_queue(params, mode, event)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        start_reply = await self._draw_start_reply_for_command(event, params, mode)
        if start_reply:
            yield event.plain_result(start_reply)
        self._schedule_draw(event, params, mode, image_path)

    @filter.command("洗图", alias=["wash"], desc="使用 Flux2 洗图工作流处理第一张输入图")
    async def command_wash(self, event: AstrMessageEvent, prompt: GreedyStr):
        async for result in self._command_flux2_tool(event, str(prompt), "wash"):
            yield result

    @filter.command("扩图", alias=["outpaint"], desc="使用 Flux2 工作流扩展第一张输入图")
    async def command_outpaint(self, event: AstrMessageEvent, prompt: GreedyStr):
        async for result in self._command_flux2_tool(event, str(prompt), "outpaint"):
            yield result

    @filter.command("多角度", alias=["multi_angle"], desc="使用 Flux2 工作流转换第一张输入图的视角")
    async def command_multi_angle(self, event: AstrMessageEvent, prompt: GreedyStr):
        async for result in self._command_flux2_tool(event, str(prompt), "multi_angle"):
            yield result


    @filter.command("高清放大", alias=["upscale"], desc="使用原始工作流进行高清放大")
    async def command_hires(self, event: AstrMessageEvent, prompt: GreedyStr):
        try:
            params = fill_params(str(prompt), "hires")
            await self._extract_inline_loras(params)
            self._extract_inline_presets(params)
        except (UsageError, ComfyError) as exc:
            yield event.plain_result(f"用法错误：{exc}")
            return
        image_path = await self._extract_image(event)
        if not image_path:
            origin = self._origin(event)
            image_path = self.last_images.get(origin, "")
        if not image_path:
            yield event.plain_result("高清放大需要附图、回复图片，或先生成一张图片")
            return
        blocked_chain = await self._moderation_input_chain(event, image_path)
        if blocked_chain is not None:
            yield event.chain_result(blocked_chain)
            return
        try:
            await self._check_draw_queue(params, "hires", event)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        start_reply = await self._draw_start_reply_for_command(event, params, "hires")
        if start_reply:
            yield event.plain_result(start_reply)
        self._schedule_draw(event, params, "hires", image_path)

    @filter.command("模型", alias=["model"], desc="查看或切换核心模型")
    async def command_model(self, event: AstrMessageEvent, arg: GreedyStr):
        client = self._client()
        try:
            env = await self._environment(client)
        except ComfyError as exc:
            yield event.plain_result(f"查询失败：{exc}")
            return
        models = self._compatible_models(env)
        value = str(arg or "").strip()
        current = str(self._get("model_name", "") or "") or (models[0] if models else "")
        if not value or value in {"列表", "list"}:
            lines = [f"当前核心模型：{current}", "可用模型："]
            lines.extend(f"{index}. {name}" for index, name in enumerate(models, 1))
            yield event.plain_result("\n".join(lines))
            return
        if value.isdigit() and 1 <= int(value) <= len(models):
            value = models[int(value) - 1]
        if value not in models:
            yield event.plain_result(f"模型不存在：{value}")
            return
        self._set("model_name", value)
        self._save_config()
        yield event.plain_result(f"核心模型已切换为：{value}")

    def _civitai_chain(self, details: list[dict[str, Any]], heading: str) -> list[Any]:
        chain: list[Any] = [Plain(heading)]
        for item in details:
            alias = item["alias"]
            command_aliases = item.get("command_aliases") or [item.get("command_alias", "")]
            command_text = "、".join(str(value) for value in command_aliases if str(value).strip()) or "无"
            label = f"{alias}（指令简称：{command_text}；文件：{item['file_name']}）"
            if not item.get("found"):
                chain.append(Plain(f"\n{label}：未找到匹配信息；搜索链接：{item['model_url']}"))
                continue
            title = "我的 CivitAI 信息" if item.get("custom_info") else ("我的 CivitAI 链接" if item.get("custom_link") else "CivitAI")
            chain.append(Plain(f"\n{label}\n{title}：{item['model_url']}"))
            if item.get("show_images", True) is False:
                continue
            for image in item.get("images", [])[:4]:
                try:
                    chain.append(Image.fromURL(image["url"]))
                except Exception:
                    continue
        return chain

    @filter.command("loraon", desc="使用指定简称开启 LoRA")
    async def command_loraon(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        if not value:
            yield event.plain_result("用法：/loraon 指令简称；例如：/loraon 1号lora")
            return
        try:
            available = await self._client().models("loras")
        except ComfyError as exc:
            yield event.plain_result(f"查询失败：{exc}")
            return
        items: list[str] = []
        missing: list[str] = []
        for token in (x.strip() for x in value.split(",") if x.strip()):
            name, separator, possible_weight = token.rpartition(":")
            if not separator or not re.fullmatch(r"\d+(?:\.\d+)?", possible_weight):
                name, possible_weight = token, "0.8"
            actual = self._resolve_lora(name, available)
            if not actual:
                missing.append(name)
                continue
            items.append(f"{actual}:{possible_weight}")
        if missing:
            yield event.plain_result("LoRA 不存在或指令简称未设置：" + "、".join(sorted(missing)))
            return
        enabled = list(self._get("lora_list", []) or [])
        merged = {x.rsplit(":", 1)[0]: x for x in enabled}
        merged.update({x.rsplit(":", 1)[0]: x for x in items})
        self._set("lora_list", list(merged.values()))
        self._save_config()
        names = "、".join(self._lora_command_alias(x.rsplit(":", 1)[0], available) for x in items)
        yield event.plain_result(f"已开启 LoRA：{names}")

    @filter.command("lora", alias=[".lora"], desc="查看和管理多个 LoRA")
    async def command_lora(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        client = self._client()
        overview_requested = not value or value == "列表" or value.casefold() in {"c站", "civitai", "图片", "图"}
        if overview_requested:
            try:
                details = await self._lora_details()
                if not details:
                    yield event.plain_result("当前没有检测到 LoRA 文件")
                    return
                image_path = await self._lora_list_image_path(details, list(self._get("lora_list", []) or []))
                yield event.chain_result([Image.fromFileSystem(str(image_path))])
            except (ComfyError, OSError, ValueError, TypeError, KeyError) as exc:
                logger.exception("[%s] 生成 LoRA 列表图片失败", PLUGIN_NAME)
                yield event.plain_result(f"LoRA 列表图片生成失败：{exc}")
            return
        try:
            available = await client.models("loras")
        except ComfyError as exc:
            yield event.plain_result(f"查询失败：{exc}")
            return
        enabled = list(self._get("lora_list", []) or [])
        sub, _, rest = value.partition(" ")
        target = self._resolve_lora(value, available)
        if target:
            try:
                details = await self._lora_details()
            except ComfyError as exc:
                yield event.plain_result(f"查询失败：{exc}")
                return
            selected = [item for item in details if item["file_name"] == target]
            yield event.chain_result(self._civitai_chain(selected, f"LoRA「{self._lora_command_alias(target, available)}」的 CivitAI 图片与链接："))
            return
        if sub in {"昵称", "别名", "alias"} and "=" in rest:
            alias, file_value = (part.strip() for part in rest.split("=", 1))
            actual = self._resolve_lora(file_value, available)
            if not actual or not alias:
                yield event.plain_result("用法：/lora 昵称 中文昵称=LoRA文件名")
                return
            self.lora_aliases[actual] = alias[:80]
            self._save_lora_aliases()
            yield event.plain_result(f"LoRA 昵称已设置：{alias}（{actual}）")
            return
        def resolve_items(raw: str) -> tuple[list[str], list[str]]:
            resolved: list[str] = []
            missing: list[str] = []
            for token in (x.strip() for x in raw.split(",") if x.strip()):
                name, separator, possible_weight = token.rpartition(":")
                if not separator or not re.fullmatch(r"\d+(?:\.\d+)?", possible_weight):
                    name, possible_weight = token, "0.8"
                actual = self._resolve_lora(name, available)
                if not actual:
                    missing.append(name)
                    continue
                resolved.append(f"{actual}:{possible_weight}")
            return resolved, missing

        if sub in {"启用", "添加", "增加", "enable", "add"}:
            items, missing = resolve_items(rest)
            if missing:
                yield event.plain_result("LoRA 不存在或昵称未设置：" + "、".join(sorted(missing)))
                return
            merged = {x.rsplit(":", 1)[0]: x for x in enabled}
            merged.update({x.rsplit(":", 1)[0]: x for x in items})
            self._set("lora_list", list(merged.values()))
            self._save_config()
            yield event.plain_result("已增加 LoRA：" + "、".join(self._lora_alias(x.rsplit(":", 1)[0]) for x in items))
            return
        if sub in {"停用", "删除", "减少", "disable", "remove", "del"}:
            targets, missing = resolve_items(rest)
            if missing:
                yield event.plain_result("LoRA 不存在或昵称未设置：" + "、".join(sorted(missing)))
                return
            target_names = {x.rsplit(":", 1)[0] for x in targets}
            kept = [x for x in enabled if x.rsplit(":", 1)[0] not in target_names]
            self._set("lora_list", kept)
            self._save_config()
            yield event.plain_result("已减少 LoRA：" + "、".join(self._lora_alias(x) for x in target_names))
            return
        if sub in {"清空", "clear"}:
            self._set("lora_list", [])
            self._save_config()
            yield event.plain_result("已清空全部 LoRA")
            return
        yield event.plain_result("用法：/lora 或 /.lora（查看全部图片）、/lora 指令简称（查看单个 C站图片）、/lora 添加、删除、昵称、清空；开启使用 /loraon 指令简称")

    @filter.command("预设", alias=["preset"], desc="管理提示词预设")
    async def command_preset(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        requested_category = ""
        if value.startswith("列表"):
            requested_category = value[2:].strip()
        elif value in self._lora_categories_list():
            requested_category = value
        if not value or value == "列表" or requested_category:
            if requested_category and requested_category not in self._lora_categories_list():
                yield event.plain_result(
                    "预设分类不存在：" + requested_category + "；可用分类：" + "、".join(self._lora_categories_list())
                )
                return
            try:
                image_path = await self._preset_list_image_path(requested_category)
                yield event.chain_result([Image.fromFileSystem(str(image_path))])
            except Exception as exc:
                logger.exception("[%s] 生成预设列表图片失败", PLUGIN_NAME)
                lines = [f"提示词预设（{requested_category or '全部分类'}）："]
                available = await self._local_lora_names()
                for file_name in self._order_loras(available or self._known_lora_preset_keys()):
                    if requested_category and self._lora_category(file_name) != requested_category:
                        continue
                    for entry in self._lora_preset_entries(file_name):
                        lines.append(f"{entry.get('tag') or '未命名简称'} = {entry.get('content') or '（空内容）'}")
                lines.extend(
                    f"{name} = {item.get('translated') or item.get('content', '')}"
                    for name, item in self.presets.items.items()
                    if isinstance(item, dict) and not requested_category
                )
                yield event.plain_result(
                    "\n".join(lines) if len(lines) > 1 else f"当前没有预设（图片生成失败：{exc}）"
                )
            return
        sub, _, rest = value.partition(" ")
        if sub in {"添加", "修改", "编辑", "add", "update", "edit"} and "=" in rest:
            name, content = rest.split("=", 1)
            try:
                old_name = name.strip()
                self.presets.add(name, content)
                self._save_linked_preset_change(
                    old_name=old_name,
                    name=name,
                    content=content,
                )
            except UsageError as exc:
                yield event.plain_result(str(exc))
                return
            verb = "修改" if sub in {"修改", "编辑", "update", "edit"} else "添加"
            yield event.plain_result(f"预设已{verb}：{name.strip()} = {content.strip()}")
            return
        if sub in {"删除", "del"}:
            name = rest.strip()
            removed = self.presets.remove(name)
            if removed:
                self._remove_lora_global_link(name)
                self._save_lora_presets()
            yield event.plain_result("预设已删除" if removed else "预设不存在")
            return
        if sub in {"翻译", "translate"}:
            name = rest.strip()
            item = self.presets.items.get(name)
            if not item:
                yield event.plain_result("预设不存在")
                return
            try:
                translated = await self._translate_prompt(
                    event,
                    item.get("content", ""),
                    force_enabled=True,
                    source_override="plugin",
                )
            except AIError as exc:
                yield event.plain_result(f"AI 翻译失败：{exc}")
                return
            item["translated"] = translated
            self.presets.save()
            self._save_linked_preset_change(
                name=name,
                content=translated,
            )
            yield event.plain_result(f"预设已翻译并保留：{name} = {translated}")
            return
        yield event.plain_result("用法：/预设 列表、/预设 列表 分类名、添加 名称=内容、修改 名称=内容、翻译 名称、删除 名称")

    @filter.command("画师串", alias=["artist_preset"], desc="管理独立画师串预设")
    async def command_artist_preset(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        active = str(self._get("artist_preset", DEFAULT_ARTIST_PRESET_NAME) or "")
        if not value or value in {"列表", "list"}:
            lines = [f"画师串预设（当前：{active or '未启用'}）："]
            for name, item in self.artist_presets.items.items():
                marker = " ✅" if name == active else ""
                lines.append(f"{name}{marker} = {item.get('content', '')}")
            yield event.plain_result("\n".join(lines) if len(lines) > 1 else "当前没有画师串预设")
            return
        sub, _, rest = value.partition(" ")
        if sub in {"添加", "修改", "编辑", "add", "update", "edit"} and "=" in rest:
            name, content = rest.split("=", 1)
            try:
                self.artist_presets.add(name, content)
            except UsageError as exc:
                yield event.plain_result(str(exc))
                return
            self._set("artist_preset", name.strip())
            self._save_config()
            yield event.plain_result(f"画师串预设已保存并启用：{name.strip()}")
            return
        if sub in {"使用", "启用", "切换", "use", "activate"}:
            name = rest.strip()
            if name.casefold() in {"", "无", "关闭", "禁用", "none", "off"}:
                self._set("artist_preset", "")
                self._save_config()
                yield event.plain_result("已关闭画师串预设")
                return
            if name not in self.artist_presets.items:
                yield event.plain_result(f"画师串预设不存在：{name}")
                return
            self._set("artist_preset", name)
            self._save_config()
            yield event.plain_result(f"已启用画师串预设：{name}")
            return
        if sub in {"删除", "del", "delete"}:
            name = rest.strip()
            removed = self.artist_presets.remove(name)
            if removed and active == name:
                self._set("artist_preset", "")
                self._save_config()
            yield event.plain_result("画师串预设已删除" if removed else "画师串预设不存在")
            return
        yield event.plain_result("用法：/画师串 列表、添加 名称=tags、使用 名称、关闭、删除 名称")

    @filter.command("工作流", alias=["workflow"], desc="查看或切换三种工作流")
    async def command_workflow(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        directory = self._workflow_dir()
        files = sorted(path.name for path in directory.glob("*.json"))
        if not value or value == "列表":
            lines = [f"工作流目录：{directory}"]
            for mode, label in MODE_NAMES.items():
                lines.append(f"{label}：{self._workflow_path(mode).name}")
            lines.append("可切换文件：" + ("、".join(files) if files else "无"))
            yield event.plain_result("\n".join(lines))
            return
        parts = value.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in MODE_ALIASES:
            yield event.plain_result("用法：/工作流 文生图 文件名、/工作流 图生图 文件名、/工作流 高清放大 文件名")
            return
        mode = MODE_ALIASES[parts[0]]
        filename = parts[1].strip()
        if not filename.endswith(".json"):
            filename += ".json"
        if not (directory / filename).is_file():
            yield event.plain_result(f"工作流不存在：{filename}")
            return
        self._set(f"workflow_{mode}", filename)
        self._save_config()
        yield event.plain_result(f"{MODE_NAMES[mode]}工作流已切换为：{filename}")

    @filter.command("画图配置", alias=["config"], desc="查询模型和工作流位置")
    async def command_config(self, event: AstrMessageEvent):
        root = detect_comfyui_root(str(self._get("comfyui_root", "")))
        source_workflow = str(self._get("source_workflow", ""))
        extra_paths_file, _ = self._extra_model_paths_sections()
        lines = [
            "当前绘画配置",
            f"ComfyUI 地址：{self._get('comfyui_url', '')}",
            f"核心模型：{self._get('model_name', '')}",
            f"LoRA：{', '.join(self._get('lora_list', []) or []) or '无'}",
            f"AI 服务：{self._get('ai_model', '') or '未配置'}",
            f"ComfyUI 根目录：{root or '未检测到'}",
        ]
        # 每类目录可能同时存在默认位置和额外路径，这里全部列出并标注来源。
        lines += self._describe_model_dirs("diffusion_models", "核心模型目录")
        lines += self._describe_model_dirs("checkpoints", "Checkpoint 模型目录")
        lines += self._describe_model_dirs("loras", "LoRA 目录")
        lines += [
            f"额外模型路径配置：{extra_paths_file or '未使用'}",
            f"工作流目录：{self._workflow_dir()}",
            f"原始工作流：{source_workflow}",
            f"输出目录：{self.output_dir}",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("comfy状态", alias=["comfy"], desc="查询 ComfyUI 状态")
    async def command_status(self, event: AstrMessageEvent):
        try:
            data = await self._client().status()
        except ComfyError as exc:
            yield event.plain_result(f"ComfyUI 未连接：{exc}\n若服务已关闭，请让插件管理员发送 /comfy打开 启动。")
            return
        yield event.plain_result(f"ComfyUI 已连接，版本：{data.get('system', {}).get('comfyui_version', '未知')}")

    @filter.command("comfy打开", alias=["comfy启动"], desc="启动本机 ComfyUI（仅插件管理员）")
    async def command_comfy_start(self, event: AstrMessageEvent):
        if not self._is_draw_limit_admin(event):
            yield event.plain_result("无权限：请在 WebUI 的绘图限额中填写自己的平台用户 ID 后再使用此指令。")
            return
        try:
            await self._client().status()
            yield event.plain_result("ComfyUI 已在运行，无需重复启动。")
            return
        except ComfyError:
            pass
        port = self._comfyui_port()
        if not port:
            yield event.plain_result("当前 ComfyUI 地址不是本机地址，插件不会远程启动服务。")
            return
        script = self._comfyui_start_script()
        if script is None:
            yield event.plain_result("未找到 ComfyUI 启动脚本。请在 WebUI 填写 ComfyUI 根目录或启动脚本路径后重试。")
            return
        try:
            await asyncio.to_thread(
                subprocess.Popen,
                ["cmd.exe", "/d", "/c", str(script)],
                cwd=str(script.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as exc:
            yield event.plain_result(f"启动 ComfyUI 失败：{exc}")
            return
        yield event.plain_result(f"已启动 ComfyUI：{script.name}。服务加载需要一些时间，稍后可用 /comfy状态 查询。")

    @filter.command("comfy关闭", alias=["comfy停止"], desc="关闭本机 ComfyUI（仅插件管理员）")
    async def command_comfy_stop(self, event: AstrMessageEvent):
        if not self._is_draw_limit_admin(event):
            yield event.plain_result("无权限：只有 WebUI 绘图限额中配置的管理员可以关闭 ComfyUI。")
            return
        port = self._comfyui_port()
        if not port:
            yield event.plain_result("当前 ComfyUI 地址不是本机地址，插件不会远程关闭服务。")
            return
        pids = await asyncio.to_thread(self._listening_pids_on_port, port)
        if not pids:
            yield event.plain_result("ComfyUI 当前未运行。需要启动时可发送 /comfy打开。")
            return
        stopped = await asyncio.to_thread(self._stop_pids, pids)
        if stopped:
            yield event.plain_result("ComfyUI 已关闭。需要继续绘图时，请让管理员发送 /comfy打开。")
            return
        yield event.plain_result("没有成功关闭 ComfyUI。请检查服务是否由其它权限账户启动。")

    @staticmethod
    def _normalize_llm_arguments(arguments: Any) -> dict[str, Any]:
        """兼容 AstrBot/中转 API 把工具参数包在 ``arguments`` 中的格式。

        标准工具调用通常直接传 ``{"prompt": "..."}``，但部分 AstrBot
        版本或 OpenAI 兼容中转会传成 ``{"arguments": {"prompt": "..."}}``，
        也有实现会把内层对象编码成 JSON 字符串。插件内部统一展开，避免
        这个协议差异直接暴露成 Python 的 unexpected keyword argument。
        """
        value: Any = arguments
        for _ in range(4):
            if value is None:
                return {}
            if isinstance(value, str):
                raw = value.strip()
                if not raw:
                    return {}
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    # 某些兼容接口把单个字符串直接放进 arguments；将其
                    # 当作 prompt 继续走原有参数构造，避免无意义的格式错误。
                    return {"prompt": raw}
                continue
            if isinstance(value, Mapping):
                payload = dict(value)
                if "arguments" in payload:
                    nested = payload.pop("arguments")
                    nested_payload = ComfyUIAIStudio._normalize_llm_arguments(nested)
                    payload.update(nested_payload)
                return {str(key): item for key, item in payload.items()}
            return {}
        return {}

    @staticmethod
    def _llm_argument_value(
        current: Any,
        arguments: Mapping[str, Any],
        name: str,
        default: Any = None,
    ) -> Any:
        """读取工具参数；显式旧版参数优先，空值才使用 ``arguments``。"""
        if current is not None and not (
            isinstance(current, str) and not current.strip()
        ):
            return current
        return arguments.get(name, default)

    @staticmethod
    def _llm_event_prompt_fallback(event: AstrMessageEvent) -> str:
        """工具参数缺失时读取本条用户消息，同时过滤无描述的裸命令。"""
        text = str(getattr(event, "message_str", "") or "").strip()
        if not text:
            return ""
        if re.fullmatch(
            r"[/／]?\s*(?:文生图|画图|生图|txt2img)\s*",
            text,
            flags=re.IGNORECASE,
        ):
            return ""
        return text

    def _llm_params(
        self,
        prompt: str,
        mode: str,
        *,
        negative_prompt: str | None = None,
        model: str | None = None,
        lora: str | None = None,
        preset: str | None = None,
        artist_preset: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        denoise: float | None = None,
        scale: float | None = None,
        ai: bool | None = None,
    ) -> DrawParams:
        """Convert structured LLM arguments to the same options as chat commands."""
        raw = str(prompt or "").strip()

        def normalize_integer(value: Any, name: str) -> int:
            """兼容 LLM 把整数参数编码成 832.0 的情况。"""
            if isinstance(value, bool):
                raise UsageError(f"{name} 必须是整数")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise UsageError(f"{name} 必须是整数：{value}") from exc
            if not number.is_integer():
                raise UsageError(f"{name} 必须是整数：{value}")
            return int(number)

        def add_option(name: str, value: Any) -> None:
            nonlocal raw
            if value is None or value == "":
                return
            if name in {"宽", "高", "步数", "种子"}:
                value = normalize_integer(value, name)
            clean = str(value).replace('"', "'")
            raw += f' --{name}="{clean}"'

        add_option("负面", negative_prompt)
        add_option("模型", model)
        add_option("lora", lora)
        add_option("预设", preset)
        add_option("画师", artist_preset)
        add_option("宽", width)
        add_option("高", height)
        add_option("步数", steps)
        add_option("cfg", cfg)
        add_option("种子", seed)
        add_option("强度", denoise)
        add_option("放大", scale)
        # ``ai`` 只为兼容旧版工具 schema 保留。结构化 LLM 的提示词来源
        # 由 WebUI 的 LLM 提示词来源设置决定，不能让模型传入的 ai=true
        # 把任务重新路由到普通 /文生图 ai 翻译链。
        params = fill_params(raw, mode)
        # 默认直接使用 AstrBot LLM 传入的 prompt、preset 和 lora；如果 WebUI
        # 选择插件 AI，_llm_execute 会在后台绘图前用用户原话重新生成一次 prompt。
        # 这里先关闭旧的 ai 字段，避免同一条任务被普通指令翻译逻辑二次处理。
        params.llm_invocation = True
        params.ai = False
        params.auto_ai = False
        params.ai_source = ""
        return params

    def _llm_plugin_ai_enabled(self) -> bool:
        value = self._get("llm_prompt_source", "astrbot")
        return str(value or "astrbot").strip().lower() in {
            "plugin",
            "plugin_ai",
            "插件",
            "插件ai",
        }

    def _img2img_llm_plugin_ai_enabled(self) -> bool:
        value = self._get("img2img_llm_prompt_source", "plugin")
        return str(value or "plugin").strip().lower() in {
            "plugin", "plugin_ai", "插件", "插件ai",
        }

    def _img2img_flux2_llm_plugin_ai_enabled(self) -> bool:
        value = self._get("img2img_flux2_llm_prompt_source", "plugin")
        return str(value or "plugin").strip().lower() in {
            "plugin", "plugin_ai", "插件", "插件ai",
        }

    def _plugin_ai_debug_enabled(self) -> bool:
        value = self._get("plugin_ai_debug", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是", "开启"}
        return bool(value)

    def _img2img_plugin_ai_debug_enabled(self) -> bool:
        value = self._get("img2img_plugin_ai_debug", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是", "开启"}
        return bool(value)

    def _img2img_flux2_plugin_ai_debug_enabled(self) -> bool:
        value = self._get("img2img_flux2_plugin_ai_debug", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是", "开启"}
        return bool(value)

    def _remove_locked_control_terms(self, text: str, params: DrawParams) -> str:
        """从交给 AI 的原话中移除已锁定的预设名和 LoRA 简称。

        这些词已经由插件转换成确定的控制项。保留它们给 AI 会让 AI
        再次查 Danbooru 并生成另一个角色 tag，造成预设和角色词条叠加。
        未命中的角色不会进入这里，因此仍可正常交给 Danbooru 处理。
        """
        result = str(text or "")
        terms: list[str] = []
        preset_store = getattr(self, "presets", None)
        for name in getattr(params, "presets", []) or []:
            name = str(name or "").strip()
            if not name:
                continue
            terms.append(name)
            if preset_store is not None:
                try:
                    terms.extend(self._preset_aliases(name))
                except Exception:
                    effective = str(preset_store.effective(name) or "").strip()
                    if effective:
                        terms.append(effective)
        if hasattr(self, "lora_command_aliases"):
            for item in getattr(params, "loras", []) or []:
                actual = str(item or "").rsplit(":", 1)[0].strip()
                terms.extend(self._lora_command_aliases(actual))
        seen: set[str] = set()
        for term in sorted(terms, key=len, reverse=True):
            term = str(term or "").strip()
            key = term.casefold()
            if not term or key in seen:
                continue
            seen.add(key)
            if re.search(r"[\u4e00-\u9fff]", term):
                result = re.sub(re.escape(term), " ", result, flags=re.IGNORECASE)
            else:
                normalized = term.replace(r"\(", "(").replace(r"\)", ")")
                result = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
                    " ",
                    result,
                    flags=re.IGNORECASE,
                )
        return re.sub(r"\s+", " ", result).strip(" ,，。；;")

    @staticmethod
    def _llm_prompt_input(original_text: str, extracted_prompt: str) -> str:
        """给插件 AI 同时提供用户原话和当前 LLM 结果，避免动作在中间步骤丢失。"""
        original = str(original_text or "").strip()
        extracted = str(extracted_prompt or "").strip()
        if original and extracted and original.casefold() != extracted.casefold():
            return (
                "用户原话：\n"
                f"{original}\n\n"
                "AstrBot LLM 已提取的画面描述：\n"
                f"{extracted}"
            )
        return original or extracted

    @staticmethod
    def _img2img_user_prompt_input(original_text: str, extracted_prompt: str) -> str:
        """只取图生图的用户编辑要求，不把完整画面摘要交给 Qwen。

        AstrBot 的工具参数有时包含一段适合文生图的画面描述。Qwen
        Image Edit 已经从参考图得到人物和场景，再把这段描述送进去会
        诱发整图重绘。图生图只使用用户原话；只有事件没有原话时才用
        工具传入的 prompt 作为兜底。
        """
        value = str(original_text or "").strip() or str(extracted_prompt or "").strip()
        if not value:
            return ""
        # 兼容上游已经拼出的“用户原话/画面描述”格式，只保留用户原话。
        match = re.search(
            r"(?:user(?:'s)?\s+original\s+words|用户原话)\s*[:：]\s*(.*?)(?=\s+(?:astrbot\s+llm\s+picture\s+description|astrbot\s+llm\s+画面描述|astrbot\s+llm\s+description)\s*[:：]|$)",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            value = match.group(1).strip()
        else:
            value = re.split(
                r"\s+(?:astrbot\s+llm\s+picture\s+description|astrbot\s+llm\s+画面描述|astrbot\s+llm\s+description)\s*[:：]",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
        value = re.sub(r"[,，、;；]\s*(?:other\s+)?unchanged\s*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+(?:其余|其它|其他)不变\s*$", "", value)
        return value.strip(" `\t\r\n,，。；;")

    async def _prepare_llm_prompt(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        original_text: str,
        mode: str = "txt2img",
    ) -> str:
        """按模式选择对应的 LLM 提示词整理器。"""
        if mode in {"img2img", "img2img_flux2"}:
            # 图生图只需告诉 Qwen Image Edit 要改什么。预设和 LoRA 已在外层
            # 锁定，不能交给 AI 改写或删除。
            source = (
                "plugin_img2img_flux2_llm"
                if mode == "img2img_flux2" and self._img2img_flux2_llm_plugin_ai_enabled()
                else "astrbot_img2img_flux2_llm"
                if mode == "img2img_flux2"
                else "plugin_img2img_llm"
                if self._img2img_llm_plugin_ai_enabled()
                else "astrbot_img2img_llm"
            )
            # Qwen 已经从 image1 读取人物和场景。把 AstrBot 的完整文生图
            # 画面描述再次塞入编辑提示词，会让模型把原图重新解释，造成
            # 成品与原图差距过大。这里只传用户真正说的编辑要求。
            source_text = self._img2img_user_prompt_input(original_text, params.prompt)
            source_text = self._remove_locked_control_terms(source_text, params)
            source_text = self._strip_style_lora_disable_terms(source_text)
            if not source_text:
                source_text = self._remove_locked_control_terms(params.prompt, params)
                source_text = self._strip_style_lora_disable_terms(source_text)
            if not source_text:
                params.img2img_edit_instruction_ready = True
                params.img2img_edit_instruction = ""
                return ""
            params.plugin_ai_debug_input = source_text
            try:
                translated = await self._translate_prompt(
                    event,
                    source_text,
                    force_enabled=True,
                    source_override=source,
                )
            except AIError as exc:
                # AI 连接异常或误返回推理文本时，只翻译用户的原始编辑要求。
                # 不回退到 AstrBot 的长画面描述，避免图生图变成文生图。
                fallback, plain_note = await self._plain_translate_prompt(
                    source_text,
                    mode="img2img_flux2" if mode == "img2img_flux2" else "img2img",
                )
                fallback = self._repair_img2img_edit_instruction(
                    fallback,
                    source_text,
                )
                params.prompt = fallback
                params.img2img_edit_instruction = fallback
                params.img2img_edit_instruction_ready = True
                params.plugin_ai_fallback_note = (
                    f"图生图插件 AI 未返回有效编辑指令，已按用户原话继续（{exc}）"
                    + (f"；{plain_note}" if plain_note else "")
                )
                logger.warning("[%s] 图生图 LLM 插件 AI 失败，已回退原编辑要求：%s", PLUGIN_NAME, exc)
                return fallback
            translated = self._repair_img2img_edit_instruction(translated, source_text)
            params.prompt = translated
            params.img2img_edit_instruction = translated
            params.plugin_ai_debug_prompt = translated
            params.img2img_edit_instruction_ready = True
            logger.info(
                "[%s] 图生图 LLM 编辑指令已生成：来源=%s；指令=%s",
                PLUGIN_NAME,
                "插件 AI" if source in {"plugin_img2img_llm", "plugin_img2img_flux2_llm"} else "AstrBot AI",
                translated,
            )
            return translated

        """文生图按 WebUI 选择决定是否由插件 AI 生成 LLM 提示词。"""
        if not self._llm_plugin_ai_enabled():
            return ""
        source_text = self._llm_prompt_input(original_text, params.prompt)
        source_text = self._remove_locked_control_terms(source_text, params)
        source_text = self._strip_style_lora_disable_terms(source_text)
        if not source_text:
            return ""
        # 仅供开发者调试回显，不参与绘图参数序列化。
        params.plugin_ai_debug_input = source_text
        try:
            translated = await self._translate_prompt(
                event,
                source_text,
                force_enabled=True,
                source_override="plugin_llm",
            )
        except AIError as exc:
            # 插件 AI 只负责优化提示词，不能让 LLM 已经识别出的绘图任务失败。
            # 回退到 LLM 提取的描述，后续仍会执行预设、LoRA 和普通翻译。
            fallback = self._strip_style_lora_disable_terms(
                str(params.prompt or original_text or "").strip()
            )
            params.prompt = fallback
            params.ai = False
            params.auto_ai = False
            params.ai_source = ""
            params.plugin_ai_fallback_note = f"插件 AI 不可用，已按普通提示词继续出图（{exc}）"
            logger.warning("[%s] LLM 插件 AI 失败，回退普通提示词：%s", PLUGIN_NAME, exc)
            return ""
        params.prompt = translated
        params.plugin_ai_debug_prompt = translated
        # 已经在工具执行前完成插件 AI 转换，后台 _generate 不再二次改写，
        # 这样动作、预设和 LoRA 控制项可以由同一条链路稳定合并。
        params.ai = False
        params.auto_ai = False
        params.ai_source = ""
        logger.info("[%s] LLM 插件 AI 提示词：%s", PLUGIN_NAME, translated)
        return translated

    def _llm_debug_reply(self, reply: str, prompt: str, source_text: str = "") -> str:
        if not prompt:
            return reply
        details = f"[开发者调试] 插件 AI 输入：\n{source_text or '无'}\n\n插件 AI 最终提示词：\n{prompt}"
        return f"{reply}\n\n{details}"

    async def _llm_execute(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        *,
        require_image: bool,
    ) -> str | None:
        # 由工具入口再次设置，兼容外部调用方直接构造 DrawParams 的情况。
        # 该标记必须在任何后台生成逻辑之前写入，确保不会触发第二次文生图 AI。
        params.llm_invocation = True
        logger.info(
            "[%s] LLM 绘图工具进入：模式=%s；必须输入图片=%s；提示词=%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            require_image,
            str(params.prompt or "")[:160],
        )
        try:
            # 先锁定预设，再解析 LoRA。这样 LLM 把“维里奈”改成
            # verina_(wuthering_waves) 后，仍会根据同名 LoRA 简称临时加载 LoRA。
            # 用户原话中的控制项依旧具有最高优先级。
            original_text = str(getattr(event, "message_str", "") or "").strip() or str(params.prompt or "").strip()
            if mode == "txt2img":
                params.prompt = self._sanitize_llm_txt2img_prompt(
                    params.prompt,
                    original_text,
                )
            if mode == "img2img_flux2":
                # Flux2 图生图的模型、LoRA 和提示词配置全部独立；不把
                # 文生图/Qwen 的预设或简称带入新工作流。
                params.presets = []
                params.loras = []
            else:
                self._extract_inline_presets(params, original_text)
                await self._extract_inline_loras(
                    params,
                    original_text,
                    allow_fuzzy=True,
                )
        except ComfyError as exc:
            return f"{MODE_NAMES[mode]}失败：无法读取 LoRA 列表：{exc}"
        if mode == "txt2img" and not params.prompt and not params.presets and not params.loras:
            return "文生图失败：请提供画面描述、提示词预设或 LoRA 指令简称。"

        image_path = ""
        second_image_path = ""
        third_image_path = ""
        if require_image:
            image_paths = await self._extract_images(event)
            if mode == "img2img_flux2" and len(image_paths) > 3:
                logger.info(
                    "[%s] LLM Flux2 图生图收到 %s 张图片，只使用前三张",
                    PLUGIN_NAME,
                    len(image_paths),
                )
                image_paths = image_paths[:3]
            image_path = image_paths[0] if image_paths else ""
            second_image_path = image_paths[1] if len(image_paths) > 1 else ""
            third_image_path = image_paths[2] if len(image_paths) > 2 else ""
            has_explicit_reference = self._event_has_explicit_image_reference(event)
            if not image_path:
                if has_explicit_reference:
                    logger.warning(
                        "[%s] LLM 图生图未读取到用户明确引用的图片，已拒绝使用最近出图替代",
                        PLUGIN_NAME,
                    )
                    return f"{MODE_NAMES[mode]}未能读取你引用的图片/合并转发，请重新回复或@那条转发消息后再试。"
                logger.warning("[%s] LLM 图生图未找到当前消息中的输入图片", PLUGIN_NAME)
                return f"{MODE_NAMES[mode]}需要在同一条消息附图，或回复一张图片（也可以回复包含图片的合并转发）后再试。"
            logger.info(
                "[%s] LLM 图生图已取得输入图片：参考图=%s",
                PLUGIN_NAME,
                image_path,
            )

        # LLM 工具路径的输入端审核在这里提前完成：命中时不发送“正在生成”
        # 开始提示，直接把警告发到群聊，并给 LLM 一个明确的拒绝结论。
        input_verdict = await self._moderation_verdict(
            event, "input", [image_path, second_image_path, third_image_path]
        )
        if input_verdict is not None:
            notice = self._moderation_notice(event, input_verdict, "input")
            try:
                await event.send(event.chain_result(notice))
                clear_result = getattr(event, "clear_result", None)
                if callable(clear_result):
                    # 防止工具执行器把本轮事件结果再发一条空消息。
                    clear_result()
            except Exception as exc:
                logger.warning("[%s] LLM 绘图审核警告发送失败：%s", PLUGIN_NAME, exc)
            return "图片安全审核未通过，已拦截本次请求并向群聊发送警告；不要用同一张图片重试。"

        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            return f"{MODE_NAMES[mode]}失败：{exc}"

        # 预设和 LoRA 是插件控制项，优先级高于任何 AI 改写。先保存已识别的
        # 结果，防止插件 AI 或翻译模型把它们从 prompt 中改掉。
        locked_presets = list(dict.fromkeys(params.presets))
        locked_loras = list(dict.fromkeys(params.loras))
        locked_lora_preset_tags = copy.deepcopy(params.lora_preset_tags)
        try:
            await self._prepare_llm_prompt(event, params, original_text, mode)
        except UsageError as exc:
            await self._release_draw_limit(params)
            return f"{MODE_NAMES[mode]}失败：{exc}"
        # _prepare_llm_prompt 只负责整理画面描述，绝不能改变已经识别的
        # 控制项。所有普通绘图模式都恢复预设、LoRA 和命中的专属预设；
        # Flux2 独立图生图在上面已经清空，因此不会串入文生图配置。
        if mode != "img2img_flux2":
            params.presets = locked_presets
            params.loras = locked_loras
            params.lora_preset_tags = locked_lora_preset_tags
        try:
            await self._check_draw_queue(params, mode, event)
        except UsageError as exc:
            await self._release_draw_limit(params)
            return f"{MODE_NAMES[mode]}失败：{exc}"
        logger.info(
            "[%s] LLM 绘图控制项已锁定：预设=%s；LoRA=%s；画面=%s",
            PLUGIN_NAME,
            ",".join(params.presets) or "无",
            ",".join(params.loras) or "无",
            params.prompt[:180],
        )
        result = await self._llm_draw(
            event,
            params,
            mode,
            image_path,
            second_image_path if mode == "img2img_flux2" else second_image_path,
            third_image_path if mode == "img2img_flux2" else "",
        )
        if result is not None:
            return result
        # 超时分支已经把“任务已提交”放进 event.result，由 AstrBot 的工具
        # 执行器直接发送；这里必须返回 None，不能再返回字符串让模型吞掉该结果。
        return None

    @staticmethod
    def _component_has_image(component: Any, depth: int = 0) -> bool:
        """同步判断消息链是否包含图片，供 LLM 工具纠正错误路由。"""
        if component is None or depth > 8:
            return False
        # Forward 只有消息 ID，实际图片需要异步调用 get_forward_msg；
        # 先把它当作图片提示，工具层再负责拉取并校验实际内容。
        if ComfyUIAIStudio._forward_component_id(component):
            return True
        if ComfyUIAIStudio._reply_component_id(component):
            return True
        if isinstance(component, Image):
            return True
        if isinstance(component, Mapping):
            component_type = str(component.get("type", "")).strip().lower()
            if component_type == "image":
                return True
            data = component.get("data")
            data = data if isinstance(data, Mapping) else component
            nested_values: list[Any] = []
            for key in ("content", "message", "messages", "nodes", "chain"):
                value = data.get(key)
                if isinstance(value, list):
                    nested_values.extend(value)
            return any(
                ComfyUIAIStudio._component_has_image(nested, depth + 1)
                for nested in nested_values
            )
        if isinstance(component, Reply):
            if any(
                ComfyUIAIStudio._component_has_image(nested, depth + 1)
                for nested in (component.chain or [])
            ):
                return True
            # 某些适配器只保留 Reply(id)，引用正文需要通过 get_msg
            # 异步拉取。先按“可能含图”处理，工具层会再确认并提取图片。
            reply_id = str(getattr(component, "id", "") or "").strip()
            return bool(reply_id and reply_id != "0")
        if isinstance(component, Node):
            return any(
                ComfyUIAIStudio._component_has_image(nested, depth + 1)
                for nested in (component.content or [])
            )
        if isinstance(component, Nodes):
            return any(
                ComfyUIAIStudio._component_has_image(nested, depth + 1)
                for node in (component.nodes or [])
                for nested in (node.content or [])
            )
        return False

    @classmethod
    def _event_has_image_hint(cls, event: AstrMessageEvent) -> bool:
        """兼容 OneBot 的 image/image_list 字段和 AstrBot 消息组件。"""
        if getattr(event, "image", None) or getattr(event, "image_list", None):
            return True
        try:
            components = cls._event_components(event)
            components.extend(cls._raw_event_components(event))
            components.extend(cls._event_level_reference_components(event))
            return any(cls._component_has_image(component) for component in components)
        except Exception:
            return False

    @staticmethod
    def _looks_like_edit_request(event: AstrMessageEvent, prompt: str) -> bool:
        text = " ".join(
            value
            for value in (
                str(getattr(event, "message_str", "") or ""),
                str(prompt or ""),
            )
            if value
        ).casefold()
        return any(
            marker in text
            for marker in (
                "图生图", "修改图片", "修改这张", "重绘", "重制", "改成", "换背景",
                "改变动作", "改变姿势", "参考这张", "基于这张", "保持原图",
                "edit image", "image to image", "img2img",
            )
        )

    @filter.llm_tool(name=LLM_TOOL_NAMES["status"])
    async def llm_status(
        self,
        event: AstrMessageEvent,
        arguments: Any = None,
    ) -> str:
        """查询本地 ComfyUI 是否在线以及当前工作流状态。用户询问绘图服务状态时调用。

        Args:
        """
        try:
            data = await self._client().status()
        except ComfyError as exc:
            return f"ComfyUI 未连接：{exc}"
        system = data.get("system", {})
        return f"ComfyUI 已连接，版本：{system.get('comfyui_version', '未知')}，地址：{self._get('comfyui_url', '')}"

    @filter.llm_tool(name=LLM_TOOL_NAMES["presets"])
    async def llm_presets(
        self,
        event: AstrMessageEvent,
        category: str = "",
        arguments: Any = None,
    ) -> str:
        """查询提示词预设，支持按 LoRA 分类筛选，不会触发绘图。

        Args:
            category(string): 可选 LoRA 分类。
        """
        llm_arguments = self._normalize_llm_arguments(arguments)
        category = self._llm_argument_value(category, llm_arguments, "category", "")
        category = str(category or "").strip()
        available = await self._local_lora_names()
        if category and category not in self._lora_categories_list():
            return "预设分类不存在。可用分类：" + "、".join(self._lora_categories_list())
        names = [
            name for name in self._order_loras(available or self._known_lora_preset_keys())
            if not category or self._lora_category(name) == category
        ]
        lines = [f"提示词预设（{category or '全部分类'}）："]
        count = 0
        for file_name in names:
            entries = self._lora_preset_entries(file_name)
            if not entries:
                continue
            lines.append(f"【{self._lora_alias(file_name)}｜{self._lora_category(file_name)}】")
            for entry in entries:
                tag = str(entry.get("tag", "") or "未命名简称").strip()
                content = str(entry.get("content", "") or "（空内容）").strip()
                lines.append(f"{tag} = {content}")
                count += 1
        if not category:
            global_entries = [
                (name, item) for name, item in self.presets.items.items()
                if isinstance(item, dict) and self.presets.effective(name)
            ]
            if global_entries:
                lines.append("【全局预设】")
                for name, item in global_entries:
                    lines.append(f"{name} = {self.presets.effective(name)}")
                    count += 1
        if count == 0:
            return f"当前分类“{category or '全部'}”没有可见预设。"
        return "\n".join(lines)

    @filter.llm_tool(name=LLM_TOOL_NAMES["generate"])
    async def llm_generate(
        self,
        event: AstrMessageEvent,
        prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        model: str | None = None,
        lora: str | None = None,
        preset: str | None = None,
        artist_preset: str | None = None,
        ai: bool | None = None,
        arguments: Any = None,
    ) -> str:
        """使用本插件基于原始工作流执行文生图。

        仅适用于没有输入图片的全新文生图。用户要求画图、生成图片、创作一张图时必须调用此工具，不要只用文字回复。
        如果用户附带图片、回复图片、转发图片，或要求修改/重绘/换背景/改变动作，禁止调用此工具，必须调用 comfyui_ai_studio_edit。
        用户用自然语言提出绘画要求时必须调用此工具。即使用户只说“帮我画一张夏空的图”，
        也要调用此工具，不要先单独回复查资料。把命中的提示词预设填入 preset，
        把命中的 LoRA 指令简称填入 lora；preset 和 lora 可以同时填写。其余画面描述填入 prompt，
        不要把“我来画一张”等聊天套话放入 prompt。prompt 必须是适配 ComfyUI Anima 工作流的英文 Danbooru 标签，
        按主体、角色、外观、服装、动作、表情、镜头、构图、场景的顺序组织，并完整保留用户要求的动作。
        任务提交后后台执行，用户可以继续聊天。

        Args:
            prompt(string): 画面描述，可以是用户自然语言；不要把预设名或 LoRA 简称遗漏在描述之外。
            width(number): 可选图片宽度；与 height 一起设置。
            height(number): 可选图片高度；与 width 一起设置。
            steps(number): 可选采样步数。
            cfg(number): 可选 CFG。
            seed(number): 可选随机种子；不传则随机生成。
            negative_prompt(string): 可选负面提示词；不传使用 WebUI 默认值。
            model(string): 可选核心模型文件名；不传使用 WebUI 当前模型。
            lora(string): 可选多个 LoRA，优先填写指令简称，格式为 1号lora:0.8,2号lora:0.6。
            preset(string): 可选一个或多个提示词预设名称，多个用逗号分隔，例如 夏空,画面预设。
            artist_preset(string): 可选画师串预设名称，不传使用 WebUI 当前画师串。
            ai(boolean): 为兼容旧工具字段而保留；是否使用插件 AI 由 WebUI 的 LLM 提示词来源决定。
        """
        llm_arguments = self._normalize_llm_arguments(arguments)
        prompt = self._llm_argument_value(prompt, llm_arguments, "prompt", "")
        width = self._llm_argument_value(width, llm_arguments, "width")
        height = self._llm_argument_value(height, llm_arguments, "height")
        steps = self._llm_argument_value(steps, llm_arguments, "steps")
        cfg = self._llm_argument_value(cfg, llm_arguments, "cfg")
        seed = self._llm_argument_value(seed, llm_arguments, "seed")
        negative_prompt = self._llm_argument_value(
            negative_prompt, llm_arguments, "negative_prompt"
        )
        model = self._llm_argument_value(model, llm_arguments, "model")
        lora = self._llm_argument_value(lora, llm_arguments, "lora")
        preset = self._llm_argument_value(preset, llm_arguments, "preset")
        artist_preset = self._llm_argument_value(
            artist_preset, llm_arguments, "artist_preset"
        )
        ai = self._llm_argument_value(ai, llm_arguments, "ai")
        if (
            not str(prompt or "").strip()
            and not str(preset or "").strip()
            and not str(lora or "").strip()
        ):
            prompt = self._llm_event_prompt_fallback(event)

        # LLM 偶尔会把带图的“修改图片”请求误选为 generate。工具层再做一次
        # 路由兜底，确保不会把图生图请求送进文生图工作流。
        if self._event_has_image_hint(event) or self._looks_like_edit_request(event, prompt):
            logger.warning(
                "[%s] LLM 错误调用文生图工具，已自动改走图生图：提示词=%s；消息含图=%s",
                PLUGIN_NAME,
                str(prompt or "")[:160],
                self._event_has_image_hint(event),
            )
            return await self.llm_edit(
                event,
                prompt=prompt,
                denoise=None,
                negative_prompt=negative_prompt,
                model=model,
                lora=lora,
                preset=preset,
                artist_preset=artist_preset,
                ai=ai,
            )
        if not str(prompt or "").strip() and not str(preset or "").strip() and not str(lora or "").strip():
            return "文生图失败：缺少画面描述、提示词预设或 LoRA 指令简称。"
        try:
            params = self._llm_params(
                prompt,
                "txt2img",
                negative_prompt=negative_prompt,
                model=model,
                lora=lora,
                preset=preset,
                artist_preset=artist_preset,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                ai=ai,
            )
        except UsageError as exc:
            return f"文生图参数错误：{exc}"
        return await self._llm_execute(event, params, "txt2img", require_image=False)

    @filter.llm_tool(name=LLM_TOOL_NAMES["edit"])
    async def llm_edit(
        self,
        event: AstrMessageEvent,
        prompt: str = "",
        denoise: float | None = None,
        negative_prompt: str | None = None,
        model: str | None = None,
        lora: str | None = None,
        preset: str | None = None,
        artist_preset: str | None = None,
        ai: bool | None = None,
        arguments: Any = None,
    ) -> str:
        """使用当前 WebUI 选择的图生图工作流基于参考图编辑。

        只要用户提供图片、回复图片、转发图片，或要求修改、重绘、换背景、改变动作，必须调用此工具，不要调用 generate，也不要只用文字回复。
        prompt 写用户明确要求编辑的内容，并尽量保留具体动作、姿势、表情和目标细节；不要写完整 Anima/Danbooru 绘图提示词。原图没有要求改变的角色、背景、镜头、姿势和画风都不应补写，但用户要求“换个姿势”时不要退化为 change her pose，应提供一个具体可执行的姿势方案。
        preset 和 lora 可以与 prompt 同时使用；它们是显式控制项，不能因提示词整理而丢失。任务后台执行，期间可以继续聊天。

        Args:
            prompt(string): 修改要求，可用中文或英文；应包含具体动作/姿势细节，例如“让她侧身站立，一只手抚摸头发，另一只手自然下垂，看向镜头”，不要只写 change her pose。
            denoise(number): 可选重绘幅度，范围为 0 到 1。
            negative_prompt(string): 可选负面提示词。
            model(string): 可选核心模型文件名。
            lora(string): 可选多个 LoRA 指令简称，格式为 1号lora:0.8,2号lora:0.6。
            preset(string): 可选一个或多个提示词预设名称，多个用逗号分隔。
            artist_preset(string): 可选画师串预设名称。
            ai(boolean): 为兼容旧工具字段而保留；是否使用插件 AI 由 WebUI 的 LLM 提示词来源决定。
        """
        llm_arguments = self._normalize_llm_arguments(arguments)
        prompt = self._llm_argument_value(prompt, llm_arguments, "prompt", "")
        denoise = self._llm_argument_value(denoise, llm_arguments, "denoise")
        negative_prompt = self._llm_argument_value(
            negative_prompt, llm_arguments, "negative_prompt"
        )
        model = self._llm_argument_value(model, llm_arguments, "model")
        lora = self._llm_argument_value(lora, llm_arguments, "lora")
        preset = self._llm_argument_value(preset, llm_arguments, "preset")
        artist_preset = self._llm_argument_value(
            artist_preset, llm_arguments, "artist_preset"
        )
        ai = self._llm_argument_value(ai, llm_arguments, "ai")
        if (
            not str(prompt or "").strip()
            and not str(preset or "").strip()
            and not str(lora or "").strip()
        ):
            prompt = self._llm_event_prompt_fallback(event)
        if not str(prompt or "").strip() and not str(preset or "").strip() and not str(lora or "").strip():
            return "图生图失败：缺少编辑要求、提示词预设或 LoRA 指令简称。"
        draw_mode = self._img2img_mode()
        try:
            params = self._llm_params(
                prompt,
                draw_mode,
                negative_prompt=negative_prompt,
                model=model,
                lora=lora,
                preset=preset,
                artist_preset=artist_preset,
                denoise=denoise,
                ai=ai,
            )
        except UsageError as exc:
            return f"图生图参数错误：{exc}"
        # AstrBot 的工具调用模型经常会在没有得到用户要求的情况下，
        # 自动填入 denoise=0.6。Qwen Image Edit 的默认完整编辑幅度是
        # 1.0；这个隐式值会让“换衣服/改动作”等 LLM 图生图退化成几乎
        # 原图直出，而同一工作流的 /图生图 指令却正常。只有用户明确在
        # 原消息中提出重绘强度时，才保留工具传入的强度。
        original_request = " ".join(
            str(value or "")
            for value in (
                getattr(event, "message_str", ""),
                prompt,
            )
        )
        if denoise is not None and not re.search(
            r"(?:denoise|重绘(?:幅度|强度)?|去噪(?:幅度|强度)?|强度)\s*[:：=]?\s*\d+(?:\.\d+)?",
            original_request,
            flags=re.IGNORECASE,
        ):
            params.denoise = 0.0
            logger.info(
                "[%s] LLM 图生图未检测到用户指定重绘强度，忽略工具自动填入的 denoise=%s，使用 WebUI 默认值",
                PLUGIN_NAME,
                denoise,
            )
        return await self._llm_execute(event, params, draw_mode, require_image=True)

    @filter.llm_tool(name=LLM_TOOL_NAMES["edit_flux2"])
    async def llm_edit_flux2(
        self,
        event: AstrMessageEvent,
        prompt: str = "",
        denoise: float | None = None,
        negative_prompt: str | None = None,
        model: str | None = None,
        ai: bool | None = None,
        arguments: Any = None,
    ) -> str:
        """强制使用独立 Flux2 Klein 工作流进行图生图，支持 1 到 3 张参考图。

        新引擎的模型、LoRA、提示词模板和参数均从 WebUI 的“图生图
        Flux2”页面读取；不会读取 Qwen 图生图或文生图的 LoRA/预设。
        第一张图片作为主参考图，第二、第三张图片会接入工作流的后续参考条件链。

        Args:
            prompt(string): 修改要求。
            denoise(number): 可选重绘幅度。
            negative_prompt(string): 可选负面提示词。
            model(string): 可选核心模型文件名。
            ai(boolean): 为兼容旧工具字段而保留。
        """
        llm_arguments = self._normalize_llm_arguments(arguments)
        prompt = self._llm_argument_value(prompt, llm_arguments, "prompt", "")
        denoise = self._llm_argument_value(denoise, llm_arguments, "denoise")
        negative_prompt = self._llm_argument_value(
            negative_prompt, llm_arguments, "negative_prompt"
        )
        model = self._llm_argument_value(model, llm_arguments, "model")
        ai = self._llm_argument_value(ai, llm_arguments, "ai")
        if not str(prompt or "").strip():
            prompt = self._llm_event_prompt_fallback(event)
        if not str(prompt or "").strip():
            return "图生图 Flux2 失败：请提供要修改的内容。"
        try:
            params = self._llm_params(
                prompt,
                "img2img_flux2",
                negative_prompt=negative_prompt,
                model=model,
                denoise=denoise,
                ai=ai,
            )
        except UsageError as exc:
            return f"图生图 Flux2 参数错误：{exc}"
        original_request = " ".join(
            str(value or "") for value in (getattr(event, "message_str", ""), prompt)
        )
        if denoise is not None and not re.search(
            r"(?:denoise|重绘(?:幅度|强度)?|去噪(?:幅度|强度)?|强度)\s*[:：=]?\s*\d+(?:\.\d+)?",
            original_request,
            flags=re.IGNORECASE,
        ):
            params.denoise = 0.0
        return await self._llm_execute(event, params, "img2img_flux2", require_image=True)

    @filter.llm_tool(name=LLM_TOOL_NAMES["upscale"])
    async def llm_upscale(
        self,
        event: AstrMessageEvent,
        prompt: str | None = None,
        scale: float | None = None,
        denoise: float | None = None,
        steps: int | None = None,
        negative_prompt: str | None = None,
        model: str | None = None,
        lora: str | None = None,
        preset: str | None = None,
        artist_preset: str | None = None,
        ai: bool | None = None,
        arguments: Any = None,
    ) -> str:
        """使用本插件基于原始工作流执行高清放大。

        用户要求放大或高清修复当前图片时必须调用此工具，不要只用文字回复。
        prompt、preset 和 lora 仍可填写，插件会和“高清放大”指令一样进入原始高清工作流。
        prompt 使用适配 ComfyUI Anima 的英文 Danbooru 标签；工具会优先使用当前消息图片、引用图片或最近出图，任务后台执行。

        Args:
            scale(number): 可选放大倍率，默认使用插件配置。
            denoise(number): 可选重绘幅度。
            steps(number): 可选高清放大步数。
            prompt(string): 可选放大时的画面描述。
            negative_prompt(string): 可选负面提示词。
            model(string): 可选核心模型文件名。
            lora(string): 可选多个 LoRA 指令简称。
            preset(string): 可选提示词预设名称，多个用逗号分隔。
            artist_preset(string): 可选画师串预设名称。
            ai(boolean): 为兼容旧工具字段而保留；是否使用插件 AI 由 WebUI 的 LLM 提示词来源决定。
        """
        llm_arguments = self._normalize_llm_arguments(arguments)
        prompt = self._llm_argument_value(prompt, llm_arguments, "prompt")
        scale = self._llm_argument_value(scale, llm_arguments, "scale")
        denoise = self._llm_argument_value(denoise, llm_arguments, "denoise")
        steps = self._llm_argument_value(steps, llm_arguments, "steps")
        negative_prompt = self._llm_argument_value(
            negative_prompt, llm_arguments, "negative_prompt"
        )
        model = self._llm_argument_value(model, llm_arguments, "model")
        lora = self._llm_argument_value(lora, llm_arguments, "lora")
        preset = self._llm_argument_value(preset, llm_arguments, "preset")
        artist_preset = self._llm_argument_value(
            artist_preset, llm_arguments, "artist_preset"
        )
        ai = self._llm_argument_value(ai, llm_arguments, "ai")
        try:
            params = self._llm_params(
                prompt or "",
                "hires",
                negative_prompt=negative_prompt,
                model=model,
                lora=lora,
                preset=preset,
                artist_preset=artist_preset,
                scale=scale,
                denoise=denoise,
                steps=steps,
                ai=ai,
            )
        except UsageError as exc:
            return f"高清放大参数错误：{exc}"
        return await self._llm_execute(event, params, "hires", require_image=True)

    def _safe_config(self) -> dict[str, Any]:
        data = {key: self._get(key, "") for key in WRITABLE_CONFIG}
        # WebUI 只需要知道是否已经配置密钥，绝不能把密钥原文下发到浏览器。
        data.pop("ai_api_key", None)
        data["ai_api_key_configured"] = bool(self._get("ai_api_key", ""))
        data.pop("civitai_token", None)
        data["civitai_token_configured"] = bool(self._get("civitai_token", ""))
        data.pop("moderation_api_key", None)
        data["moderation_api_key_configured"] = bool(self._get("moderation_api_key", ""))
        data["moderation_ready"] = self._moderator().available
        data["workflow_selected"] = {mode: self._workflow_path(mode).name for mode in MODE_NAMES}
        data["plugin_ai_anima_context_preview"] = str(
            self._get("plugin_ai_anima_context", "")
            or build_anima_context(self.plugin_dir, 6000, self._anima_template_override())
        )
        template_path = find_template_path(self.plugin_dir, self._anima_template_override())
        data["anima_engineer_plugin"] = {
            "installed": True,
            "builtin": True,
            "template_path": str(template_path) if template_path else "",
            "mode": "内置 Anima 提示词工程师模板" if template_path else "内置 Anima 精简规则",
        }
        data["img2img_prompt_defaults"] = {
            "llm_tool_prompt": DEFAULT_IMG2IMG_LLM_TOOL_PROMPT,
            "plugin_system_prompt": DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT,
            "astrbot_system_prompt": DEFAULT_IMG2IMG_ASTRBOT_LLM_SYSTEM_PROMPT,
            "knowledge": DEFAULT_IMG2IMG_PLUGIN_AI_KNOWLEDGE,
            "plugin_user_prompt_template": DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE,
            "astrbot_user_prompt_template": DEFAULT_IMG2IMG_ASTRBOT_USER_PROMPT_TEMPLATE,
            "output_format": DEFAULT_IMG2IMG_OUTPUT_FORMAT,
            "system_prompt": DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT,
        }
        data["img2img_flux2_prompt_defaults"] = {
            "system_prompt": DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT,
            "knowledge": DEFAULT_IMG2IMG_PLUGIN_AI_KNOWLEDGE,
            "prompt_template": DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE,
            "output_format": DEFAULT_IMG2IMG_OUTPUT_FORMAT,
            "default_positive": FLUX2_IMG2IMG_DEFAULT_POSITIVE,
            "default_negative": FLUX2_IMG2IMG_DEFAULT_NEGATIVE,
        }
        return data

    def _ai_request_config(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """合并 WebUI 临时输入和已保存配置；API Key 永远不放进响应。"""
        result = self._config_dict()
        if isinstance(data, dict):
            for key in ("ai_base_url", "ai_model"):
                if key in data and data[key] is not None:
                    result[key] = data[key]
            # 密码框留空代表沿用已保存密钥，便于先获取模型或测试连接。
            if str(data.get("ai_api_key", "") or "").strip():
                result["ai_api_key"] = str(data["ai_api_key"]).strip()
        return result

    async def api_status(self):
        from astrbot.api.web import json_response
        root = detect_comfyui_root(str(self._get("comfyui_root", "")))
        try:
            status = await self._client().status()
            comfy = {"ok": True, "version": status.get("system", {}).get("comfyui_version", "未知")}
        except ComfyError as exc:
            comfy = {"ok": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("[%s] 读取控制台状态失败", PLUGIN_NAME)
            comfy = {"ok": False, "error": f"插件状态接口异常：{type(exc).__name__}: {exc}"}
        # 模型目录可能来自手动覆盖、extra_model_paths.yaml 或默认位置，
        # 因此统一走解析器，并把来源一并发给控制台用于标注。
        model_paths = {
            category: str(entries[0].path)
            for category in MODEL_CATEGORIES
            if (entries := self._model_dirs(category))
        }
        extra_paths_file, _ = self._extra_model_paths_sections()
        return json_response({"version": f"v{PLUGIN_VERSION}", "comfy": comfy, "config": self._safe_config(), "draw_limit": self._draw_limit_status(), "paths": {
            "comfyui_root": root, **model_paths,
            "extra_model_paths_file": extra_paths_file,
            "workflow_dir": str(self._workflow_dir()),
            "source_workflow": str(self._get("source_workflow", "")),
            "output_dir": str(self.output_dir), "data_dir": str(self.data_dir),
        }, "path_sources": self._path_sources()})

    async def api_models(self):
        from astrbot.api.web import json_response
        try:
            env = await self._environment(self._client())
            # classes 只用于运行时诊断，不能把 set 直接交给 JSON 编码器。
            compatible_models = self._compatible_models(env)
            return json_response({
                "diffusion_models": env["diffusion_models"],
                "checkpoints": env["checkpoints"],
                "loras": env["loras"],
                "upscale_models": env["upscale_models"],
                "unet": env["unet"],
                "text_encoders": env["text_encoders"],
                "vae": env["vae"],
                "compatible_models": compatible_models,
                "current_model": self._model_name(env, mode="txt2img"),
                "model_profiles": {
                    name: self._model_profile(name) for name in compatible_models
                },
                "anima_29b_patch_installed": self._anima_29b_patch_installed(),
                "img2img_models": list(dict.fromkeys(env["diffusion_models"] + env["unet"])),
                "img2img_clip_models": env.get("clip_gguf") or env["text_encoders"],
                "img2img_vae_models": env["vae"],
                "current_img2img": self._get("img2img_unet_name", QWEN_IMG2IMG_UNET_DEFAULT),
                "current_img2img_clip": self._get("img2img_clip_name", QWEN_IMG2IMG_CLIP_DEFAULT),
                "current_img2img_vae": self._get("img2img_vae_name", QWEN_IMG2IMG_VAE_DEFAULT),
                "img2img_flux2_models": list(dict.fromkeys(env["diffusion_models"] + env["unet"] + env["checkpoints"])),
                "img2img_flux2_clip_models": env.get("flux2_text_encoders") or env["text_encoders"],
                "img2img_flux2_vae_models": env["vae"],
                "img2img_flux2_loras": env["loras"],
                "current_img2img_flux2": self._get("img2img_flux2_unet_name", FLUX2_IMG2IMG_UNET_DEFAULT),
                "current_img2img_flux2_clip": self._get("img2img_flux2_clip_name", FLUX2_IMG2IMG_CLIP_DEFAULT),
                "current_img2img_flux2_vae": self._get("img2img_flux2_vae_name", FLUX2_IMG2IMG_VAE_DEFAULT),
                "current_img2img_flux2_lora": self._get("img2img_flux2_lora_name", FLUX2_IMG2IMG_LORA_DEFAULT),
                "current_img2img_engine": self._img2img_mode(),
                "enabled_loras": list(self._get("lora_list", []) or []),
            })
        except (ComfyError, UsageError) as exc:
            return json_response({"error": str(exc)}, status_code=500)

    async def api_workflows(self):
        from astrbot.api.web import json_response
        directory = str(self._workflow_dir())
        return json_response({
            "directory": directory,
            "workflow_dir": directory,
            "files": sorted(path.name for path in self._workflow_dir().glob("*.json")),
            "selected": {mode: self._workflow_path(mode).name for mode in MODE_NAMES},
        })

    async def api_config(self):
        from astrbot.api.web import json_response, request
        if request.method == "GET":
            return json_response(self._safe_config())
        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        changed = []
        for key, value in data.items():
            if key in WRITABLE_CONFIG:
                if key == "civitai_base_url":
                    try:
                        value = self._civitai_base_url(value)
                    except UsageError as exc:
                        return json_response({"error": str(exc)}, status_code=400)
                if key in {"ai_api_key", "civitai_token", "moderation_api_key"} and not str(value or "").strip():
                    continue
                self._set(key, value)
                changed.append(key)
        self._save_config()
        return json_response({
            "ok": True,
            "changed": changed,
            "lora_list": list(self._get("lora_list", []) or []),
        })

    async def api_ai_models(self):
        """提供 WebUI 的获取模型和连接测试，不依赖 AstrBot 当前人格。"""
        from astrbot.api.web import json_response, request

        data: dict[str, Any] = {}
        if request.method != "GET":
            raw = await self._request_json(request, {})
            if not isinstance(raw, dict):
                return json_response({"error": "请求体必须是对象"}, status_code=400)
            data = raw
        action = str(data.get("action", "list_models") or "list_models").strip().lower()
        try:
            translator = AITranslator(self._ai_request_config(data))
            if action in {"list", "models", "list_models"}:
                models = await translator.list_models()
                return json_response({"ok": True, "models": models})
            if action in {"test", "test_connection", "connect"}:
                await translator.test_connection()
                return json_response({"ok": True, "model": translator.model, "message": "连接成功，模型响应正常"})
            return json_response({"error": "未知操作，请选择获取模型列表或测试连接"}, status_code=400)
        except AIError as exc:
            return json_response({"error": str(exc)}, status_code=400)

    async def api_moderation_test(self):
        """WebUI 的审核链路测试：用一张纯色小图验证地址、密钥和模型是否可用。

        测试使用临时输入，允许在保存之前先验证；密码框留空表示沿用已保存密钥。
        """
        from astrbot.api.web import json_response, request

        data: dict[str, Any] = {}
        if request.method != "GET":
            raw = await self._request_json(request, {})
            if isinstance(raw, dict):
                data = raw
        merged = {key: self._get(key, "") for key in MODERATION_CONFIG_KEYS}
        for key in (
            "moderation_base_url",
            "moderation_model",
            "moderation_strictness",
            "moderation_timeout",
            "moderation_fail_open",
            "moderation_max_side",
        ):
            if data.get(key) not in (None, ""):
                merged[key] = data[key]
        if str(data.get("moderation_api_key", "") or "").strip():
            merged["moderation_api_key"] = str(data["moderation_api_key"]).strip()
        config = ModerationConfig.from_mapping(merged)
        if not config.configured:
            return json_response(
                {"error": "请先填写审核模型的服务地址和模型名"}, status_code=400
            )
        moderator = ImageModerator(config)
        try:
            message = await moderator.test_connection()
        except Exception as exc:
            return json_response({"error": f"审核链路测试失败：{exc}"}, status_code=400)
        return json_response(
            {
                "ok": True,
                "message": message,
                "strictness": config.strictness,
                "model": config.model,
            }
        )

    def _artist_preset_payload(self) -> dict[str, Any]:
        return {
            "presets": self.artist_presets.items,
            "active": str(self._get("artist_preset", DEFAULT_ARTIST_PRESET_NAME) or ""),
        }

    async def api_artist_presets(self):
        from astrbot.api.web import json_response, request

        if request.method == "GET":
            return json_response(self._artist_preset_payload())
        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        action = str(data.get("action", "") or "").strip().lower()
        name = str(data.get("name", "") or "").strip()
        try:
            if action in {"add", "update"}:
                self.artist_presets.add(name, str(data.get("content", "") or ""))
            elif action in {"remove", "delete"}:
                self.artist_presets.remove(name)
                if str(self._get("artist_preset", "") or "") == name:
                    self._set("artist_preset", "")
                    self._save_config()
            elif action in {"activate", "use"}:
                if name.casefold() in {"", "无", "关闭", "禁用", "none", "off"}:
                    self._set("artist_preset", "")
                elif name not in self.artist_presets.items:
                    return json_response({"error": f"画师串预设不存在：{name}"}, status_code=404)
                else:
                    self._set("artist_preset", name)
                self._save_config()
            else:
                return json_response({"error": "未知操作"}, status_code=400)
        except UsageError as exc:
            return json_response({"error": str(exc)}, status_code=400)
        return json_response({"ok": True, **self._artist_preset_payload()})

    async def api_presets(self):
        from astrbot.api.web import json_response, request
        if request.method == "GET":
            return json_response({"presets": self.presets.items})
        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        action = data.get("action", "")
        try:
            if action in {"add", "update"}:
                name = str(data.get("name", ""))
                content = str(data.get("content", ""))
                old_name = str(data.get("old_name", name))
                if action == "update":
                    self.presets.update(old_name, name, content)
                else:
                    self.presets.add(name, content)
                self._save_linked_preset_change(
                    old_name=old_name if action == "update" else "",
                    name=name,
                    content=content,
                )
            elif action == "remove":
                name = str(data.get("name", ""))
                removed = self.presets.remove(name)
                if removed:
                    self._remove_lora_global_link(name)
                    self._save_lora_presets()
            elif action == "translate":
                name = str(data.get("name", ""))
                item = self.presets.items.get(name)
                if not item:
                    return json_response({"error": "预设不存在"}, status_code=404)
                ai_config = self._config_dict()
                ai_config["ai_enabled"] = True
                item["translated"] = await AITranslator(ai_config).generate(
                    item.get("content", ""),
                    system_prompt=str(
                        self._get("plugin_ai_command_system_prompt", "")
                        or DANBOORU_SYSTEM_PROMPT
                    ).strip(),
                    max_tokens=512,
                )
                self.presets.save()
                self._save_linked_preset_change(
                    name=name,
                    content=item["translated"],
                )
                return json_response({"ok": True, "translated": item["translated"]})
            else:
                return json_response({"error": "未知操作"}, status_code=400)
        except (UsageError, AIError) as exc:
            return json_response({"error": str(exc)}, status_code=400)
        return json_response({"ok": True})

    async def api_recent(self):
        from astrbot.api.web import json_response
        files = []
        for path in self.output_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                files.append({"name": str(path.relative_to(self.output_dir)), "mtime": path.stat().st_mtime})
        files.sort(key=lambda item: item["mtime"], reverse=True)
        output_dir = str(self.output_dir)
        return json_response({"items": files[:50], "directory": output_dir, "output_dir": output_dir})
