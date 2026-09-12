from __future__ import annotations

import asyncio
import base64
import copy
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
import uuid
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain, Reply
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
from .paths import detect_comfyui_root, model_dir
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
    adapt_original,
    anima_sampling_defaults,
    copy_and_repair_original,
    is_anima_29b_model,
    load_api_workflow,
    load_workflow,
)

PLUGIN_NAME = "astrbot_plugin_comfyui_ai_studio"
PLUGIN_VERSION = "0.8.0"
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
MODE_NAMES = {"txt2img": "文生图", "img2img": "图生图", "hires": "高清放大"}
MODE_ALIASES = {"文生图": "txt2img", "图生图": "img2img", "高清放大": "hires", "高清": "hires"}
MODE_FILES = {"txt2img": "文生图.json", "img2img": "图生图_qwen_edit.json", "hires": "高清放大.json"}
QWEN_IMG2IMG_SOURCE_DEFAULT = r"E:\112121121.json"
QWEN_IMG2IMG_SOURCE_LEGACY = (
    r"E:\222.json",
    r"E:\11111图生图.json",
    r"E:\QwenImageEdit2511局部重绘替换万物.json",
    r"E:\▶QwenImageEdit2511-AIO图生图10G.json",
    r"E:\No2图生图模型\8G-10G显存下载\工作流\▶QwenImageEdit2511-AIO图生图10G.json",
)
QWEN_IMG2IMG_UNET_DEFAULT = "Qwen-Rapid-NSFW-v23_Q3_K.gguf"
QWEN_IMG2IMG_CLIP_DEFAULT = "Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf"
QWEN_IMG2IMG_VAE_DEFAULT = "qwen_image_vae.safetensors"
QWEN_IMG2IMG_LORA_DEFAULT = ""
LLM_TOOL_NAMES = {
    "status": "comfyui_ai_studio_status",
    "generate": "comfyui_ai_studio_generate",
    "edit": "comfyui_ai_studio_edit",
    "upscale": "comfyui_ai_studio_upscale",
}
WRITABLE_CONFIG = {
    "comfyui_url", "comfyui_root", "source_workflow", "workflow_dir", "model_name",
    "workflow_txt2img", "workflow_img2img", "workflow_hires",
    "img2img_source_workflow", "img2img_unet_name", "img2img_clip_name", "img2img_vae_name",
    "img2img_lora_name", "img2img_lora_strength", "img2img_default_positive", "img2img_default_negative",
    "img2img_width", "img2img_height", "img2img_keep_aspect_ratio", "img2img_steps", "img2img_cfg", "img2img_seed",
    "img2img_sampler_name", "img2img_scheduler", "img2img_denoise", "img2img_scale_method",
    "img2img_megapixels", "img2img_largest_size", "img2img_crop", "img2img_resolution_steps", "img2img_reference_method", "img2img_sampling_shift",
    "img2img_cfg_norm_strength", "img2img_pre_cfg", "img2img_tile_size", "img2img_tile_overlap",
    "img2img_temporal_size", "img2img_temporal_overlap", "img2img_second_image", "img2img_filename_prefix",
    "lora_list", "default_positive", "default_negative", "quality_prefix", "artist_preset", "ai_base_url",
    "ai_api_key", "ai_model", "civitai_token", "width", "height", "steps", "cfg", "seed",
    "sampler_name", "scheduler", "denoise", "hires_scale",
    "hires_steps", "hires_denoise", "hires_upscale_model", "max_concurrent",
    "anima_teacache", "draw_start_reply", "draw_reply_mode", "draw_reply_custom", "draw_delivery_mode",
    "draw_reply_timeout", "draw_attach_prompt", "nsfw_group_blacklist",
    "plain_translate_enabled", "plain_translate_url",
    "llm_prompt_source", "plugin_ai_command_system_prompt", "plugin_ai_llm_system_prompt",
    "plugin_ai_anima_context", "plugin_ai_debug", "llm_wait_timeout",
    "img2img_llm_prompt_source", "img2img_plugin_ai_llm_system_prompt",
    "img2img_plugin_ai_knowledge", "img2img_plugin_ai_debug",
    "img2img_llm_tool_prompt", "img2img_plugin_ai_user_prompt_template",
    "img2img_plugin_ai_output_format",
    "img2img_astrbot_llm_system_prompt", "img2img_astrbot_user_prompt_template",
    "draw_limit_count", "draw_limit_window_seconds", "draw_limit_admin_ids",
    "comfyui_start_script",
}

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
只输出一行小写英文、逗号分隔的最终提示词，不要解释、Markdown、代码块、质量词、画师名、LoRA 语法或权重语法。用户原话中用于控制预设和 LoRA 的名称由绘图插件单独处理，不要因为翻译而删除或改写它们。"""

DEFAULT_IMG2IMG_PLUGIN_AI_KNOWLEDGE = """这是 Qwen Image Edit 图生图，不是文生图。
原图是最高优先级：除用户明确要求修改的项目外，人物身份、脸、发型、姿势、镜头、构图、背景、光线、色彩与画风都必须保持原样。
输出只描述要改动的内容；不要补充质量词、角色名、场景、镜头、风格、画师、LoRA、预设或未被要求的细节。"""

DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT = """你是 Qwen Image Edit 的图生图编辑指令整理器。
把用户对已有图片的要求改写为一句简洁的英文编辑指令。只保留用户明确想改动的内容，不要扩写成完整绘图提示词。
例如“把身上衣服换成裙子”只输出“change the outfit to a dress”。
不要输出人物名、质量词、背景、镜头、画风、动作或其它原图未要求修改的内容；不要输出解释、思考过程、Markdown、代码块、LoRA 语法或预设名。
最终只能输出一行，格式必须严格为：EDIT: 英文编辑指令。"""

DEFAULT_IMG2IMG_ASTRBOT_LLM_SYSTEM_PROMPT = """你是 Qwen Image Edit 的图生图编辑指令整理器。
你正在把用户对已有图片的修改要求转换成 Qwen Image Edit 能执行的英文编辑指令。
只保留用户明确要求改变的内容，必须保留具体的动作、姿势、服装修改、表情修改、物体替换、背景替换或其它明确编辑目标。
不要输出完整绘图提示词，不要补充人物身份、原图已有的服装、背景、镜头、画风、质量词或任何用户没有要求改变的内容。
不要解释、分析、复述规则或输出 Markdown。最终只输出一行，格式严格为：EDIT: 简洁英文编辑指令。
例如：用户说“把身上衣服换成裙子”，只输出“EDIT: change the outfit to a dress”。"""
DEFAULT_IMG2IMG_LLM_TOOL_PROMPT = """当前请求是图生图编辑。必须调用 comfyui_ai_studio_edit，不能调用文生图工具，也不能只文字回复。
prompt 只填写用户明确想改动的内容，例如“把身上衣服换成裙子”只填写“换成裙子”或等价英文。
禁止补充角色、质量词、场景、镜头、画风、动作、服装细节或任何原图未要求改变的内容。
用户原话中的提示词预设名和 LoRA 指令简称仍要分别填入 preset 和 lora，它们可以同时填写。"""
DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE = """用户对已有图片的编辑要求：
{request}"""
DEFAULT_IMG2IMG_ASTRBOT_USER_PROMPT_TEMPLATE = """用户对已有图片的编辑要求：
{request}"""
DEFAULT_IMG2IMG_OUTPUT_FORMAT = """最终只能输出一行，格式严格为：EDIT: 简洁英文编辑指令。
不要输出解释、分析、复述规则、Markdown、代码块或其它前后缀。"""
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
        # 读取 LoRA safetensors 头部得到的架构信息。只缓存文件名对应的结果，
        # 不读取权重正文；这样每次绘图不会因为检查兼容性而重复打开大文件。
        self._lora_profile_cache: dict[str, str] = {}
        self._register_web_api()

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
        is_edit = self._event_has_image_hint(event) or self._looks_like_edit_request(event, message)
        if not is_drawing_request(message) and not is_edit:
            return
        try:
            if is_edit:
                instruction = str(
                    self._get("img2img_llm_tool_prompt", "")
                    or DEFAULT_IMG2IMG_LLM_TOOL_PROMPT
                ).strip()
                req.system_prompt = f"【图生图编辑规则】\n{instruction}\n\n{str(req.system_prompt or '').strip()}".strip()
                return
            context = str(self._get("plugin_ai_anima_context", "") or "").strip()
            if not context:
                context = build_anima_context(self.plugin_dir, 2600)
            instruction = (
                "当前请求涉及绘图。请把自己当作 Anima3 提示词工程师：先理解用户真正想画什么，"
                "再按 Anima 规则生成提示词；调用本插件的绘图工具时，prompt 必须是适配 Anima 工作流的"
                "一行小写英文 Danbooru 标签。必须保留用户原话中的动作、姿势、正在进行的行为、表情、"
                "镜头、构图和场景。调用 comfyui_ai_studio_generate、comfyui_ai_studio_edit 或 "
                "comfyui_ai_studio_upscale 时，检查用户原话里的提示词预设名和 LoRA 指令简称，分别填入 "
                "preset 和 lora；两者可以同时填写，多个值用逗号分隔。不要把明确的绘图请求改成只查询规则。"
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
            ("ai_models", self.api_ai_models, ["GET", "POST"], "获取 AI 模型并测试连接"),
            ("presets", self.api_presets, ["GET", "POST"], "管理提示词预设"),
            ("artist_presets", self.api_artist_presets, ["GET", "POST"], "管理独立画师串预设"),
            ("lora_info", self.api_lora_info, ["GET", "POST"], "读取 LoRA 别名和 CivitAI 信息"),
            ("open_folder", self.api_open_folder, ["POST"], "打开本地模型文件夹"),
            ("open_lora", self.api_open_lora, ["POST"], "打开 LoRA 所在文件夹"),
            ("upload_lora", self.api_upload_lora, ["POST"], "上传 LoRA 文件"),
            ("download_lora", self.api_download_lora, ["POST"], "从 CivitAI 下载 LoRA 文件"),
            ("download_lora_progress", self.api_download_lora_progress, ["POST"], "读取 LoRA 下载进度"),
            ("delete_lora", self.api_delete_lora, ["POST"], "删除 LoRA 文件"),
            ("recent", self.api_recent, ["GET"], "读取最近生成图片"),
        ]
        for name, handler, methods, description in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{name}", handler, methods, description
            )

    async def initialize(self) -> None:
        source = Path(str(self._get("source_workflow", r"E:\难工作流.json")))
        if source.is_file():
            try:
                # 原始工作流是唯一基准。每次加载都刷新默认副本，避免插件继续
                # 使用旧副本；用户在 WebUI 选择的自定义文件不会被覆盖。
                copy_and_repair_original(source, self._workflow_dir())
                logger.info("[%s] 已按原始工作流刷新三套默认工作流：%s", PLUGIN_NAME, source)
            except Exception as exc:
                logger.warning("[%s] 从原始工作流刷新副本失败：%s", PLUGIN_NAME, exc)
        # 图生图使用独立的 Qwen Image Edit 工作流和模型链。
        # 这里迁移旧版配置，但不修改用户提供的源文件。
        configured_img2img_source = str(self._get("img2img_source_workflow", "") or "").strip()
        legacy_sources = {item.casefold() for item in QWEN_IMG2IMG_SOURCE_LEGACY}
        changed = False
        if not configured_img2img_source or configured_img2img_source.casefold() in legacy_sources:
            configured_img2img_source = QWEN_IMG2IMG_SOURCE_DEFAULT
            self._set("img2img_source_workflow", configured_img2img_source)
            changed = True
        img2img_source = Path(configured_img2img_source)
        img2img_target = self._workflow_dir() / MODE_FILES["img2img"]
        if img2img_source.is_file():
            try:
                img2img_workflow = load_workflow(img2img_source)
                img2img_target.write_text(
                    json.dumps(img2img_workflow, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("[%s] 已加载独立 Qwen Image Edit 图生图工作流：%s", PLUGIN_NAME, img2img_source)
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
        )
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
            "text_encoders": categories[4],
            "vae": categories[5],
            "classes": classes,
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
        values = [str(name).replace("\\", "/") for name in (names or []) if str(name).strip()]
        download_order = getattr(self, "lora_download_order", {})
        if not isinstance(download_order, dict):
            download_order = {}
        downloaded: list[tuple[float, int, str]] = []
        old: list[str] = []
        for index, name in enumerate(values):
            raw_time = download_order.get(name)
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

    def _img2img_output_size(self, image_path: str, width: int, height: int) -> tuple[int, int]:
        """Fit the output rectangle to the input image aspect ratio.

        The Qwen workflow still receives an explicit size, so this keeps the
        existing width/height controls useful as a maximum bounding box.
        """
        keep_ratio = self._get("img2img_keep_aspect_ratio", True)
        if isinstance(keep_ratio, str):
            keep_ratio = keep_ratio.strip().lower() not in {"0", "false", "off", "no", "否", "关闭"}
        if not keep_ratio or not image_path:
            return width, height
        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as image:
                source_width, source_height = image.size
            if source_width <= 0 or source_height <= 0:
                return width, height
            scale = min(float(width) / source_width, float(height) / source_height)
            fitted_width = max(64, int(source_width * scale) // 8 * 8)
            fitted_height = max(64, int(source_height * scale) // 8 * 8)
            logger.info(
                "[%s] 图生图保持原图比例：原图=%sx%s；目标框=%sx%s；实际输出=%sx%s",
                PLUGIN_NAME,
                source_width,
                source_height,
                width,
                height,
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
        if source == "plugin_img2img_llm":
            custom_prompt = str(
                self._get("img2img_plugin_ai_llm_system_prompt", "")
                or ""
            ).strip()
            system_prompt = custom_prompt or DEFAULT_IMG2IMG_PLUGIN_AI_LLM_SYSTEM_PROMPT
            edit_knowledge = str(self._get("img2img_plugin_ai_knowledge", "") or "").strip()
            system_prompt = (
                f"{system_prompt}\n\n【图生图编辑知识】\n"
                f"{edit_knowledge or DEFAULT_IMG2IMG_PLUGIN_AI_KNOWLEDGE}"
            )
        elif source == "astrbot_img2img_llm":
            # 图生图的 AstrBot 来源也必须使用专用编辑规则。不能把
            # AstrBot 工具参数原样当作最终 prompt，否则“换衣服”等中文
            # 原话会直接进入 Qwen，效果等同于普通图生图指令。
            system_prompt = str(
                self._get("img2img_astrbot_llm_system_prompt", "")
                or DEFAULT_IMG2IMG_ASTRBOT_LLM_SYSTEM_PROMPT
            ).strip()
            edit_knowledge = str(self._get("img2img_plugin_ai_knowledge", "") or "").strip()
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
        if source not in {"plugin_img2img_llm", "astrbot_img2img_llm"}:
            # context.llm_generate() 不会再次经过当前插件的 on_llm_request
            # 过滤器，故在文生图提示词请求中主动复用内置 Anima 精简知识。
            anima_context = str(self._get("plugin_ai_anima_context", "") or "").strip()
            if not anima_context:
                anima_context = build_anima_context(self.plugin_dir, 2600)
            try:
                system_prompt = (
                    f"{system_prompt}\n\n【Anima 提示词工程师知识】\n"
                    f"{anima_context}"
                )
            except Exception as exc:
                logger.debug("[%s] Anima 提示词知识加载失败，使用基础规则：%s", PLUGIN_NAME, exc)
        if source == "astrbot_img2img_llm":
            prompt_template = str(
                self._get("img2img_astrbot_user_prompt_template", "")
                or DEFAULT_IMG2IMG_ASTRBOT_USER_PROMPT_TEMPLATE
            ).strip()
            user_prompt = self._format_img2img_llm_template(prompt_template, text)
            output_format = str(
                self._get("img2img_plugin_ai_output_format", "")
                or DEFAULT_IMG2IMG_OUTPUT_FORMAT
            ).strip()
            system_prompt = f"{system_prompt}\n\n【输出格式要求】\n{output_format}".strip()
            result = await self._astrbot_generate(
                event,
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=96 if source == "astrbot_img2img_llm" else 512,
                preserve_newlines=source in {"plugin_img2img_llm", "astrbot_img2img_llm"},
            )
            return (
                self._clean_img2img_edit_instruction(result)
                if source == "astrbot_img2img_llm"
                else result
            )
        if source == "astrbot":
            result = await self._astrbot_generate(
                event,
                f"画面描述：{text}",
                system_prompt=system_prompt,
                max_tokens=512,
            )
            return result
        ai_config = self._config_dict()
        if source == "plugin_img2img_llm":
            prompt_template = str(
                self._get("img2img_plugin_ai_user_prompt_template", "")
                or DEFAULT_IMG2IMG_PLUGIN_AI_USER_PROMPT_TEMPLATE
            ).strip()
            prompt = self._format_img2img_llm_template(prompt_template, text)
            output_format = str(
                self._get("img2img_plugin_ai_output_format", "")
                or DEFAULT_IMG2IMG_OUTPUT_FORMAT
            ).strip()
            system_prompt = f"{system_prompt}\n\n【输出格式要求】\n{output_format}".strip()
        else:
            prompt = text
        result = await AITranslator(ai_config).generate(
            prompt,
            system_prompt=system_prompt,
            max_tokens=96 if source == "plugin_img2img_llm" else 768 if source == "plugin_llm" else 512,
            preserve_newlines=source == "plugin_img2img_llm",
        )
        if source == "plugin_img2img_llm":
            return self._clean_img2img_edit_instruction(result)
        return result

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

    async def _plain_translate_prompt(self, text: str) -> tuple[str, str]:
        """非 AI 模式下翻译用户中文；失败时保留原文，不中断绘图。"""
        enabled = self._get("plain_translate_enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "off", "no", "否", "关闭"}
        if not enabled or not contains_chinese(text):
            return text, ""
        try:
            result = await PlainTranslator(
                str(
                    self._get(
                        "plain_translate_url",
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
    def _clean_img2img_edit_instruction(value: str) -> str:
        """只接受短英文编辑指令，阻止模型推理文本进入 Qwen 工作流。"""
        raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip(" `\n\t")
        if not raw:
            raise AIError("图生图插件 AI 返回了空内容")

        candidates: list[str] = []
        line_marker_pattern = re.compile(
            r"(?:^|\n)\s*(?:edit|final(?:\s+edit)?|instruction|编辑指令|最终(?:编辑指令|结果)?)\s*[:：]\s*([^\n]+)",
            re.IGNORECASE,
        )
        candidates.extend(match.group(1) for match in line_marker_pattern.finditer(raw))
        # 一些推理模型会把“分析过程\nEDIT: ...”压成“分析过程, EDIT: ...”，
        # 再用逗号/分号起点兼容这种返回。单独的行匹配优先，保留编辑
        # 指令内部可能存在的逗号。
        inline_marker_pattern = re.compile(
            r"[,，;；]\s*(?:edit|final(?:\s+edit)?|instruction|编辑指令|最终(?:编辑指令|结果)?)\s*[:：]\s*([^\n,，;；]+)",
            re.IGNORECASE,
        )
        candidates.extend(match.group(1) for match in inline_marker_pattern.finditer(raw))

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
        # 一些推理模型把最终短句夹在解释的引号中；只取纯英文候选。
        candidates.extend(re.findall(r"[\"“'`]([^\"”'`\n]{3,180})[\"”'`]", raw))
        # 无格式标记时逐行尝试，避免把多行分析文本整体交给 Qwen。
        candidates.extend(line.strip() for line in raw.split("\n") if line.strip())
        if "\n" not in raw:
            candidates.append(raw)

        disallowed = (
            "we need", "user", "original", "astrbot", "should", "output", "prompt",
            "system", "because", "therefore", "note", "think", "analysis", "reasoning",
            "用户", "原话", "输出", "提示词", "系统", "注意", "需要", "我们",
        )
        for candidate in candidates:
            clean = re.sub(
                r"^(?:edit|final(?:\s+edit)?|instruction)\s*[:：]\s*",
                "",
                str(candidate),
                flags=re.IGNORECASE,
            )
            clean = clean.strip(" `\"'，,。.;:：")
            if not clean or len(clean) > 180 or re.search(r"[\u4e00-\u9fff]", clean):
                continue
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.'()/_-]*", clean):
                continue
            lowered = clean.casefold()
            if any(marker in lowered for marker in disallowed):
                continue
            if len(re.findall(r"[A-Za-z]+", clean)) > 18:
                continue
            return clean
        raise AIError("图生图插件 AI 返回了推理内容，未提取到有效英文编辑指令")

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
        text = self.presets.expand(params.prompt)
        preset_values: list[str] = []
        for name in params.presets:
            value = self.presets.effective(name)
            if not value:
                raise UsageError(f"预设不存在：{name}")
            preset_values.append(value)
        use_ai = params.auto_ai or params.ai is True
        is_img2img = mode == "img2img"
        if is_img2img and params.img2img_edit_instruction_ready:
            # LLM 图生图的编辑结果是独立字段。params.prompt 可能仍是
            # AstrBot 工具传入的自然语言或旧版长描述，不能让它覆盖最终编辑指令。
            text = str(params.img2img_edit_instruction or text or "").strip()
        source = str(params.ai_source or ("astrbot" if params.auto_ai else "plugin")).lower()
        if is_img2img and use_ai:
            # 图生图不适用 Anima 文生图扩写；即使是 /图生图 ai，也使用
            # 专用编辑知识把需求收敛成“只改什么”。
            source = "plugin_img2img_llm"
        note = ""
        # 预设或 LoRA 控制项已经消耗掉角色名时，不再对空文本调用 AI，
        # 避免模型凭空补出 Danbooru 身份词。
        if text.strip() and not (is_img2img and params.img2img_edit_instruction_ready) and use_ai and (contains_chinese(text) or params.ai is True):
            try:
                text = await self._translate_prompt(
                    event,
                    text,
                    force_enabled=params.ai is True or params.auto_ai,
                    source_override=source,
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
                # ai/noai 是可选增强功能，AI 服务异常不应阻断普通绘图。
                # 保留用户原文，若包含中文再尝试无 AI 翻译网站；两者都失败时
                # 仍把原文交给 ComfyUI，而不是把一次绘图变成“格式解析失败”。
                logger.warning("[%s] AI 提示词处理失败，回退普通绘图：%s", PLUGIN_NAME, exc)
                note = f"AI 不可用，已按普通方式继续出图（{exc}）"
                text, plain_note = await self._plain_translate_prompt(text)
                if plain_note:
                    note = f"{note}；{plain_note}"
        elif contains_chinese(text) and not (is_img2img and params.img2img_edit_instruction_ready):
            text, plain_note = await self._plain_translate_prompt(text)
            note = plain_note
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
            quality = str(self._get("img2img_default_positive", "") or "").strip()
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
        positive = ", ".join(
            value.strip(" ,，")
            for value in fragments
            if value.strip(" ,，")
        )
        negative = params.negative or str(
            self._get("img2img_default_negative", "")
            if mode == "img2img"
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
        category = str(self._lora_map_value(self.lora_categories, file_name, "") or "").strip()
        return category if category in self.lora_category_entries else "未分类"

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
        value = str(value or "").strip()
        if not value:
            return None
        exact = {name.lower(): name for name in available}
        if value.lower() in exact:
            return exact[value.lower()]
        aliases = {self._lora_alias(name).lower(): name for name in available}
        resolved = aliases.get(value.lower())
        if resolved:
            return resolved
        command_aliases = {
            alias.lower(): name
            for name in available
            for alias in self._lora_command_aliases_for(name, available)
        }
        return command_aliases.get(value.lower())

    def _resolve_lora_command_alias_only(self, value: str, available: list[str]) -> str | None:
        """只按 LoRA 指令简称解析，避免误删普通提示词中的同名单词。"""
        target = str(value or "").strip().casefold()
        if not target:
            return None
        for name in available:
            if any(alias.casefold() == target for alias in self._lora_command_aliases_for(name, available)):
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
        """记录本次任务命中的一个 LoRA 专属预设。"""
        tag = self._lora_preset_tag_for_alias(file_name, alias)
        if not tag:
            return
        mapping = getattr(params, "lora_preset_tags", None)
        if not isinstance(mapping, dict):
            mapping = {}
            params.lora_preset_tags = mapping
        key = str(file_name or "").replace("\\", "/").casefold()
        values = mapping.setdefault(key, [])
        if not isinstance(values, list):
            values = []
            mapping[key] = values
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

    async def _extract_inline_loras(self, params: DrawParams, extra_text: str = "") -> None:
        """提取提示词和用户原话中的简称，并只对当前任务临时加载。

        LLM 工具经常只把英文提示词放进 prompt，而把用户原话里的 LoRA 简称
        丢在工具参数之外。因此这里同时检查当前消息原文，保证第二简称也能命中。
        """
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
        if not params.loras and not any(str(alias).casefold() in source_text for alias in known_aliases):
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
        alias_pairs = sorted(
            [
                (alias, name)
                for name in available
                for alias in self._lora_command_aliases_for(name, available)
            ],
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
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

        # 预设名称和 LoRA 指令简称相同时，预设不能吞掉 LoRA 控制项。
        # 这里只按完整简称匹配，不会因为普通提示词中的英文标签误加载 LoRA。
        for preset_name in params.presets:
            target = str(preset_name or "").strip().casefold()
            if not target:
                continue
            for alias, actual in alias_pairs:
                if alias.casefold() == target:
                    append_lora(actual, "0.8")
                    self._remember_lora_preset_match(params, actual, alias)
                    break
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

    def _civitai_search_url(self, file_name: str) -> str:
        from urllib.parse import quote_plus

        return "https://civitai.com/search/models?query=" + quote_plus(Path(file_name).stem)

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

    def _civitai_headers(self, accept: str) -> dict[str, str]:
        """构造 CivitAI API 和 CDN 下载都能使用的请求头。"""
        headers = {
            "Accept": accept,
            "User-Agent": f"AstrBot-ComfyUI-AI-Studio/{PLUGIN_VERSION}",
            "Referer": "https://civitai.com/",
        }
        token = str(self._get("civitai_token", "") or "").strip()
        if token:
            headers["Authorization"] = (
                token if token.lower().startswith("bearer ") else f"Bearer {token}"
            )
        return headers

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
        if custom_url and "trigger_words" in override:
            result["trigger_words"] = self._normalize_trigger_words(override.get("trigger_words", []))
        if custom_url and "tags" in override:
            result["tags"] = self._normalize_civitai_tags(override.get("tags", []))
        if custom_name:
            result["model_name"] = custom_name
        if custom_images:
            result["images"] = custom_images[:8]
        elif custom_url:
            # 使用自定义 CivitAI 链接时，不能继续显示旧文件名匹配到的图片。
            result["images"] = []
        else:
            result["images"] = list(result.get("images", []) or [])[:1]
        result["custom_url"] = custom_url
        result["custom_name"] = custom_name
        result["custom_images"] = [item["url"] for item in custom_images]
        result["civitai_tags"] = self._normalize_civitai_tags(result.get("tags", []))
        result["custom_link"] = bool(custom_url)
        result["custom_info"] = bool(custom_url or custom_name or custom_images)
        result["show_images"] = override.get("show_images", True) is not False
        if custom_url:
            result["model_url"] = custom_url
            result["found"] = True
            if not result.get("model_name"):
                result["model_name"] = "我的自定义链接"
        elif custom_name or custom_images:
            result["found"] = True
        return result

    async def _fetch_civitai_lora(self, file_name: str, *, force: bool = False) -> dict[str, Any]:
        cached = self.civitai_cache.get(file_name)
        if (
            not force
            and isinstance(cached, dict)
            and time.time() - float(cached.get("fetched_at", 0) or 0) < 86400
            and cached.get("url_format") == "versioned-model-url-v2"
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
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                response = await client.get(
                    "https://civitai.com/api/v1/models",
                    params={"query": query, "types": "LORA", "limit": 10},
                )
                response.raise_for_status()
                items = response.json().get("items", [])
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
                    result["model_url"] = f"https://civitai.com/models/{model_id}"
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
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            result["error"] = f"CivitAI 查询失败：{exc}"
        async with self._civitai_cache_lock:
            self.civitai_cache[file_name] = {
                "fetched_at": time.time(),
                "url_format": "versioned-model-url-v2",
                "data": result,
            }
            self._save_civitai_cache()
        return self._apply_civitai_link_override(file_name, result)

    async def _fetch_civitai_link_info(self, link: str) -> dict[str, Any]:
        """按用户填写的 CivitAI 模型链接立即读取名称和首张预览图。"""
        from urllib.parse import urlparse

        parsed = urlparse(link)
        if parsed.netloc.lower().split(":", 1)[0] not in {"civitai.com", "www.civitai.com"}:
            return {}
        model_match = re.search(r"/models/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        version_match = re.search(r"/model-versions/(\d+)(?:/|$)", parsed.path, re.IGNORECASE)
        requested_version = parse_qs(parsed.query).get("modelVersionId", [""])[0]
        version_id = requested_version or (version_match.group(1) if version_match else "")
        if not model_match and not version_id:
            return {}
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                if model_match:
                    response = await client.get(f"https://civitai.com/api/v1/models/{model_match.group(1)}")
                    response.raise_for_status()
                    item = response.json()
                else:
                    response = await client.get(f"https://civitai.com/api/v1/model-versions/{version_id}")
                    response.raise_for_status()
                    version_item = response.json()
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
                            model_response = await client.get(f"https://civitai.com/api/v1/models/{model_id}")
                            model_response.raise_for_status()
                            model_item = model_response.json()
                            if isinstance(model_item, dict):
                                item = model_item
                        except (httpx.HTTPError, ValueError, TypeError):
                            pass
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.debug("[%s] 读取自定义 CivitAI 链接失败：%s", PLUGIN_NAME, exc)
            return {}
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
    def _civitai_reference(link: str) -> tuple[str, str]:
        """从 CivitAI 模型页提取模型 ID 和可选版本 ID。"""
        parsed = urlparse(link)
        host = parsed.netloc.lower().split(":", 1)[0]
        if host not in {"civitai.com", "www.civitai.com", "civitai.ai", "www.civitai.ai"}:
            raise UsageError("下载地址必须是 civitai.com 的模型链接")
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
        model_id, version_id = self._civitai_reference(link)
        parsed_link = urlparse(link)
        is_direct_download = bool(re.search(r"/(?:api/)?download/models/\d+(?:/|$)", parsed_link.path, re.IGNORECASE))
        headers = self._civitai_headers("application/json")

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
                        raise UsageError("CivitAI 模型信息返回格式不是 JSON") from exc
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
                    model = await get_json(client, f"https://civitai.com/api/v1/models/{model_id}")
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
                version = await get_json(client, f"https://civitai.com/api/v1/model-versions/{version_id}")
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
            download_url = f"https://civitai.com/api/download/models/{version_id}"
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
        limiter = asyncio.Semaphore(4)

        async def fetch(name: str) -> dict[str, Any]:
            async with limiter:
                info = await self._fetch_civitai_lora(name, force=force)
            # CivitAI 触发词只作为资料展示，不自动写入专属预设或最终提示词。
            visible_info = dict(info)
            return {
                "file_name": name,
                "alias": self._lora_alias(name),
                "command_aliases": self._lora_command_aliases_for(name, names),
                "command_alias": self._lora_command_alias(name, names),
                "category": self._lora_category(name),
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

    def _model_folder_path(self, kind: str) -> Path:
        kind = str(kind or "").strip()
        root = detect_comfyui_root(str(self._get("comfyui_root", "")))
        categories = {"diffusion_models", "checkpoints", "loras", "upscale_models"}
        if kind in categories:
            if not root:
                raise UsageError("未检测到 ComfyUI 根目录")
            return Path(model_dir(root, kind))
        if kind == "workflows":
            return self._workflow_dir()
        if kind == "source_workflow":
            return Path(str(self._get("source_workflow", r"E:\难工作流.json"))).parent
        raise UsageError("不支持的文件夹类型")

    async def api_lora_info(self):
        from astrbot.api.web import json_response, request

        if request.method == "GET":
            try:
                return json_response(await self._lora_payload())
            except (ComfyError, OSError) as exc:
                return json_response({"error": str(exc)}, status_code=500)
        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        action = str(data.get("action", "") or "")
        try:
            available = await self._local_lora_names()
            file_name = str(data.get("file_name", "") or "").strip()
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
                return json_response({"ok": True, "items": (await self._lora_payload())["items"], "categories": self._lora_categories_list()})
            if action in {"civitai_override", "set_civitai_override"}:
                if file_name not in available:
                    return json_response({"error": "LoRA 不存在"}, status_code=404)
                try:
                    custom_url = self._validate_civitai_link(data.get("url", data.get("civitai_url", "")))
                except UsageError as exc:
                    return json_response({"error": f"CivitAI 信息无效：{exc}"}, status_code=400)
                custom_name = str(data.get("name", data.get("model_name", "")) or "").strip()[:120]
                link_info = await self._fetch_civitai_link_info(custom_url) if custom_url else {}
                if not custom_name:
                    custom_name = str(link_info.get("model_name", "") or "")[:120]
                custom_images = [
                    str(item.get("url", "") or "")
                    for item in (link_info.get("images", []) if isinstance(link_info, dict) else [])
                    if isinstance(item, dict) and str(item.get("url", "") or "").strip()
                ][:1]
                self.civitai_overrides[file_name] = {
                    "url": custom_url,
                    "name": custom_name,
                    "images": custom_images,
                    "trigger_words": self._normalize_trigger_words(link_info.get("trigger_words", [])),
                    "tags": self._normalize_civitai_tags(link_info.get("tags", [])),
                    "show_images": bool(data.get("show_images", True)),
                }
                self.civitai_links.pop(file_name, None)
                self._save_civitai_overrides()
                self._save_civitai_links()
                return json_response({"ok": True, "lora_list": list(self._get("lora_list", []) or []), **(await self._lora_payload())})
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
                self.civitai_overrides[file_name] = {
                    "url": custom_url,
                    "name": custom_name,
                    "images": custom_images,
                    "trigger_words": self._normalize_trigger_words(link_info.get("trigger_words", [])),
                    "tags": self._normalize_civitai_tags(link_info.get("tags", [])),
                    "show_images": bool(data.get("show_images", True)),
                }

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
        from astrbot.api.web import json_response, request

        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        try:
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
            directory = self._model_folder_path("loras").resolve()
            target = (directory / relative).resolve()
            if directory not in target.parents or not target.is_file():
                return json_response({"error": "LoRA 路径无效"}, status_code=400)
            startfile = getattr(os, "startfile", None)
            if not callable(startfile):
                raise UsageError("当前系统不支持打开本地文件夹")
            await asyncio.to_thread(startfile, str(target.parent))
            return json_response({"ok": True, "path": str(target.parent), "file_name": filename})
        except (OSError, UsageError, ComfyError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

    async def api_upload_lora(self):
        from astrbot.api.web import json_response, request

        try:
            directory = self._model_folder_path("loras")
            directory.mkdir(parents=True, exist_ok=True)
            files = await self._request_files(request)
            upload = files.get("file")
            filename = Path(str(getattr(upload, "filename", "") or "")).name
            if not upload or not filename:
                return json_response({"error": "请选择 LoRA 文件"}, status_code=400)
            if Path(filename).suffix.lower() not in {".safetensors", ".pt", ".ckpt", ".bin"}:
                return json_response({"error": "只支持 safetensors、pt、ckpt、bin 文件"}, status_code=400)
            target = directory / filename
            if target.exists():
                return json_response({"error": f"文件已存在：{filename}，请先处理原文件"}, status_code=409)
            await upload.save(str(target))
            self._lora_names_cache = None
            return json_response({"ok": True, "file_name": filename, "path": str(target)})
        except (OSError, UsageError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

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
        """后台下载 LoRA 并持续记录字节数，避免 WebUI 请求被大文件阻塞。"""
        temp_path: Path | None = None
        try:
            self._update_download_job(job_id, status="running", stage="正在读取 CivitAI 模型信息")
            download_url, filename, version_id = await self._civitai_download_file(link)
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
            )
            headers = self._civitai_headers("application/octet-stream, */*;q=0.8")
            retry_statuses = {408, 429, 500, 502, 503, 504}
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60, read=180),
                follow_redirects=True,
            ) as client:
                for attempt in range(3):
                    current_url = download_url
                    if attempt:
                        # CivitAI 的 CDN 地址带短时签名；重试时重新获取，避免复用过期地址。
                        current_url, _, _ = await self._civitai_download_file(link)
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
            temp_path.replace(target)
            self.lora_download_order[filename] = time.time()
            self._save_lora_download_order()
            self._lora_names_cache = None
            self._update_download_job(
                job_id,
                status="done",
                stage="下载完成，等待 WebUI 刷新",
                progress=100,
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
            directory = self._model_folder_path("loras")
            target = (directory / relative).resolve()
            if directory.resolve() not in target.parents:
                return json_response({"error": "LoRA 路径无效"}, status_code=400)
            if not target.is_file():
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
        """读取本地 LoRA 文件名，管理操作不依赖 ComfyUI 的缓存列表。"""
        directory = self._model_folder_path("loras")
        if not directory.is_dir():
            return []
        allowed = {".safetensors", ".pt", ".ckpt", ".bin"}
        names = [
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        ]
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
        """按 ComfyUI 的 LoRA 相对路径找到本地文件。"""
        root = detect_comfyui_root(str(self._get("comfyui_root", "")))
        if not root:
            return None
        lora_root = Path(root) / "models" / "loras"
        relative = str(file_name or "").replace("\\", "/").strip(" /")
        if not relative:
            return None
        candidate = lora_root.joinpath(*[part for part in relative.split("/") if part not in {"", ".", ".."}])
        if candidate.is_file():
            return candidate
        # Windows 通常不区分大小写，但 ComfyUI 的 API 列表可能来自额外模型
        # 路径，做一次文件名回退，兼容旧配置写入的大小写或斜杠差异。
        target_name = Path(relative).name.casefold()
        try:
            for path in lora_root.rglob(Path(relative).name):
                if path.is_file() and path.name.casefold() == target_name:
                    return path
        except OSError:
            return None
        return None

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

    async def _persist_image_ref(self, image_ref: str) -> str:
        """将本地路径、file URI 或远程图片引用缓存到插件目录。"""
        value = str(image_ref or "").strip()
        if not value:
            return ""
        if os.path.isfile(value):
            return await self._persist_input_image(value)
        try:
            path = await Image(file=value).convert_to_file_path()
        except Exception as exc:
            logger.debug("[%s] 转换图片引用失败：%s：%s", PLUGIN_NAME, value[:160], exc)
            return ""
        return await self._persist_input_image(path)

    async def _extract_component_image(self, component: Any, depth: int = 0) -> str:
        """递归读取直接图片、引用链和合并转发节点中的第一张图片。"""
        if component is None or depth > 8:
            return ""
        if isinstance(component, Image):
            try:
                path = await component.convert_to_file_path()
            except Exception:
                return ""
            return await self._persist_image_ref(path)
        if isinstance(component, Reply):
            for nested in component.chain or []:
                cached = await self._extract_component_image(nested, depth + 1)
                if cached:
                    return cached
            return ""
        if isinstance(component, Node):
            for nested in component.content or []:
                cached = await self._extract_component_image(nested, depth + 1)
                if cached:
                    return cached
            return ""
        if isinstance(component, Nodes):
            for node in component.nodes or []:
                cached = await self._extract_component_image(node, depth + 1)
                if cached:
                    return cached
        return ""

    async def _extract_image(self, event: AstrMessageEvent) -> str:
        """提取消息或转发消息中的图片并立即持久化。"""
        components = list(event.get_messages() or [])
        logger.info(
            "[%s] 开始提取图生图输入图片：消息组件=%d；原文=%s",
            PLUGIN_NAME,
            len(components),
            str(getattr(event, "message_str", "") or "")[:120],
        )
        for component in components:
            cached = await self._extract_component_image(component)
            if cached:
                logger.info("[%s] 已缓存图生图输入图片：%s", PLUGIN_NAME, cached)
                return cached

        # 部分适配器会把图片只写入 event.image/image_list，而不放回
        # get_messages()。直接处理这两个标准字段，避免命令已经触发却拿不到图。
        direct_refs: list[str] = []
        direct_image = getattr(event, "image", None)
        if direct_image:
            direct_refs.append(str(direct_image))
        direct_refs.extend(str(value) for value in (getattr(event, "image_list", None) or []) if value)
        for image_ref in direct_refs:
            cached = await self._persist_image_ref(image_ref)
            if cached:
                logger.info("[%s] 已从 event.image/image_list 缓存图生图输入图片：%s", PLUGIN_NAME, cached)
                return cached

        # AstrBot 对“只有 reply id、正文是 [转发消息]”的消息提供了远程兜底，
        # 可以继续调用 get_msg/get_forward_msg 拉取转发节点里的图片 URL。
        try:
            from astrbot.core.utils.quoted_message import extract_quoted_message_images

            for image_ref in await extract_quoted_message_images(event):
                cached = await self._persist_image_ref(image_ref)
                if cached:
                    logger.info("[%s] 已从引用/转发消息缓存图生图输入图片：%s", PLUGIN_NAME, cached)
                    return cached
        except Exception as exc:
            logger.warning("[%s] 解析引用/转发图片失败：%s", PLUGIN_NAME, exc)
        logger.warning("[%s] 未找到可用的图生图输入图片", PLUGIN_NAME)
        return ""

    async def _generate(self, event: AstrMessageEvent, params: DrawParams, mode: str, image_path: str = "") -> list[Path]:
        logger.info(
            "[%s] 进入生成流程：模式=%s；输入图片=%s；提示词=%s",
            PLUGIN_NAME,
            MODE_NAMES.get(mode, mode),
            image_path or "无",
            str(params.prompt or "")[:160],
        )
        if mode == "img2img" and not image_path:
            raise UsageError("图生图没有拿到输入图片，已阻止提交空白图任务；请在同一条消息附图或回复图片")
        client = self._client()
        env = await self._environment(client)
        model = ""
        if mode != "img2img":
            model = self._model_name(env, params.model, mode)
        # 临时简称 LoRA 与 WebUI 中长期启用的 LoRA 合并；同一文件以临时权重为准。
        # 图生图的长期 LoRA 在 img2img_lora_name 中单独配置；本次指令写入的
        # LoRA 仍可临时覆盖它，文生图的长期 LoRA 不会串入 Qwen 工作流。
        configured_img2img_loras: list[str] = []
        if mode == "img2img":
            configured_img2img_lora = str(self._get("img2img_lora_name", "") or "").strip()
            if configured_img2img_lora:
                configured_img2img_loras.append(
                    f"{configured_img2img_lora}:{float(self._get('img2img_lora_strength', 0.8) or 0.0):g}"
                )
        requested_loras = (
            configured_img2img_loras + list(params.loras)
            if mode == "img2img"
            else list(self._get("lora_list", []) or []) + list(params.loras)
        )
        loras: list[str] = []
        resolved_loras: dict[str, str] = {}
        for item in requested_loras:
            name, separator, weight = str(item).rpartition(":")
            if not separator or not re.fullmatch(r"\d+(?:\.\d+)?", weight):
                name, weight = str(item), "0.8"
            actual = self._resolve_lora(name, list(env["loras"]))
            if not actual:
                raise UsageError(f"LoRA 不存在或昵称未设置：{name}；请先使用 /lora 列表")
            resolved_loras[actual.casefold()] = f"{actual}:{weight}"
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
                params.denoise or self._get("img2img_denoise", 1.0),
            )
        configured_seed = self._get("img2img_seed", -1) if mode == "img2img" else self._get("seed", -1)
        try:
            configured_seed = int(configured_seed)
        except (TypeError, ValueError):
            configured_seed = -1
        seed = params.seed if params.seed >= 0 else configured_seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        width = params.width or int(self._get("img2img_width", 512) if mode == "img2img" else self._get("width", 832) or 832)
        height = params.height or int(self._get("img2img_height", 960) if mode == "img2img" else self._get("height", 1216) or 1216)
        if mode == "img2img" and image_path:
            width, height = self._img2img_output_size(image_path, width, height)
        scale = params.scale or float(self._get("hires_scale", 2.0) or 2.0)
        workflow = load_workflow(self._workflow_path(mode))
        is_qwen_img2img = mode == "img2img" and all(
            node_id in workflow
            for node_id in ("1", "10", "11", "13", "14", "30", "39", "119", "152", "158", "160", "161")
        )
        if mode == "img2img" and not is_qwen_img2img:
            raise WorkflowError(
                "当前图生图工作流不是 Qwen Image Edit 工作流；请在 WebUI 重新加载 E:\\112121121.json"
            )
        image_name = ""
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
        else:
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
                env["text_encoders"],
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
            patched, report = adapt_qwen_img2img(
                workflow,
                positive=positive,
                negative=negative,
                image_name=image_name,
                unet_name=qwen_unet,
                clip_name=qwen_clip,
                vae_name=qwen_vae,
                lora_name=configured_img2img_lora,
                lora_strength=float(self._get("img2img_lora_strength", 0.8) or 0.0),
                loras=prompt_loras,
                width=width,
                height=height,
                steps=params.steps or int(self._get("img2img_steps", 4) or 4),
                cfg=params.cfg or float(self._get("img2img_cfg", 1.0) or 1.0),
                seed=seed,
                sampler_name=params.sampler or str(self._get("img2img_sampler_name", "euler") or "euler"),
                scheduler=params.scheduler or str(self._get("img2img_scheduler", "simple") or "simple"),
                denoise=params.denoise or float(self._get("img2img_denoise", 1.0) or 1.0),
                scale_method=str(self._get("img2img_scale_method", "lanczos") or "lanczos"),
                largest_size=max(
                    64,
                    int(self._get("img2img_largest_size", max(width, height, 64)) or max(width, height, 64)),
                ),
                crop=str(self._get("img2img_crop", "center") or "center"),
                filename_prefix=str(self._get("img2img_filename_prefix", "astrbot/img2img_qwen") or "astrbot/img2img_qwen"),
            )
            logger.info(
                "[%s] Qwen 节点1 prompt 已写入：%s；节点39 negative=%s；节点152 denoise=%s",
                PLUGIN_NAME,
                str(patched.get("1", {}).get("inputs", {}).get("prompt", ""))[:500],
                str(patched.get("39", {}).get("inputs", {}).get("prompt", ""))[:300],
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

    def _draw_start_reply(self, mode: str, prompt: str) -> str:
        fallback = f"{MODE_NAMES[mode]}任务已提交，生成期间可以继续聊天，完成后会发送结果。"
        template = str(self._get("draw_start_reply", "") or "")
        return self._format_reply_template(
            template,
            {"mode": MODE_NAMES[mode], "prompt": prompt or ""},
            fallback,
        )

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
        fallback = self._custom_draw_reply(mode, count, params.prompt)
        source = str(self._get("draw_reply_mode", "astrbot") or "astrbot").lower()
        if source == "custom":
            return fallback
        default_user_prompt = (
            f"绘画类型：{MODE_NAMES[mode]}\n"
            f"用户原始需求：{params.prompt or '用户未提供文字描述'}\n"
            f"本次实际发送图片数量：{count}\n"
            "请写一句自然的中文回复，结合用户需求描述画面已经完成；不要复述固定的完成模板。"
        )
        user_prompt = default_user_prompt
        try:
            if source in {"astrbot", "plugin"}:
                timeout_value = self._get("draw_reply_timeout", 6)
                try:
                    timeout = max(0.0, float(timeout_value or 0))
                except (TypeError, ValueError):
                    timeout = 6.0
                request = self._astrbot_generate(
                    event,
                    user_prompt,
                    system_prompt=None,
                    max_tokens=80,
                    use_event_context=True,
                )
                if timeout <= 0:
                    return fallback
                return await asyncio.wait_for(request, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[%s] 完成回复超过 %.1f 秒，已先发送图片并使用固定回复", PLUGIN_NAME, timeout)
        except Exception as exc:
            logger.warning("[%s] %s 完成回复失败，已回退自定义内容：%s", PLUGIN_NAME, source, exc)
        return fallback

    async def _run(self, event: AstrMessageEvent, params: DrawParams, mode: str, image_path: str = "") -> list:
        semaphore = self._build_semaphore()
        async with semaphore:
            try:
                images = await self._generate(event, params, mode, image_path)
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
                if image_path:
                    try:
                        cached_path = Path(image_path).resolve()
                        cache_root = self.input_cache_dir.resolve()
                        if cached_path.parent == cache_root:
                            cached_path.unlink(missing_ok=True)
                    except (OSError, RuntimeError):
                        logger.debug("[%s] 清理输入图片缓存失败：%s", PLUGIN_NAME, image_path)
            reply = await self._draw_reply(event, params, mode, len(images))
            if self._get("draw_attach_prompt", False):
                prompt_attachment = self._prompt_attachment(params)
                if prompt_attachment:
                    reply = f"{reply}\n\n{prompt_attachment}"
            if str(self._get("draw_delivery_mode", "normal") or "normal").lower() == "forward":
                return self._forward_result_chain(event, reply, images)
            result = [Image.fromFileSystem(str(path)) for path in images]
            result.append(Plain(reply))
            quote = self._completion_quote(event)
            if quote:
                result.insert(0, quote)
            return result

    def _forward_result_chain(
        self,
        event: AstrMessageEvent,
        reply: str,
        images: list[Path],
    ) -> list:
        """把完成回复和每张图片组织成群聊常见的合并转发消息。"""
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
        nodes.append(Node(uin=uin, name="ComfyUI 绘图", content=[Plain(reply)]))
        chain = [Nodes(nodes)]
        quote = self._completion_quote(event)
        return [quote, *chain] if quote else chain

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
            max_bytes = 4 * 1024 * 1024
            if source.stat().st_size <= max_bytes:
                return source

            from PIL import Image as PILImage

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            target = cache_dir / f"forward_{uuid.uuid4().hex}.jpg"
            with PILImage.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((2048, 2048), PILImage.Resampling.LANCZOS)
                quality = 86
                while True:
                    image.save(target, format="JPEG", quality=quality, optimize=True)
                    if target.stat().st_size <= max_bytes or quality <= 62:
                        break
                    quality -= 8
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

    async def _background_draw(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        mode: str,
        image_path: str = "",
    ) -> None:
        chain = None
        try:
            chain = await self._run(event, params, mode, image_path)
            if any(isinstance(component, Nodes) for component in chain):
                # 引用消息和合并转发在 OneBot 中是两个发送动作。这样只重试
                # 合并转发本身，不会在失败重试时重复引用用户原消息。
                for component in chain:
                    if isinstance(component, Nodes):
                        await self._send_forward_component(event, component)
                    else:
                        await event.send(event.chain_result([component]))
            else:
                await event.send(event.chain_result(chain))
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
    ):
        """为 LLM 工具提交绘图任务，并由插件后台可靠发送最终图片。

        AstrBot 的本地 LLM 工具结果会经过 ``tool_direct_result`` 包装。部分
        平台适配器会把其中的 Nodes/图片当成工具结果处理，导致文本能看到但
        图片丢失。因此 LLM 工具只返回“已提交”状态，实际结果统一走插件的
        普通发送路径；这样也保留了“画图期间可以继续聊天”。
        """
        task = asyncio.create_task(self._run(event, params, mode, image_path))
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
            if (
                chain
                and params is not None
                and (
                    self._img2img_plugin_ai_debug_enabled()
                    if params.mode == "img2img"
                    else self._plugin_ai_debug_enabled()
                )
                and getattr(params, "plugin_ai_debug_prompt", "")
            ):
                chain.append(Plain(self._llm_debug_reply(
                    "",
                    str(getattr(params, "plugin_ai_debug_prompt", "") or ""),
                    str(getattr(params, "plugin_ai_debug_input", "") or ""),
                )))
            if any(isinstance(component, Nodes) for component in chain):
                for component in chain:
                    if isinstance(component, Nodes):
                        await self._send_forward_component(event, component)
                    else:
                        await event.send(event.chain_result([component]))
            else:
                await event.send(event.chain_result(chain))
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
    ) -> None:
        task = asyncio.create_task(self._background_draw(event, params, mode, image_path))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _help_text(self) -> str:
        """帮助图片不可用时的文字版兜底内容。"""
        return (
            "ComfyUI AI 绘画台指令\n"
            "/文生图 描述\n"
            "/图生图 描述（同一条消息附图片）\n"
            "/高清放大（附图片、回复图片，或放大最近出图）\n"
            "在画图指令中单独写 ai 可开启本次 AI 提示词优化，例如：/文生图 ai 夏空；ai 不会进入最终提示词。也可写 noai 或继续使用 --ai=开、--noai。\n"
            "画图指令末尾可直接写 LoRA 指令简称临时加载，例如：/文生图 夏空海边 1号lora；也支持 1号lora:0.6，任务结束后不会保存为默认 LoRA\n"
            "如果 LoRA 指令简称与普通提示词预设同名，直接写名称会同时启用 LoRA 和预设；只使用预设可写 --预设=名称\n"
            "/模型 列表 或 /模型 名称\n"
            "/lora 或 /.lora 查看全部 LoRA 总览图片（含每个 LoRA 一张预览图、分类、简称和开启状态）；/lora 1号lora 查看单个 LoRA；/loraon 1号lora 开启 LoRA；/lora 添加或删除 指令简称\n"
            "/预设 列表 查看按 LoRA 分类的指令简称与预设图片；/预设 添加 夏空=ciaccona；/预设 修改 夏空=新内容；/预设 翻译 夏空；/预设 删除 夏空\n"
            "/画师串 列表；/画师串 使用 画风001；/画师串 添加 名称=画师 tags；/画师串 关闭\n"
            "/工作流 列表；/工作流 文生图 文件名；/工作流 图生图 文件名；/工作流 高清放大 文件名\n"
            "/画图配置 查询模型位置、工作流位置和当前配置\n"
            "/comfy状态 查询 ComfyUI 状态；管理员可用 /comfy打开 启动本机 ComfyUI、/comfy关闭 关闭服务\n"
            "直接用自然语言要求 AstrBot 画图时，会自动识别提示词预设和 LoRA 指令简称；预设与 LoRA 可同时生效，其余内容作为画面描述。\n"
            "生成结果的发送方式可在 WebUI 回复设置中选择普通消息或群聊合并转发；不支持合并转发的平台会自动退回普通消息。\n"
            "鸣潮角色知识和 Anima 提示词工程师由独立插件提供；本插件只负责绘图执行，并保留临时 LoRA 和提示词预设的最高优先级。\n"
            "绘画选项：--模型=名称 --lora=a:0.8,b:0.5 --预设=夏空 --画师=画风001 --负面=内容 --宽=832 --高=1216 --步数=24 --种子=-1 --ai=开 --noai --强度=0.6 --放大=2"
        )

    @staticmethod
    def _list_image_font_paths() -> list[str]:
        return [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    @classmethod
    def _list_image_font(cls, size: int, bold: bool = False):
        from PIL import ImageFont

        paths = cls._list_image_font_paths()
        if bold:
            paths = paths[1:] + paths[:1]
        for font_path in paths:
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

    async def _preset_list_image_path(self) -> Path:
        """生成按 LoRA 分组的简称与专属预设长图。"""
        from PIL import Image as PILImage
        from PIL import ImageDraw

        available = await self._local_lora_names()
        names = self._order_loras(available or self._known_lora_preset_keys())
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
        draw.text((margin, 100), f"按 LoRA 分类 · 共 {sum(len(item[2]) for item in groups)} 条 · 插件 v{PLUGIN_VERSION}", font=footer_font, fill=(164, 187, 190))
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

        font_paths = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        def load_font(size: int, bold: bool = False):
            ordered = font_paths[1:] + font_paths[:1] if bold else font_paths
            for font_path in ordered:
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
                    "/模型 列表 或 /模型 名称",
                    "/工作流 列表；/工作流 文生图 文件名",
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
                    "可上传 LoRA，也可用 CivitAI 链接下载并查看进度",
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
        draw.text((margin, 48), "ComfyUI AI 绘画台", font=title_font, fill=(238, 248, 246))
        draw.text(
            (margin, 116),
            f"AstrBot 绘图指令速查 · 插件版本 {PLUGIN_VERSION}",
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

    @filter.command("helpd", desc="查询 ComfyUI AI 绘画指令")
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
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        self._schedule_draw(event, params, "txt2img")
        yield event.plain_result(self._draw_start_reply("txt2img", params.prompt))

    @filter.command("图生图", alias=["img2img", "局部重绘", "inpaint"], desc="使用 Qwen Image Edit 工作流进行图生图或参考图编辑")
    async def command_img2img(self, event: AstrMessageEvent, prompt: GreedyStr):
        logger.info(
            "[%s] 图生图命令已触发：原文=%s；消息图片字段=%s；图片数量=%s；回复=%s",
            PLUGIN_NAME,
            str(getattr(event, "message_str", "") or "")[:160],
            bool(getattr(event, "image", None)),
            len(getattr(event, "image_list", None) or []),
            getattr(event, "reply", None) or getattr(event, "reply_id", None) or "无",
        )
        try:
            params = fill_params(str(prompt), "img2img")
            await self._extract_inline_loras(params)
            self._extract_inline_presets(params)
        except (UsageError, ComfyError) as exc:
            yield event.plain_result(f"用法错误：{exc}")
            return
        image_path = await self._extract_image(event)
        if not image_path:
            logger.warning("[%s] 图生图命令已触发，但未提取到输入图片", PLUGIN_NAME)
            yield event.plain_result("图生图需要在同一条消息附图，或回复一张图片后发送指令")
            return
        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        logger.info("[%s] 图生图命令已拿到输入图片，开始后台提交：%s", PLUGIN_NAME, image_path)
        self._schedule_draw(event, params, "img2img", image_path)
        yield event.plain_result(self._draw_start_reply("img2img", params.prompt))

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
        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            yield event.plain_result(f"生成失败：{exc}")
            return
        self._schedule_draw(event, params, "hires", image_path)
        yield event.plain_result(self._draw_start_reply("hires", params.prompt))

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
        if not value or value == "列表":
            try:
                image_path = await self._preset_list_image_path()
                yield event.chain_result([Image.fromFileSystem(str(image_path))])
            except Exception as exc:
                logger.exception("[%s] 生成预设列表图片失败", PLUGIN_NAME)
                lines = ["提示词预设："]
                lines.extend(
                    f"{name} = {item.get('translated') or item.get('content', '')}"
                    for name, item in self.presets.items.items()
                    if isinstance(item, dict)
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
        yield event.plain_result("用法：/预设 列表、添加、修改 名称=内容、翻译 名称、删除 名称")

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
        source_workflow = str(self._get("source_workflow", r"E:\难工作流.json"))
        yield event.plain_result(
            "当前绘画配置\n"
            f"ComfyUI 地址：{self._get('comfyui_url', '')}\n"
            f"核心模型：{self._get('model_name', '')}\n"
            f"LoRA：{', '.join(self._get('lora_list', []) or []) or '无'}\n"
            f"AI 服务：{self._get('ai_model', '') or '未配置'}\n"
            f"ComfyUI 根目录：{root or '未检测到'}\n"
            f"核心模型目录：{model_dir(root, 'diffusion_models')}\n"
            f"Checkpoint 模型目录：{model_dir(root, 'checkpoints')}\n"
            f"LoRA 目录：{model_dir(root, 'loras')}\n"
            f"工作流目录：{self._workflow_dir()}\n"
            f"原始工作流：{source_workflow}\n"
            f"输出目录：{self.output_dir}"
        )

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
        if ai is not None:
            add_option("ai", "开" if ai else "关")
        params = fill_params(raw, mode)
        # 默认直接使用 AstrBot LLM 传入的 prompt、preset 和 lora；如果 WebUI
        # 选择插件 AI，_llm_execute 会在后台绘图前用用户原话重新生成一次 prompt。
        # 这里先关闭旧的 ai 字段，避免同一条任务被普通指令翻译逻辑二次处理。
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

    async def _prepare_llm_prompt(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        original_text: str,
        mode: str = "txt2img",
    ) -> str:
        """按模式选择对应的 LLM 提示词整理器。"""
        if mode == "img2img":
            # 图生图只需告诉 Qwen Image Edit 要改什么。预设和 LoRA 已在外层
            # 锁定，不能交给 AI 改写或删除。
            source = (
                "plugin_img2img_llm"
                if self._img2img_llm_plugin_ai_enabled()
                else "astrbot_img2img_llm"
            )
            # 只把用户原始编辑要求交给图生图 AI。AstrBot 提取出的长画面描述
            # 可能含有原图的角色、服装或场景，传入后会诱导模型错误扩写。
            source_text = self._remove_locked_control_terms(original_text, params)
            if not source_text:
                source_text = self._remove_locked_control_terms(params.prompt, params)
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
                fallback, plain_note = await self._plain_translate_prompt(source_text)
                params.prompt = fallback
                params.img2img_edit_instruction = fallback
                params.img2img_edit_instruction_ready = True
                params.plugin_ai_fallback_note = (
                    f"图生图插件 AI 未返回有效编辑指令，已按用户原话继续（{exc}）"
                    + (f"；{plain_note}" if plain_note else "")
                )
                logger.warning("[%s] 图生图 LLM 插件 AI 失败，已回退原编辑要求：%s", PLUGIN_NAME, exc)
                return fallback
            params.prompt = translated
            params.img2img_edit_instruction = translated
            params.plugin_ai_debug_prompt = translated
            params.img2img_edit_instruction_ready = True
            logger.info(
                "[%s] 图生图 LLM 编辑指令已生成：来源=%s；指令=%s",
                PLUGIN_NAME,
                "插件 AI" if source == "plugin_img2img_llm" else "AstrBot AI",
                translated,
            )
            return translated

        """文生图按 WebUI 选择决定是否由插件 AI 生成 LLM 提示词。"""
        if not self._llm_plugin_ai_enabled():
            return ""
        source_text = self._llm_prompt_input(original_text, params.prompt)
        source_text = self._remove_locked_control_terms(source_text, params)
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
            fallback = str(params.prompt or original_text or "").strip()
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
    ) -> str:
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
            original_text = str(getattr(event, "message_str", "") or "")
            self._extract_inline_presets(params, original_text)
            await self._extract_inline_loras(params, original_text)
            self._extract_inline_presets(params, original_text)
        except ComfyError as exc:
            return f"{MODE_NAMES[mode]}失败：无法读取 LoRA 列表：{exc}"
        if mode == "txt2img" and not params.prompt and not params.presets and not params.loras:
            return "文生图失败：请提供画面描述、提示词预设或 LoRA 指令简称。"

        image_path = ""
        if require_image:
            image_path = await self._extract_image(event)
            if not image_path:
                origin = self._origin(event)
                image_path = self.last_images.get(origin, "")
            if not image_path:
                logger.warning("[%s] LLM 图生图未找到输入图片或最近出图", PLUGIN_NAME)
                return f"{MODE_NAMES[mode]}需要用户附图、回复图片，或已有最近出图。"
            logger.info("[%s] LLM 图生图已取得输入图片：%s", PLUGIN_NAME, image_path)

        try:
            await self._reserve_draw_limit(event, params)
        except UsageError as exc:
            return f"{MODE_NAMES[mode]}失败：{exc}"

        start_reply = self._draw_start_reply(mode, params.prompt)
        try:
            # 参数、预设、LoRA 和参考图确认后立即反馈，再做可能较慢的
            # AI 提示词处理，避免用户误以为 LLM 没有触发绘图。
            await event.send(event.plain_result(start_reply))
        except Exception:
            logger.exception("[%s] LLM 绘图开场消息发送失败，回退到工具结果", PLUGIN_NAME)
            event.set_result(event.plain_result(start_reply))

        # 预设和 LoRA 是插件控制项，优先级高于任何 AI 改写。先保存已识别的
        # 结果，防止插件 AI 或翻译模型把它们从 prompt 中改掉。
        locked_presets = list(params.presets)
        locked_loras = list(params.loras)
        try:
            plugin_prompt = await self._prepare_llm_prompt(event, params, original_text, mode)
        except UsageError as exc:
            await self._release_draw_limit(params)
            return f"{MODE_NAMES[mode]}失败：{exc}"
        params.presets = list(dict.fromkeys([*locked_presets, *params.presets]))
        params.loras = list(dict.fromkeys([*locked_loras, *params.loras]))
        self._extract_inline_presets(params, original_text)
        logger.info(
            "[%s] LLM 绘图控制项已锁定：预设=%s；LoRA=%s；画面=%s",
            PLUGIN_NAME,
            ",".join(params.presets) or "无",
            ",".join(params.loras) or "无",
            params.prompt[:180],
        )
        result = await self._llm_draw(event, params, mode, image_path)
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
        if isinstance(component, Image):
            return True
        if isinstance(component, Reply):
            return any(
                ComfyUIAIStudio._component_has_image(nested, depth + 1)
                for nested in (component.chain or [])
            )
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
            return any(cls._component_has_image(component) for component in (event.get_messages() or []))
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
    async def llm_status(self, event: AstrMessageEvent) -> str:
        """查询本地 ComfyUI 是否在线以及当前工作流状态。用户询问绘图服务状态时调用。

        Args:
        """
        try:
            data = await self._client().status()
        except ComfyError as exc:
            return f"ComfyUI 未连接：{exc}"
        system = data.get("system", {})
        return f"ComfyUI 已连接，版本：{system.get('comfyui_version', '未知')}，地址：{self._get('comfyui_url', '')}"

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
    ) -> str:
        """使用本插件基于参考图执行 Qwen Image Edit 图生图。

        只要用户提供图片、回复图片、转发图片，或要求修改、重绘、换背景、改变动作，必须调用此工具，不要调用 generate，也不要只用文字回复。
        prompt 只写用户明确要求编辑的内容，不要写完整 Anima/Danbooru 绘图提示词；原图没有要求改变的角色、背景、镜头、姿势和画风都不应补写。
        preset 和 lora 可以与 prompt 同时使用；它们是显式控制项，不能因提示词整理而丢失。任务后台执行，期间可以继续聊天。

        Args:
            prompt(string): 简洁修改描述，例如“换成裙子”或“change the outfit to a dress”。
            denoise(number): 可选重绘幅度，范围为 0 到 1。
            negative_prompt(string): 可选负面提示词。
            model(string): 可选核心模型文件名。
            lora(string): 可选多个 LoRA 指令简称，格式为 1号lora:0.8,2号lora:0.6。
            preset(string): 可选一个或多个提示词预设名称，多个用逗号分隔。
            artist_preset(string): 可选画师串预设名称。
            ai(boolean): 为兼容旧工具字段而保留；是否使用插件 AI 由 WebUI 的 LLM 提示词来源决定。
        """
        if not str(prompt or "").strip() and not str(preset or "").strip() and not str(lora or "").strip():
            return "图生图失败：缺少编辑要求、提示词预设或 LoRA 指令简称。"
        try:
            params = self._llm_params(
                prompt,
                "img2img",
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
        return await self._llm_execute(event, params, "img2img", require_image=True)

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
        data["workflow_selected"] = {mode: self._workflow_path(mode).name for mode in MODE_NAMES}
        data["plugin_ai_anima_context_preview"] = str(
            self._get("plugin_ai_anima_context", "") or build_anima_context(self.plugin_dir, 6000)
        )
        template_path = find_template_path(self.plugin_dir)
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
        return json_response({"version": f"v{PLUGIN_VERSION}", "comfy": comfy, "config": self._safe_config(), "draw_limit": self._draw_limit_status(), "paths": {
            "comfyui_root": root, "diffusion_models": model_dir(root, "diffusion_models"),
            "checkpoints": model_dir(root, "checkpoints"), "loras": model_dir(root, "loras"),
            "upscale_models": model_dir(root, "upscale_models"), "unet": model_dir(root, "unet"),
            "text_encoders": model_dir(root, "text_encoders"), "vae": model_dir(root, "vae"),
            "workflow_dir": str(self._workflow_dir()),
            "source_workflow": str(self._get("source_workflow", r"E:\难工作流.json")),
            "output_dir": str(self.output_dir), "data_dir": str(self.data_dir),
        }})

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
                "img2img_clip_models": env["text_encoders"],
                "img2img_vae_models": env["vae"],
                "current_img2img": self._get("img2img_unet_name", QWEN_IMG2IMG_UNET_DEFAULT),
                "current_img2img_clip": self._get("img2img_clip_name", QWEN_IMG2IMG_CLIP_DEFAULT),
                "current_img2img_vae": self._get("img2img_vae_name", QWEN_IMG2IMG_VAE_DEFAULT),
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
                if key in {"ai_api_key", "civitai_token"} and not str(value or "").strip():
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
