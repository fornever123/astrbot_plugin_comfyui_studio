from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import random
import re
import shutil
import struct
import time
import uuid
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Node, Nodes, Plain, Reply
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .ai import DANBOORU_SYSTEM_PROMPT, AIError, AITranslator
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
    adapt_original,
    copy_and_repair_original,
    load_api_workflow,
)

PLUGIN_NAME = "astrbot_plugin_comfyui_ai_studio"
PLUGIN_VERSION = "0.6.8"
DEFAULT_ARTIST_PRESET_NAME = "画风001"
DEFAULT_ARTIST_PRESET_TAGS = (
    "@yukisiannn, @kani biimu, @ixy, @shnva, @shiromochi sakura, @stmast,"
)
MODE_NAMES = {"txt2img": "文生图", "img2img": "图生图", "hires": "高清放大"}
MODE_ALIASES = {"文生图": "txt2img", "图生图": "img2img", "高清放大": "hires", "高清": "hires"}
MODE_FILES = {"txt2img": "文生图.json", "img2img": "图生图.json", "hires": "高清放大.json"}
LLM_TOOL_NAMES = {
    "status": "comfyui_ai_studio_status",
    "generate": "comfyui_ai_studio_generate",
    "edit": "comfyui_ai_studio_edit",
    "upscale": "comfyui_ai_studio_upscale",
}
WRITABLE_CONFIG = {
    "comfyui_url", "comfyui_root", "source_workflow", "workflow_dir", "model_name",
    "workflow_txt2img", "workflow_img2img", "workflow_hires",
    "lora_list", "default_positive", "default_negative", "quality_prefix", "artist_preset", "ai_base_url",
    "ai_api_key", "ai_model", "civitai_token", "width", "height", "steps", "cfg", "seed",
    "sampler_name", "scheduler", "denoise", "hires_scale",
    "hires_steps", "hires_denoise", "hires_upscale_model", "max_concurrent",
    "anima_teacache", "draw_start_reply", "draw_reply_mode", "draw_reply_custom", "draw_delivery_mode",
    "draw_reply_timeout", "plain_translate_enabled", "plain_translate_url",
    "llm_prompt_source", "plugin_ai_command_system_prompt", "plugin_ai_llm_system_prompt",
    "plugin_ai_debug",
}

DEFAULT_PLUGIN_AI_LLM_SYSTEM_PROMPT = """你是 ComfyUI Anima 工作流的提示词工程师，负责把用户的完整绘图要求转换成可直接用于 Anima 的英文 Danbooru 标签。
你会同时收到用户原话和 AstrBot LLM 提取的画面描述。必须以用户原话为最高优先级，补回 LLM 遗漏的关键内容。
完整保留并具体表达：角色、主体、动作、姿势、正在进行的行为、表情、服装、镜头、构图、场景、时间、天气和光线。
例如用户说“洗澡的夏空”，必须输出与洗澡动作相关的标签，不能只输出角色名；用户说“坐在浴缸里洗澡”，必须保留坐姿、浴缸和洗澡行为。
不要把“帮我画一张”“请生成图片”等聊天套话写进提示词，也不要臆造用户没有要求的角色、服装或场景。
角色名、作品名优先转换为稳定的 Danbooru 标签；已经存在的英文标签保留且不要重复。
输出顺序遵循：人数与性别、角色与作品、外观、服装与状态、动作与姿势、表情、镜头与构图、场景环境、细节氛围。
只输出一行小写英文、逗号分隔的最终提示词，不要解释、Markdown、代码块、质量词、画师名、LoRA 语法或权重语法。用户原话中用于控制预设和 LoRA 的名称由绘图插件单独处理，不要因为翻译而删除或改写它们。"""
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
        # LoRA 独立预设与旧版全局 presets.json 分开保存，旧预设永远不迁移、不覆盖。
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
        self._civitai_cache_lock = asyncio.Lock()
        self.last_images: dict[str, str] = {}
        self.semaphore: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._download_jobs: dict[str, dict[str, Any]] = {}
        self._lora_names_cache: tuple[float, list[str]] | None = None
        self._register_web_api()

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
        self._write_map(self.lora_presets_path, self.lora_presets)

    def _save_lora_download_order(self) -> None:
        self._write_map(self.lora_download_order_path, self.lora_download_order)

    def _set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def _save_config(self) -> None:
        saver = getattr(self.config, "save_config", None)
        if callable(saver):
            saver()

    @staticmethod
    async def _request_json(request_obj: Any, default: Any = None) -> Any:
        """兼容 AstrBot 新旧 Web API 的 JSON 请求读取方式。"""
        try:
            reader = getattr(request_obj, "json")
            value = reader(default=default) if callable(reader) else reader
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
        )
        self._lora_names_cache = (time.monotonic(), list(categories[2]))
        return {
            "diffusion_models": categories[0],
            "checkpoints": categories[1],
            "loras": categories[2],
            "upscale_models": categories[3],
            "classes": classes,
        }

    async def _available_loras(self, *, force: bool = False) -> list[str]:
        """短时间缓存 LoRA 列表，避免每条绘图指令都等待 ComfyUI 查询。"""
        now = time.monotonic()
        if not force and self._lora_names_cache is not None:
            cached_at, names = self._lora_names_cache
            if now - cached_at < 30:
                return list(names)
        names = await self._client().models("loras")
        self._lora_names_cache = (now, list(names))
        return list(names)

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
        if source == "plugin_llm":
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
        if source == "astrbot":
            result = await self._astrbot_generate(
                event,
                f"画面描述：{text}",
                system_prompt=system_prompt,
            )
            return result
        ai_config = self._config_dict()
        result = await AITranslator(ai_config).generate(
            text,
            system_prompt=system_prompt,
            max_tokens=768 if source == "plugin_llm" else 512,
        )
        return result

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

    async def _prompt_text(
        self,
        event: AstrMessageEvent,
        params: DrawParams,
        lora_trigger_words: list[str] | None = None,
        lora_prompt_values: list[str] | None = None,
    ) -> tuple[str, str, str]:
        # 预设必须在任何翻译或扩写前识别，避免 AI 把预设名称翻译成普通描述。
        self._extract_inline_presets(params)
        text = self.presets.expand(params.prompt)
        preset_values: list[str] = []
        trigger_values = self._normalize_trigger_words(lora_trigger_words or [])
        trigger_keys = {value.casefold() for value in trigger_values}
        for name in params.presets:
            value = self.presets.effective(name)
            if not value:
                # CivitAI 触发词是独立配置，可能没有同名提示词预设；
                # 这时仍然允许它和其它预设、LoRA 一起使用。
                if str(name or "").strip().casefold() in trigger_keys:
                    continue
                raise UsageError(f"预设不存在：{name}")
            preset_values.append(value)
        use_ai = params.auto_ai or params.ai is True
        source = str(params.ai_source or ("astrbot" if params.auto_ai else "plugin")).lower()
        note = ""
        if use_ai and (contains_chinese(text) or params.ai is True):
            try:
                text = await self._translate_prompt(
                    event,
                    text,
                    force_enabled=params.ai is True or params.auto_ai,
                    source_override=source,
                )
                note = (
                    "提示词已由插件 AI 根据 LLM 绘图要求整理为 Danbooru 标签"
                    if source == "plugin_llm"
                    else "提示词已由插件 AI 优化为 Danbooru 标签并修正角色词条"
                    if source == "plugin"
                    else "提示词已由 AstrBot 当前 AI 优化为 Danbooru 标签并修正角色词条"
                )
            except AIError as exc:
                raise UsageError(f"AI 翻译失败：{exc}；可使用 --noai 直接出图") from exc
        elif contains_chinese(text):
            text, plain_note = await self._plain_translate_prompt(text)
            note = plain_note
        artist_name = str(
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
        quality = str(
            self._get("default_positive", "")
            or self._get("quality_prefix", "")
            or ""
        ).strip()
        # LoRA 的触发词和专属预设只在 LoRA 实际启用时加入。
        # 普通 CivitAI tag 不再进入提示词；旧版全局预设仍按原顺序保留。
        fragments = [quality, artist_text, *preset_values, *(lora_prompt_values or [])]
        combined = ", ".join(value.strip(" ,，") for value in fragments if value.strip(" ,，"))
        for trigger in trigger_values:
            if trigger.casefold() not in combined.casefold() and trigger.casefold() not in text.casefold():
                fragments.append(trigger)
                combined = f"{combined}, {trigger}" if combined else trigger
        fragments.append(text)
        positive = ", ".join(
            value.strip(" ,，")
            for value in fragments
            if value.strip(" ,，")
        )
        negative = params.negative or str(self._get("default_negative", "") or "")
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
        raw = self._lora_map_value(self.lora_command_aliases, file_name, [])
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
        if raw != result:
            self.lora_command_aliases[file_name] = result
        return result

    def _lora_command_aliases_for(self, file_name: str, available: list[str] | None = None) -> list[str]:
        aliases = self._lora_command_aliases(file_name)
        if aliases:
            return aliases
        names = sorted(available or [file_name], key=str.casefold)
        try:
            index = names.index(file_name) + 1
        except ValueError:
            index = 1
        return [f"{index}号lora"]

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
        """读取已缓存的 CivitAI 触发词；触发词不依赖提示词预设存在。"""
        candidates = [file_name, Path(file_name).name]
        for key in candidates:
            override = self.civitai_overrides.get(key)
            if isinstance(override, dict) and "trigger_words" in override:
                return self._normalize_trigger_words(override.get("trigger_words", []))
            cached = self.civitai_cache.get(key)
            if not isinstance(cached, dict):
                continue
            data = cached.get("data", cached)
            if isinstance(data, dict) and "trigger_words" in data:
                return self._normalize_trigger_words(data.get("trigger_words", []))
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

    def _lora_preset_entries(self, file_name: str) -> list[dict[str, str]]:
        """读取单个 LoRA 的 tag -> 预设内容映射，并兼容早期实验格式。"""
        mapping = getattr(self, "lora_presets", {})
        raw = self._lora_map_value(mapping, file_name, []) if isinstance(mapping, dict) else []
        if isinstance(raw, dict):
            raw = raw.get("entries", raw.get("presets", raw))
        if isinstance(raw, dict):
            raw = [{"tag": key, "content": value} for key, value in raw.items()]
        if not isinstance(raw, list):
            return []
        result: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
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
            result.append({"tag": tag, "content": content})
        return result

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
        """生成当前 LoRA 的专属预设内容，不再把普通 CivitAI tag 当提示词。"""
        values: list[str] = []
        for entry in self._lora_preset_entries(file_name):
            # 左侧简称只是这条预设的分类/触发标识，右侧才是实际加入的内容。
            value = entry["content"]
            if value:
                values.append(value)
        return self._normalize_civitai_tags(values)

    async def _extract_inline_loras(self, params: DrawParams, extra_text: str = "") -> None:
        """提取提示词和用户原话中的简称，并只对当前任务临时加载。

        LLM 工具经常只把英文提示词放进 prompt，而把用户原话里的 LoRA 简称
        丢在工具参数之外。因此这里同时检查当前消息原文，保证第二简称也能命中。
        """
        available = await self._available_loras()
        extracted: list[str] = []
        remaining: list[str] = []

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

        def resolve_piece(piece: str) -> tuple[str | None, str | None]:
            value = piece.strip(" \t\r\n,，")
            name, separator, possible_weight = value.rpartition(":")
            weight = "0.8"
            if separator and re.fullmatch(r"\d+(?:\.\d+)?", possible_weight):
                value, weight = name.strip(), possible_weight
            actual = self._resolve_lora_command_alias_only(value, available)
            if not actual:
                return None, None
            # 同名普通提示词预设要保留给 PresetStore 展开，这样一个词可以同时
            # 触发临时 LoRA 和提示词预设；带权重时保留不带权重的预设名称。
            keep_for_prompt_preset = value if value in self.presets.items else None
            return f"{actual}:{weight}", keep_for_prompt_preset

        for token in params.prompt.split():
            pieces = [piece for piece in re.split(r"[,，]", token) if piece.strip()]
            resolved = [resolve_piece(piece) for piece in pieces]
            if pieces and all(item[0] is not None for item in resolved):
                for item in resolved:
                    if item[0] is not None:
                        actual_name, _, actual_weight = item[0].rpartition(":")
                        append_lora(actual_name, actual_weight)
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
        params.prompt = re.sub(r"\s+", " ", params.prompt).strip(" ,，。；;")

    def _extract_inline_presets(self, params: DrawParams, extra_text: str = "") -> None:
        """从 LLM 提示词和当前用户原话提取正式预设。"""
        text = params.prompt
        for name in sorted(self.presets.items, key=len, reverse=True):
            if not name:
                continue
            name_folded = name.casefold()
            in_prompt = name_folded in text.casefold()
            in_original = name_folded in str(extra_text or "").casefold()
            if not in_prompt and not in_original:
                continue
            if name not in params.presets:
                params.presets.append(name)
            if in_prompt:
                # 用大小写不敏感的方式移除命中的名称，避免英文预设被重复送入 AI。
                pattern = re.compile(re.escape(name), re.IGNORECASE)
                text = pattern.sub(" ", text)
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
        """让“指令简称 = CivitAI 触发词”成为可追踪的自动预设。"""
        if aliases is None:
            aliases = self._lora_command_aliases(file_name)
        if trigger_words is None:
            info = await self._fetch_civitai_lora(file_name)
            # CivitAI 临时限流、网络错误或搜索不到模型时，不删除上一次
            # 已保存的自动预设；否则每次 WebUI 刷新都会导致预设闪失。
            if not info.get("found") and info.get("error"):
                return
            trigger_words = self._civitai_trigger_words(info)
        normalized = self._normalize_trigger_words(trigger_words)
        # 没有触发词只代表本次资料没有提供触发词，不应把稳定配置清空。
        if not normalized:
            return
        self.presets.sync_auto(
            source="civitai_lora",
            source_key=file_name,
            names=aliases,
            content=", ".join(normalized),
        )

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
        model_id = model_match.group(1) if model_match else ""
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
                            f"CivitAI 模型信息请求失败（HTTP {response.status_code}）：{detail}"
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
            await self._sync_lora_trigger_presets(
                name,
                trigger_words=self._civitai_trigger_words(info),
            )
            return {
                "file_name": name,
                "alias": self._lora_alias(name),
                "command_aliases": self._lora_command_aliases_for(name, names),
                "command_alias": self._lora_command_alias(name, names),
                "category": self._lora_category(name),
                "civitai_tags": self._civitai_tags(info),
                "lora_presets": self._lora_preset_entries(name),
                **info,
            }

        return await asyncio.gather(*(fetch(name) for name in names))

    async def _lora_payload(self, *, force: bool = False) -> dict[str, Any]:
        return {
            "items": await self._lora_details(force=force),
            "categories": self._lora_categories_list(),
            # 仅新增读取字段；旧的 presets.json 仍由 /presets 原样管理。
            "lora_presets": getattr(self, "lora_presets", {}),
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
                await self._sync_lora_trigger_presets(
                    file_name,
                    trigger_words=self._normalize_trigger_words(link_info.get("trigger_words", [])),
                )
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
                self.lora_command_aliases[file_name] = command_aliases

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

                if "lora_presets" in data or "presets" in data:
                    try:
                        self.lora_presets[file_name] = self._normalize_lora_preset_entries(
                            data.get("lora_presets", data.get("presets", []))
                        )
                    except UsageError as exc:
                        return json_response({"error": str(exc)}, status_code=400)
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
                await self._sync_lora_trigger_presets(
                    file_name,
                    aliases=command_aliases,
                    trigger_words=(
                        self._normalize_trigger_words(link_info.get("trigger_words", []))
                        if custom_url
                        else None
                    ),
                )
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
                await self._sync_lora_trigger_presets(file_name, aliases=aliases)
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
                await self._sync_lora_trigger_presets(file_name)
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
        from astrbot.api.web import json_response, request

        try:
            data = await self._request_json(request, {})
            if not isinstance(data, dict):
                return json_response({"error": "请求体必须是对象"}, status_code=400)
            link = self._validate_civitai_link(data.get("url", data.get("civitai_url", "")))
            if not link:
                return json_response({"error": "请输入 CivitAI 模型链接"}, status_code=400)
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
        except (UsageError, ValueError) as exc:
            return json_response({"error": str(exc)}, status_code=400)

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
                                    f"CivitAI 下载失败（HTTP {response.status_code}）：{detail}"
                                )
                            if response.is_error:
                                payload = await response.aread()
                                detail = self._civitai_error_detail(payload)
                                raise UsageError(
                                    f"CivitAI 下载失败（HTTP {response.status_code}）：{detail}"
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
        from astrbot.api.web import json_response, request

        data = await self._request_json(request, {})
        if not isinstance(data, dict):
            return json_response({"error": "请求体必须是对象"}, status_code=400)
        job_id = str(data.get("job_id", "") or "").strip()
        job = self._download_jobs.get(job_id)
        if not job:
            return json_response({"error": "下载任务不存在或已过期"}, status_code=404)
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
        names = sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed
        )
        # 只调整通过本插件新下载的文件；没有下载记录的旧 LoRA 继续保持原有顺序。
        downloaded = []
        old_names = []
        for index, name in enumerate(names):
            raw_time = self.lora_download_order.get(name)
            try:
                download_time = float(raw_time)
            except (TypeError, ValueError):
                download_time = 0
            if download_time > 0:
                downloaded.append((download_time, index, name))
            else:
                old_names.append(name)
        downloaded.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, _, name in downloaded] + old_names

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
                load_api_workflow(target)
            except WorkflowError:
                target.unlink(missing_ok=True)
                return json_response({"error": "工作流格式无效，请上传 ComfyUI 的 API 导出 JSON"}, status_code=400)
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

    async def _extract_image(self, event: AstrMessageEvent) -> str:
        """提取消息图片并立即持久化，避免后台任务引用 AstrBot 临时路径。"""
        for component in event.get_messages():
            if isinstance(component, Image):
                try:
                    path = await component.convert_to_file_path()
                    if path and os.path.isfile(path):
                        cached = await self._persist_input_image(path)
                        if cached:
                            return cached
                except Exception:
                    continue
            if isinstance(component, Reply):
                for quoted in component.chain or []:
                    if not isinstance(quoted, Image):
                        continue
                    try:
                        path = await quoted.convert_to_file_path()
                        if path and os.path.isfile(path):
                            cached = await self._persist_input_image(path)
                            if cached:
                                return cached
                    except Exception:
                        continue
        return ""

    async def _generate(self, event: AstrMessageEvent, params: DrawParams, mode: str, image_path: str = "") -> list[Path]:
        client = self._client()
        env = await self._environment(client)
        model = self._model_name(env, params.model, mode)
        # 临时简称 LoRA 与 WebUI 中长期启用的 LoRA 合并；同一文件以临时权重为准。
        requested_loras = list(self._get("lora_list", []) or []) + list(params.loras)
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
        lora_trigger_words: list[str] = []
        lora_prompt_values: list[str] = []
        for item in loras:
            actual = item.rsplit(":", 1)[0]
            # 首次通过指令使用 LoRA 时也自动读取 CivitAI 触发词；24 小时缓存命中时
            # 不会重复联网，网络失败则继续使用本地已有配置。
            try:
                await self._fetch_civitai_lora(actual)
            except Exception as exc:
                logger.debug("[%s] 绘图前读取 LoRA CivitAI 信息失败：%s", PLUGIN_NAME, exc)
            lora_trigger_words.extend(self._lora_trigger_words(actual))
            lora_prompt_values.extend(self._lora_prompt_values(actual))
        lora_trigger_words = self._normalize_trigger_words(lora_trigger_words)
        lora_prompt_values = self._normalize_civitai_tags(lora_prompt_values)
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
            lora_trigger_words,
            lora_prompt_values,
        )
        configured_seed = self._get("seed", -1)
        try:
            configured_seed = int(configured_seed)
        except (TypeError, ValueError):
            configured_seed = -1
        seed = params.seed if params.seed >= 0 else configured_seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        width = params.width or int(self._get("width", 832) or 832)
        height = params.height or int(self._get("height", 1216) or 1216)
        scale = params.scale or float(self._get("hires_scale", 2.0) or 2.0)
        image_name = ""
        if image_path:
            image_name = (await client.upload_image(image_path))["name"]
        else:
            image_name = await self._ensure_blank_image(client, width, height, scale)
        upscale = params.upscale_model or str(self._get("hires_upscale_model", "") or "")
        if not upscale and env["upscale_models"]:
            upscale = env["upscale_models"][0]
        workflow = load_api_workflow(self._workflow_path(mode))
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
            sampler_name=params.sampler or str(self._get("sampler_name", "er_sde") or "er_sde"),
            scheduler=str(self._get("scheduler", "normal") or "normal"),
            batch=params.batch,
            filename_prefix=f"astrbot/{mode}",
        )
        prompt_id = await client.queue(patched)
        history = await client.wait(prompt_id, timeout=900)
        images = await client.download_outputs(history, self.output_dir)
        if not images:
            raise ComfyError("ComfyUI 任务完成，但没有找到 SaveImage 输出")
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
                return [Plain(f"生成失败：{exc}")]
            except Exception as exc:
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
        nodes = [
            Node(
                uin=uin,
                name="ComfyUI 绘图",
                content=[Image.fromFileSystem(str(path))],
            )
            for path in images
        ]
        nodes.append(Node(uin=uin, name="ComfyUI 绘图", content=[Plain(reply)]))
        chain = [Nodes(nodes)]
        quote = self._completion_quote(event)
        return [quote, *chain] if quote else chain

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
        try:
            chain = await self._run(event, params, mode, image_path)
            try:
                await event.send(event.chain_result(chain))
            except Exception:
                if not any(isinstance(component, Nodes) for component in chain):
                    raise
                logger.warning("[%s] 当前平台不支持合并转发，已回退为普通图片消息", PLUGIN_NAME)
                await event.send(event.chain_result(self._flatten_forward_chain(chain)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[%s] 后台绘图发送结果失败", PLUGIN_NAME)
            try:
                await event.send(event.plain_result(f"生成失败：{exc}"))
            except Exception:
                logger.exception("[%s] 无法发送后台绘图错误", PLUGIN_NAME)

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
            "/lora 查看全部 LoRA、指令简称和开启状态；/lora 1号lora 查看该 LoRA 的 C站图片；/loraon 1号lora 开启 LoRA；/lora 添加或删除 指令简称；/lora c站 查看全部 C站图片\n"
            "/预设 列表；/预设 添加 夏空=ciaccona；/预设 修改 夏空=新内容；/预设 翻译 夏空；/预设 删除 夏空\n"
            "/画师串 列表；/画师串 使用 画风001；/画师串 添加 名称=画师 tags；/画师串 关闭\n"
            "/工作流 列表；/工作流 文生图 文件名；/工作流 图生图 文件名；/工作流 高清放大 文件名\n"
            "/画图配置 查询模型位置、工作流位置和当前配置\n"
            "/comfy状态 查询 ComfyUI 状态\n"
            "直接用自然语言要求 AstrBot 画图时，会自动识别提示词预设和 LoRA 指令简称；预设与 LoRA 可同时生效，其余内容作为画面描述。\n"
            "生成结果的发送方式可在 WebUI 回复设置中选择普通消息或群聊合并转发；不支持合并转发的平台会自动退回普通消息。\n"
            "鸣潮角色知识和 Anima 提示词工程师由独立插件提供；本插件只负责绘图执行，并保留临时 LoRA 和提示词预设的最高优先级。\n"
            "绘画选项：--模型=名称 --lora=a:0.8,b:0.5 --预设=夏空 --画师=画风001 --负面=内容 --宽=832 --高=1216 --步数=24 --种子=-1 --ai=开 --noai --强度=0.6 --放大=2"
        )

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
                    "/lora 查看 LoRA、简称、启用状态和 C站图片",
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
                    "/预设 列表；/预设 添加 名称=内容",
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
        self._schedule_draw(event, params, "txt2img")
        yield event.plain_result(self._draw_start_reply("txt2img", params.prompt))

    @filter.command("图生图", alias=["img2img"], desc="使用原始工作流进行图生图")
    async def command_img2img(self, event: AstrMessageEvent, prompt: GreedyStr):
        try:
            params = fill_params(str(prompt), "img2img")
            await self._extract_inline_loras(params)
            self._extract_inline_presets(params)
        except (UsageError, ComfyError) as exc:
            yield event.plain_result(f"用法错误：{exc}")
            return
        image_path = await self._extract_image(event)
        if not image_path:
            yield event.plain_result("图生图需要在同一条消息中附上一张图片")
            return
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

    @filter.command("lora", desc="查看和管理多个 LoRA")
    async def command_lora(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        client = self._client()
        try:
            available = await client.models("loras")
        except ComfyError as exc:
            yield event.plain_result(f"查询失败：{exc}")
            return
        enabled = list(self._get("lora_list", []) or [])
        sub, _, rest = value.partition(" ")
        if sub in {"c站", "civitai", "图片", "图"}:
            try:
                details = await self._lora_details()
            except ComfyError as exc:
                yield event.plain_result(f"查询失败：{exc}")
                return
            yield event.chain_result(self._civitai_chain(details, "LoRA 的 CivitAI 图片与链接（每个 LoRA 最多展示 4 张）："))
            return
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
        if not value or value == "列表":
            enabled_map: dict[str, str] = {}
            for item in enabled:
                file_name, _, weight = str(item).rpartition(":")
                if not file_name:
                    file_name, weight = str(item), "1"
                enabled_map[file_name] = weight
            lines = ["全部 LoRA（✅ 表示已开启）："]
            ordered_available = sorted(available, key=str.casefold)
            for index, name in enumerate(ordered_available, 1):
                state = f"✅ 已开启，权重 {enabled_map[name]}" if name in enabled_map else "⬜ 未开启"
                lines.append(
                    f"{index}. {state}｜分类：{self._lora_category(name)}｜显示昵称：{self._lora_alias(name)}｜指令简称：{self._lora_command_alias(name, available)}｜文件：{name}"
                )
            yield event.plain_result(
                "\n".join(lines) +
                "\n查看单个 C站图片：/lora 指令简称；开启：/loraon 指令简称；批量管理：/lora 添加/删除 指令简称"
            )
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
        yield event.plain_result("用法：/lora（查看全部）、/lora 指令简称（查看单个 C站图片）、/lora 添加、删除、昵称、清空、c站；开启使用 /loraon 指令简称")

    @filter.command("预设", alias=["preset"], desc="管理提示词预设")
    async def command_preset(self, event: AstrMessageEvent, arg: GreedyStr):
        value = str(arg or "").strip()
        if not value or value == "列表":
            lines = ["提示词预设："]
            lines.extend(f"{name} = {item.get('translated') or item.get('content', '')}" for name, item in self.presets.items.items())
            yield event.plain_result("\n".join(lines) if len(lines) > 1 else "当前没有预设")
            return
        sub, _, rest = value.partition(" ")
        if sub in {"添加", "修改", "编辑", "add", "update", "edit"} and "=" in rest:
            name, content = rest.split("=", 1)
            try:
                self.presets.add(name, content)
            except UsageError as exc:
                yield event.plain_result(str(exc))
                return
            verb = "修改" if sub in {"修改", "编辑", "update", "edit"} else "添加"
            yield event.plain_result(f"预设已{verb}：{name.strip()} = {content.strip()}")
            return
        if sub in {"删除", "del"}:
            yield event.plain_result("预设已删除" if self.presets.remove(rest.strip()) else "预设不存在")
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
            yield event.plain_result(f"ComfyUI 未连接：{exc}")
            return
        yield event.plain_result(f"ComfyUI 已连接，版本：{data.get('system', {}).get('comfyui_version', '未知')}")

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

    def _plugin_ai_debug_enabled(self) -> bool:
        value = self._get("plugin_ai_debug", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "是", "开启"}
        return bool(value)

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
    ) -> str:
        """按 WebUI 选择决定是否由插件 AI 生成 LLM 绘图提示词。"""
        if not self._llm_plugin_ai_enabled():
            return ""
        source_text = self._llm_prompt_input(original_text, params.prompt)
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
            raise UsageError(f"LLM 插件 AI 提示词生成失败：{exc}") from exc
        params.prompt = translated
        # 已经在工具执行前完成插件 AI 转换，后台 _generate 不再二次改写，
        # 这样动作、预设和 LoRA 控制项可以由同一条链路稳定合并。
        params.ai = False
        params.auto_ai = False
        params.ai_source = ""
        logger.info("[%s] LLM 插件 AI 提示词：%s", PLUGIN_NAME, translated)
        return translated

    def _llm_debug_reply(self, reply: str, prompt: str, source_text: str = "") -> str:
        if not prompt or not self._plugin_ai_debug_enabled():
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
        try:
            # 解析工具参数、LLM 生成 prompt，以及当前用户原话中的控制项。
            # 这样 LLM 没有填写 preset/lora 字段时，用户原话仍然具有最高优先级。
            original_text = str(getattr(event, "message_str", "") or "")
            await self._extract_inline_loras(params, original_text)
            self._extract_inline_presets(params, original_text)
        except ComfyError as exc:
            return f"{MODE_NAMES[mode]}失败：无法读取 LoRA 列表：{exc}"
        if mode == "txt2img" and not params.prompt and not params.presets and not params.loras:
            return "文生图失败：请提供画面描述、提示词预设或 LoRA 指令简称。"
        try:
            plugin_prompt = await self._prepare_llm_prompt(event, params, original_text)
        except UsageError as exc:
            return f"{MODE_NAMES[mode]}失败：{exc}"
        image_path = ""
        if require_image:
            image_path = await self._extract_image(event)
            if not image_path:
                origin = self._origin(event)
                image_path = self.last_images.get(origin, "")
            if not image_path:
                return f"{MODE_NAMES[mode]}需要用户附图、回复图片，或已有最近出图。"

        self._schedule_draw(event, params, mode, image_path)
        return self._llm_debug_reply(
            self._draw_start_reply(mode, params.prompt),
            plugin_prompt,
            str(getattr(params, "plugin_ai_debug_input", "") or ""),
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

        用户要求画图、生成图片、创作一张图时必须调用此工具，不要只用文字回复。
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
        if not str(prompt or "").strip() and not str(preset or "").strip():
            return "文生图失败：缺少画面描述或提示词预设。"
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
        """使用本插件基于原始工作流执行图生图。

        用户要求修改、重绘或改变当前图片时必须调用此工具，不要只用文字回复。
        用户要求修改、重绘或改变当前图片时调用此工具。preset 和 lora 可以与 prompt 同时使用；
        其余画面描述放入 prompt，prompt 使用适配 ComfyUI Anima 的英文 Danbooru 标签并保留动作和构图，
        任务后台执行，期间可以继续聊天。

        Args:
            prompt(string): 修改描述，可以是用户自然语言。
            denoise(number): 可选重绘幅度，范围为 0 到 1。
            negative_prompt(string): 可选负面提示词。
            model(string): 可选核心模型文件名。
            lora(string): 可选多个 LoRA 指令简称，格式为 1号lora:0.8,2号lora:0.6。
            preset(string): 可选一个或多个提示词预设名称，多个用逗号分隔。
            artist_preset(string): 可选画师串预设名称。
            ai(boolean): 为兼容旧工具字段而保留；是否使用插件 AI 由 WebUI 的 LLM 提示词来源决定。
        """
        if not str(prompt or "").strip() and not str(preset or "").strip():
            return "图生图失败：缺少画面描述或提示词预设。"
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
        return json_response({"version": f"v{PLUGIN_VERSION}", "comfy": comfy, "config": self._safe_config(), "paths": {
            "comfyui_root": root, "diffusion_models": model_dir(root, "diffusion_models"),
            "checkpoints": model_dir(root, "checkpoints"), "loras": model_dir(root, "loras"),
            "upscale_models": model_dir(root, "upscale_models"), "workflow_dir": str(self._workflow_dir()),
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
                "compatible_models": compatible_models,
                "current_model": self._model_name(env, mode="txt2img"),
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
                self.presets.add(str(data.get("name", "")), str(data.get("content", "")))
            elif action == "remove":
                self.presets.remove(str(data.get("name", "")))
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
