from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALIASES = {
    "模型": "model", "model": "model",
    "lora": "lora", "预设": "preset", "preset": "preset",
    "画师": "artist", "画风": "artist", "artist": "artist",
    "负面": "negative", "neg": "negative",
    "宽": "width", "width": "width", "高": "height", "height": "height",
    "步数": "steps", "steps": "steps", "cfg": "cfg", "种子": "seed", "seed": "seed",
    "强度": "denoise", "denoise": "denoise", "批次": "batch", "batch": "batch",
    "放大": "scale", "scale": "scale", "ai": "ai", "noai": "noai",
    "采样": "sampler", "sampler": "sampler", "调度": "scheduler", "scheduler": "scheduler",
    "放大模型": "upscale_model", "upscale_model": "upscale_model",
}


class UsageError(Exception):
    pass


@dataclass
class DrawParams:
    mode: str = "txt2img"
    prompt: str = ""
    negative: str | None = None
    model: str = ""
    loras: list[str] = field(default_factory=list)
    presets: list[str] = field(default_factory=list)
    # 本次任务实际命中的 LoRA 专属预设，键为 LoRA 文件名，值为简称列表。
    # 它与 loras 分开保存，避免一个 LoRA 的多个专属预设被全部拼入提示词。
    lora_preset_tags: dict[str, list[str]] = field(default_factory=dict)
    artist_preset: str = ""
    ai: bool | None = None
    # LLM 工具可以把自然语言交给当前 AstrBot 内置 AI 优化；普通指令不使用这两个字段。
    ai_source: str = ""
    auto_ai: bool = False
    width: int = 0
    height: int = 0
    steps: int = 0
    cfg: float = 0.0
    seed: int = -1
    denoise: float = 0.0
    scale: float = 0.0
    upscale_model: str = ""
    batch: int = 1
    sampler: str = ""
    scheduler: str = ""
    # 图生图的编辑指令已经由专用 LLM/插件 AI 整理，后续不再套用文生图
    # 的 Anima 标签扩写或普通翻译。
    img2img_edit_instruction_ready: bool = False
    # 图生图 LLM 整理出的最终编辑指令。单独保存，避免后续预设/LoRA
    # 识别或普通参数整理把它覆盖回 LLM 工具传入的原始描述。
    img2img_edit_instruction: str = ""
    # 用户绘图限额的本次预留标识。任务提交失败时可精确归还额度。
    draw_limit_user_id: str = ""
    draw_limit_token: str = ""
    draw_limit_cost: int = 0


def _tokens(text: str) -> list[str]:
    # 支持 --负面="多个单词"，同时保留中文文本。
    return re.findall(r"""(?:[^\s"']+|"[^"]*"|'[^']*')+""", text.strip())


def parse_options(text: str) -> tuple[str, dict[str, str]]:
    tokens = _tokens(text)
    prompt: list[str] = []
    options: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            prompt.append(token)
            index += 1
            continue
        body = token[2:]
        if "=" in body:
            key, value = body.split("=", 1)
        else:
            key, value = body, ""
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                value = tokens[index + 1]
                index += 1
        options[ALIASES.get(key, key)] = value.strip("\"'")
        index += 1
    return " ".join(prompt).strip(), options


def parse_loras(value: str) -> list[str]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, strength = item.rsplit(":", 1)
            try:
                number = float(strength)
            except ValueError as exc:
                raise UsageError(f"LoRA 权重不是数字：{item}") from exc
            if not 0 <= number <= 2:
                raise UsageError(f"LoRA 权重应在 0 到 2 之间：{item}")
            result.append(f"{name.strip()}:{number:g}")
        else:
            result.append(item)
    return result


def as_int(options: dict[str, str], key: str, default: int) -> int:
    value = options.get(key, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise UsageError(f"--{key} 必须是整数：{value}") from exc


def as_float(options: dict[str, str], key: str, default: float) -> float:
    value = options.get(key, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise UsageError(f"--{key} 必须是数字：{value}") from exc


def fill_params(text: str, mode: str) -> DrawParams:
    prompt, options = parse_options(text)
    # `ai`/`noai` are standalone command switches. Remove them from the
    # prompt before preset expansion so they can never reach ComfyUI.
    prompt_tokens = _tokens(prompt)
    inline_ai = False
    inline_noai = False
    clean_prompt_tokens: list[str] = []
    for token in prompt_tokens:
        marker = token.strip("\"'").casefold()
        if marker == "ai":
            inline_ai = True
        elif marker == "noai":
            inline_noai = True
        else:
            clean_prompt_tokens.append(token)
    prompt = " ".join(clean_prompt_tokens).strip()
    if "ai" not in options and "noai" not in options:
        if inline_noai:
            options["noai"] = "1"
        elif inline_ai:
            options["ai"] = "开"
    params = DrawParams(mode=mode, prompt=prompt)
    params.model = options.get("model", "")
    params.loras = parse_loras(options.get("lora", ""))
    params.presets = [x.strip() for x in options.get("preset", "").split(",") if x.strip()]
    params.artist_preset = options.get("artist", "").strip()
    params.negative = options.get("negative") or None
    params.width = as_int(options, "width", 0)
    params.height = as_int(options, "height", 0)
    params.steps = as_int(options, "steps", 0)
    params.cfg = as_float(options, "cfg", 0)
    params.seed = as_int(options, "seed", -1)
    params.denoise = as_float(options, "denoise", 0)
    params.scale = as_float(options, "scale", 0)
    params.batch = max(1, min(4, as_int(options, "batch", 1)))
    params.sampler = options.get("sampler", "")
    params.scheduler = options.get("scheduler", "")
    params.upscale_model = options.get("upscale_model", "")
    if "ai" in options:
        params.ai = options["ai"].lower() in {"1", "true", "on", "开", "是"}
    if "noai" in options:
        params.ai = False
    if params.width < 0 or params.height < 0:
        raise UsageError("宽度和高度不能为负数")
    return params


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


class PresetStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.items: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        with self.lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.items = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self.items = {}

    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def add(self, name: str, content: str) -> None:
        name, content = name.strip(), content.strip()
        if not name or not content:
            raise UsageError("预设名称和内容都不能为空")
        current = self.items.get(name)
        previous = current if isinstance(current, dict) else {}
        # 更新预设时保留同内容的翻译和其它管理字段，避免 WebUI 每次保存
        # 都让已翻译内容消失；内容真正改变时清掉旧翻译，防止使用过期结果。
        translated = str(previous.get("translated") or "") if previous.get("content") == content else ""
        self.items[name] = {
            "content": content,
            "translated": translated,
        }
        self.save()

    def update(self, old_name: str, name: str, content: str) -> None:
        """更新预设，支持 WebUI 修改名称且保留翻译缓存。"""
        old_name = str(old_name or "").strip()
        name, content = str(name or "").strip(), str(content or "").strip()
        if not name or not content:
            raise UsageError("预设名称和内容都不能为空")
        previous = self.items.get(old_name, {}) if old_name else {}
        if old_name and old_name != name and old_name in self.items:
            del self.items[old_name]
        translated = (
            str(previous.get("translated") or "")
            if isinstance(previous, dict) and previous.get("content") == content
            else ""
        )
        self.items[name] = {"content": content, "translated": translated}
        self.save()

    def sync_auto(
        self,
        *,
        source: str,
        source_key: str,
        names: list[str],
        content: str,
    ) -> bool:
        """同步插件生成的预设，不覆盖用户手动建立的同名预设。"""
        source = str(source or "").strip()
        source_key = str(source_key or "").strip()
        wanted = {
            str(name or "").strip()
            for name in names
            if str(name or "").strip()
        }
        content = str(content or "").strip()
        changed = False

        for name, item in list(self.items.items()):
            if not isinstance(item, dict):
                continue
            if (
                item.get("_auto_source") == source
                and item.get("_auto_source_key") == source_key
                and (not content or name not in wanted)
            ):
                del self.items[name]
                changed = True

        if content:
            for name in wanted:
                current = self.items.get(name)
                if current is not None and not (
                    isinstance(current, dict)
                    and current.get("_auto_source") == source
                    and current.get("_auto_source_key") == source_key
                ):
                    # 同名预设由用户维护时，自动预设不得覆盖它。
                    continue
                translated = ""
                if isinstance(current, dict) and current.get("content") == content:
                    translated = str(current.get("translated") or "")
                next_item = {
                    "content": content,
                    "translated": translated,
                    "_auto_source": source,
                    "_auto_source_key": source_key,
                }
                if current != next_item:
                    self.items[name] = next_item
                    changed = True

        if changed:
            self.save()
        return changed

    def remove(self, name: str) -> bool:
        if name not in self.items:
            return False
        del self.items[name]
        self.save()
        return True

    def effective(self, name: str) -> str:
        item = self.items.get(name, {})
        return str(item.get("translated") or item.get("content") or "")

    def expand(self, prompt: str) -> str:
        result = prompt
        for name in sorted(self.items, key=len, reverse=True):
            if name:
                result = result.replace(name, self.effective(name))
        return result
