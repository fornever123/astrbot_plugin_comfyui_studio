from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def looks_like_comfyui(path: str) -> bool:
    root = Path(path)
    return root.is_dir() and (root / "models").is_dir() and (
        (root / "main.py").is_file() or (root / "folder_paths.py").is_file()
    )


def detect_comfyui_root(explicit: str = "") -> str:
    if explicit:
        path = Path(explicit)
        for candidate in (path, path / "ComfyUI"):
            if looks_like_comfyui(str(candidate)):
                return str(candidate)
    candidates = []
    for drive in "CDEFGH":
        base = Path(f"{drive}:\\")
        candidates.extend(
            [
                base / "ComfyUI",
                base / "ComfyUI_windows_portable" / "ComfyUI",
                base / "1aaaaaai" / "ComfyUI-aki-v3" / "ComfyUI",
            ]
        )
    candidates.append(Path.home() / "ComfyUI")
    for candidate in candidates:
        if looks_like_comfyui(str(candidate)):
            return str(candidate)
    return ""


def model_dir(root: str, category: str) -> str:
    return str(Path(root) / "models" / category) if root else ""


def platform_name() -> str:
    return sys.platform


# --------------------------------------------------------------------------
# 模型目录解析
#
# ComfyUI 允许用 extra_model_paths.yaml 把任意类别（loras / vae / unet …）指到
# 别的磁盘，所以「<根目录>/models/<类别>」只是默认约定，不能当成事实。下面这套
# 解析器把三类来源统一起来，优先级为：
#     手动覆盖（WebUI 配置） > extra_model_paths.yaml > <根目录>/models/<类别>
# --------------------------------------------------------------------------

# ComfyUI 的 folder name 存在新旧两套叫法，查询时互相兜底。
MODEL_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "diffusion_models": ("diffusion_models", "unet"),
    "unet": ("unet", "diffusion_models"),
    "text_encoders": ("text_encoders", "clip"),
    "clip": ("clip", "text_encoders"),
}

# 插件会展示、也可能写入的模型类别（顺序与控制台文件夹卡片保持一致）。
# 「clip」是 text_encoders 的旧名，通过 MODEL_CATEGORY_ALIASES 兼容，不单独展示。
MODEL_CATEGORIES: tuple[str, ...] = (
    "diffusion_models",
    "checkpoints",
    "loras",
    "upscale_models",
    "unet",
    "text_encoders",
    "vae",
    "controlnet",
    "ipadapter",
    "clip_vision",
)

EXTRA_MODEL_PATHS_FILENAMES = ("extra_model_paths.yaml", "extra_model_paths.yml")

SOURCE_LABELS = {
    "override": "手动指定",
    "extra": "额外路径",
    "default": "默认位置",
    "auto": "自动探测",
    "builtin": "内置副本",
    "runtime": "运行时",
    "none": "未找到",
}


# 各类别的中文名，供控制台「路径设置」展示（顺序与 MODEL_CATEGORIES 一致）。
MODEL_CATEGORY_LABELS: dict[str, str] = {
    "diffusion_models": "核心大模型（diffusion_models）",
    "checkpoints": "Checkpoint",
    "loras": "LoRA",
    "upscale_models": "放大模型",
    "unet": "UNet",
    "text_encoders": "文本编码器（text_encoders）",
    "vae": "VAE",
    "controlnet": "ControlNet",
    "ipadapter": "IPAdapter",
    "clip_vision": "CLIP Vision",
}

# 各类别的模型文件后缀，用于「检测」时统计目录里到底有没有可用模型。
MODEL_CATEGORY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "diffusion_models": (".safetensors", ".sft", ".ckpt", ".pt", ".gguf", ".bin"),
    "checkpoints": (".safetensors", ".sft", ".ckpt", ".pt", ".bin"),
    "loras": (".safetensors", ".sft", ".pt", ".ckpt", ".bin"),
    "upscale_models": (".safetensors", ".pth", ".pt", ".ckpt", ".onnx", ".bin"),
    "unet": (".safetensors", ".sft", ".pt", ".gguf", ".bin"),
    "text_encoders": (".safetensors", ".sft", ".pt", ".bin"),
    "vae": (".safetensors", ".pt", ".ckpt", ".bin"),
    "controlnet": (".safetensors", ".pth", ".pt", ".ckpt", ".bin"),
    "ipadapter": (".safetensors", ".bin", ".pt"),
    "clip_vision": (".safetensors", ".bin", ".pt"),
}

# 未统计文件数时的占位值（大目录不做全量遍历，避免面板卡住）。
UNKNOWN_FILE_COUNT = -1


def scan_path(
    path: str | Path,
    *,
    kind: str = "dir",
    suffixes: tuple[str, ...] = (),
    limit: int = 20000,
) -> tuple[bool, int]:
    """检测一个路径是否可用，返回（是否存在，命中文件数）。

    - ``kind="file"``：只看单个文件，存在即计 1。
    - ``kind="dir"``：给了 ``suffixes`` 才递归统计匹配文件；没给后缀说明只是
      「要一个目录」，直接返回 ``UNKNOWN_FILE_COUNT``，不做全量遍历。
    """
    try:
        text = str(path or "").strip()
    except (TypeError, ValueError):
        return False, 0
    if not text:
        return False, 0
    try:
        target = Path(text).expanduser()
    except (TypeError, ValueError):
        return False, 0

    if kind == "file":
        try:
            return (True, 1) if target.is_file() else (False, 0)
        except OSError:
            return False, 0

    try:
        if not target.is_dir():
            return False, 0
    except OSError:
        return False, 0

    wanted = tuple(str(item).lower() for item in suffixes if str(item).strip())
    if not wanted:
        return True, UNKNOWN_FILE_COUNT

    count = 0
    for _current, dirs, files in os.walk(target, onerror=lambda _exc: None):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            if not name.lower().endswith(wanted):
                continue
            count += 1
            if count >= limit:
                return True, count
    return True, count

@dataclass(frozen=True)
class ModelDir:
    """一个已解析的模型目录，以及它的来源。"""

    path: Path
    source: str

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    @property
    def is_custom(self) -> bool:
        return self.source != "default"


def category_aliases(category: str) -> tuple[str, ...]:
    key = str(category or "").strip()
    return MODEL_CATEGORY_ALIASES.get(key, (key,)) if key else ()


def _strip_comment(line: str) -> str:
    """去掉行内注释；引号内的 # 视为普通字符。"""
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def parse_extra_model_paths(text: str) -> dict[str, dict[str, list[str]]]:
    """解析 ComfyUI 的 extra_model_paths.yaml。

    只取这份文件里实际用得到的结构，因此不引入 YAML 依赖：顶层是配置段名，
    段内 ``base_path`` 加若干「类别: 路径」条目，路径既可以是单个字符串，
    也可以是 ``|`` 块里的多行。
    """
    sections: dict[str, dict[str, list[str]]] = {}
    current: dict[str, list[str]] | None = None
    block_key = ""
    block_indent = 0

    for raw_line in str(text or "").splitlines():
        line = _strip_comment(raw_line.replace("\t", "    ")).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if block_key and indent > block_indent:
            current[block_key].append(_unquote(stripped))
            continue
        block_key = ""

        if indent == 0 and stripped.endswith(":"):
            name = _unquote(stripped[:-1]).strip()
            current = sections.setdefault(name, {}) if name else None
            continue
        if current is None or ":" not in stripped:
            continue

        key, _, value = stripped.partition(":")
        key = _unquote(key).strip()
        value = value.strip()
        if not key:
            continue
        if value in {"|", "|-", "|+", ">", ">-", ">+"}:
            current[key] = []
            block_key = key
            block_indent = indent
            continue
        if not value:
            continue
        current.setdefault(key, []).append(_unquote(value))

    return {
        name: {key: list(values) for key, values in items.items() if values}
        for name, items in sections.items()
        if items
    }


def find_extra_model_paths_file(root: str) -> str:
    """自动寻找 ComfyUI 的 extra_model_paths.yaml：先根目录，再上一层。"""
    if not root:
        return ""
    base = Path(root)
    for directory in (base, base.parent):
        for filename in EXTRA_MODEL_PATHS_FILENAMES:
            candidate = directory / filename
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
    return ""


def read_extra_model_paths(path: str) -> dict[str, dict[str, list[str]]]:
    if not path:
        return {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return {}
    return parse_extra_model_paths(text)


def _resolve_against(base_path: str, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or not base_path:
        return candidate
    return Path(base_path) / candidate


def resolve_model_dirs(
    root: str,
    category: str,
    *,
    override: str = "",
    sections: dict[str, dict[str, list[str]]] | None = None,
) -> list[ModelDir]:
    """解析某个类别的全部模型目录，按优先级排序并去重。"""
    resolved: list[ModelDir] = []

    def add(value: str | Path, source: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        try:
            path = Path(text).expanduser()
        except (TypeError, ValueError):
            return
        key = str(path)
        if any(str(item.path) == key for item in resolved):
            return
        resolved.append(ModelDir(path=path, source=source))

    add(override, "override")

    for section in (sections or {}).values():
        if not isinstance(section, dict):
            continue
        base_path = ""
        for name in ("base_path", "basePath"):
            values = section.get(name) or []
            if values:
                base_path = str(values[0])
                break
        for alias in category_aliases(category):
            for value in section.get(alias) or []:
                add(_resolve_against(base_path, value), "extra")

    if root:
        add(model_dir(root, category), "default")

    return resolved


def primary_model_dir(
    root: str,
    category: str,
    *,
    override: str = "",
    sections: dict[str, dict[str, list[str]]] | None = None,
) -> ModelDir | None:
    entries = resolve_model_dirs(root, category, override=override, sections=sections)
    return entries[0] if entries else None
