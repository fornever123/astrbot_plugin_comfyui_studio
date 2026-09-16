"""输入 / 输出双向图片安全审核。

本模块只负责“把图片交给一个 OpenAI 兼容的多模态模型并解析判定结果”，
不关心群聊上下文。群名单、开关和警告文案由 ``main.py`` 决定。

判定协议（硬约束，便于解析）：

    {"level": "safe|borderline|suggestive|explicit",
     "guro":  "none|mild|severe",
     "reason": "不超过 20 字的中文原因"}

``level`` 描述成人内容强度，``guro`` 描述血腥猎奇强度，两者独立判定。
是否拦截由 ``strictness``（loose / standard / strict）在本模块内映射，
这样“调节敏感度”不需要重新训练或改写提示词。
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from .ai import AITranslator, normalize_base_url

try:  # Pillow 是插件的既有依赖，缺失时退化为原图直传。
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - 仅在依赖损坏时触发
    PILImage = None


DEFAULT_STRICTNESS = "standard"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_SIDE = 1024
# 超过这个体积先等比缩小，避免 base64 后请求体过大被网关拒绝。
_RAW_SHRINK_BYTES = 1_800_000
_CACHE_LIMIT = 512

LEVEL_SAFE = "safe"
LEVEL_BORDERLINE = "borderline"
LEVEL_SUGGESTIVE = "suggestive"
LEVEL_EXPLICIT = "explicit"
LEVELS = (LEVEL_SAFE, LEVEL_BORDERLINE, LEVEL_SUGGESTIVE, LEVEL_EXPLICIT)

GURO_NONE = "none"
GURO_MILD = "mild"
GURO_SEVERE = "severe"
GURO_LEVELS = (GURO_NONE, GURO_MILD, GURO_SEVERE)

STRICTNESS_CHOICES = ("loose", "standard", "strict")

#: 判定尺度 -> (需要拦截的 level 集合, 需要拦截的 guro 集合)
_BLOCK_RULES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # 宽松：只拦明确露骨和严重血腥。
    "loose": (frozenset({LEVEL_EXPLICIT}), frozenset({GURO_SEVERE})),
    # 标准：拦性暗示及以上，以及轻微以上的血腥猎奇；放行泳装、运动装等擦边内容。
    "standard": (
        frozenset({LEVEL_SUGGESTIVE, LEVEL_EXPLICIT}),
        frozenset({GURO_MILD, GURO_SEVERE}),
    ),
    # 严格：擦边也拦。
    "strict": (
        frozenset({LEVEL_BORDERLINE, LEVEL_SUGGESTIVE, LEVEL_EXPLICIT}),
        frozenset({GURO_MILD, GURO_SEVERE}),
    ),
}

_STRICTNESS_HINT = {
    "loose": "只把明确露骨的性内容和严重的血腥、肢解判定为不安全；泳装、内衣、擦边暗示一律视为安全。",
    "standard": (
        "把明确裸露、性行为、性暗示画面，以及明显血腥、肢解、恶心猎奇的画面判定为不安全；"
        "普通泳装、运动装、不露骨的擦边二次元图视为安全。"
    ),
    "strict": "任何擦边、暗示性姿势、泳装内衣、轻微血迹或伤口都判定为不安全。",
}

# 不同模型可能返回同义词，这里统一归一到协议值。
_LEVEL_ALIASES = {
    "safe": LEVEL_SAFE,
    "normal": LEVEL_SAFE,
    "none": LEVEL_SAFE,
    "ok": LEVEL_SAFE,
    "sfw": LEVEL_SAFE,
    "borderline": LEVEL_BORDERLINE,
    "tease": LEVEL_BORDERLINE,
    "mild": LEVEL_BORDERLINE,
    "suggestive": LEVEL_SUGGESTIVE,
    "nsfw": LEVEL_SUGGESTIVE,
    "questionable": LEVEL_SUGGESTIVE,
    "explicit": LEVEL_EXPLICIT,
    "porn": LEVEL_EXPLICIT,
    "sexual": LEVEL_EXPLICIT,
    "hentai": LEVEL_EXPLICIT,
    "unsafe": LEVEL_EXPLICIT,
}

_GURO_ALIASES = {
    "none": GURO_NONE,
    "no": GURO_NONE,
    "safe": GURO_NONE,
    "mild": GURO_MILD,
    "light": GURO_MILD,
    "slight": GURO_MILD,
    "gore": GURO_SEVERE,
    "severe": GURO_SEVERE,
    "extreme": GURO_SEVERE,
    "heavy": GURO_SEVERE,
}


def normalize_strictness(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in STRICTNESS_CHOICES else DEFAULT_STRICTNESS


def build_prompt(strictness: str) -> str:
    """构造审核系统提示词。"""
    hint = _STRICTNESS_HINT.get(normalize_strictness(strictness), _STRICTNESS_HINT[DEFAULT_STRICTNESS])
    return (
        "你是内容安全审核员，负责判断一张图片能否在普通聊天群里传播。\n"
        f"本次判定尺度：{hint}\n"
        "只输出一个 JSON 对象。禁止输出 Markdown、代码块、前后缀或任何解释文字。\n"
        '格式：{"level":"safe|borderline|suggestive|explicit","guro":"none|mild|severe","reason":"不超过20字的中文原因"}\n'
        "level 含义：\n"
        "- safe：完全正常的图片\n"
        "- borderline：擦边但不露骨，例如泳装、运动装、轻微暧昧\n"
        "- suggestive：性暗示，例如明显挑逗姿势、内衣贴身、性暗示构图\n"
        "- explicit：明确露骨，例如裸露性器官、性行为、色情画面\n"
        "guro 含义：\n"
        "- none：没有血腥猎奇内容\n"
        "- mild：轻微血迹、伤口、惊悚\n"
        "- severe：严重血腥、肢解、内脏、极端恶心猎奇\n"
        "两个字段都必须给出。判断不确定时选择更安全的一档。"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型回复中提取第一个 JSON 对象。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    # 去掉 ```json ... ``` 围栏
    raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    candidates: list[str] = []
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        candidates.append(match.group(0))
    candidates.append(raw)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _normalize_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in LEVELS:
        return text
    return _LEVEL_ALIASES.get(text, LEVEL_SAFE)


def _normalize_guro(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in GURO_LEVELS:
        return text
    return _GURO_ALIASES.get(text, GURO_NONE)


def judge(level: str, guro: str, strictness: str) -> tuple[bool, str]:
    """按判定尺度换算是否拦截，并返回命中类别。"""
    levels, guros = _BLOCK_RULES[normalize_strictness(strictness)]
    nsfw_hit = level in levels
    guro_hit = guro in guros
    if nsfw_hit and guro_hit:
        return True, "nsfw+guro"
    if nsfw_hit:
        return True, "nsfw"
    if guro_hit:
        return True, "guro"
    return False, ""


@dataclass(frozen=True)
class ModerationVerdict:
    """一次审核的结果。``blocked`` 为 True 表示应拦截并回传警告。"""

    blocked: bool = False
    level: str = LEVEL_SAFE
    guro: str = GURO_NONE
    reason: str = ""
    category: str = ""
    error: str = ""
    skipped: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.error)

    def describe(self) -> str:
        """给日志用的简短描述。"""
        if self.skipped:
            return "已跳过"
        if self.error:
            return f"检测失败：{self.error}"
        parts = [f"level={self.level}", f"guro={self.guro}"]
        if self.reason:
            parts.append(f"原因={self.reason}")
        return "；".join(parts)


def _safe_verdict(reason: str = "") -> ModerationVerdict:
    return ModerationVerdict(blocked=False, level=LEVEL_SAFE, guro=GURO_NONE, reason=reason)


@dataclass
class ModerationConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    strictness: str = DEFAULT_STRICTNESS
    timeout: int = DEFAULT_TIMEOUT
    fail_open: bool = True
    max_side: int = DEFAULT_MAX_SIDE

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "ModerationConfig":
        source: Mapping[str, Any] = cfg or {}

        def _bool(value: Any, default: bool) -> bool:
            if value is None or value == "":
                return default
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on", "是", "开启"}

        def _int(value: Any, default: int) -> int:
            try:
                number = int(float(value))
            except (TypeError, ValueError):
                return default
            return number if number > 0 else default

        return cls(
            base_url=normalize_base_url(str(source.get("moderation_base_url", "") or "")),
            api_key=str(source.get("moderation_api_key", "") or ""),
            model=str(source.get("moderation_model", "") or "").strip(),
            strictness=normalize_strictness(source.get("moderation_strictness", DEFAULT_STRICTNESS)),
            timeout=_int(source.get("moderation_timeout", DEFAULT_TIMEOUT), DEFAULT_TIMEOUT),
            fail_open=_bool(source.get("moderation_fail_open", True), True),
            max_side=_int(source.get("moderation_max_side", DEFAULT_MAX_SIDE), DEFAULT_MAX_SIDE),
        )


class ImageModerator:
    """OpenAI 兼容多模态审核客户端。"""

    def __init__(self, config: ModerationConfig):
        self.config = config
        self._cache: dict[str, ModerationVerdict] = {}

    # ---------- 对外能力 ----------

    @property
    def available(self) -> bool:
        """审核模型是否配置完整。未配置时调用方应跳过检测而不是拦截。"""
        return self.config.configured

    def clear_cache(self) -> None:
        self._cache.clear()

    async def check(self, path: str | Path) -> ModerationVerdict:
        """审核单张图片。任何异常都会按 ``fail_open`` 转成安全或不安全。"""
        if not self.available:
            return ModerationVerdict(skipped=True, error="审核模型未配置")

        file_path = Path(str(path))
        try:
            if not file_path.is_file():
                return ModerationVerdict(skipped=True, error="图片文件不存在")
            raw = file_path.read_bytes()
        except OSError as exc:
            return self._failure(f"读取图片失败：{exc}")
        if not raw:
            return ModerationVerdict(skipped=True, error="图片内容为空")

        digest = hashlib.sha1(raw).hexdigest()
        cache_key = f"{self.config.strictness}:{digest}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        verdict = await self._request(raw)
        self._remember(cache_key, verdict)
        return verdict

    async def check_many(self, paths: Iterable[str | Path]) -> ModerationVerdict:
        """审核多张图片，命中任意一张即返回该张的拦截结果。"""
        last = _safe_verdict()
        for path in paths:
            if not str(path or "").strip():
                continue
            verdict = await self.check(path)
            if verdict.blocked:
                return verdict
            last = verdict
        return last

    async def test_connection(self) -> str:
        """用一张纯色小图验证审核链路，返回成功说明。"""
        if not self.available:
            raise RuntimeError("请先填写审核模型的服务地址和模型名")
        raw = self._probe_image()
        verdict = await self._request(raw)
        if verdict.error:
            raise RuntimeError(verdict.error)
        return f"审核链路正常：level={verdict.level}；guro={verdict.guro}"

    # ---------- 内部实现 ----------

    @staticmethod
    def _probe_image() -> bytes:
        """生成一张 32x32 的纯灰图，作为连通性探测样本（不含任何敏感内容）。"""
        if PILImage is not None:
            try:
                buffer = BytesIO()
                PILImage.new("RGB", (32, 32), (128, 128, 128)).save(buffer, "PNG")
                return buffer.getvalue()
            except Exception:  # pragma: no cover
                pass
        # 1x1 灰色 PNG 兜底，避免 Pillow 不可用时探测直接失败。
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+P//PwAF/gL+0l7CkwAAAABJRU5ErkJggg=="
        )

    def _failure(self, message: str) -> ModerationVerdict:
        """按失败策略生成结果：fail_open 放行，fail_closed 拦截。"""
        blocked = not self.config.fail_open
        return ModerationVerdict(
            blocked=blocked,
            reason="审核服务不可用" if blocked else "",
            category="error" if blocked else "",
            error=message,
        )

    def _remember(self, key: str, verdict: ModerationVerdict) -> None:
        # 失败结果不写缓存，避免一次网络抖动长期放行同一张图。
        if verdict.failed:
            return
        if len(self._cache) >= _CACHE_LIMIT:
            for stale in list(self._cache)[: _CACHE_LIMIT // 4]:
                self._cache.pop(stale, None)
        self._cache[key] = verdict

    def _data_url(self, raw: bytes) -> str:
        """把图片压到可接受的大小并编码成 data URL。"""
        mime = "image/png"
        payload = raw
        if PILImage is not None:
            try:
                with PILImage.open(BytesIO(raw)) as image:
                    image.load()
                    fmt = str(image.format or "").upper()
                    mime = {
                        "JPEG": "image/jpeg",
                        "JPG": "image/jpeg",
                        "WEBP": "image/webp",
                        "PNG": "image/png",
                        "GIF": "image/gif",
                        "BMP": "image/bmp",
                    }.get(fmt, "image/png")
                    too_large_side = max(image.size) > self.config.max_side
                    too_large_bytes = len(raw) > _RAW_SHRINK_BYTES
                    if too_large_side or too_large_bytes:
                        # 转 RGB 后存 JPEG：无损格式（如带透明通道的 PNG）
                        # 转 JPEG 会失败，必须先转换色彩模式。
                        shrunk = image.convert("RGB")
                        shrunk.thumbnail(
                            (self.config.max_side, self.config.max_side),
                            PILImage.LANCZOS if hasattr(PILImage, "LANCZOS") else PILImage.BICUBIC,
                        )
                        buffer = BytesIO()
                        shrunk.save(buffer, "JPEG", quality=85, optimize=True)
                        payload = buffer.getvalue()
                        mime = "image/jpeg"
            except Exception:
                # 无法解码时保留原始字节，交给上游模型判断。
                payload = raw
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AstrBot-ComfyUI-AI-Studio-Moderation/1.0",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _error_text(self, exc: BaseException) -> str:
        # 错误信息可能被上游回显请求内容，先抹掉密钥。
        message = str(exc)
        key = self.config.api_key
        if key:
            message = message.replace(key, "[已隐藏]")
        return message[:300]

    async def _request(self, raw: bytes) -> ModerationVerdict:
        base_url = self.config.base_url
        if not base_url or not self.config.model:
            return ModerationVerdict(skipped=True, error="审核模型未配置")
        try:
            data_url = self._data_url(raw)
        except Exception as exc:  # pragma: no cover - 编码极少失败
            return self._failure(f"图片编码失败：{self._error_text(exc)}")

        body = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": build_prompt(self.config.strictness)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请审核这张图片，只返回 JSON。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json=body,
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return self._failure(f"审核请求失败：{self._error_text(exc)}")

        text = AITranslator._content_value(payload)
        parsed = _extract_json_object(text)
        if parsed is None:
            return self._failure(f"审核结果无法解析：{text[:160] or '空响应'}")

        level = _normalize_level(parsed.get("level", parsed.get("nsfw", "")))
        guro = _normalize_guro(parsed.get("guro", parsed.get("gore", "")))
        reason = str(parsed.get("reason", "") or "").strip()[:60]
        blocked, category = judge(level, guro, self.config.strictness)
        return ModerationVerdict(
            blocked=blocked,
            level=level,
            guro=guro,
            reason=reason,
            category=category,
        )
