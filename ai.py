from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class AIError(Exception):
    pass


def normalize_base_url(value: str) -> str:
    """把用户输入的服务地址统一成 OpenAI 兼容 API 的基础地址。"""
    raw = str(value or "").strip()
    if not raw:
        return ""
    # 用户常会直接复制完整接口地址；统一去掉已知的接口后缀，避免重复拼接。
    try:
        parsed = urlsplit(raw)
        if parsed.scheme and parsed.netloc:
            path = parsed.path.rstrip("/")
            for suffix in ("/chat/completions", "/models"):
                if path.lower().endswith(suffix):
                    path = path[: -len(suffix)].rstrip("/")
                    break
            # 服务地址上的查询参数通常来自一次性测试链接，不能拼到 /models 或 /chat/completions 后。
            return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    except ValueError:
        pass
    return raw


DANBOORU_SYSTEM_PROMPT = """你是 Stable Diffusion / Danbooru 提示词专家。
把用户的中文画面描述改写为一行英文、逗号分隔的标签。
保留用户明确给出的角色、主体、动作、服装、镜头、构图、背景和天气；已存在的英文标签不要重复。
角色名称优先使用 Danbooru 官方词条；只输出最终提示词，不要解释、Markdown、代码块或换行。
不要输出质量词、画师名、LoRA 语法或权重语法；不要擅自添加用户没有要求的剧情、场景或服装。"""

PLAIN_SYSTEM_PROMPT = """你是 Stable Diffusion 提示词专家。
把用户的中文或自然语言画面描述改写为一行英文、逗号分隔的提示词。
保留主体、角色、动作、服装、镜头、构图、背景和光线。
输入中已有英文标签时保留它们，不要重复。"""

# 保留旧名称，避免第三方代码直接导入 SYSTEM_PROMPT 时失效。
SYSTEM_PROMPT = DANBOORU_SYSTEM_PROMPT


class AITranslator:
    def __init__(self, cfg: dict[str, Any]):
        self.base_url = normalize_base_url(str(cfg.get("ai_base_url", "")))
        self.api_key = str(cfg.get("ai_api_key", ""))
        self.model = str(cfg.get("ai_model", "gpt-4o-mini"))
        # 0.6.3 不再要求单独的 ai_enabled 字段。兼容旧配置：未显式关闭时，
        # 只要服务地址和模型存在就允许调用；明确写 false 时仍然保持关闭。
        configured = bool(self.base_url and self.model)
        self.enabled = bool(cfg.get("ai_enabled", configured))
        self.temperature = max(0.0, min(2.0, float(cfg.get("ai_temperature", 0.2) or 0.2)))
        self.to_danbooru = bool(cfg.get("ai_danbooru", True))

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AstrBot-ComfyUI-AI-Studio/0.8.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        if not self.base_url:
            return ""
        return f"{self.base_url}/{path.lstrip('/')}"

    def _error_text(self, exc: Exception) -> str:
        # 不让 API Key 进入错误信息，即使上游错误文本意外回显了请求内容。
        message = str(exc)
        if self.api_key:
            message = message.replace(self.api_key, "[已隐藏]")
        return message[:500]

    @staticmethod
    def _content_value(data: Any) -> str:
        """从常见 OpenAI 兼容响应中提取文本。

        不同的本地网关和推理模型会把结果放在 ``message.content``、
        ``message.reasoning_content``、``choice.text``，或把 content 返回成
        多模态数组。以前只读取第一种格式，模型实际已经返回内容时也会被
        错误判定为“返回格式无法解析”。
        """

        def flatten(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        nested = item.get("text") or item.get("content") or item.get("value")
                        if nested:
                            parts.append(flatten(nested))
                return "".join(parts)
            if isinstance(value, dict):
                for key in ("text", "content", "value", "output_text", "reasoning_content", "reasoning"):
                    if value.get(key):
                        return flatten(value[key])
            return str(value or "") if value is not None else ""

        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message")
            if isinstance(message, dict):
                for key in ("content", "text", "reasoning_content", "reasoning"):
                    value = flatten(message.get(key))
                    if value.strip():
                        return value
            elif message:
                value = flatten(message)
                if value.strip():
                    return value
            for key in ("text", "content", "reasoning_content", "reasoning"):
                value = flatten(choice.get(key))
                if value.strip():
                    return value
        for key in ("output_text", "content", "response", "text"):
            value = flatten(data.get(key))
            if value.strip():
                return value
        return ""

    async def list_models(self) -> list[str]:
        """读取 OpenAI 兼容服务的 GET /models 列表。"""
        if not self.base_url:
            raise AIError("请先填写 AI 服务地址")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(self._url("models"), headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AIError(f"获取模型列表失败：{self._error_text(exc)}") from exc

        values = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
        if not isinstance(values, list):
            raise AIError("获取模型列表失败：服务返回格式不是模型列表")
        models: list[str] = []
        for item in values:
            model_id = item.get("id") if isinstance(item, dict) else item
            model_id = str(model_id or "").strip()
            if model_id and model_id not in models:
                models.append(model_id)
        if not models:
            raise AIError("服务返回的模型列表为空")
        return models

    async def test_connection(self) -> None:
        """用当前模型发送最小请求，成功返回；失败抛出中文错误。"""
        if not self.base_url:
            raise AIError("请先填写 AI 服务地址")
        if not self.model:
            raise AIError("请先选择或填写 AI 模型")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "请只回复：连接成功"}],
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(self._url("chat/completions"), json=body, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AIError(f"连接测试失败：{self._error_text(exc)}") from exc
        value = self._content_value(data)
        if not value.strip():
            raise AIError("连接测试失败：模型返回格式无法解析")

    async def translate(self, text: str) -> str:
        if not self.enabled or not self.base_url or not self.model:
            raise AIError("AI 未配置，请在 AstrBot WebUI 中填写 AI 地址、模型并启用 AI")
        return await self.generate(
            f"画面描述：{text}",
            system_prompt=DANBOORU_SYSTEM_PROMPT if self.to_danbooru else PLAIN_SYSTEM_PROMPT,
            require_enabled=True,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str,
        require_enabled: bool = False,
        max_tokens: int | None = None,
        preserve_newlines: bool = False,
    ) -> str:
        """调用插件配置的 OpenAI 兼容接口生成文本。"""
        if require_enabled and not self.enabled:
            raise AIError("插件独立 AI 未启用")
        if not self.base_url or not self.model:
            raise AIError("插件独立 AI 未配置，请在 AstrBot WebUI 中填写 AI 地址和模型")
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if max_tokens is not None:
            body["max_tokens"] = max(1, int(max_tokens))
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(self._url("chat/completions"), json=body, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AIError(f"AI 请求失败：{self._error_text(exc)}") from exc
        value = self._content_value(data)
        if not value.strip():
            raise AIError("AI 返回格式无法解析")
        result = (
            value.strip(" `\n\t")
            if preserve_newlines
            else value.replace("\n", ", ").strip(" `,，。；;")
        )
        if not result:
            raise AIError("AI 返回了空内容")
        return result
