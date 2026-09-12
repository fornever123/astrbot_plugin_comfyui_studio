from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlparse

import httpx


DEFAULT_TRANSLATION_ENDPOINTS = (
    "https://translate.googleapis.com/translate_a/single",
    "https://translate.google.com/translate_a/single",
    "https://api.mymemory.translated.net/get",
)


class PlainTranslationError(Exception):
    """普通翻译网站不可用或返回格式异常。"""


class PlainTranslator:
    """使用公开翻译网站，把中文提示词翻译成英文，不调用 AI。

    自定义地址始终优先；自定义服务不可用时，自动尝试 Google 和 MyMemory。
    这样可以避免单一翻译网站限流导致整次绘图只能把中文原文交给 ComfyUI。
    """

    def __init__(self, endpoint: str = ""):
        self.endpoint = str(endpoint or "").strip()

    def _endpoints(self) -> list[str]:
        result: list[str] = []
        for endpoint in (self.endpoint, *DEFAULT_TRANSLATION_ENDPOINTS):
            value = str(endpoint or "").strip()
            if value and value not in result:
                result.append(value)
        return result

    @staticmethod
    def _endpoint_label(endpoint: str) -> str:
        parsed = urlparse(endpoint)
        return parsed.netloc or endpoint[:80]

    @staticmethod
    def _clean_result(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        value = html.unescape(value).replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", value).strip(" ,，。；;\t")

    @classmethod
    def _parse_payload(cls, payload: Any) -> str:
        """兼容 Google、MyMemory、LibreTranslate 及简单自建接口。"""
        if isinstance(payload, dict):
            response_data = payload.get("responseData")
            if isinstance(response_data, dict):
                result = cls._clean_result(response_data.get("translatedText"))
                if result:
                    return result
            for key in ("translatedText", "translated_text", "translation", "text", "result"):
                result = cls._clean_result(payload.get(key))
                if result:
                    return result
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        result = cls._clean_result(message.get("content"))
                        if result:
                            return result
                    result = cls._clean_result(choice.get("text"))
                    if result:
                        return result
            return ""

        # Google translate_a/single 的主要结构是：[[[译文, 原文], ...], ...]
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            parts = []
            for item in payload[0]:
                if isinstance(item, list) and item:
                    result = cls._clean_result(item[0])
                    if result:
                        parts.append(result)
            return ", ".join(parts).strip(" ,，。；;")
        return ""

    @classmethod
    def _parse_response(cls, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            raw = cls._clean_result(getattr(response, "text", ""))
            if raw and not raw.startswith(("<", "{")):
                return raw
            return ""
        return cls._parse_payload(payload)

    async def _request_endpoint(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        value: str,
    ) -> str:
        host = urlparse(endpoint).netloc.lower()
        if "mymemory.translated.net" in host:
            response = await client.get(
                endpoint,
                params={"q": value, "langpair": "zh-CN|en"},
            )
            response.raise_for_status()
            result = self._parse_response(response)
            if result:
                return result
            raise PlainTranslationError("MyMemory 返回空内容")

        params = {
            "client": "gtx",
            "sl": "zh-CN",
            "tl": "en",
            "dt": "t",
            "q": value,
        }
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        result = self._parse_response(response)
        if result:
            return result

        # 允许 WebUI 填入 LibreTranslate 或兼容接口地址。
        response = await client.post(
            endpoint,
            json={"q": value, "source": "zh", "target": "en", "format": "text"},
        )
        response.raise_for_status()
        result = self._parse_response(response)
        if result:
            return result
        raise PlainTranslationError("翻译接口返回空内容")

    async def _translate_once(
        self,
        client: httpx.AsyncClient,
        value: str,
    ) -> str:
        errors: list[str] = []
        for endpoint in self._endpoints():
            try:
                result = await self._request_endpoint(client, endpoint, value)
                if result:
                    return result
            except (httpx.HTTPError, ValueError, PlainTranslationError) as exc:
                errors.append(f"{self._endpoint_label(endpoint)}：{exc}")
        detail = "；".join(errors[-3:])
        raise PlainTranslationError(f"普通翻译服务均不可用{f'（{detail}）' if detail else ''}")

    @staticmethod
    def _split_long_text(value: str, limit: int = 420) -> list[str]:
        parts = re.split(r"(?<=[，。！？；,!?;\n])", value)
        chunks: list[str] = []
        current = ""
        for part in parts:
            if not part:
                continue
            if current and len(current) + len(part) > limit:
                chunks.append(current.strip())
                current = ""
            if len(part) > limit:
                for index in range(0, len(part), limit):
                    piece = part[index : index + limit].strip()
                    if piece:
                        chunks.append(piece)
            else:
                current += part
        if current.strip():
            chunks.append(current.strip())
        return chunks or [value]

    async def translate(self, text: str) -> str:
        value = str(text or "").strip()
        if not value or not any("\u4e00" <= char <= "\u9fff" for char in value):
            return value

        headers = {"User-Agent": "AstrBot-ComfyUI-AI-Studio/0.8.0"}
        timeout = httpx.Timeout(12.0, connect=6.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                try:
                    return await self._translate_once(client, value)
                except PlainTranslationError:
                    # 长提示词可能超过翻译网站的单次 URL 限制；短文本继续
                    # 抛出原始错误，长文本再按标点分段重试。
                    if len(value) <= 450:
                        raise
                    chunks = self._split_long_text(value)
                    translated_chunks = []
                    for chunk in chunks:
                        if any("\u4e00" <= char <= "\u9fff" for char in chunk):
                            translated_chunks.append(await self._translate_once(client, chunk))
                        else:
                            translated_chunks.append(chunk)
                    return ", ".join(item for item in translated_chunks if item).strip(" ,，。；;")
        except (httpx.HTTPError, ValueError, PlainTranslationError) as exc:
            if isinstance(exc, PlainTranslationError):
                raise
            raise PlainTranslationError(f"普通翻译网站请求失败：{exc}") from exc


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))
