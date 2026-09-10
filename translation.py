from __future__ import annotations

from typing import Any

import httpx


class PlainTranslationError(Exception):
    """普通翻译网站不可用或返回格式异常。"""


class PlainTranslator:
    """使用公开翻译网页接口，把中文提示词翻译成英文，不调用任何 AI。"""

    def __init__(self, endpoint: str = "https://translate.googleapis.com/translate_a/single"):
        self.endpoint = str(endpoint or "").strip()

    async def translate(self, text: str) -> str:
        value = str(text or "").strip()
        if not value or not any("\u4e00" <= char <= "\u9fff" for char in value):
            return value
        if not self.endpoint:
            raise PlainTranslationError("未配置普通翻译网站地址")
        params = {
            "client": "gtx",
            "sl": "zh-CN",
            "tl": "en",
            "dt": "t",
            "q": value,
        }
        headers = {"User-Agent": "AstrBot-ComfyUI-AI-Studio/0.6.4"}
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                response = await client.get(self.endpoint, params=params, headers=headers)
                response.raise_for_status()
                data: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise PlainTranslationError(f"普通翻译网站请求失败：{exc}") from exc

        try:
            parts = [str(item[0]) for item in data[0] if isinstance(item, list) and item and item[0]]
        except (IndexError, TypeError, KeyError) as exc:
            raise PlainTranslationError("普通翻译网站返回格式无法解析") from exc
        result = ", ".join(part.strip() for part in parts if part.strip()).strip(" ,，")
        if not result:
            raise PlainTranslationError("普通翻译网站返回了空内容")
        return result


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))
