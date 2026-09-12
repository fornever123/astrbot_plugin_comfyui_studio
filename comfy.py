from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx


class ComfyError(Exception):
    pass


MODEL_TYPES = (
    "diffusion_models", "checkpoints", "loras", "upscale_models", "vae",
    "text_encoders", "unet", "unet_gguf", "clip_gguf",
)


class ComfyClient:
    def __init__(self, url: str, timeout: float = 60):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = f"astrbot-{int(time.time() * 1000)}"

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(method, f"{self.url}{path}", **kwargs)
                response.raise_for_status()
                if not response.content:
                    return None
                try:
                    return response.json()
                except ValueError:
                    return response.content
        except httpx.ConnectError as exc:
            raise ComfyError(f"无法连接 ComfyUI：{self.url}，请确认服务已启动") from exc
        except httpx.TimeoutException as exc:
            raise ComfyError("ComfyUI 请求超时") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise ComfyError(f"ComfyUI 接口错误 HTTP {exc.response.status_code}：{detail}") from exc
        except httpx.HTTPError as exc:
            raise ComfyError(f"ComfyUI 请求失败：{exc}") from exc

    async def status(self) -> dict[str, Any]:
        return await self.request("GET", "/system_stats")

    async def object_info(self) -> dict[str, Any]:
        data = await self.request("GET", "/object_info")
        return data if isinstance(data, dict) else {}

    async def models(self, category: str) -> list[str]:
        if category not in MODEL_TYPES:
            return []
        try:
            data = await self.request("GET", f"/models/{category}")
            values = data.get("models", data) if isinstance(data, dict) else data
            if isinstance(values, list):
                return self._unique_model_names(values)
        except ComfyError as exc:
            # ComfyUI 0.30+ 默认不再提供 /models/{category}，但 /object_info
            # 仍包含所有加载器的实际下拉选项。不要让旧接口 404 造成“未检测到”。
            if "HTTP 404" not in str(exc):
                raise
        return self.models_from_object_info(await self.object_info(), category)

    @staticmethod
    def _unique_model_names(values: list[Any]) -> list[str]:
        names: list[str] = []
        for value in values:
            name = value if isinstance(value, str) else value.get("name") if isinstance(value, dict) else ""
            name = str(name or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    @classmethod
    def models_from_object_info(cls, data: dict[str, Any], category: str) -> list[str]:
        """从 ComfyUI /object_info 提取模型加载器的下拉值。"""
        node_inputs: dict[str, tuple[str, ...]] = {
            "diffusion_models": (("UNETLoader", "unet_name"),),
            "checkpoints": (("CheckpointLoaderSimple", "ckpt_name"),),
            "loras": (("LoraLoader", "lora_name"), ("LoraLoaderModelOnly", "lora_name")),
            "upscale_models": (("UpscaleModelLoader", "model_name"),),
            "clip_gguf": (("CLIPLoaderGGUF", "clip_name"),),
            "vae": (("VAELoader", "vae_name"),),
            "unet_gguf": (("UnetLoaderGGUF", "unet_name"),),
        }
        names: list[str] = []
        for node_name, input_name in node_inputs.get(category, ()):
            node = data.get(node_name, {}) if isinstance(data, dict) else {}
            required = node.get("input", {}).get("required", {}) if isinstance(node, dict) else {}
            spec = required.get(input_name) if isinstance(required, dict) else None
            values = spec[0] if isinstance(spec, list) and spec and isinstance(spec[0], list) else []
            for value in values:
                name = str(value or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    async def upload_image(self, path: str) -> dict[str, str]:
        if not os.path.isfile(path):
            raise ComfyError(f"图片不存在：{path}")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(path, "rb") as image_file:
                    response = await client.post(
                        f"{self.url}/upload/image",
                        files={"image": (Path(path).name, image_file)},
                        data={"overwrite": "true"},
                    )
                    response.raise_for_status()
                    data = response.json()
        except httpx.HTTPError as exc:
            raise ComfyError(f"上传图片失败：{exc}") from exc
        return {
            "name": str(data.get("name", "")),
            "subfolder": str(data.get("subfolder", "")),
            "type": str(data.get("type", "input")),
        }

    async def queue(self, workflow: dict[str, Any]) -> str:
        data = await self.request("POST", "/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if not isinstance(data, dict) or not data.get("prompt_id"):
            node_errors = data.get("node_errors") if isinstance(data, dict) else None
            raise ComfyError(f"ComfyUI 拒绝工作流：{node_errors or data}")
        return str(data["prompt_id"])

    async def wait(self, prompt_id: str, timeout: int = 900) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1.2)
            data = await self.request("GET", f"/history/{prompt_id}")
            item = data.get(prompt_id) if isinstance(data, dict) else None
            if not isinstance(item, dict):
                continue
            status = item.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return item
            for message in status.get("messages", []):
                if isinstance(message, list) and message and message[0] in {"execution_error", "execution_interrupted"}:
                    payload = message[1] if len(message) > 1 else {}
                    raise ComfyError(f"ComfyUI 生成失败：{payload}")
        raise ComfyError(f"生成超时：{timeout} 秒")

    async def download_outputs(self, history: dict[str, Any], directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        result: list[Path] = []
        for node_output in (history.get("outputs") or {}).values():
            if not isinstance(node_output, dict):
                continue
            for image in node_output.get("images", []):
                if not isinstance(image, dict) or not image.get("filename"):
                    continue
                # PreviewImage and Image Comparer nodes also expose an
                # ``images`` list, but those are temporary previews of the
                # same picture. Only forward files written by SaveImage.
                if str(image.get("type", "output")).lower() != "output":
                    continue
                query = urlencode({
                    "filename": image.get("filename", ""),
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                })
                data = await self.request("GET", f"/view?{query}")
                if not isinstance(data, bytes):
                    continue
                target = directory / str(image["filename"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                result.append(target)
        return result
