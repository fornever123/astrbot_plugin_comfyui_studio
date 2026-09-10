from __future__ import annotations

import os
import sys
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
