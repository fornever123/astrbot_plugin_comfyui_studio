"""ComfyUI 插件内置的 Anima 提示词工程师规则。

模板随本插件发布，绘图插件的命令翻译、LLM 绘图和预设翻译都使用同一份
精简规则。模板路径可在 WebUI 中覆盖，默认使用插件内置的 knowledge 目录。
"""

from __future__ import annotations

import re
from pathlib import Path


COMPACT_RULES = """你是 Anima3 模型的提示词工程师。把用户的绘图要求转换为可供 Anima 使用的英文 Danbooru 标签。
只输出一行小写英文、逗号分隔的具体画面提示词；不输出 Markdown、解释、质量词、画师名、LoRA 语法或预设名。
顺序遵循：人数与性别、角色与作品、外观、服装与状态、动作与姿势、表情、镜头与构图、场景环境、细节氛围。
必须保留用户明确要求的动作、姿势、正在进行的行为和场景，不得把“洗澡、奔跑、坐着”等动作遗漏成只有角色名。
角色资料只用于确认身份，不要擅自补充用户没有要求的服装、动作、场景或性格。用户明确提供的预设名和 LoRA 简称由绘图插件单独处理。"""


def _section(text: str, start_title: str, end_title: str) -> str:
    start = text.find(start_title)
    if start < 0:
        return ""
    end = text.find(end_title, start + len(start_title)) if end_title else -1
    return text[start:] if end < 0 else text[start:end]


def _template_candidates(plugin_dir: Path, extra_path: str = "") -> list[Path]:
    """用户配置的模板优先；未配置或读不到时回退到插件内置模板。"""
    candidates: list[Path] = []
    text = str(extra_path or "").strip()
    if text:
        custom = Path(text)
        try:
            is_dir = custom.is_dir()
        except OSError:
            is_dir = False
        # 允许直接指向文件，也允许指向存放「提示词模版.txt」的目录。
        candidates.extend([custom / "提示词模版.txt", custom] if is_dir else [custom])
    candidates.append(plugin_dir / "knowledge" / "提示词模版.txt")
    return candidates


def find_template_path(plugin_dir: Path, extra_path: str = "") -> Path | None:
    """定位已配置的模板，其次使用随插件发布的模板。"""
    for path in _template_candidates(plugin_dir, extra_path):
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def build_context(plugin_dir: Path, maximum: int = 2600, extra_path: str = "") -> str:
    """返回内置 Anima 模板的精简上下文，读取失败时仍返回核心规则。"""
    source = ""
    path = find_template_path(plugin_dir, extra_path)
    if path:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            source = ""
    if not source:
        return COMPACT_RULES
    sections = [
        _section(source, "## 1. ROLE", "## 2. OUTPUT PROTOCOL"),
        _section(source, "## 2. OUTPUT PROTOCOL", "## 3. FINAL SELF-CHECK"),
        _section(source, "## 3. FINAL SELF-CHECK", "## 3.1 CONFLICT TABLE"),
        _section(source, "## 4. SLOT ORDER", "## 5. ASSEMBLY DECISION TREE"),
        _section(source, "## 5. ASSEMBLY DECISION TREE", "## 6. COUNT & IDENTITY"),
    ]
    reference = "\n\n".join(item.strip() for item in sections if item.strip())
    limit = max(800, min(6000, int(maximum or 2600)))
    return f"{COMPACT_RULES}\n\n以下是内置 Anima 提示词工程师模板的精简参考，只遵守结构，不要逐字复述：\n{reference[:limit]}"


def is_drawing_request(message: str) -> bool:
    """判断当前主 LLM 请求是否属于绘图或提示词请求。"""
    return bool(
        re.search(
            r"(帮我|请|想要|给我)?(画|绘|生成|出图|图片|提示词|文生图|图生图|高清放大)",
            str(message or ""),
        )
    )
