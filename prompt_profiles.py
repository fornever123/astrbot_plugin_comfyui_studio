"""模型家族的提示词写法档案（prompt profiles）。

同一句需求交给不同架构的模型，正确的写法并不一样：

* **Anima / Stable Diffusion 系**：英文 Danbooru 标签，逗号分隔，槽位顺序固定。
* **Qwen Image 2.1**：中文自然语言，Qwen3VL 文本编码器直接理解完整句子。
* **Krea 2 Turbo**：中文自然语言，Qwen3VL-4B 编码器 + 强提示跟随，摄影语言与
  具体名词收益最大；CFG 固定 1，负面提示词不参与引导。

本模块只依赖标准库，供 ``prompt_enhancer``（AI 优化）与 ``main``（LLM 工具
规则注入）共用，避免两处各写一套规则而漂移。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ANIMA = "anima"
QWEN21 = "qwen21"
KREA2 = "krea2"

PROFILE_KEYS = (ANIMA, QWEN21, KREA2)

TAGS = "tags"
NATURAL = "natural"


@dataclass(frozen=True)
class PromptProfile:
    """一个模型家族的提示词写法契约。"""

    key: str
    label: str
    #: ``tags`` = 逗号分隔的英文标签；``natural`` = 自然语言整句。
    family: str
    language: str
    #: 提示词里各要素的推荐先后顺序（供 AI 优化与人工参考）。
    slot_order: tuple[str, ...] = ()
    #: CFG=1 时负面提示词不参与引导，必须靠正面描述表达"不要什么"。
    negative_effective: bool = True
    #: 发给提示词优化器的「输出格式」一行。
    enhancer_format: str = ""
    #: 追加在优化模板后的模型专属要求（文生图）。
    enhancer_extra: str = ""
    #: 图生图时追加的编辑要点；留空表示沿用 Qwen 2.1 编辑要求。
    edit_extra: str = ""
    #: 注入 AstrBot 主 LLM 的一句话写法说明。
    llm_style: str = ""
    #: 明确禁止的写法，供文档与测试使用。
    avoid: tuple[str, ...] = field(default_factory=tuple)


ANIMA_PROFILE = PromptProfile(
    key=ANIMA,
    label="Anima / SD 系",
    family=TAGS,
    language="en",
    slot_order=(
        "人数与性别",
        "角色与作品",
        "外观（发色、瞳色、体型）",
        "服装与状态",
        "动作与姿势",
        "表情",
        "镜头与构图",
        "场景与环境",
        "细节与氛围",
    ),
    negative_effective=True,
    enhancer_format=(
        "输出格式：适配 Anima / Stable Diffusion，使用一行英文逗号分隔的 Danbooru 标签，"
        "按人数与性别、角色与作品、外观、服装与状态、动作与姿势、表情、镜头与构图、场景与环境的顺序组织。"
        "全部小写英文，不加 Markdown、不加括号权重、不写句式句子。"
        "不要添加质量词和画师名；由插件的默认正面词与画风设置合并。"
    ),
    enhancer_extra=(
        "本模型额外要求：\n"
        "1. 用户明确要求的动作、姿势、正在进行的行为（洗澡、奔跑、坐着、回头）必须落成标签，"
        "不能退化成只有角色名或只有服装。\n"
        "2. 角色资料只用于确认身份，不要补写用户没要求的服装、动作、场景或性格。\n"
        "3. 标签用 Danbooru 惯例（1girl、solo、upper body、from side），不要生造词。"
    ),
    llm_style=(
        "Anima 模式：prompt 使用一行小写英文 Danbooru 标签，遵循内置 Anima 提示词工程规则。"
    ),
    avoid=("括号权重", "质量词", "画师名", "自然语言长句"),
)

QWEN21_PROFILE = PromptProfile(
    key=QWEN21,
    label="Qwen Image 2.1",
    family=NATURAL,
    language="zh",
    slot_order=(
        "主体与动作",
        "空间关系",
        "环境与时间",
        "光线与氛围",
        "镜头与构图",
        "风格与材质",
    ),
    # 内置 qwen21 工作流 KSampler cfg=1.0：负面分支被 CFG 完全忽略，写负面等于白写。
    negative_effective=False,
    enhancer_format=(
        "输出格式：使用简洁自然的中文自然语言，一段话写完，不用标签堆砌、不用括号权重、不写参数。"
        "保留用户给出的英文专有词与引号内文字。"
        "按主体与动作、空间关系、环境、光线、镜头构图、风格材质的顺序组织。"
    ),
    enhancer_extra=(
        "本模型额外要求：\n"
        "1. 第一句先定框架：媒介 + 风格 + 主体 + 背景色板（例如“这是一张雨夜街头的人像摄影作品，冷蓝与暖橙对撞”），"
        "再按主体、环境、光线、材质、构图、风格逐层展开。\n"
        "2. 用户明确指定的主体、数量、动作、服装、表情、构图、场景、文字必须逐项保留，"
        "不能漏动作、不能换成另一个动作、不能换成另一个角色。\n"
        "3. 颜色必须带修饰词（deep navy 而不是 blue），材质必须具体（brushed metal 而不是 metal），"
        "光线要写来源与方向（“北向窗户进来的柔光”，而不是“好光线”）。\n"
        "4. 图内文字逐字加英文双引号，只写用户给出的原文；透明背景要用官方三句式"
        "“This is an RGBA image with transparency. … The image has alpha channel and the background is transparent.”。\n"
        "5. 该模型内置工作流 cfg=1，负面提示词不参与引导：要排除的内容改写正面表述，不写“不要/避免”。\n"
        "6. 只补少量有助于表达的视觉细节，不添加无关人物、服装或剧情。\n"
        "7. 不写分辨率、宽高比、步数、CFG、模型名、LoRA 权重与预设名。"
    ),
    edit_extra=(
        "本模型额外要求：\n"
        "1. 以编辑动作开头，写清改哪里、改成什么；只改用户指定的属性。\n"
        "2. 未指定的身份、脸、姿势、构图、画风与背景保持原图不变，不重新描述整张原图。\n"
        "3. 多图用 <Picture 1> 表示主图、<Picture 2> 表示参考图，写明从哪张取什么；"
        "未提到的图片不要编造用途。\n"
        "4. 不写分辨率、宽高比、步数、CFG、模型名、LoRA 权重与预设名。"
    ),
    llm_style=(
        "Qwen Image 2.1 模式：输入 prompt 使用简洁、自然的中文画面描述，第一句先定媒介与主体，"
        "不要转换成 Danbooru 标签，不要运行 Anima 模板，也不要重复注入角色预设内容；"
        "该模型 cfg=1，负面提示词不生效，需要排除的内容写进正面描述。"
    ),
    avoid=("Danbooru 标签", "括号权重", "参数", "质量词堆砌", "负面提示词（CFG=1 无效）"),
)

KREA2_PROFILE = PromptProfile(
    key=KREA2,
    label="Krea 2 Turbo",
    family=NATURAL,
    language="zh",
    slot_order=(
        "主体与动作（一句话说清）",
        "环境与时间天气",
        "光线（方向、软硬、色温）",
        "镜头（机位、焦距、景深）",
        "材质与色彩",
        "媒介与风格",
    ),
    # Krea 2 Turbo 采样 cfg=1，负面提示词不参与引导。
    negative_effective=False,
    enhancer_format=(
        "输出格式：使用具体、可拍的中文自然语言，一段话写完（40–120 字为宜），"
        "不用 Danbooru 标签、不用括号权重、不写参数与画师名。"
        "按主体与动作、环境与时间、光线、镜头、材质色彩、媒介风格的顺序组织。"
    ),
    enhancer_extra=(
        "本模型额外要求：\n"
        "1. 文本编码器是 Qwen3VL-4B，中文原句可直接理解；不要翻译成英文标签，也不要把句子拆成关键词。\n"
        "2. 用具体名词和可观察的细节代替抽象赞美：写“傍晚侧逆光，墙面是潮湿的灰绿色瓷砖”，"
        "不要写“绝美光影、极致画质、大师作品”。\n"
        "3. 摄影语言收益最大：机位（低角度/俯视/过肩）、焦距（24mm 广角、85mm 人像）、景深（浅景深虚化背景）、"
        "光线方向（逆光、侧光、柔和窗光）。按需写一到两项即可，不要堆满。\n"
        "4. 该模型 CFG 固定为 1，负面提示词不参与引导：用户想排除的东西（不要有人、不要文字、不要水印）"
        "必须改写成正面表述，例如“空无一人的街道”“画面干净无字”。\n"
        "5. Turbo 只需 8 步，不需要用“超精细、8K、极致细节”堆料；主体与构图写清楚比堆词更有效。\n"
        "6. 用户明确指定的动作、数量、服装、场景必须保留；不补写用户没要求的剧情与人物。\n"
        "7. 词预算即变焦：谁先写、谁写得多，谁就占画面。主体开头且细节密集 = 收紧构图；"
        "环境开头 = 拉远全景；想让人物的脚进画面就写“脚踩在起毛的地毯上”这类锚点短语，不要只写 full body。\n"
        "8. 图内文字用英文双引号写死，长度限一个词到一块短招牌；整段话必然糊。\n"
        "9. 保持用户指定的媒介（照片/插画/油画/3D 渲染），不要为了更容易画而换成别的媒介。"
    ),
    llm_style=(
        "Krea 2 Turbo 模式：prompt 使用具体的中文自然语言整句，写清主体动作、环境、光线与镜头，"
        "不要转成英文 Danbooru 标签；负面提示词在该模型上不生效，需要排除的内容要写进正面描述。"
    ),
    avoid=("Danbooru 标签", "括号权重", "质量词", "画师名", "负面提示词（CFG=1 无效）"),
)

_PROFILES = {
    ANIMA: ANIMA_PROFILE,
    QWEN21: QWEN21_PROFILE,
    KREA2: KREA2_PROFILE,
}

#: 用文件名片段识别模型家族。顺序重要：先判专有家族，再回落 Anima。
_MODEL_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (KREA2, ("krea2", "krea_2", "krea-2", "krea 2")),
    (QWEN21, ("qwen_image_2.1", "qwen_image_2_1", "qwen-image-2.1", "qwen21")),
)


def normalise_engine(value: str) -> str:
    """把 ``engine`` 字段规整成档案 key，未知值返回空串。"""
    key = str(value or "").strip().casefold()
    return key if key in _PROFILES else ""


def detect_profile_key(model: str = "", engine: str = "") -> str:
    """按 engine → 模型文件名 → 默认 Anima 的顺序判定提示词档案。

    Args:
        model: ComfyUI 里的扩散模型文件名（可含目录）。
        engine: ``DrawParams.engine``，例如 ``krea2``。

    Returns:
        ``anima`` / ``qwen21`` / ``krea2`` 之一。
    """
    explicit = normalise_engine(engine)
    if explicit:
        return explicit
    name = str(model or "").replace("\\", "/").casefold()
    for key, markers in _MODEL_MARKERS:
        if any(marker in name for marker in markers):
            return key
    return ANIMA


def resolve_prompt_profile(model: str = "", engine: str = "") -> PromptProfile:
    """返回命中的档案对象；永不返回 ``None``，便于调用方直接用。"""
    return _PROFILES[detect_profile_key(model, engine)]


def get_profile(key: str) -> PromptProfile:
    """按 key 取档案，未知 key 回落到 Anima。"""
    return _PROFILES.get(normalise_engine(key) or ANIMA, ANIMA_PROFILE)


def is_natural_language(model: str = "", engine: str = "", mode: str = "txt2img") -> bool:
    """图生图一律自然语言；文生图看档案家族。"""
    if str(mode or "").strip().casefold() == "img2img":
        return True
    return resolve_prompt_profile(model, engine).family == NATURAL


def negative_is_effective(model: str = "", engine: str = "") -> bool:
    """判断该模型上负面提示词是否参与引导（CFG=1 的 Turbo 系不参与）。"""
    return resolve_prompt_profile(model, engine).negative_effective


def build_enhancer_format_rule(
    model: str = "", engine: str = "", mode: str = "txt2img"
) -> str:
    """给提示词优化器的「输出格式」一行。图生图统一走编辑指令格式。"""
    if str(mode or "").strip().casefold() == "img2img":
        return (
            "输出格式：使用简洁自然的中文自然语言，只输出一段最终编辑指令。"
            "以编辑动作开头，写清改哪里、改成什么；未指定的部分以一句保留说明保护原图，"
            "不逐项重画原图细节。保留用户给出的英文专有词与引号内文字。"
        )
    return resolve_prompt_profile(model, engine).enhancer_format


def build_enhancer_extra(model: str = "", mode: str = "txt2img", engine: str = "") -> str:
    """追加在优化模板后的模型专属要求。

    图生图只有 Qwen 2.1 一条链路，统一用它的编辑要点；没有编辑要点的家族
    （如 Krea 2）留在文生图，不会走到这里。
    """
    if str(mode or "").strip().casefold() == "img2img":
        return QWEN21_PROFILE.edit_extra
    return resolve_prompt_profile(model, engine).enhancer_extra


def build_llm_style(model: str = "", engine: str = "") -> str:
    """注入 AstrBot 主 LLM 的一句话写法说明。"""
    return resolve_prompt_profile(model, engine).llm_style


#: 引擎专业模板文件名（随插件发布在 knowledge/ 下），与 Anima 的模板同一套机制。
KNOWLEDGE_FILES: dict[str, str] = {
    QWEN21: "qwen21_提示词模版.txt",
    KREA2: "krea2_提示词模版.txt",
}

#: 成人向增量包。只在 ``nsfw_prompt_enabled`` 打开、且会话不在黑名单群时叠加。
KNOWLEDGE_FILES_NSFW: dict[str, str] = {
    QWEN21: "qwen21_提示词模版_nsfw.txt",
    KREA2: "krea2_提示词模版_nsfw.txt",
}

#: 各引擎在不同模式下要注入的章节标题；章节按「到下一个 ## 标题为止」截取。
_KNOWLEDGE_SECTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    QWEN21: {
        "txt2img": (
            "## 1. ROLE",
            "## 2. OUTPUT PROTOCOL",
            "## 3. SLOT ORDER",
            "## 4. TEXT RENDERING",
            "## 6. ANTI-PATTERNS",
        ),
        "img2img": ("## 1. ROLE", "## 5. EDIT MODE", "## 6. ANTI-PATTERNS"),
    },
    KREA2: {
        "txt2img": (
            "## 1. ROLE",
            "## 2. OUTPUT PROTOCOL",
            "## 3. SLOT ORDER",
            "## 4. WORD BUDGET",
            "## 5. DECISION TREE",
            "## 7. TEXT IN IMAGE",
        ),
        "img2img": (
            "## 1. ROLE",
            "## 2. OUTPUT PROTOCOL",
            "## 3. SLOT ORDER",
            "## 4. WORD BUDGET",
        ),
    },
}


def _heading_block(text: str, start: str) -> str:
    """截取从 ``start`` 标题起、到下一个二级标题之前的内容。"""
    index = text.find(start)
    if index < 0:
        return ""
    end = text.find("\n## ", index + len(start))
    block = text[index:] if end < 0 else text[index:end]
    return block.strip()


def find_knowledge_path(plugin_dir: Path | str | None, engine: str = "") -> Path | None:
    """定位该引擎随插件发布的专业模板；没有对应档案时返回 ``None``。"""
    key = normalise_engine(engine)
    name = KNOWLEDGE_FILES.get(key)
    if not name or plugin_dir is None:
        return None
    path = Path(plugin_dir) / "knowledge" / name
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def _nsfw_blocks(text: str) -> str:
    """收集 ``## N<数字>.`` 形式的成人向章节（主模板章节用阿拉伯数字，不会冲突）。"""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^## N\d+\.", line):
            if current:
                blocks.append("\n".join(current).strip())
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return "\n\n".join(block for block in blocks if block)


def build_knowledge_context(
    plugin_dir: Path | str | None,
    model: str = "",
    engine: str = "",
    mode: str = "txt2img",
    maximum: int = 2200,
    nsfw: bool = False,
) -> str:
    """返回该引擎专业模板的章节上下文，供提示词优化器与 Anima 知识注入对称使用。

    标签族（Anima）返回空串，由 ``anima_knowledge.build_context`` 负责；
    模板缺失或读取失败时同样返回空串，调用方回落到 ``enhancer_extra`` 的紧凑规则。

    Args:
        plugin_dir: 插件根目录，模板位于其 ``knowledge/`` 子目录。
        model: 已选扩散模型文件名。
        engine: ``DrawParams.engine``，优先级高于文件名。
        mode: ``txt2img`` 或 ``img2img``。
        maximum: 主模板返回文本的字符上限。
        nsfw: 是否叠加成人向增量包（调用方须已通过黑名单与开关判定）。

    Returns:
        由若干章节拼成的参考文本；无可用模板时为空串。
    """
    profile = resolve_prompt_profile(model, engine)
    if profile.family != NATURAL:
        return ""
    path = find_knowledge_path(plugin_dir, profile.key)
    if path is None:
        return ""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    mode_key = "img2img" if str(mode or "").strip().casefold() == "img2img" else "txt2img"
    headings = _KNOWLEDGE_SECTIONS.get(profile.key, {}).get(mode_key, ())
    blocks = [_heading_block(source, heading) for heading in headings]
    body = "\n\n".join(block for block in blocks if block)
    if not body:
        return ""
    limit = max(600, min(6000, int(maximum or 2200)))
    body = body[:limit]
    if nsfw:
        extra = _load_nsfw_context(plugin_dir, profile.key)
        if extra:
            body = f"{body}\n\n{extra}"
    return body


def _load_nsfw_context(
    plugin_dir: Path | str | None, key: str, maximum: int = 3800
) -> str:
    """读取并截取成人向增量包；文件缺失时返回空串（不抛异常）。

    上限按整份增量包取（约 3.7k 字），确保引擎差异与自查章节不会被截断。
    """
    name = KNOWLEDGE_FILES_NSFW.get(key)
    if not name or plugin_dir is None:
        return ""
    path = Path(plugin_dir) / "knowledge" / name
    try:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    return _nsfw_blocks(text)[: max(600, min(6000, int(maximum or 3800)))]


def describe(model: str = "", engine: str = "") -> dict[str, object]:
    """给 WebUI / 命令回复用的可序列化摘要。"""
    profile = resolve_prompt_profile(model, engine)
    return {
        "key": profile.key,
        "label": profile.label,
        "family": profile.family,
        "language": profile.language,
        "slot_order": list(profile.slot_order),
        "negative_effective": profile.negative_effective,
        "avoid": list(profile.avoid),
        "knowledge": KNOWLEDGE_FILES.get(profile.key, ""),
    }
