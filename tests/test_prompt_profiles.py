from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


def test_detects_profile_from_engine_and_model_name():
    from prompt_profiles import detect_profile_key

    assert detect_profile_key("", "krea2") == "krea2"
    assert detect_profile_key("krea2_turbo-Q4_K_M.gguf") == "krea2"
    assert detect_profile_key("krea2_turbo_4step_rank_64_lora.safetensors") == "krea2"
    for name in (
        "qwen_image_2.1_bf16.safetensors",
        "qwen_image_2_1_bf16.safetensors",
        "Qwen-Image-2.1-Turbo.safetensors",
    ):
        assert detect_profile_key(name) == "qwen21"
    assert detect_profile_key("anima-2.9b.safetensors") == "anima"
    assert detect_profile_key("") == "anima"
    # engine 优先于文件名：Krea 2 用的是 qwen3vl 文本编码器，不能被名字带偏。
    assert detect_profile_key("qwen_image_2.1_bf16.safetensors", "krea2") == "krea2"


def test_krea2_is_natural_language_and_ignores_negative_prompt():
    from prompt_profiles import negative_is_effective, resolve_prompt_profile

    profile = resolve_prompt_profile("krea2_turbo-Q4_K_M.gguf")
    assert profile.family == "natural"
    assert profile.language == "zh"
    assert negative_is_effective("krea2_turbo-Q4_K_M.gguf") is False
    # 内置 qwen21 工作流 cfg=1.0，负面分支被 CFG 完全忽略，不能当有效。
    assert negative_is_effective("qwen_image_2.1_bf16.safetensors") is False
    assert negative_is_effective("anima-2.9b.safetensors") is True


def test_unknown_engine_falls_back_to_filename_detection():
    from prompt_profiles import detect_profile_key, get_profile

    assert detect_profile_key("krea2_turbo.gguf", "not-an-engine") == "krea2"
    assert get_profile("nope").key == "anima"


def test_enhancer_prompt_uses_tags_only_for_anima():
    from prompt_enhancer import build_system_prompt

    anima = build_system_prompt({}, "txt2img", "anima-2.9b.safetensors")
    qwen = build_system_prompt({}, "txt2img", "qwen_image_2.1_bf16.safetensors")
    krea = build_system_prompt({}, "txt2img", "krea2_turbo-Q4_K_M.gguf")
    # engine 显式指定时即便文件名像 Qwen 也走 Krea 2 档案。
    krea_by_engine = build_system_prompt(
        {}, "txt2img", "qwen_image_2.1_bf16.safetensors", "krea2"
    )

    assert "Danbooru" in anima and "英文" in anima
    assert "自然语言" in qwen and "Danbooru" not in qwen.split("输出格式")[-1]
    krea_rule = krea.split("输出格式")[-1]
    assert "自然语言" in krea_rule
    # Krea 2 的格式要求是「不要 Danbooru 标签」，不是「使用 Danbooru 标签」。
    assert "不用 Danbooru 标签" in krea_rule
    assert "负面提示词" in krea and "CFG" in krea
    assert krea_by_engine == krea


def test_krea2_enhancer_spells_out_positive_phrasing_for_exclusions():
    from prompt_enhancer import build_system_prompt

    prompt = build_system_prompt({}, "txt2img", "", "krea2")
    assert "空无一人的街道" in prompt
    assert "负面提示词" in prompt
    # Anima 的默认正面词/画师串不该出现在 Krea 2 规则里。
    assert "masterpiece" not in prompt


def test_img2img_rules_stay_edit_first_for_every_family():
    from prompt_enhancer import build_system_prompt

    for model, engine in (
        ("qwen_image_2.1_bf16.safetensors", ""),
        ("krea2_turbo-Q4_K_M.gguf", "krea2"),
        ("anima-2.9b.safetensors", ""),
    ):
        prompt = build_system_prompt({}, "img2img", model, engine)
        assert "<Picture 1>" in prompt
        assert "以编辑动作开头" in prompt
        assert "不重新描述整张原图" in prompt


def test_describe_reports_slot_order_and_negative_policy():
    from prompt_profiles import describe

    summary = describe("krea2_turbo-Q4_K_M.gguf")
    assert summary["key"] == "krea2"
    assert summary["negative_effective"] is False
    assert summary["slot_order"]
    assert "负面提示词（CFG=1 无效）" in summary["avoid"]
    assert summary["knowledge"] == "krea2_提示词模版.txt"


def test_engine_knowledge_templates_ship_with_the_plugin():
    from prompt_profiles import KNOWLEDGE_FILES, build_knowledge_context

    for engine, filename in KNOWLEDGE_FILES.items():
        assert (PLUGIN_DIR / "knowledge" / filename).is_file()

    krea = build_knowledge_context(PLUGIN_DIR, "", "krea2", "txt2img")
    # Krea 2 的反直觉规则必须进上下文，否则等于没装专业模板。
    assert "WORD BUDGET" in krea and "变焦" in krea
    assert "光源" in krea
    assert "双引号" in krea

    qwen_t2i = build_knowledge_context(PLUGIN_DIR, "", "qwen21", "txt2img")
    assert "首句定框架" in qwen_t2i
    assert "RGBA image with transparency" in qwen_t2i

    qwen_edit = build_knowledge_context(PLUGIN_DIR, "", "qwen21", "img2img")
    assert "EDIT MODE" in qwen_edit
    assert "属性解耦" in qwen_edit
    assert "泄漏" in qwen_edit and "改不到位" in qwen_edit
    # 编辑模式不该把文生图的分层章节一起塞进来。
    assert "SLOT ORDER" not in qwen_edit

    # 标签族没有引擎模板，返回空串，由 Anima 自己的知识库负责。
    assert build_knowledge_context(PLUGIN_DIR, "anima-2.9b.safetensors") == ""
    # 模板缺失时不能抛异常。
    assert build_knowledge_context(None, "", "krea2") == ""
