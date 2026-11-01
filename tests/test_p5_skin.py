"""WebUI P5 风格皮肤的回归测试。

皮肤只改展示层：额外的样式、字体和几个小脚本挂在现有控制台上，
不改 app.js 的任何逻辑。这里锁住几条容易在后续改版中被悄悄破坏的约定：

* 皮肤文件在根目录和 pages/console 各有一份，内容必须一致，且都进发布包；
* 页面先加载字体、再加载样式；皮肤脚本必须排在 app.js 之后，按 fx → ui → bg → forms 顺序；
* 图标统一走页面内的 P5 图标精灵（<symbol> + <use>），不再混用 lucide 内联线稿；
* 表单增强（数字框按钮、滑杆、计数、复制路径）只加不带 id / data-* 的展示控件，
  不改 app.js 读写的字段，标签点击仍然聚焦原输入框；
* 字体以 base64 内嵌，不依赖外网或额外的二进制文件；
* AstrBot 的插件页面跑在 sandbox iframe 里，localStorage 可能直接抛错，必须 try/catch；
* 皮肤脚本不发请求、不碰插件接口，只做展示。
"""

from __future__ import annotations

import base64
import importlib.util
import re
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
CONSOLE_DIR = PLUGIN_DIR / "pages" / "console"
SKIN_FILES = ("p5-fonts.css", "p5-fx.js", "p5-ui.js", "p5-bg.js", "p5-forms.js")
SKIN_SCRIPTS = ("p5-fx.js", "p5-ui.js", "p5-bg.js", "p5-forms.js")
PAGES = ("index.html", "pages/console/index.html")


def _read(relative: str) -> str:
    return (PLUGIN_DIR / relative).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", SKIN_FILES)
def test_skin_files_exist_in_both_copies_and_stay_identical(name: str) -> None:
    root_copy = PLUGIN_DIR / name
    page_copy = CONSOLE_DIR / name
    assert root_copy.is_file() and page_copy.is_file(), name
    assert root_copy.read_bytes() == page_copy.read_bytes(), name
    assert b"\r\n" not in page_copy.read_bytes(), f"{name} 应使用 LF 换行"


def test_skin_files_are_shipped_in_release_package() -> None:
    script = PLUGIN_DIR / "scripts" / "build_release.py"
    spec = importlib.util.spec_from_file_location("release_builder_p5", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in SKIN_FILES:
        assert name in module.RELEASE_FILES, name
        assert f"pages/console/{name}" in module.RELEASE_FILES, name


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_fonts_before_style_and_skin_scripts_after_app(page: str) -> None:
    html = _read(page)
    fonts = html.index('href="p5-fonts.css?v=')
    style = html.index('href="style.css?v=')
    assert fonts < style
    app = html.index('src="app.js?v=')
    positions = [html.index(f'src="{name}?v=') for name in SKIN_SCRIPTS]
    assert app < positions[0] < positions[1] < positions[2] < positions[3]
    # 皮肤脚本不能抢在 AstrBot 桥接脚本前面，否则 app.js 的初始化顺序会变。
    assert html.index("bridge-sdk.js") < app


@pytest.mark.parametrize("page", PAGES)
def test_page_keeps_existing_hooks_and_adds_skin_mount_points(page: str) -> None:
    html = _read(page)
    # app.js 依赖的原有元素和文字不能因为换皮被改名或删掉。
    for hook in ('id="themeToggle"', 'id="status"', 'id="statusText"', 'id="toast"', 'id="version"', "普通 LoRA</h2>"):
        assert hook in html, hook
    for mount in ('id="p5Bg"', 'id="p5Fx"', 'id="p5Motion"', 'id="p5PaletteBtn"', 'id="p5OffText"'):
        assert html.count(mount) == 1, mount
    # 装饰层对读屏软件隐藏
    for decorative in ('<canvas id="p5Bg" class="p5-bg" aria-hidden="true">', 'class="p5-marq" aria-hidden="true"', 'id="p5Fx" class="p5-fx" aria-hidden="true"'):
        assert decorative in html, decorative


def test_fonts_are_embedded_as_valid_woff2_and_style_has_no_external_fonts() -> None:
    css = _read("pages/console/p5-fonts.css")
    faces = re.findall(r'font-family: "([^"]+)";\s*src: url\("data:font/woff2;base64,([A-Za-z0-9+/=]+)"\)', css)
    assert {family for family, _ in faces} == {"P5 Sans", "P5 Serif"}
    for _, payload in faces:
        assert base64.b64decode(payload)[:4] == b"wOF2"
    style = _read("pages/console/style.css")
    assert "@font-face" not in style
    assert not re.search(r"url\(\s*[\"']?(https?:)?//", style), "样式不应引用外网资源"
    assert '"P5 Sans"' in style and '"P5 Serif"' in style


@pytest.mark.parametrize("name", SKIN_SCRIPTS)
def test_skin_scripts_guard_storage_and_never_call_plugin_apis(name: str) -> None:
    js = _read(f"pages/console/{name}")
    for line in js.splitlines():
        if "localStorage." in line:
            assert "try {" in line and "catch" in line, line.strip()
    for forbidden in ("fetch(", "XMLHttpRequest", "AstrBotPluginPage", "bridge", "apiGet", "apiPost", "eval("):
        assert forbidden not in js, f"{name} 不应包含 {forbidden}"


def test_skin_respects_lite_motion_and_reduced_motion() -> None:
    ui = _read("pages/console/p5-ui.js")
    style = _read("pages/console/style.css")
    assert "prefers-reduced-motion" in ui
    assert '"comfyui-ai-studio-motion"' in ui
    assert ':root[data-motion="lite"]' in style
    assert "@media (prefers-reduced-motion: reduce)" in style


def test_skin_keeps_theme_tokens_for_light_and_dark() -> None:
    style = _read("pages/console/style.css")
    assert "T4-E 换肤层" in style
    assert ':root[data-ui-theme="dark"]' in style
    assert ".lora-simple-switch" in style
    # 按钮的基础样式用 :where() 降权，全局 reset 也必须同样降权，否则按钮文字颜色会被继承覆盖。
    assert ":where(button, input, select, textarea) { font: inherit; color: inherit; }" in style
    assert "\nbutton, input, select, textarea {" not in style


@pytest.mark.parametrize("page", PAGES)
def test_page_icons_come_from_single_p5_sprite(page: str) -> None:
    html = _read(page)
    assert html.count('class="p5-sprite"') == 1
    sprite = html[html.index('class="p5-sprite"'):]
    sprite = sprite[: sprite.index("</svg>")]
    defined = set(re.findall(r'<symbol id="(i-[a-z-]+)" viewBox="0 0 24 24">', sprite))
    used = re.findall(r'<svg class="ui-ic" aria-hidden="true"><use href="#(i-[a-z-]+)"/></svg>', html)
    assert len(used) >= 50
    assert set(used) <= defined, set(used) - defined
    # 精灵要放在 body 最前面，<use> 引用时符号已经存在；不能再出现旧的 lucide 内联线稿图标。
    assert html.index("<body>") < html.index('class="p5-sprite"') < html.index('class="sidebar')
    assert not re.search(r'<svg class="ui-ic" viewBox=', html)
    assert 'stroke-linecap="round"' not in html
    # p5-forms.js 里复制按钮用到的图标也必须在精灵里
    assert "i-copy" in defined


def test_form_enhancements_only_add_presentational_controls() -> None:
    js = _read("pages/console/p5-forms.js")
    # 增强控件不能带 id 或 data-*，否则可能和 app.js 的选择器撞上
    for created in re.findall(r'make\("(?:button|input|span|i|textarea)", "[^"]*"(?:, \{([^}]*)\})?\)', js):
        assert "id:" not in created and "data-" not in created, created
    # 新增按钮都是 type=button 且不进 Tab 顺序（输入框本身仍可键盘操作）
    assert js.count('type: "button", tabindex: "-1"') == 2
    assert 'type: "range", tabindex: "-1", "aria-hidden": "true"' in js
    # 程序回填数值时同步滑杆：只在单个元素上包一层 value，并调用原生 setter
    assert 'Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value")' in js
    assert "desc.set.call(this, v)" in js
    # 剪贴板在 sandbox iframe 里可能被拒绝，必须有 execCommand 兜底且吞掉异常
    assert 'document.execCommand("copy")' in js
    assert "catch (_)" in js
    # 数字框必须保持在标签里的第一个可标注控件位置
    assert "wrap.appendChild(inp)" in js and "wrap.append(dec, inc)" in js


@pytest.mark.parametrize("page", PAGES)
def test_form_sections_keep_every_original_field(page: str) -> None:
    html = _read(page)
    for field in ("width", "height", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
                  "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_model",
                  "img2img_unet_name", "img2img_lora_strength", "img2img_largest_size", "img2img_crop",
                  "img2img_match_input_size", "img2img_sampling_shift", "img2img_filename_prefix"):
        assert html.count(f'id="{field}"') == 1, field
    assert html.count('<div class="form-sub" role="heading" aria-level="3">') >= 9


def test_style_marks_unsaved_pages_and_keeps_lite_motion_quiet() -> None:
    style = _read("pages/console/style.css")
    for view in ("txt2img", "img2img", "lora", "presets", "artist", "workflow", "paths", "ai"):
        assert f':root:has(#view-{view} ' in style and f'[data-view-tab="{view}"] > .p5-tab-dot' in style, view
    assert 'content: "未保存"' in style
    assert "field-sizing: content" in style and "@supports (field-sizing: content)" in style
    assert ':root[data-motion="lite"] :is(.p5-tab-dot' in style
