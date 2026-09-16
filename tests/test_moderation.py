"""图片安全审核（输入 / 输出双向）的回归测试。

审核引擎与插件接线是两块独立逻辑：

* ``moderation.py`` 只负责“问一次多模态模型并解析判定”，用假的 httpx 客户端
  覆盖成功、JSON 围栏、网络异常和 fail_open / fail_closed 四条路径。
* ``main.py`` 只负责“哪些会话要检测、命中后发什么”，用 ``object.__new__``
  构造插件、注入假审核器来验证名单匹配、@ 提醒和结果链解析。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[1]
ASTRBOT_DIR = Path(r"E:\mcmbot\AstrBot\AstrBot")
sys.path.insert(0, str(ASTRBOT_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))


# ----------------------------------------------------------------------------
# moderation.py：判定协议与失败策略
# ----------------------------------------------------------------------------


def test_strictness_block_rules_are_progressive() -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import judge, normalize_strictness

    # 泳装 / 擦边：宽松和标准都放行，只有严格才拦。
    assert judge("borderline", "none", "loose") == (False, "")
    assert judge("borderline", "none", "standard") == (False, "")
    assert judge("borderline", "none", "strict") == (True, "nsfw")

    # 性暗示：标准起拦。
    assert judge("suggestive", "none", "loose") == (False, "")
    assert judge("suggestive", "none", "standard") == (True, "nsfw")

    # 猎奇独立判定：轻微血迹在标准下即拦。
    assert judge("safe", "mild", "loose") == (False, "")
    assert judge("safe", "mild", "standard") == (True, "guro")
    assert judge("safe", "severe", "loose") == (True, "guro")

    # 两轴同时命中时给出合并类别，方便日志和警告文案区分。
    assert judge("explicit", "severe", "standard") == (True, "nsfw+guro")

    # 未知尺度回退到标准，绝不因为拼错而变成“全部放行”。
    assert normalize_strictness("unknown") == "standard"
    assert normalize_strictness("") == "standard"
    assert normalize_strictness("STRICT") == "strict"


def test_prompt_declares_json_protocol_and_scale() -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import build_prompt

    prompt = build_prompt("strict")
    assert '"level"' in prompt and '"guro"' in prompt and '"reason"' in prompt
    assert "只输出一个 JSON 对象" in prompt
    # 尺度提示必须真的随配置改变，否则 WebUI 上的三档就是摆设。
    assert "任何擦边" in prompt
    assert "普通泳装" in build_prompt("standard")
    assert "泳装、内衣、擦边暗示一律视为安全" in build_prompt("loose")


@pytest.mark.parametrize(
    "raw, expected_level, expected_guro",
    [
        ('{"level": "safe", "guro": "none"}', "safe", "none"),
        ('```json\n{"level":"explicit","guro":"severe"}\n```', "explicit", "severe"),
        ('结论如下：{"level":"NSFW","guro":"无"} 以上。', "suggestive", "none"),
        ('{"nsfw": "porn", "gore": "gore"}', "explicit", "severe"),
        ('{"level": "questionable", "guro": "light"}', "suggestive", "mild"),
    ],
)
def test_model_reply_aliases_are_normalized(raw: str, expected_level: str, expected_guro: str) -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import (
        _extract_json_object,
        _normalize_guro,
        _normalize_level,
    )

    parsed = _extract_json_object(raw)
    assert parsed is not None
    level = _normalize_level(parsed.get("level", parsed.get("nsfw", "")))
    guro = _normalize_guro(parsed.get("guro", parsed.get("gore", "")))
    assert (level, guro) == (expected_level, expected_guro)


def test_unparsable_reply_is_not_reported_as_safe() -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import _extract_json_object

    assert _extract_json_object("") is None
    assert _extract_json_object("图片看起来没问题") is None
    assert _extract_json_object("[1, 2, 3]") is None


def _fake_httpx(monkeypatch, *, content: str = "", error: Exception | None = None):
    """替换 httpx.AsyncClient，记录请求并返回固定响应或抛出固定异常。"""
    from astrbot_plugin_comfyui_ai_studio import moderation

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    class FakeClient:
        def __init__(self, timeout=None) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["body"] = json
            captured["headers"] = headers
            if error is not None:
                raise error
            return FakeResponse()

    monkeypatch.setattr(moderation.httpx, "AsyncClient", FakeClient)
    return captured


def _write_probe_image(tmp_path: Path) -> str:
    from astrbot_plugin_comfyui_ai_studio.moderation import ImageModerator

    target = tmp_path / "probe.png"
    target.write_bytes(ImageModerator._probe_image())
    return str(target)


def test_request_sends_image_data_url_and_blocks_explicit_image(monkeypatch, tmp_path) -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import ImageModerator, ModerationConfig

    captured = _fake_httpx(
        monkeypatch,
        content='```json\n{"level":"explicit","guro":"none","reason":"裸露"}\n```',
    )
    config = ModerationConfig(
        base_url="https://example.com/v1",
        api_key="secret-key",
        model="vision-x",
        strictness="standard",
    )
    verdict = asyncio.run(ImageModerator(config).check(_write_probe_image(tmp_path)))

    assert verdict.blocked is True
    assert verdict.category == "nsfw"
    assert (verdict.level, verdict.guro) == ("explicit", "none")
    assert verdict.reason == "裸露"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    body = captured["body"]
    assert body["model"] == "vision-x"
    image_part = body["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/")
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


def test_fail_open_releases_and_fail_closed_blocks_on_network_error(monkeypatch, tmp_path) -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import ImageModerator, ModerationConfig

    image = _write_probe_image(tmp_path)
    _fake_httpx(monkeypatch, error=httpx.ConnectError("连接被拒绝"))

    open_config = ModerationConfig(
        base_url="https://example.com/v1", model="vision-x", fail_open=True
    )
    open_verdict = asyncio.run(ImageModerator(open_config).check(image))
    assert open_verdict.blocked is False
    assert open_verdict.failed is True
    assert "审核请求失败" in open_verdict.error

    closed_config = ModerationConfig(
        base_url="https://example.com/v1", model="vision-x", fail_open=False
    )
    closed_verdict = asyncio.run(ImageModerator(closed_config).check(image))
    assert closed_verdict.blocked is True
    assert closed_verdict.category == "error"


def test_unconfigured_moderator_skips_instead_of_blocking() -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import ImageModerator, ModerationConfig

    moderator = ImageModerator(ModerationConfig())
    assert moderator.available is False
    verdict = asyncio.run(moderator.check("不存在的文件.png"))
    assert verdict.skipped is True
    assert verdict.blocked is False


def test_moderation_config_reads_webui_keys() -> None:
    from astrbot_plugin_comfyui_ai_studio.moderation import ModerationConfig

    config = ModerationConfig.from_mapping(
        {
            "moderation_base_url": "https://api.example.com/v1/chat/completions/",
            "moderation_api_key": "k",
            "moderation_model": "vision-x",
            "moderation_strictness": "strict",
            "moderation_timeout": "45",
            "moderation_fail_open": "false",
            "moderation_max_side": 768,
        }
    )
    assert config.configured is True
    # 用户常直接粘贴完整接口地址，必须剥离 /chat/completions 后缀。
    assert config.base_url == "https://api.example.com/v1"
    assert config.strictness == "strict"
    assert config.timeout == 45
    assert config.fail_open is False
    assert config.max_side == 768


# ----------------------------------------------------------------------------
# main.py：名单匹配、@ 提醒、结果链解析与接线
# ----------------------------------------------------------------------------


class _Event:
    def __init__(self, group_id: str = "", sender_id: str = "", origin: str = "") -> None:
        self.group_id = group_id
        self.sender_id = sender_id
        self.origin = origin


def _star(**config):
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    star = object.__new__(ComfyUIAIStudio)
    star.config = dict(config)
    star._get = lambda key, default=None: star.config.get(key, default)
    star._group_id = lambda event: getattr(event, "group_id", "")
    star._sender_id = lambda event: getattr(event, "sender_id", "")
    star._origin = lambda event: getattr(event, "origin", "")
    return star


def test_input_and_output_lists_are_independent() -> None:
    star = _star(
        moderation_enabled=True,
        moderation_input_groups="111",
        moderation_output_groups="222",
    )
    in_group = _Event(group_id="111")
    out_group = _Event(group_id="222")
    other = _Event(group_id="333")

    assert star._moderation_enabled_for(in_group, "input") is True
    assert star._moderation_enabled_for(in_group, "output") is False
    assert star._moderation_enabled_for(out_group, "input") is False
    assert star._moderation_enabled_for(out_group, "output") is True
    assert star._moderation_enabled_for(other, "input") is False
    assert star._moderation_enabled_for(other, "output") is False


def test_wildcard_and_sender_id_both_select_sessions() -> None:
    wildcard = _star(
        moderation_enabled=True,
        moderation_input_groups="全部",
        moderation_output_groups="",
    )
    assert wildcard._moderation_enabled_for(_Event(group_id="123456"), "input") is True
    # 输出端名单为空表示不检测，不能被 `*` 的语义误伤。
    assert wildcard._moderation_enabled_for(_Event(group_id="123456"), "output") is False

    by_sender = _star(
        moderation_enabled=True,
        moderation_input_groups="",
        moderation_output_groups="999\n888",
    )
    assert by_sender._moderation_enabled_for(
        _Event(group_id="123456", sender_id="999"), "output"
    ) is True
    assert by_sender._moderation_enabled_for(
        _Event(group_id="123456", sender_id="777"), "output"
    ) is False


def test_private_session_can_be_listed_by_full_origin() -> None:
    star = _star(
        moderation_enabled=True,
        moderation_input_groups="aiocqhttp:FriendMessage:12345",
        moderation_output_groups="",
    )
    assert star._moderation_enabled_for(
        _Event(origin="aiocqhttp:FriendMessage:12345"), "input"
    ) is True
    assert star._moderation_enabled_for(
        _Event(origin="aiocqhttp:FriendMessage:99999"), "input"
    ) is False


def test_moderation_disabled_or_unlisted_never_detects() -> None:
    disabled = _star(
        moderation_enabled=False,
        moderation_input_groups="111",
        moderation_output_groups="*",
    )
    assert disabled._moderation_enabled_for(_Event(group_id="111"), "input") is False

    empty = _star(moderation_enabled=True)
    assert empty._moderation_enabled_for(_Event(group_id="111"), "input") is False


def test_verdict_lists_default_to_no_detection() -> None:
    """名单为空字符串时必须返回空集合，而不是“全部会话”。"""
    star = _star(moderation_enabled=True)
    assert star._moderation_targets("") == set()
    assert star._moderation_targets("  ") == set()


class _FakeModerator:
    def __init__(self, verdict, available: bool = True) -> None:
        self.verdict = verdict
        self.available = available
        self.checked: list[str] = []

    async def check_many(self, paths):
        self.checked.extend(str(path) for path in paths)
        return self.verdict


def _blocked_verdict():
    from astrbot_plugin_comfyui_ai_studio.moderation import ModerationVerdict

    return ModerationVerdict(
        blocked=True, level="explicit", guro="severe", reason="裸露与血腥", category="nsfw+guro"
    )


def test_input_gate_returns_at_warning_chain_and_skips_when_allowed() -> None:
    from astrbot.api.message_components import At, Plain

    star = _star(moderation_enabled=True, moderation_input_groups="111")
    star._moderator = lambda: _FakeModerator(_blocked_verdict())
    event = _Event(group_id="111", sender_id="999")

    chain = asyncio.run(star._moderation_input_chain(event, "a.png", "", "b.png"))
    assert chain is not None
    assert isinstance(chain[0], At) and str(chain[0].qq) == "999"
    text = "".join(component.text for component in chain if isinstance(component, Plain))
    assert "成人内容 + 血腥猎奇" in text
    assert "裸露与血腥" in text

    # 未列入名单的群直接放行，并且不会去调用审核模型。
    fake = _FakeModerator(_blocked_verdict())
    star._moderator = lambda: fake
    assert asyncio.run(star._moderation_input_chain(_Event(group_id="222"), "a.png")) is None
    assert fake.checked == []


def test_warning_omits_at_when_sender_is_unknown() -> None:
    from astrbot.api.message_components import At, Plain

    star = _star(moderation_enabled=True, moderation_input_groups="111")
    star._moderator = lambda: _FakeModerator(_blocked_verdict())
    # AstrBot 没有发送者字段时 _sender_id 会退化成“会话:xxx”，这种值不能用于 @。
    star._sender_id = lambda event: "会话:default"

    chain = asyncio.run(star._moderation_input_chain(_Event(group_id="111"), "a.png"))
    assert chain is not None
    assert not any(isinstance(component, At) for component in chain)
    assert any(isinstance(component, Plain) for component in chain)


def test_output_gate_marks_params_so_completion_reply_is_skipped() -> None:
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    star = _star(moderation_enabled=True, moderation_output_groups="111")
    fake = _FakeModerator(_blocked_verdict())
    star._moderator = lambda: fake
    params = DrawParams(mode="txt2img", prompt="一只猫")

    from astrbot.api.message_components import Image

    chain = [Image.fromFileSystem("out/1.png")]
    result = asyncio.run(star._moderation_output_chain(_Event(group_id="111"), params, chain))

    assert result is not None
    assert params.moderation_blocked is True
    assert params.moderation_blocked_reason == "output"
    assert len(fake.checked) == 1 and fake.checked[0].endswith("1.png")


def test_output_gate_ignores_text_only_and_empty_chains() -> None:
    from astrbot.api.message_components import Plain
    from astrbot_plugin_comfyui_ai_studio.prompting import DrawParams

    star = _star(moderation_enabled=True, moderation_output_groups="111")
    fake = _FakeModerator(_blocked_verdict())
    star._moderator = lambda: fake
    params = DrawParams()

    assert asyncio.run(star._moderation_output_chain(_Event(group_id="111"), params, [Plain("生成失败")])) is None
    assert asyncio.run(star._moderation_output_chain(_Event(group_id="111"), params, [])) is None
    assert params.moderation_blocked is False
    assert fake.checked == []


def test_chain_image_paths_expands_forward_nodes_and_skips_remote_urls() -> None:
    from astrbot.api.message_components import Image, Node, Nodes, Plain
    from astrbot_plugin_comfyui_ai_studio.main import ComfyUIAIStudio

    chain = [
        Image.fromFileSystem("out/1.png"),
        Plain("说明"),
        Image(file="https://example.com/2.png"),
        Nodes(
            nodes=[
                Node(content=[Image.fromFileSystem("out/3.png")], name="a", uin="1"),
                Node(content=[Plain("文字")], name="a", uin="1"),
            ]
        ),
    ]
    names = [Path(path).name for path in ComfyUIAIStudio._chain_image_paths(chain)]
    assert names == ["1.png", "3.png"]
    assert ComfyUIAIStudio._chain_image_paths(None) == []


def test_unconfigured_moderator_does_not_block_detection() -> None:
    """审核器已配置开关但没填地址/模型时必须放行，而不是把所有图拦下来。"""
    star = _star(moderation_enabled=True, moderation_input_groups="111")
    star._moderator = lambda: _FakeModerator(_blocked_verdict(), available=False)
    event = _Event(group_id="111", sender_id="999")
    assert asyncio.run(star._moderation_input_chain(event, "a.png")) is None


# ----------------------------------------------------------------------------
# 接线与配置：这些断言保证“改一处忘一处”不会静默失效
# ----------------------------------------------------------------------------


def test_runner_and_send_paths_call_both_gates() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    run_body = source[source.index("    async def _run(") : source.index("    def _style_lora_usage_text(")]
    assert "_moderation_input_chain(" in run_body

    for handler in ("_background_draw", "_send_llm_task_result"):
        start = source.index(f"    async def {handler}(")
        end = source.index("\n    async def ", start + 10)
        body = source[start:end]
        assert "_moderation_output_chain(" in body, handler
        assert "await event.send(event.chain_result(blocked_chain))" in body, handler
        # 命中输出审核后必须 return，绝不能落到发送图片的分支。
        assert body.index("_moderation_output_chain(") < body.index("await self._send_forward_chain")


def test_command_entries_check_input_before_start_reply() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    for start_marker, end_marker in (
        ('    async def _command_img2img_mode(', '    @filter.command("图生图"'),
        ("    async def _command_flux2_tool(", '    @filter.command("洗图"'),
        ('    @filter.command("高清放大"', "    @filter.command(\"模型\""),
    ):
        start = source.index(start_marker)
        end = source.index(end_marker, start)
        body = source[start:end]
        assert "_moderation_input_chain(" in body, start_marker
        assert body.index("_moderation_input_chain(") < body.index(
            "start_reply = await self._draw_start_reply_for_command"
        ), start_marker


def test_llm_execute_blocks_before_reserving_draw_limit() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    start = source.index("    async def _llm_execute(")
    end = source.index("        try:\n            await self._reserve_draw_limit(event, params)", start)
    body = source[start:end]
    assert 'self._moderation_verdict(\n            event, "input"' in body
    assert "图片安全审核未通过" in body


def test_webui_and_schema_expose_moderation_settings() -> None:
    page = (PLUGIN_DIR / "pages" / "console" / "index.html").read_text(encoding="utf-8")
    app = (PLUGIN_DIR / "pages" / "console" / "app.js").read_text(encoding="utf-8")
    schema = json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8-sig"))

    config_keys = (
        "moderation_enabled",
        "moderation_input_groups",
        "moderation_output_groups",
        "moderation_base_url",
        "moderation_api_key",
        "moderation_model",
        "moderation_strictness",
        "moderation_timeout",
        "moderation_fail_open",
        "moderation_max_side",
    )
    for element_id in config_keys:
        assert f'id="{element_id}"' in page, element_id
        assert element_id in schema, element_id

    for element_id in ("testModerationConnection", "saveModeration"):
        assert f'id="{element_id}"' in page, element_id

    assert "${API}/config" in app
    assert "moderation_test" in app
    assert "saveModeration" in app


def test_safe_config_hides_moderation_api_key() -> None:
    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    start = source.index("    def _safe_config(")
    end = source.index("    def _ai_request_config(", start)
    body = source[start:end]
    assert 'data.pop("moderation_api_key", None)' in body
    assert 'data["moderation_api_key_configured"]' in body
    # 保存接口对空密钥的处理必须包含审核密钥，否则 WebUI 留空会把已存密钥清掉。
    assert '"ai_api_key", "civitai_token", "moderation_api_key"' in source
    assert '("moderation_test", self.api_moderation_test' in source


def test_writable_config_covers_every_moderation_key() -> None:
    from astrbot_plugin_comfyui_ai_studio.main import MODERATION_CONFIG_KEYS, WRITABLE_CONFIG

    assert set(MODERATION_CONFIG_KEYS) <= WRITABLE_CONFIG
    for key in MODERATION_CONFIG_KEYS:
        assert key in WRITABLE_CONFIG


def test_root_and_page_console_copies_stay_identical() -> None:
    """插件同时保留根目录和 pages/console 两份前端，必须防止只改一份。"""
    for name in ("index.html", "app.js", "style.css"):
        assert (PLUGIN_DIR / name).read_bytes() == (
            PLUGIN_DIR / "pages" / "console" / name
        ).read_bytes(), name
