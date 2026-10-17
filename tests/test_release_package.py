"""Behavioral checks for clean, directly installable release archives."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"
ROOT_FILES = (
    "__init__.py",
    "main.py",
    "ai.py",
    "anima_knowledge.py",
    "comfy.py",
    "bundled_resources.py",
    "bundled_workflows.py",
    "moderation.py",
    "model_profiles.py",
    "paths.py",
    "prompt_enhancer.py",
    "prompt_profiles.py",
    "prompting.py",
    "qwen21_workflows.py",
    "krea2_workflows.py",
    "translation.py",
    "workflow.py",
    "metadata.yaml",
    "_conf_schema.json",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "index.html",
    "app.js",
    "style.css",
)
SHIPPED_ASSETS = (
    "pages/console/index.html",
    "pages/console/app.js",
    "pages/console/style.css",
    "knowledge/提示词模版.txt",
    "knowledge/qwen21_提示词模版.txt",
    "knowledge/qwen21_提示词模版_nsfw.txt",
    "knowledge/krea2_提示词模版.txt",
    "knowledge/krea2_提示词模版_nsfw.txt",
    "workflows/文生图.json",
    "workflows/高清放大.json",
    "workflows/Anima_ControlNet文生图.json",
    "workflows/anima_native_api.json",
    "workflows/qwen21_edit_api.json",
    "workflows/qwen21_edit_turbo_api.json",
    "workflows/qwen21_edit_turbo_model_api.json",
    "workflows/qwen21_t2i_api.json",
    "workflows/qwen21_t2i_turbo_api.json",
    "workflows/qwen21_t2i_turbo_model_api.json",
    "workflows/krea2_t2i_api.json",
)


def _load_release_module():
    assert SCRIPT.is_file(), "The release packaging script has not been implemented"
    spec = importlib.util.spec_from_file_location("release_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "plugin"
    for name in ROOT_FILES + SHIPPED_ASSETS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = '{"default_model": "qwen21.safetensors"}'
        if path.suffix != ".json":
            content = f"public shipped content: {name}\n"
        if path.suffix == ".py":
            content = "# " + content
        path.write_text(content, encoding="utf-8")
    return root


def test_archive_has_installable_root_and_preserves_shipped_defaults(
    source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "release.zip"
    result = _load_release_module().build_release(source, output)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == set(ROOT_FILES + SHIPPED_ASSETS)
        assert archive.testzip() is None
        assert json.loads(archive.read("_conf_schema.json")) == {
            "default_model": "qwen21.safetensors"
        }
        assert (
            archive.read("knowledge/提示词模版.txt")
            == (source / "knowledge/提示词模版.txt").read_bytes()
        )
    assert result["output"] == str(output.resolve())
    assert set(result["files"]) == set(ROOT_FILES + SHIPPED_ASSETS)


def test_private_runtime_weights_and_user_source_are_never_included(
    source: Path, tmp_path: Path
) -> None:
    private_files = (
        ".git/config",
        ".env",
        "credentials.json",
        "settings.json",
        "user_source.py",
        "secrets.py",
        "main.py.bak",
        "app.js.bak_ui3",
        "tests/test_private.py",
        "scripts/internal.py",
        "desktop/desktop_app.py",
        "venv/lib/private.py",
        ".venv/lib/private.py",
        "__pycache__/main.pyc",
        "data/settings.json",
        "data/model_profiles.json",
        "data/workflows/qwen21_t2i_api.json",
        "logs/runtime.log",
        "backups/main.py",
        "backup_0.7.5_before_qwen_reset/main.py",
        "output/private.png",
        "weights/model.safetensors",
        "workflows/model.safetensors",
        "workflows/credentials.json",
        "workflows/user_workflow.json",
        "workflows/backups/qwen21_t2i_api.json",
        "pages/console/private.js",
        "pages/console/data/settings.json",
        "knowledge/user_notes.txt",
        "knowledge/credentials.json",
    )
    for name in private_files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRIVATE_MARKER_DO_NOT_SHIP", encoding="utf-8")
    output = tmp_path / "release.zip"
    _load_release_module().build_release(source, output)
    with ZipFile(output) as archive:
        assert set(archive.namelist()).isdisjoint(private_files)
        assert all(
            b"PRIVATE_MARKER_DO_NOT_SHIP" not in archive.read(name)
            for name in archive.namelist()
        )


def test_untracked_required_modules_and_all_qwen21_templates_are_included(
    source: Path, tmp_path: Path
) -> None:
    # There is deliberately no Git index or commit in this working directory.
    (source / ".git").mkdir()
    (source / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (source / "model_profiles.py").write_text(
        "CURRENT_WORKTREE_DEFAULTS", encoding="utf-8"
    )
    output = tmp_path / "release.zip"
    _load_release_module().build_release(source, output)
    with ZipFile(output) as archive:
        assert archive.read("model_profiles.py") == b"CURRENT_WORKTREE_DEFAULTS"
        assert "prompt_enhancer.py" in archive.namelist()
        assert "qwen21_workflows.py" in archive.namelist()
        assert "bundled_resources.py" in archive.namelist()
        assert "bundled_workflows.py" in archive.namelist()
        assert {
            "workflows/qwen21_edit_api.json",
            "workflows/qwen21_edit_turbo_api.json",
            "workflows/qwen21_edit_turbo_model_api.json",
            "workflows/qwen21_t2i_api.json",
            "workflows/qwen21_t2i_turbo_api.json",
            "workflows/qwen21_t2i_turbo_model_api.json",
        }.issubset(archive.namelist())


@pytest.mark.parametrize(
    "missing",
    ROOT_FILES + SHIPPED_ASSETS,
)
def test_missing_essential_file_is_rejected_before_output_creation(
    source: Path, tmp_path: Path, missing: str
) -> None:
    (source / missing).unlink()
    output = tmp_path / "not-created" / "release.zip"
    with pytest.raises(ValueError, match="Missing essential") as error:
        _load_release_module().build_release(source, output)
    assert missing in str(error.value)
    assert not output.parent.exists()


def test_existing_output_is_not_overwritten(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "release.zip"
    output.write_bytes(b"existing release")
    with pytest.raises(FileExistsError):
        _load_release_module().build_release(source, output)
    assert output.read_bytes() == b"existing release"


def test_parallel_catalog_and_native_defaults_are_bundled(
    source: Path, tmp_path: Path
) -> None:
    (source / "bundled_resources.py").write_text("# public catalog\n", encoding="utf-8")
    output = tmp_path / "release.zip"
    result = _load_release_module().build_release(source, output)
    with ZipFile(output) as archive:
        assert (
            archive.read("bundled_resources.py")
            == (source / "bundled_resources.py").read_bytes()
        )
        assert "bundled_workflows.py" in archive.namelist()
        assert "workflows/anima_native_api.json" in archive.namelist()
    assert "bundled_resources.py" in result["files"]


@pytest.mark.parametrize(
    "statement",
    (
        "from .bundled_resources import builtin_catalog\n",
        "from . import bundled_resources\n",
        "import bundled_resources\n",
    ),
)
def test_missing_catalog_referenced_by_main_is_rejected(
    source: Path, tmp_path: Path, statement: str
) -> None:
    (source / "main.py").write_text(statement, encoding="utf-8")
    (source / "bundled_resources.py").unlink()
    output = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="Missing essential.*bundled_resources.py"):
        _load_release_module().build_release(source, output)
    assert not output.exists()


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="Source directory"):
        _load_release_module().build_release(tmp_path / "missing", output)
    assert not output.exists()


def test_allowlisted_symlink_cannot_leak_private_content(
    source: Path, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE", encoding="utf-8")
    target = source / "main.py"
    target.unlink()
    try:
        target.symlink_to(secret)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    output = tmp_path / "release.zip"
    with pytest.raises(ValueError, match="link|outside"):
        _load_release_module().build_release(source, output)
    assert not output.exists()


def test_cli_exact_output_and_refusal_to_overwrite(
    source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "chosen" / "release.zip"
    command = [
        sys.executable,
        "-B",
        str(SCRIPT),
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["output"] == str(output.resolve())
    original = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert output.read_bytes() == original


def test_cli_defaults_to_distinct_timestamped_files_in_cwd(
    source: Path, tmp_path: Path
) -> None:
    command = [sys.executable, "-B", str(SCRIPT), "--source", str(source)]
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        output = Path(json.loads(completed.stdout)["output"])
        assert output.parent == tmp_path
        assert output.name.startswith("astrbot_plugin_comfyui_ai_studio-")
        assert output.suffix == ".zip"
        assert output.is_file()
        outputs.append(output)
    assert outputs[0] != outputs[1]
