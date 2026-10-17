"""Build an installable ZIP from audited public files, never from Git HEAD.

Only the files listed below are shipped. Add new public modules or assets to
this allowlist deliberately; do not replace it with a recursive directory copy.
Default model names and workflows are source assets, not model weights. Runtime
profiles, credentials, user workflows, desktop files and all data/ are omitted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Exact root Python modules prevent unrelated user source from being bundled.
# Untracked runtime modules are intentionally included from the working tree.
RELEASE_FILES = (
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
    "pages/console/index.html",
    "pages/console/app.js",
    "pages/console/style.css",
    "knowledge/提示词模版.txt",
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


def build_release(source: Path, output: Path) -> dict:
    """Package the public working-tree source without an enclosing ZIP folder.

    Args:
        source: Plugin root containing the required public source assets.
        output: Exact ZIP target; existing files are never overwritten.

    Returns:
        A JSON-serializable manifest with output, source and archive member names.

    Raises:
        ValueError: Source or essential assets are missing, or assets use links.
        FileExistsError: The requested output already exists.
        OSError: Reading source or creating the ZIP fails.
    """
    source = Path(source).resolve()
    output = Path(output).absolute()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")

    # Validate the entire allowlist before creating any output directory or ZIP.
    missing = [name for name in RELEASE_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError("Missing essential release files: " + ", ".join(missing))
    for name in RELEASE_FILES:
        path = source / name
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(
                f"Release asset uses a link or points outside source: {name}"
            )

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose another target: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another builder claiming the path
    # after the existence check. Never remove an output we failed to create.
    stream = output.open("xb")
    try:
        with stream, ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
            for name in sorted(RELEASE_FILES):
                archive.write(source / name, arcname=name)
    except BaseException:
        output.unlink(missing_ok=True)
        raise

    return {
        "source": str(source),
        "output": str(output.resolve()),
        "files": sorted(RELEASE_FILES),
    }


def main() -> int:
    """Build a release from CLI arguments and print its JSON manifest.

    Returns:
        Zero on success; argparse reports invalid paths with a nonzero exit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Plugin source root (default: the parent of this scripts directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Exact ZIP target (must not exist); default: timestamped ZIP in cwd.",
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path.cwd() / f"astrbot_plugin_comfyui_ai_studio-{timestamp}.zip"
    try:
        result = build_release(args.source, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
