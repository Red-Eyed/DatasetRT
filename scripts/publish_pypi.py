from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"


class PublishConfig(BaseSettings):
    """Configuration for publishing prebuilt release artifacts."""

    model_config = SettingsConfigDict(env_prefix="DATASETRT_")

    repository_url: str = ""
    """Optional custom Python package repository URL."""

    skip_existing: bool = False
    """Tell maturin upload to ignore already-uploaded files."""


def main() -> None:
    config = PublishConfig()
    require_command("uv")
    artifacts = release_artifacts()
    command = upload_command(config, artifacts)
    run(command)


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"required command not found: {command}")


def release_artifacts() -> list[Path]:
    artifacts = sorted(path for path in DIST_DIR.glob("*") if path.is_file())
    if not artifacts:
        raise SystemExit(f"no release artifacts found in {DIST_DIR}")
    return artifacts


def upload_command(config: PublishConfig, artifacts: Sequence[Path]) -> list[str | Path]:
    command: list[str | Path] = [
        "uv",
        "run",
        "--python",
        "3.11",
        "--extra",
        "dev",
        "maturin",
        "upload",
    ]
    if config.skip_existing:
        command.append("--skip-existing")
    if config.repository_url:
        command.extend(["--repository-url", config.repository_url])
    command.extend(artifacts)
    return command


def run(command: Sequence[str | Path]) -> None:
    normalized = [str(part) for part in command]
    subprocess.run(normalized, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
