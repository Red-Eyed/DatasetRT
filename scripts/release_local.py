from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import tomllib
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13")


class ReleaseConfig(BaseSettings):
    """Configuration for local release artifact builds."""

    model_config = SettingsConfigDict(env_prefix="DATASETRT_")

    python_versions: str = " ".join(DEFAULT_PYTHON_VERSIONS)
    """Space-separated Python versions to build wheels for."""

    linux_targets: str = ""
    """Space-separated `rust-target:container-platform` Linux target specs."""

    container_image: str = "ghcr.io/pyo3/maturin:latest"
    """Maturin container image used for Linux wheels."""

    compatibility: str = "manylinux_2_28"
    """Linux wheel compatibility tag passed to maturin."""

    skip_macos: bool = False
    """Skip host macOS wheel builds."""

    skip_linux: bool = False
    """Skip Podman Linux wheel builds."""

    @property
    def parsed_python_versions(self) -> tuple[str, ...]:
        return split_nonempty(self.python_versions)

    @property
    def parsed_linux_targets(self) -> tuple[str, ...]:
        if self.linux_targets:
            return split_nonempty(self.linux_targets)
        return (default_linux_target(),)


def main() -> None:
    config = ReleaseConfig()
    require_command("uv")
    require_command("cargo")
    if not config.skip_linux:
        require_command("podman")

    clean_dist()
    build_sdist()
    if not config.skip_macos:
        build_macos_wheels(config.parsed_python_versions)
    if not config.skip_linux:
        build_linux_wheels(config)
    print_artifacts()


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"required command not found: {command}")


def clean_dist() -> None:
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def build_sdist() -> None:
    run(["uv", "run", "--python", "3.11", "--extra", "dev", "maturin", "sdist", "--out", DIST_DIR])


def build_macos_wheels(python_versions: Sequence[str]) -> None:
    for python_version in python_versions:
        run(
            [
                "uv",
                "run",
                "--python",
                python_version,
                "--extra",
                "dev",
                "maturin",
                "build",
                "--release",
                "--out",
                DIST_DIR,
            ]
        )


def build_linux_wheels(config: ReleaseConfig) -> None:
    for target_spec in config.parsed_linux_targets:
        target = parse_linux_target(target_spec)
        build_linux_target(config, target)


def parse_linux_target(target_spec: str) -> tuple[str, str]:
    parts = target_spec.split(":", maxsplit=1)
    if len(parts) != 2:
        raise SystemExit(
            f"linux target must use `rust-target:container-platform`, got: {target_spec}"
        )
    return parts[0], parts[1]


def build_linux_target(config: ReleaseConfig, target: tuple[str, str]) -> None:
    rust_target, container_platform = target
    interpreters = [manylinux_python_path(version) for version in config.parsed_python_versions]
    run(
        [
            "podman",
            "run",
            "--rm",
            "--platform",
            container_platform,
            "--volume",
            f"{PROJECT_ROOT}:/io",
            "--workdir",
            "/io",
            config.container_image,
            "build",
            "--release",
            "--out",
            "/io/dist",
            "--target",
            rust_target,
            "--compatibility",
            config.compatibility,
            "--interpreter",
            *interpreters,
        ]
    )


def default_linux_target() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64-unknown-linux-gnu:linux/arm64"
    if machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-gnu:linux/amd64"
    raise SystemExit(f"unsupported host architecture for default Linux target: {machine}")


def manylinux_python_path(version: str) -> str:
    compact = version.replace(".", "")
    return f"/opt/python/cp{compact}-cp{compact}/bin/python"


def split_nonempty(value: str) -> tuple[str, ...]:
    parts = tuple(part for part in value.split() if part)
    if not parts:
        raise SystemExit("release configuration list must not be empty")
    return parts


def print_artifacts() -> None:
    version = project_version()
    print(f"Built DatasetRT {version} artifacts:")
    for artifact in sorted(DIST_DIR.iterdir()):
        if artifact.is_file():
            print(artifact)


def project_version() -> str:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    version = data["project"]["version"]
    if not isinstance(version, str):
        raise SystemExit("project version must be a string")
    return version


def run(command: Sequence[str | Path]) -> None:
    normalized = [str(part) for part in command]
    subprocess.run(normalized, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
