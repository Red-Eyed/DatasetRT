from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import tomllib
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_MACOS_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
)
DEFAULT_LINUX_TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
)


class BuildConfig(BaseSettings):
    """Configuration for local artifact builds."""

    model_config = SettingsConfigDict(env_prefix="DATASETRT_")

    linux_targets: str = ""
    """Space-separated Rust Linux target triples."""

    macos_targets: str = ""
    """Space-separated Rust macOS target triples."""

    compatibility: str = "manylinux_2_17"
    """Linux wheel compatibility tag passed to maturin."""

    skip_macos: bool = False
    """Skip host macOS wheel builds."""

    skip_linux: bool = False
    """Skip Zig Linux wheel builds."""

    @property
    def parsed_linux_targets(self) -> tuple[str, ...]:
        if self.linux_targets:
            return split_nonempty(self.linux_targets)
        return DEFAULT_LINUX_TARGETS

    @property
    def parsed_macos_targets(self) -> tuple[str, ...]:
        if self.macos_targets:
            return split_nonempty(self.macos_targets)
        return DEFAULT_MACOS_TARGETS


def main() -> None:
    config = BuildConfig()
    require_command("uv")
    require_command("cargo")
    require_command("rustup")

    clean_dist()
    build_sdist()
    if not config.skip_macos:
        build_macos_wheels(config)
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


def build_macos_wheels(config: BuildConfig) -> None:
    if platform.system() != "Darwin":
        raise SystemExit("macOS wheels must be built on macOS; set DATASETRT_SKIP_MACOS=true")
    for target in config.parsed_macos_targets:
        ensure_rust_target(target)
        run(
            [
                "uv",
                "run",
                "--python",
                "3.11",
                "--extra",
                "dev",
                "maturin",
                "build",
                "--release",
                "--out",
                DIST_DIR,
                "--target",
                target,
            ]
        )


def build_linux_wheels(config: BuildConfig) -> None:
    for target in config.parsed_linux_targets:
        build_linux_target(config, target)


def build_linux_target(config: BuildConfig, target: str) -> None:
    ensure_rust_target(target)
    run(
        [
            "uv",
            "run",
            "--python",
            "3.11",
            "--extra",
            "dev",
            "maturin",
            "build",
            "--release",
            "--out",
            DIST_DIR,
            "--target",
            target,
            "--zig",
            "--compatibility",
            config.compatibility,
        ],
        env=zig_cache_env(),
    )


def ensure_rust_target(target: str) -> None:
    run(["rustup", "target", "add", target])


def zig_cache_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CARGO_ZIGBUILD_CACHE_DIR", "/tmp/cargo-zigbuild")
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", "/tmp/zig-global-cache")
    env.setdefault("ZIG_LOCAL_CACHE_DIR", "/tmp/zig-local-cache")
    return env


def split_nonempty(value: str) -> tuple[str, ...]:
    parts = tuple(part for part in value.split() if part)
    if not parts:
        raise SystemExit("build configuration list must not be empty")
    return parts


def print_artifacts() -> None:
    version = project_version()
    print(f"Built DatasetRT {version} artifacts:")
    for artifact in sorted(DIST_DIR.iterdir()):
        if artifact.is_file():
            print(artifact)


def project_version() -> str:
    data = tomllib.loads((PROJECT_ROOT / "Cargo.toml").read_text())
    version = data["package"]["version"]
    if not isinstance(version, str):
        raise SystemExit("project version must be a string")
    return version


def run(command: Sequence[str | Path], *, env: dict[str, str] | None = None) -> None:
    normalized = [str(part) for part in command]
    subprocess.run(normalized, cwd=PROJECT_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
