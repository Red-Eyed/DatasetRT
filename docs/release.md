# Release

DatasetRT releases are built locally to avoid spending GitHub Actions minutes on wheel builds.

GitHub CI remains checks-only. It validates Rust, Python linting, type checking, and tests on Python 3.11, but it does not publish artifacts.

## Local Build Matrix

DatasetRT uses PyO3 `abi3-py310`. Each wheel is tagged `cp310-abi3`, so one wheel per platform supports Python 3.10, 3.11, 3.12, and 3.13.

`scripts/release_local.py` builds:

- source distribution
- macOS `arm64` wheel
- macOS `x86_64` wheel
- Linux `x86_64` wheel with Zig cross-compilation
- Linux `aarch64` wheel with Zig cross-compilation

Artifacts are written to `dist/`.

```bash
just release-all
```

The Linux builds use `maturin --zig`, so they do not require QEMU or a running Podman machine.

To request a custom Linux target set, set `DATASETRT_LINUX_TARGETS`:

```bash
DATASETRT_LINUX_TARGETS="x86_64-unknown-linux-gnu" \
  uv run --python 3.11 --extra dev scripts/release_local.py
```

Useful release environment variables:

- `DATASETRT_LINUX_TARGETS`: space-separated Rust Linux target triples, default `x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu`.
- `DATASETRT_MACOS_TARGETS`: space-separated Rust macOS target triples, default `aarch64-apple-darwin x86_64-apple-darwin`.
- `DATASETRT_SKIP_MACOS`: set to `true` to skip host macOS wheels.
- `DATASETRT_SKIP_LINUX`: set to `true` to skip Zig Linux wheels.
- `DATASETRT_COMPATIBILITY`: Linux wheel compatibility tag, default `manylinux_2_17`.

## Publish

Publishing is a separate explicit step:

```bash
just publish
```

The publish script uploads the files already present in `dist/` through `maturin upload`. Configure PyPI credentials in the local environment before running it.

If publishing was interrupted and PyPI reports `File already exists`, resume with:

```bash
just publish-resume
```

`publish-resume` passes `--skip-existing` to maturin upload and continues with missing files. PyPI files are immutable: if an existing filename needs different contents, bump the project version and rebuild instead of trying to overwrite it.

Set `DATASETRT_REPOSITORY_URL` to upload to a custom package repository.

## Recommended Release Flow

```bash
just check
just release-all
```

After inspecting `dist/`, tag and push the release:

```bash
git tag v0.1.0
git push origin v0.1.0
```
