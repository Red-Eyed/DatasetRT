# Release

DatasetRT releases are built locally to avoid spending GitHub Actions minutes on wheel matrices.

GitHub CI remains checks-only. It validates Rust, Python linting, type checking, and tests on Python 3.11, but it does not publish artifacts.

## Local Build Matrix

`scripts/release_local.py` builds:

- source distribution
- macOS wheels on the local host architecture for Python 3.10, 3.11, 3.12, and 3.13
- Linux wheels for the matching host architecture in Podman for Python 3.10, 3.11, 3.12, and 3.13

Artifacts are written to `dist/`.

```bash
uv run --python 3.11 --extra dev scripts/release_local.py
```

The Linux builds use the `ghcr.io/pyo3/maturin:latest` manylinux container through Podman. On macOS, Podman must have a running machine before Linux wheels can be built.

To request multiple Linux architectures from a machine with working emulation, set `DATASETRT_LINUX_TARGETS`:

```bash
DATASETRT_LINUX_TARGETS="x86_64-unknown-linux-gnu:linux/amd64 aarch64-unknown-linux-gnu:linux/arm64" \
  uv run --python 3.11 --extra dev scripts/release_local.py
```

For the most reliable full matrix, run the default script once on an arm64 machine and once on an x86_64 machine, then combine the `dist/` artifacts before publishing.

Useful release environment variables:

- `DATASETRT_PYTHON_VERSIONS`: space-separated Python versions, default `3.10 3.11 3.12 3.13`.
- `DATASETRT_LINUX_TARGETS`: space-separated `rust-target:container-platform` values.
- `DATASETRT_SKIP_MACOS`: set to `true` to skip host macOS wheels.
- `DATASETRT_SKIP_LINUX`: set to `true` to skip Podman Linux wheels.
- `DATASETRT_CONTAINER_IMAGE`: maturin container image, default `ghcr.io/pyo3/maturin:latest`.
- `DATASETRT_COMPATIBILITY`: Linux wheel compatibility tag, default `manylinux_2_28`.

## Publish

Publishing is a separate explicit step:

```bash
uv run --python 3.11 --extra dev scripts/publish_pypi.py
```

The publish script uploads the files already present in `dist/` through `maturin upload`. Configure PyPI credentials in the local environment before running it.

Set `DATASETRT_SKIP_EXISTING=true` to pass `--skip-existing` to maturin upload. Set `DATASETRT_REPOSITORY_URL` to upload to a custom package repository.

## Recommended Release Flow

```bash
cargo test
cargo clippy --all-targets -- -D warnings
uv run --python 3.11 --extra dev ruff format --check dataset_rt tests scripts
uv run --python 3.11 --extra dev ruff check dataset_rt tests scripts
uv run --python 3.11 --extra dev pyrefly check
uv run --python 3.11 --extra dev pytest
uv run --python 3.11 --extra dev scripts/release_local.py
```

After inspecting `dist/`, tag and push the release:

```bash
git tag v0.1.0
git push origin v0.1.0
```
