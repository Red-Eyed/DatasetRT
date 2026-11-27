# Build Artifacts

DatasetRT release artifacts are built locally. Only the user publishes packages.

GitHub CI validates Rust, linting and types once, builds a standard wheel once,
then installs that same artifact on CPython 3.10–3.15. Tests run outside the
checkout so the source package cannot shadow the installed wheel. Dependencies
must resolve to published binaries. CI never publishes packages.

## Local Build Matrix

DatasetRT uses PyO3 `abi3-py310`. Each standard wheel is tagged `cp310-abi3`,
so one wheel per platform supports standard CPython 3.10 through 3.15.

`scripts/build_local.py` builds:

- source distribution
- macOS `arm64` wheel
- macOS `x86_64` wheel
- Linux `x86_64` wheel with Zig cross-compilation
- Linux `aarch64` wheel with Zig cross-compilation

Artifacts are written to `dist/`.

```bash
just build-all
```

The Linux builds use `maturin --zig`, so they do not require QEMU or a running Podman machine.

To request a custom Linux target set, set `DATASETRT_LINUX_TARGETS`:

```bash
DATASETRT_LINUX_TARGETS="x86_64-unknown-linux-gnu" \
  uv run --python 3.11 --extra dev scripts/build_local.py
```

Useful build environment variables:

- `DATASETRT_LINUX_TARGETS`: space-separated Rust Linux target triples, default `x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu`.
- `DATASETRT_MACOS_TARGETS`: space-separated Rust macOS target triples, default `aarch64-apple-darwin x86_64-apple-darwin`.
- `DATASETRT_SKIP_MACOS`: set to `true` to skip host macOS wheels.
- `DATASETRT_SKIP_LINUX`: set to `true` to skip Zig Linux wheels.
- `DATASETRT_COMPATIBILITY`: Linux wheel compatibility tag, default `manylinux_2_17`.

Free-threaded Python support is postponed. Release wheels support standard
CPython only.

## Publishing

Publishing is handled outside the project `justfile`.

PyPI files are immutable: if an existing filename needs different contents, bump the project version and rebuild instead of trying to overwrite it.

## Recommended Build Flow

```bash
just check
just build-all
```

After inspecting `dist/`, tag and push the release:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```
