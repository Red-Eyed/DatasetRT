# Development

DatasetRT is implemented as a Python package backed by a Rust extension module.

## Toolchain

- Rust stable.
- PyO3.
- maturin.
- Polars for the public Python weight table.
- Python package metadata in `pyproject.toml`.

The local development interpreter is pinned with `.python-version`. The current project default is CPython 3.11 because it has mature wheels for the Python dependencies used in the test environment.

Python commands use `uv`.

## Checks

Run the relevant checks before committing:

```bash
just check
```

If `ruff format` changes files, re-read touched files before making further edits.

CI runs the same checks on Python 3.11 and Rust stable. The editable extension is built with maturin before Python tests run.

Release wheel matrices are built locally with Podman instead of GitHub Actions. See [Release](release.md).

## Smoke Benchmark

Use the benchmark smoke script to verify the recommended factory path and get rough throughput numbers:

```bash
just smoke
```

The script writes a temporary synthetic cache through `CachedDataset.from_cache_sources`, reads it back in physical order, and prints JSON.

## Design Rules

Correctness-sensitive logic belongs in Rust. When in doubt, ask whether the behavior affects reproducibility, integrity, concurrency, sampling, or cache validity. If yes, implement it in Rust.

Keep Python wrappers thin. Python may validate user ergonomics at the edge, but Rust must validate again before writing or reading a cache.

Rust project code denies panic-style constructs with clippy: `unwrap`, `expect`, `panic`, `todo`, `unimplemented`, and unchecked indexing/slicing. Recoverable problems must be returned as typed errors.

Prefer strict types:

- Dates and timestamps are typed values, never strings.
- Domain absence is modeled explicitly, not with bare `None`.
- Metadata values use a closed primitive enum.

Keep optional or experimental components isolated in their own files with minimal guarded wiring.
