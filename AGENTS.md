# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

DatasetRT is a correctness-first dataset cache runtime for machine learning training systems. It provides immutable on-disk caches, deterministic sampling, metadata-aware weighting, and bounded Rust-owned read/write pipelines behind a small Python API.

This repository should be treated as production infrastructure, not an experiment script. Changes must preserve restart speed, bounded memory, deterministic behavior, and cache integrity for large datasets spread across many cache directories.

Core vocabulary:

- **Cache source**: A Python object with `name` and `__iter__` that yields `CacheInput` values. Python owns source discovery and domain serialization into bytes.
- **Cache directory**: One immutable DatasetRT cache containing `manifest.json`, `metadata.arrow`, `index.bin`, and `shards/`.
- **Manifest**: The publication marker for a complete cache. Readers reject caches without a valid manifest.
- **Metadata**: Primitive sample-level values stored columnarly in `metadata.arrow` and redundantly embedded in each shard record.
- **Physical sample**: A concrete `(cache_id, sample_id)` pair. `cache_id` is the position of the cache path passed to `CachedDataset`; `sample_id` is the row within that cache.
- **Weight table**: A Polars table with `cache_id`, `sample_id`, metadata columns, and `weight`. Rust owns the authoritative weight vector and validates all updates.
- **Epoch plan**: The ordered list of physical samples emitted by an iterator. With `shuffle=True`, it is deterministic weighted multinomial sampling with replacement for the dataset seed, epoch, and weight vector; it is not a permutation.
- **Shard**: A binary file containing concatenated sample records. The index maps each sample to a shard id, offset, and byte length.
- **Stored payload**: The bytes stored in the shard after optional per-record compression. DatasetRT does not decode images, tensors, or domain records.

Important external tools:

- **PyO3**: Rust bindings for exposing the native runtime as the `dataset_rt._dataset_rt` Python extension.
- **maturin**: Builds and installs the mixed Rust/Python package during development and release.
- **Polars**: Columnar DataFrame engine used by the Python API for metadata-aware weight editing.
- **Apache Arrow IPC**: Columnar file format used for `metadata.arrow` and in-memory weight table transfer.
- **crossbeam-channel**: Bounded Rust channels used for reader/writer backpressure.
- **indicatif**: Progress bar rendering for cache writes.
- **lz4_flex**: Per-record LZ4 compression for random-access payload reads.
- **uv**: Python environment and command runner. Use it for every Python command.
- **pyrefly**: Python type checker used by `just typecheck`.

Central formats:

- `manifest.json`: JSON cache manifest with format version, schema, counts, shard metadata, and checksums.
- `metadata.arrow`: Arrow IPC file with one metadata row per sample.
- `index.bin`: Fixed-width little-endian binary table. Each row is `(shard_id: u64, offset: u64, byte_len: u64)`.
- `shards/*.bin`: Concatenated records. Each record is metadata JSON length, metadata JSON, then stored payload bytes.

## Non-Negotiable Design Rules

Rust owns correctness-sensitive behavior:

- cache publication and validation
- manifest, metadata schema, index, shard layout, and checksums
- deterministic sampling, epoch state, and weight validation
- worker pools, queues, backpressure, ordering, and cancellation
- authoritative dataset state and weight vectors

Python owns ergonomics and integration:

- `CacheSource` implementations
- Pydantic configuration models
- domain object serialization into bytes before `CacheInput`
- Polars expressions users write to inspect or edit weights
- optional framework adapters such as PyTorch wrappers

Scalable paths must not materialize Python row objects. Avoid `list[dict]`, `iter_rows`, per-row callbacks, or Python-owned lookup loops for dataset-scale operations. Columnar data should stay columnar across the Python/Rust boundary, using Polars, Arrow IPC, typed buffers, or a deliberately designed native bridge.

Do not add temp-file data bridges on hot or scalable paths. Temporary files are acceptable only for build/test fixtures or explicit artifact outputs, not as an internal transport between Python and Rust.

Defaults must be safe for normal production restarts. Expensive integrity checks such as hashing all shards belong behind explicit `validate_cache=True` configuration.

## Development Commands

Environment:

```bash
just sync          # install Python dev dependencies with uv
just develop       # build and install the Rust extension in editable mode
```

Formatting and checks:

```bash
just fmt           # format Python and Rust
just fmt-check     # check Python and Rust formatting
just lint          # run ruff
just typecheck     # run pyrefly
just test-rust     # run cargo test
just clippy        # run cargo clippy with warnings denied
just test-python   # rebuild extension, then run pytest
just check         # full local validation before commit/release
```

Focused commands:

```bash
cargo test                         # Rust unit tests
cargo clippy --all-targets -- -D warnings
uv run --python 3.11 --extra dev pytest tests/test_dataset_rt.py
uv run --python 3.11 --extra dev pytest tests/test_integrity.py
uv run --python 3.11 --extra dev ruff check dataset_rt tests scripts
uv run --python 3.11 --extra dev pyrefly check
```

Benchmarks and builds:

```bash
just smoke       # synthetic write/read smoke benchmark
just build-all   # local sdist and platform wheel build
```

Use `uv run --python 3.11 --extra dev ...` for Python commands. Do not call bare `python`, `pip`, or `pytest` from the system environment.

## Architecture

Python package:

- `dataset_rt/api.py` defines the public API, Pydantic configs, typed `NamedTuple` return values, and Polars weight-table helpers.
- `dataset_rt/_dataset_rt.pyi` mirrors the Rust extension API for type checking.
- `dataset_rt/__init__.py` re-exports the public surface.

Rust modules:

- `types.rs`: validated IDs, config newtypes, metadata enums, samples, and `CacheError`.
- `storage.rs`: cache format, manifests, metadata Arrow files, indexes, shards, compression envelopes, checksums, and cache loading.
- `writer.rs` and `writer/pipeline.rs`: source ingestion, bounded queues, serialization workers, ordered commit, progress, profiling, and publication.
- `dataset.rs`: Python-facing `CachedDataset`, schema checks, weight state, weight validation, and iterator construction.
- `runtime.rs`: Rust-owned synchronous reader pipeline with scheduler, workers, bounded result queue, and reorder buffer.
- `sampling.rs`: deterministic weighted epoch planning.
- `compression.rs`: per-record compression/decompression.

Data flow:

1. Python `CacheSource` yields `CacheInput(data, metadata)`.
2. Rust validates metadata schema, compresses payloads if configured, writes shard records, writes Arrow metadata and binary index, then publishes manifest.
3. `CachedDataset` loads manifests and indexes, validates cache shape, and initializes Rust dataset state.
4. Iteration creates an epoch plan, schedules bounded read tasks, reads addressed shard records, decompresses payload bytes, and returns `CachedSample` to Python.
5. `weight_table()` exposes metadata and weights through Polars; `set_weight_table()` returns only identity and weight data to Rust for validation.

## Checklist For Every Change

Before editing code, answer these questions in the plan:

- Does this affect correctness, determinism, cache validity, sampling, ordering, concurrency, or backpressure? If yes, Rust must own the authoritative behavior.
- What is the expected complexity in samples, caches, shards, and metadata columns? State whether the change is O(1), O(n), O(n log n), or worse.
- Does this introduce eager loading at dataset construction? Startup should read only what is necessary for manifests, index shape, and configured validation.
- Does this allocate per sample, per row, or per metadata cell? If yes, explain why it is unavoidable and bounded.
- Does any scalable path cross Python as row objects, dictionaries, tuples, callbacks, or `iter_rows`? If yes, redesign.
- Does the change preserve bounded queues and backpressure? No unbounded channels, task lists, worker output buffers, or hidden Python queues.
- Does the change preserve deterministic ordering for identical cache contents, seed, epoch, and weights?
- Does it keep cache identity stable as `(cache_id, sample_id)`?
- Does it avoid default shard checksum hashing unless `validate_cache=True`?
- Does it avoid temp files as internal transport between Python and Rust?
- Does it avoid `O(n^2)` lookup patterns such as scanning all physical samples for each weight row?
- Does it handle 10K cache directories and million-row metadata tables without Python object expansion?
- Are all public Python conveniences backed by Rust-side validation before mutating authoritative state?
- Are error cases returned as typed `CacheError` values rather than panics or unchecked indexing?
- Does every new function or method have a short docstring/doc comment that states its production invariant or boundary responsibility?

When changing storage format:

- Preserve backward compatibility or bump `FORMAT_VERSION`.
- Update `docs/storage-format.md`.
- Add corruption/shape tests in `tests/test_integrity.py`.
- Verify missing files, bad lengths, bad checksums, bad schema, and row-count mismatches fail clearly.

When changing sampling or weights:

- Keep weight validation in Rust.
- Preserve `shuffle=True` semantics as weighted multinomial sampling with replacement. A sample may appear multiple times in one epoch, and another sample may be absent.
- For weight table updates, validate exact coverage: every physical sample appears once, no duplicates, no unknown identities.
- Keep weights positive and finite.
- Add determinism tests for same seed/epoch/weights.
- Avoid Python row iteration for full-table updates.

When changing reader runtime:

- Keep queues bounded by `PrefetchSize`.
- Preserve cancellation on iterator drop.
- Preserve planned output order even when workers finish out of order.
- Account for memory as `prefetch_size * max_loaded_sample_size` plus bounded reorder state.

When changing writer runtime:

- Keep source ingestion bounded.
- Preserve ordered commit and deterministic manifests.
- Preserve Ctrl-C/SystemExit behavior.
- Do not publish a cache until all required files are durable and manifest is written.
- Clean temporary cache directories on handled failures.

When changing Python API:

- Keep Python as orchestration and ergonomics, not authoritative runtime state.
- Prefer Polars expressions and Arrow/typed buffers over Python row loops.
- Keep `Path` at API edges and pass strings only where the Rust extension requires them.
- Add docstrings to new Python functions and doc comments to new Rust functions. The text should explain the invariant or boundary, not restate the implementation.
- Update `_dataset_rt.pyi` whenever the PyO3 surface changes.
- Rebuild with `just develop` or `just test-python` before trusting Python tests.

## Anti-Patterns

Avoid these unless the user explicitly asks for a prototype and the code is clearly marked non-production:

- `pl.DataFrame(list_of_dicts)` for dataset-scale data.
- `DataFrame.iter_rows(...)` or `Series.to_list()` on full dataset tables.
- Python `dict` per sample for metadata table construction.
- Eagerly loading all `metadata.arrow` rows during dataset construction.
- Building a full epoch plan unnecessarily for non-shuffled physical-order reads.
- Per-row lookup with `.position(...)` or linear scans over all physical samples.
- Internal temp-file transport between Python and Rust.
- Unbounded worker queues or channels.
- Hidden Python threads/queues in the runtime path.
- Bare `unwrap`, `expect`, `panic`, `todo`, `unimplemented`, or unchecked indexing/slicing in Rust.
- `validate_cache=True` behavior becoming the default path.
- Adding framework-specific logic to Rust core.

## Testing Expectations

For any non-trivial change, run at least:

```bash
cargo test
uv run --python 3.11 --extra dev ruff check dataset_rt tests scripts
just test-python
```

Before commit or release, run:

```bash
just check
```

If touching scalable paths, add or run a benchmark-style smoke test that exercises many samples and multiple caches. The test does not need production volume, but it must prove the shape of the implementation: no row-object bridge, no unbounded queue, no accidental quadratic lookup, and no eager metadata expansion.

If `ruff format` changes a file, re-read the changed sections before continuing.

## Release Notes

Distribution artifacts are built locally with `just build-all`; GitHub Actions is checks-only. Wheels use PyO3 `abi3-py310`, so one wheel per platform supports Python 3.10 through 3.13. PyPI artifacts are immutable; bump the version before rebuilding a changed release artifact.
