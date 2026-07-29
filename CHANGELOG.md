# Changelog

All notable changes to DatasetRT are documented here.

## 0.1.11 - 2026-07-29

### Changed

- Cache checksum validation is now opt-in with `WriterConfig(validate_cache=True)` and `ReaderConfig(validate_cache=True)`.
- Default dataset construction still reads manifests, metadata, indexes, and shard file lengths, but no longer hashes every payload shard during restart.

## 0.1.10 - 2026-07-29

### Changed

- Multi-source cache writes now reuse one bounded writer pipeline across the whole source list.
- Python ingestion now continues across source boundaries until the configured queue is full, while cache commit and publish remain sequential.
- Writer progress rendering was split into focused Rust modules to keep writer internals easier to read.

## 0.1.9 - 2026-07-29

### Added

- Multi-source cache writes now show a source-level progress bar with completed source count, source throughput, and ETA when `show_progress=True`.

## 0.1.8 - 2026-07-29

### Fixed

- Writer Ctrl-C handling now preserves `KeyboardInterrupt` and `SystemExit` instead of converting them into per-source cache write errors.
- Long writer ingestion waits now poll Python signals so interrupted writes stop promptly and clean up temporary cache directories.

## 0.1.7 - 2026-07-29

### Changed

- `write_cache` now returns per-source `CacheWriteSuccess` or `CacheWriteError` results instead of failing the whole batch on source-level errors.
- `CachedDataset.from_cache_sources` now returns a matchable dataset result containing all source outcomes, so partial source failures can be logged while successful caches are still loaded.

## 0.1.6 - 2026-07-29

### Changed

- Cache directories now use the plain `CacheSource.name` instead of a hashed suffix.
- Writers now stage cache builds under `base_cache_dir/tmp` and publish with an atomic rename after a successful write.
- Multi-source writes now reject duplicate generated cache paths before writing any cache.

## 0.1.5 - 2026-07-29

### Changed

- Bumped the cache storage format to v2 and embedded each sample's metadata redundantly in its shard record for debugging, visualization, and raw record inspection.
- Readers now validate embedded shard metadata against `metadata.arrow` when materializing a sample.

## 0.1.4 - 2026-07-28

### Added

- Added optional Rust-owned write progress with committed samples/s and MB/s.
- Added per-payload LZ4 shard compression with transparent Rust-side decompression on read.

## 0.1.3 - 2026-07-28

### Changed

- Bumped the package patch version for a fresh PyPI artifact set.

## 0.1.2 - 2026-07-28

### Changed

- Rebuilt artifact packaging with `cp310-abi3` wheels for Python 3.10 through 3.13.
- Added local Linux `x86_64` and `aarch64` wheel builds through `maturin --zig`.
- Clarified local artifact building with `just build-all`.

## 0.1.0 - 2026-07-28

### Added

- Rust-backed immutable cache storage with manifests, checksums, Arrow metadata, binary indexes, and payload shards.
- Thin Python API with `CacheSource`, `CacheInput`, `CachedDataset`, `ReaderConfig`, `WriterConfig`, and `ShardCompression`.
- `CachedDataset.from_cache_sources` factory that creates missing caches, reuses valid existing caches, and loads the dataset.
- Deterministic weighted sampling with Polars `weight_table` and `set_weight_table`.
- Reader configuration for `prefetch_size`, `num_workers`, and `shuffle`.
- Rust-owned writer prefetching, worker threads, ordered commits, and multi-source cache writes.
- Optional sized PyTorch `IterableDataset` adapter through `to_torch_iterable_dataset`.
- Stable ABI wheels (`cp310-abi3`) for Python 3.10 through 3.13.
- Local artifact build tooling for macOS arm64, macOS x86_64, Linux x86_64, Linux aarch64, and sdist artifacts.
- GitHub CI for checks, plus `just` project commands for common workflows.

### Limits

- Payloads are bytes-like only: `bytes`, `bytearray`, or `memoryview`.
- Metadata values are primitive only: `bool`, `int`, `float`, or `str`.
- Shard compression is intentionally limited to `ShardCompression(algo="none", ratio=1.0)`.
- Domain decoding, such as JPEG/tensor/object decoding, stays in Python.
