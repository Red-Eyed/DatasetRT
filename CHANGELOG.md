# Changelog

All notable changes to DatasetRT are documented here.

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
- Local release tooling for macOS arm64, macOS x86_64, Linux x86_64, Linux aarch64, and sdist artifacts.
- GitHub CI for checks, plus `just` project commands for common workflows.

### Limits

- Payloads are bytes-like only: `bytes`, `bytearray`, or `memoryview`.
- Metadata values are primitive only: `bool`, `int`, `float`, or `str`.
- Shard compression is intentionally limited to `ShardCompression(algo="none", ratio=1.0)`.
- Domain decoding, such as JPEG/tensor/object decoding, stays in Python.
