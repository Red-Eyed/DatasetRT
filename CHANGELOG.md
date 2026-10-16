# Changelog

All notable changes to DatasetRT are documented here.

## 0.4.7

### Documentation

- Organize documentation into a persistent sidebar with getting-started, guide,
  concept, API, and benchmark sections. Keep the quickstart on-site and remove
  automatic GitHub edit links.
- Distinguish serialized payload storage from the separate metadata table, and
  explain metadata analysis, filtering, weighted sampling, native read speed,
  and mmap sharing of metadata and indexes across loading processes.
- Clarify DatasetRT's Python/Rust dataset caching purpose, add a runnable
  quickstart, and expose documentation, repository, issue, and changelog URLs
  in package metadata.
- Add a static public documentation build with canonical URLs, page descriptions,
  a search index, a sitemap, and a GitHub Pages workflow. Exclude internal
  investigations, agent review checklists, and maintainer instructions.
- Add runnable caching, Polars weighting, and PyTorch loading tutorials, with
  application-facing runtime and architecture guides. Show iterable sources
  packing Python records with pickle or dill and read transforms unpacking them.
- Document reproducible reader measurements with raw JSON evidence and compare
  dataset tools using their upstream documentation.

### Developers

- Add `just docs-build`, `just docs-serve`, and strict documentation validation
  to `just check`, using an isolated documentation dependency group.
- Add `just bench-reader` to build the extension automatically before benchmarking.

## 0.4.0 - 2026-10-15

This release consolidates the 0.4 release candidates since 0.3.2.

### Highlights

- Add finite, metadata-aware PyTorch DataLoaders with persistent sampling streams,
  automatic worker sizing, and distributed sample budgets.
- Add optional spawned cache-source writers and portable dataset reconstruction
  settings that reference existing cache payloads.
- Reduce reader memory with shared immutable Arrow metadata and mapped indexes,
  and bound outstanding reads together with queued and reordered results.

### Backwards Incompatible Changes

- Write new caches with v3 manifests and persistent name-derived `cache_id`
  values. Existing v2 caches remain readable with positional IDs. Treat IDs as
  opaque; use `CachedDataset.update_manifests(version=3)` for explicit migration,
  constructing legacy datasets in their original path order to preserve IDs.
  Older releases that only read v2 cannot read migrated caches.
- Make `CachedDataset.to_torch_dataloader()` epochs finite instead of the
  infinite streams in early release candidates. Set `samples_per_epoch` for a
  global sample budget, or omit it to inherit `len(dataset)` at construction.
  Iterate the loader directly instead of externally limiting an infinite stream.
  `batch_size` must be a positive integer; use `1` for single-sample collation.
- Use explicit automatic modes for `num_workers`, `native_num_workers`,
  `persistent_workers`, and `prefetch_factor` (`"auto"`), and for seeds
  (`"random"`). Replace previous release-candidate `None` selections accordingly.
  Sequential loading rejects an explicit seed. Automatic persistence is enabled
  with loading workers; automatic prefetch uses two batches per worker and is
  disabled with zero workers. Explicit persistence or integer prefetch with
  zero workers raises an error.
- Default loading processes to `multiprocessing_context="spawn"`. Use a guarded
  application entry point and picklable callbacks. Replace release-candidate
  `None` contexts with `"inherit"` to select Python's platform default.
- Default shuffled loading to replicated populations for global weighted
  sampling; sequential loading defaults to split populations. Explicit
  `worker_partition="split"` remains available, but equal consumer quotas can
  approximate the global weight distribution. Replication retains private
  weights and sampling state per consumer; 10M f64 weights require 80 MB per
  consumer, in addition to sampling state.
- Divide global sample budgets equally across initialized DDP ranks, padding
  with `drop_last=False` or discarding with `drop_last=True` to complete local
  batches. Rank zero warns when the effective budget changes. Rank lengths and
  reported loader lengths match emitted batches.
- Replace release-candidate v1 reconstruction bundles containing
  `metadata.parquet` with v2 bundles containing `metadata.arrow`. Recreate old
  exports with `CachedDataset.dump_config()` before calling `from_config()`.

### New Features

- Add `WriterConfig.num_processes` for bounded spawned source writers with
  input-order outcomes; zero retains serial writing. Sources must be picklable
  and importable. Worker death or timeout stops processing without retries.
- Add `CachedDataset.dump_config()` and `CachedDataset.from_config()` for
  reconstruction settings and immutable active metadata snapshots. Loader
  `work_dir` controls retained snapshots; the default manages temporary output.
- Add `python -m dataset_rt.run_benchmark` with the `benchmark` extra for local
  rank launch or external `torchrun`, generated or existing caches, CPU/Gloo or
  CUDA/NCCL execution, correctness checks, and JSON resource/performance reports.

### Improvements

- Resolve automatic loader counts from CPU affinity and visible container quota
  and CPU-set limits, accounting for local ranks. Choose up to four loading
  processes and 1–4 native threads per consumer; a one-CPU budget uses zero
  loading processes. Set explicit counts for a reproducible worker topology.
- Log loader configuration and reader lifecycle through `dataset_rt.loader`.
  `verbose=False` suppresses diagnostics while retaining the warning that
  explicitly nonpersistent workers repeat their sampling stream each epoch.
- Validate worker options, finite timeouts, sample budgets, and multiprocess
  transform picklability before exporting loader reconstruction artifacts.

### Bug Fixes

- Continue sampling across short epochs by retaining loading workers by default.
  Explicit `persistent_workers=False` still restarts reader progress and now
  warns about repeated samples.
- Preserve bounded reader memory when an early read completes late, while
  retaining planned emission order and iterator-drop cancellation.
- Allow fresh native runtimes after macOS `fork` without reusing an inherited
  thread parker; bounded channels retain FIFO delivery and backpressure.
- Assign whole-batch worker quotas so `len(loader)` matches emitted batches and
  `drop_last=True` drops at most one local tail.

### Performance

- Share immutable memory-mapped Arrow metadata, selected row views, cache
  indexes, and shuffled split permutations across consumers on one host.
  Weights and sampling state remain private. Avoid whole-table IPC copies in
  metadata access and updates, and stream reconstruction exports in bounded
  chunks.

### Developers

- Add paired reader/writer benchmarks, CPU DDP acceptance examples, and
  `just test-correctness` for fast behavioral checks with spawned workers.
- Preserve public imports and pickle paths while splitting Python modules;
  enforce precise annotations and run ordinary tests with pytest-xdist.

### Validation coverage

- CPU acceptance is verified. CUDA/NCCL and physical multi-host acceptance
  remain planned validation work on suitable hardware.

## 0.3.2 - 2026-09-28

### Added

- Add `CachedDataset.get_item(cache_id, sample_id)` to read an immutable physical sample directly, including samples excluded from the active metadata table, without advancing an iterator or sampling cursor.

## 0.3.0 - 2026-08-08

### Added

- Added `CachedDataset.set_epoch_len()` to control how many samples each future iterator emits.
- Added cyclic non-shuffled epoch windows so short epochs continue through active rows across iterator boundaries.
- Added continuous shuffled draw streams so short shuffled epochs split one deterministic multinomial stream.

### Changed

- Made `len(dataset)` report the configured epoch length while metadata updates continue to control only the active sampling population.
- Reset ordered and shuffled stream positions after `update_metadata()` without changing the configured epoch length.

## 0.2.6 - 2026-08-04

### Changed

- Allowed duplicate `(cache_id, sample_id)` rows in `update_metadata()` so callers can duplicate rows for balancing, OHEM, or other custom sampling strategies.
- Made active metadata row count, including duplicates, the dataset length used by future iterators.
- Documented `get_metadata()` and `update_metadata()` directly from Python API docstrings to keep agent-facing API docs aligned with code.
- Added an API docs sync check to `just check` so Markdown docs cannot drift from the Python docstrings.

## 0.2.5 - 2026-08-04

### Added

- Added `CachedDataset.get_metadata()` and `update_metadata()` as the runtime metadata table API.
- Added support for runtime-only metadata table edits that can keep optional columns alongside required identity, cache metadata, and `weight` columns.

### Changed

- Made the active metadata table authoritative for dataset length and future sampling, so removing rows limits subsequent iterators to the remaining physical samples.
- Kept `update_metadata()` in memory only; it does not rewrite cache files and existing iterators continue using the metadata snapshot they were created with.
- Kept `samples_metadata()` and `set_samples_metadata()` as compatibility aliases.
- Rewrote the README as a shorter project overview and moved detailed behavior to the docs.

### Fixed

- Validated metadata updates through Rust before mutating runtime state, including required columns, stable physical sample identities, unknown rows, and positive finite weights.

## 0.2.3 - 2026-07-31

### Added

- Added `DatasetRuntime(num_workers=...)` as the explicit owner of one fixed, reusable Rust worker pool.

### Changed

- Moved cache writing, cache loading, and source-to-dataset construction to `DatasetRuntime` methods so `num_workers` is configured exactly once.
- Kept finite per-operation task and result windows bounded by the runtime worker count and each operation's `prefetch_size`.
- Kept loaded datasets bound to their runtime pool for all subsequent sample reads.

### Fixed

- Removed the contradictory global-pool worker-count validation that could reject an operation even when both counts were equal.

### Removed

- Removed hidden process-global worker-pool state and implicit first-call initialization.
- Removed operation-level `num_workers` arguments, the public `write_cache` function, direct `CachedDataset` construction, and `CachedDataset.from_cache_sources`.
- Removed deprecated `weight_table` and `set_weight_table` compatibility aliases.

## 0.2.2 - 2026-07-31

### Changed

- Reused one explicitly sized process-wide Rust worker pool for cache loading, sample reads, and writer jobs.
- Made `num_workers` a required top-level argument to `write_cache`, `CachedDataset`, and `CachedDataset.from_cache_sources`.
- Loaded cache directories concurrently with bounded work while preserving constructor order and stable `cache_id` values.
- Kept per-operation task and result windows bounded by `num_workers` and `prefetch_size`.
- Preserved deterministic read ordering, iterator cancellation, ordered writer commits, and interruption cleanup.

### Removed

- Removed per-operation scheduler, reader, cache-loader, and writer thread creation.
- Removed worker-count fields and legacy aliases from `ReaderConfig` and `WriterConfig`.

## 0.2.0 - 2026-07-30

### Added

- Added `CachedDataset.samples_metadata()` and `set_samples_metadata()` as the canonical metadata-aware weight editing API.
- Added deprecated compatibility aliases for `weight_table()` and `set_weight_table()`.

### Changed

- Moved samples metadata table construction to Rust-owned Arrow IPC instead of Python reading `metadata.arrow`.
- Reused Arrow metadata batches directly when exporting samples metadata, avoiding Python row objects and Rust metadata-cell expansion.
- Kept default uniform weights implicit instead of materializing an all-ones vector at dataset load.
- Streamed physical-order and shuffled epoch planning to avoid unnecessary full plan vectors.
- Preserved `shuffle=True` as deterministic weighted multinomial sampling with replacement.
- Cached shard readers per worker to reduce repeated file-open overhead during random reads.

### Removed

- Removed the row-based Python/Rust weight bridge and unused private weight-vector bindings.

## 0.1.12 - 2026-07-29

### Added

- Added `WriterProfilerConfig` for optional JSON writer timing summaries.
- Writer profiling now separates Python iterator time, Python-to-Rust extraction, queue backpressure, compression, disk writes, finish steps, and cache publish time.
- Enabled best-effort profiler summary flushing for handled writer failures, including Ctrl-C/`KeyboardInterrupt`.

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
