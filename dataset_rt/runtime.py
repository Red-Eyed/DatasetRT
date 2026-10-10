"""Native worker-pool ownership and serial cache orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dataset_rt._dataset_rt import DatasetRuntime as _RustDatasetRuntime
from dataset_rt._dataset_rt import write_cache as _write_cache
from dataset_rt.config import DEFAULT_WRITER_CONFIG, ReaderConfig, WriterConfig
from dataset_rt.dataset import CachedDataset
from dataset_rt.records import (
    CacheSource,
    CacheSourcesDatasetError,
    CacheSourcesDatasetResult,
    CacheSourcesDatasetSuccess,
    CacheWriteError,
    CacheWriteResult,
    CacheWriteSuccess,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataset_rt._dataset_rt import CacheWriteRecord as _RawCacheWriteResult


def _cache_write_result(result: _RawCacheWriteResult) -> CacheWriteResult:
    """Convert one native outcome while rejecting unknown status tags."""
    status, source_name, detail = result
    match status:
        case "success":
            return CacheWriteSuccess(source_name, Path(detail))
        case "error":
            return CacheWriteError(source_name, detail)
        case _:
            raise ValueError(f"unknown cache write result status: {status}")


def _successful_cache_paths(results: Sequence[CacheWriteResult]) -> list[Path]:
    """Retain successful destinations in original source order."""
    paths = []
    for result in results:
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case CacheWriteError():
                continue
    return paths


def _cache_sources_dataset_error(results: list[CacheWriteResult]) -> CacheSourcesDatasetError:
    """Summarize source failures when no dataset can be loaded."""
    return CacheSourcesDatasetError(results, _format_cache_sources_dataset_error(results))


def _format_cache_sources_dataset_error(results: Sequence[CacheWriteResult]) -> str:
    """Describe failures without discarding per-source outcome records."""
    messages = []
    for result in results:
        match result:
            case CacheWriteError(source_name=source_name, message=message):
                messages.append(f"{source_name}: {message}")
            case CacheWriteSuccess():
                continue
    if not messages:
        return "no cache sources were provided"
    details = "; ".join(messages)
    return f"no caches were written: {details}"


class DatasetRuntime:
    """Owner of the fixed Rust worker pool used by DatasetRT operations.

    Create one runtime per process or training job, then call its methods to
    write caches, load datasets, and iterate samples. The worker count is chosen
    once at construction and reused; per-operation APIs do not resize the pool.
    """

    # Existing pickles resolve public classes through the compatibility facade.
    __module__ = "dataset_rt.api"

    __slots__ = ("_inner", "_num_workers")

    def __init__(self, *, num_workers: int) -> None:
        """Create exactly `num_workers` reusable Rust worker threads.

        `num_workers` must be positive. Rust validates the value before creating
        the native pool. The pool is owned by this runtime and kept alive by any
        datasets loaded through it.
        """
        self._num_workers = num_workers
        self._inner = _RustDatasetRuntime(num_workers)

    @property
    def num_workers(self) -> int:
        """Return the fixed worker count selected when this runtime was created."""
        return self._num_workers

    def write_cache(
        self,
        sources: CacheSource | list[CacheSource],
        path: str | Path,
        *,
        writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
    ) -> list[CacheWriteResult]:
        """Write one immutable cache per source under `path`.

        `sources` may be one `CacheSource` or a list of sources. Each source
        must expose a plain `name` and yield `CacheInput` values. Rust writes
        source `name` under `path / name`, validates a stable metadata schema,
        stores payload bytes in shards, writes `metadata.arrow` and `index.bin`,
        then publishes a manifest only after the cache is complete.

        Returns one `CacheWriteSuccess` or `CacheWriteError` per source in input
        order. Per-source failures are reported as values instead of exceptions
        when Rust can handle them cleanly.

        `writer_config.num_processes=0` keeps this native serial path. A positive
        count spawns source-writing processes, each with `self.num_workers`
        native threads. Sources must then be picklable and importable. Jobs
        and outcomes cross processes; sample payloads do not. Worker death or
        `process_timeout_seconds` stops the pool without retrying unfinished
        sources. Profiling keeps serial execution even with a positive count.

        Each source has one owning worker; callers must exclude concurrent write
        invocations targeting the same destinations. Parallel mode refuses
        pre-existing temporary cache paths and reserves the `tmp` source name.
        `show_progress` reports completed source count from the parent; child
        sample progress is disabled. Use a main guard when launching processes.
        """
        return self._write_sources(sources, Path(path), writer_config, reuse_existing=False)

    def _write_sources(
        self,
        sources: CacheSource | list[CacheSource],
        path: Path,
        config: WriterConfig,
        *,
        reuse_existing: bool,
    ) -> list[CacheWriteResult]:
        """Select optional process supervision without changing native serial writing."""
        match config:
            case WriterConfig(num_processes=count) if count > 0 and not config.profiler.enabled:
                from dataset_rt.writer import write_parallel

                return write_parallel(sources, path, config, self.num_workers, reuse_existing)
        return [
            _cache_write_result(result)
            for result in _write_cache(self._inner, sources, str(path), config, reuse_existing)
        ]

    def cached_dataset(
        self,
        paths: Sequence[str | Path],
        *,
        reader_config: ReaderConfig,
    ) -> CachedDataset:
        """Load immutable cache directories into a `CachedDataset`.

        V3 manifests define persistent `cache_id` values. Legacy v2 IDs remain
        positions in `paths` until explicit `update_manifests(version=3)` migration.
        DatasetRT validates manifests, schemas, metadata/index shape, and shard
        lengths while loading. Expensive checksum hashing is controlled by
        `reader_config.validate_cache`.

        The returned dataset keeps this runtime's Rust worker pool alive and
        uses it for every future iterator.
        """
        return CachedDataset._load(self._inner, paths, reader_config, num_workers=self.num_workers)

    def from_cache_sources(
        self,
        sources: CacheSource | list[CacheSource],
        path: str | Path,
        *,
        reader_config: ReaderConfig,
        writer_config: WriterConfig = DEFAULT_WRITER_CONFIG,
    ) -> CacheSourcesDatasetResult:
        """Create or reuse source caches, then load all successful cache paths.

        Existing complete caches under `path / source.name` are reused. Missing
        caches are written with `writer_config`. The method then loads every
        successful cache with `reader_config`.

        Returns `CacheSourcesDatasetSuccess` when at least one cache is loaded;
        returns `CacheSourcesDatasetError` when every source failed or no source
        was provided. The result always includes per-source write outcomes so
        callers can audit partial success.

        Parallel execution follows `write_cache`'s process, timeout, ownership,
        and serialization contract. Successful paths preserve input source order,
        regardless of completion order; IDs come from v3 manifests or v2 positions.
        """
        results = self._write_sources(sources, Path(path), writer_config, reuse_existing=True)
        cache_paths = _successful_cache_paths(results)
        if not cache_paths:
            return _cache_sources_dataset_error(results)
        dataset = self.cached_dataset(cache_paths, reader_config=reader_config)
        return CacheSourcesDatasetSuccess(dataset, results)
