"""Optional source-level process orchestration around the native cache writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dataset_rt._dataset_rt import DatasetRuntime as RustRuntime
from dataset_rt._dataset_rt import validate_source_paths, write_cache

if TYPE_CHECKING:
    from dataset_rt._dataset_rt import CacheWriteRecord
    from dataset_rt.api import CacheSource, WriterConfig


@dataclass(frozen=True)
class CacheWriter:
    """Keep the default native path and opt into bounded source partitions."""

    runtime: RustRuntime
    num_workers: int

    def __call__(
        self,
        sources: CacheSource | list[CacheSource],
        path: str,
        config: WriterConfig,
        reuse_existing: bool,
    ) -> list[CacheWriteRecord]:
        """Preserve native results and validate all destinations before spawning."""
        if getattr(config, "num_processes", 0) == 0:
            return write_cache(self.runtime, sources, path, config, reuse_existing)
        source_list = sources if isinstance(sources, list) else [sources]
        if not source_list:
            return []
        validate_source_paths(source_list, path)
        return self._write_parallel(source_list, path, config, reuse_existing)

    def _write_parallel(
        self,
        sources: list[CacheSource],
        path: str,
        config: WriterConfig,
        reuse_existing: bool,
    ) -> list[CacheWriteRecord]:
        """Submit at most one partition per process and collect in source order."""
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        count = min(config.num_processes, len(sources))
        with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as pool:
            futures = [
                pool.submit(
                    _write_partition,
                    sources[index * len(sources) // count : (index + 1) * len(sources) // count],
                    path,
                    _partition_config(config, index),
                    reuse_existing,
                    self.num_workers,
                )
                for index in range(count)
            ]
            return [record for future in futures for record in future.result()]


def _partition_config(config: WriterConfig, index: int) -> WriterConfig:
    """Let only the first partition render native bars and isolate profiler files."""
    config = config.model_copy(update={"show_progress": config.show_progress and index == 0})
    if not config.profiler.enabled:
        return config
    path = config.profiler.path
    profiler = config.profiler.model_copy(
        update={"path": path.with_name(f"{path.stem}.{index}{path.suffix}")}
    )
    return config.model_copy(update={"profiler": profiler})


def _write_partition(
    sources: list[CacheSource],
    path: str,
    config: WriterConfig,
    reuse_existing: bool,
    num_workers: int,
) -> list[CacheWriteRecord]:
    """Create native worker state inside the child and stream sources directly to Rust."""
    runtime = RustRuntime(num_workers)
    return write_cache(runtime, sources, path, config, reuse_existing)
