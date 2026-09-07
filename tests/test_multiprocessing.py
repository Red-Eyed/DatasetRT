"""Exercise source processes through the public API and native cache validation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from dataset_rt import (
    CacheInput,
    CacheSource,
    CacheSourcesDatasetSuccess,
    CacheWriteError,
    CacheWriteSuccess,
    DatasetRuntime,
    ReaderConfig,
    WriterConfig,
    WriterProfilerConfig,
)


@dataclass
class Source:
    """Record the executing process and optionally fail after a partial write."""

    name: str
    fail: bool = False

    def __iter__(self) -> Iterator[CacheInput]:
        """Expose child execution while keeping payloads deterministic."""
        yield CacheInput(self.name.encode(), {"pid": os.getpid()})
        if self.fail:
            raise ValueError("source failed")


@dataclass
class InterruptedSource:
    """Raise control flow after a sample to exercise native cleanup in a child."""

    name = "interrupted"
    exception: type[BaseException]

    def __iter__(self) -> Iterator[CacheInput]:
        """Ensure a partially created cache never becomes published."""
        yield CacheInput(b"partial", {"pid": os.getpid()})
        raise self.exception("stop child")


@pytest.fixture
def runtime() -> DatasetRuntime:
    """Keep native worker ownership scoped to each test."""
    return DatasetRuntime(num_workers=2)


@pytest.fixture
def process_config() -> WriterConfig:
    """Use a small fixed pool without terminal output."""
    return WriterConfig(num_processes=2, show_progress=False)


def test_zero_processes_keeps_iteration_in_parent(runtime: DatasetRuntime, tmp_path: Path) -> None:
    """The default retains sources that cannot be sent through a process queue."""

    class LocalSource(Source):
        """Remain local to prove the default never serializes a source."""

    assert WriterConfig().num_processes == 0
    results = runtime.write_cache(LocalSource("local"), tmp_path)
    assert isinstance(results[0], CacheWriteSuccess)
    dataset = runtime.cached_dataset([results[0].path], reader_config=ReaderConfig(seed=1))
    assert next(iter(dataset)).metadata["pid"] == os.getpid()


def test_negative_process_count_is_rejected() -> None:
    """Reject an invalid concurrency bound at the config boundary."""
    with pytest.raises(ValidationError):
        WriterConfig.model_validate({"num_processes": -1})


def test_processes_preserve_source_order_and_integrity(
    runtime: DatasetRuntime, process_config: WriterConfig, tmp_path: Path
) -> None:
    """Verify every source, payload, and identity after independent child writes."""
    sources: list[CacheSource] = [Source(f"source_{index}") for index in range(20)]
    results = runtime.write_cache(sources, tmp_path, writer_config=process_config)
    assert [result.source_name for result in results] == [source.name for source in sources]
    paths = [result.path for result in results if isinstance(result, CacheWriteSuccess)]
    assert len(paths) == len(sources)
    dataset = runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=1, shuffle=False, validate_cache=True)
    )
    pids = set()
    for index, sample in enumerate(dataset):
        assert sample.data == sources[index].name.encode()
        assert (sample.cache_id, sample.sample_id) == (index, 0)
        pids.add(sample.metadata["pid"])
    assert os.getpid() not in pids
    assert 1 <= len(pids) <= process_config.num_processes


def test_duplicate_names_fail_before_any_process_writes(
    runtime: DatasetRuntime, process_config: WriterConfig, tmp_path: Path
) -> None:
    """Global native validation prevents destinations colliding across partitions."""
    root = tmp_path / "caches"
    with pytest.raises(ValueError, match="duplicate generated cache path"):
        runtime.write_cache([Source("same"), Source("same")], root, writer_config=process_config)
    assert not root.exists()


def test_process_failure_preserves_other_sources_and_cleans_partial_cache(
    runtime: DatasetRuntime, process_config: WriterConfig, tmp_path: Path
) -> None:
    """Per-source errors remain ordered results without publishing partial data."""
    results = runtime.write_cache(
        [Source("bad", fail=True), Source("good")], tmp_path, writer_config=process_config
    )
    assert isinstance(results[0], CacheWriteError)
    assert "source failed" in results[0].message
    assert isinstance(results[1], CacheWriteSuccess)
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "tmp" / "bad").exists()


def test_process_cache_reuse_does_not_iterate_sources(
    runtime: DatasetRuntime, process_config: WriterConfig, tmp_path: Path
) -> None:
    """Reuse existing complete caches through the same process dispatch path."""
    runtime.write_cache([Source("a"), Source("b")], tmp_path, writer_config=process_config)
    outcome = runtime.from_cache_sources(
        [Source("a", fail=True), Source("b", fail=True)],
        tmp_path,
        writer_config=process_config,
        reader_config=ReaderConfig(seed=1, shuffle=False, validate_cache=True),
    )
    assert isinstance(outcome, CacheSourcesDatasetSuccess)
    assert len(outcome.dataset) == 2


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_process_interrupt_reaches_parent_and_cleans_partial_cache(
    runtime: DatasetRuntime,
    process_config: WriterConfig,
    tmp_path: Path,
    exception: type[BaseException],
) -> None:
    """Do not convert child control flow into an ordinary source error."""
    with pytest.raises(exception, match="stop child"):
        runtime.write_cache(InterruptedSource(exception), tmp_path, writer_config=process_config)
    assert not (tmp_path / "interrupted").exists()
    assert not (tmp_path / "tmp" / "interrupted").exists()


def test_empty_sources_do_not_start_processes(
    runtime: DatasetRuntime, process_config: WriterConfig, tmp_path: Path
) -> None:
    """An empty request returns immediately without creating any cache files."""
    assert runtime.write_cache([], tmp_path, writer_config=process_config) == []
    assert list(tmp_path.iterdir()) == []


def test_process_profiles_have_distinct_paths(runtime: DatasetRuntime, tmp_path: Path) -> None:
    """Concurrent profiler writers must not overwrite one another's summaries."""
    config = WriterConfig(
        num_processes=2,
        show_progress=False,
        profiler=WriterProfilerConfig(enabled=True, path=tmp_path / "profile.json"),
    )
    runtime.write_cache([Source("a"), Source("b")], tmp_path / "caches", writer_config=config)
    assert (tmp_path / "profile.0.json").exists()
    assert (tmp_path / "profile.1.json").exists()
    assert not (tmp_path / "profile.json").exists()
