"""Real spawned writer acceptance with bounded source and failure fixtures."""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import pytest

from dataset_rt import (
    CacheInput,
    CacheSourcesDatasetSuccess,
    CacheWriteError,
    CacheWriteSuccess,
    DatasetRuntime,
    ReaderConfig,
    WriterConfig,
    WriterProfilerConfig,
)
from dataset_rt._dataset_rt import DatasetRuntime as NativeRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.connection import Connection
    from multiprocessing.queues import Queue

    from dataset_rt.records import CacheSource


@dataclass
class Source:
    """Stream identifiable payloads and exercise source-local process behavior."""

    name: str
    count: int = 8
    delay: float = 0.0
    behavior: Literal["normal", "error", "death", "blocked", "keyboard", "exit"] = "normal"
    observations: Path = Path("/private/tmp")

    def __iter__(self) -> Iterator[CacheInput]:
        """Failures happen after native temporary-cache creation, never in the parent."""
        if self.observations != Path("/private/tmp"):
            (self.observations / self.name).write_text(str(os.getpid()))
        time.sleep(self.delay)
        match self.behavior:
            case "death":
                os._exit(17)
            case "blocked":
                time.sleep(120)
            case "keyboard":
                raise KeyboardInterrupt("source interrupt")
            case "exit":
                raise SystemExit(23)
            case "error":
                yield CacheInput(b"partial", {"index": 0})
                raise ValueError("source failure")
            case "normal":
                pass
        for index in range(self.count):
            yield CacheInput(f"{self.name}:{index}".encode(), {"index": index, "pid": os.getpid()})


@dataclass
class UnpicklableSource:
    """Serial source descriptors are not required to support pickle."""

    name: str = "unpicklable"

    def __getstate__(self) -> dict[str, str]:
        """Expose a recoverable serialization failure in parallel mode."""
        raise ValueError("cannot pickle this source")

    def __iter__(self) -> Iterator[CacheInput]:
        """The serial writer still consumes this source normally."""
        yield CacheInput(b"serial", {"index": 0})


@pytest.fixture
def runtime() -> DatasetRuntime:
    """Keep the caller's native thread budget explicit."""
    return DatasetRuntime(num_workers=1)


@pytest.mark.parametrize("name", ["source", "é", "a" * 255, ".dataset-rt-locks"])
def test_source_name_validation_accepts_plain_segments(name: str) -> None:
    """Validated names retain their exact identity rather than being rewritten."""
    from dataset_rt.writer import Ok, validate_source_name

    assert validate_source_name(name) == Ok(name)


@pytest.mark.parametrize("name", ["", ".", "..", "Tmp", "../escape", "a\x00b", "a" * 256, "\ud800"])
def test_source_name_validation_returns_errors(name: str) -> None:
    """Invalid source names produce typed errors without exception-driven control flow."""
    from dataset_rt.writer import Err, validate_source_name

    result = validate_source_name(name)
    assert isinstance(result, Err)
    assert result.error.startswith("CacheSource.name")


@pytest.fixture
def parallel_config() -> WriterConfig:
    """Finite test supervision avoids a blocked source hanging pytest."""
    return WriterConfig(num_processes=2, process_timeout_seconds=10, show_progress=False)


@pytest.fixture
def observations(tmp_path: Path) -> Path:
    """Use explicit fixture artifacts to observe actual source process identities."""
    path = tmp_path / "observations"
    path.mkdir()
    return path


@pytest.mark.parametrize("processes", [0, 1, 2, 4])
def test_ordered_sources_and_integrity(
    runtime: DatasetRuntime,
    tmp_path: Path,
    processes: int,
) -> None:
    """Every public source outcome preserves order and native payload integrity."""
    sources: list[CacheSource] = [Source(f"source-{index}", count=1000) for index in range(5)]
    result = runtime.from_cache_sources(
        sources,
        tmp_path / "cache",
        writer_config=WriterConfig(num_processes=processes, show_progress=False),
        reader_config=ReaderConfig(seed=7, shuffle=False, validate_cache=True),
    )
    assert isinstance(result, CacheSourcesDatasetSuccess)
    assert [outcome.source_name for outcome in result.results] == [
        source.name for source in sources
    ]
    assert all(isinstance(outcome, CacheWriteSuccess) for outcome in result.results)
    count = 0
    pids: set[int] = set()
    for sample in result.dataset:
        assert sample.data == f"source-{count // 1000}:{sample.sample_id}".encode()
        pids.add(int(sample.metadata["pid"]))
        count += 1
    assert count == 5000
    if processes == 0:
        assert pids == {os.getpid()}
    else:
        assert os.getpid() not in pids
        assert 1 <= len(pids) <= processes
    assert not multiprocessing.active_children()


def test_source_failure_keeps_healthy_jobs(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """A handled source error publishes no cache and does not break the pool."""
    sources: list[CacheSource] = [Source("bad", behavior="error"), Source("good"), Source("next")]
    results = runtime.write_cache(sources, tmp_path, writer_config=parallel_config)
    assert isinstance(results[0], CacheWriteError)
    assert all(isinstance(result, CacheWriteSuccess) for result in results[1:])
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "tmp" / "bad").exists()


def test_serial_unpicklable_source(runtime: DatasetRuntime, tmp_path: Path) -> None:
    """The zero-process branch must never attempt source serialization."""
    results = runtime.write_cache(
        UnpicklableSource(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )
    assert isinstance(results[0], CacheWriteSuccess)


def test_parallel_serialization_failure(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """Serialization failures remain ordered per-source outcomes; other jobs run."""
    sources: list[CacheSource] = [UnpicklableSource(), Source("good")]
    results = runtime.write_cache(sources, tmp_path, writer_config=parallel_config)
    assert isinstance(results[0], CacheWriteError)
    assert "serialization" in results[0].message
    assert isinstance(results[1], CacheWriteSuccess)


@pytest.mark.parametrize("behavior", ["death", "blocked"])
def test_broken_worker_is_bounded_and_cleans_owned_paths(
    runtime: DatasetRuntime,
    tmp_path: Path,
    behavior: Literal["death", "blocked"],
) -> None:
    """A dead or blocked child cannot strand native temporary caches or live children."""
    started = time.monotonic()
    result = runtime.write_cache(
        Source("bad", behavior=behavior),
        tmp_path,
        writer_config=WriterConfig(num_processes=1, process_timeout_seconds=2, show_progress=False),
    )
    assert isinstance(result[0], CacheWriteError)
    assert time.monotonic() - started < 12
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "tmp" / "bad").exists()
    assert not multiprocessing.active_children()
    retry = runtime.write_cache(
        Source("bad"), tmp_path, writer_config=WriterConfig(num_processes=1, show_progress=False)
    )
    assert isinstance(retry[0], CacheWriteSuccess)


@pytest.mark.parametrize(
    ("behavior", "exception"), [("keyboard", KeyboardInterrupt), ("exit", SystemExit)]
)
def test_child_control_flow_propagates_after_cleanup(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
    behavior: Literal["keyboard", "exit"],
    exception: type[KeyboardInterrupt] | type[SystemExit],
) -> None:
    """Source-level control flow stops the operation rather than becoming an error row."""
    with pytest.raises(exception) as caught:
        runtime.write_cache(
            Source("interrupt", behavior=behavior), tmp_path, writer_config=parallel_config
        )
    assert not multiprocessing.active_children()
    assert not (tmp_path / "tmp" / "interrupt").exists()
    if isinstance(caught.value, SystemExit):
        with pytest.raises(SystemExit) as serial:
            runtime.write_cache(
                Source("serial-exit", behavior="exit"),
                tmp_path,
                writer_config=WriterConfig(show_progress=False),
            )
        assert caught.value.code == serial.value.code


def _recording_worker(
    jobs: Queue[bytes],
    results: Connection,
    path: Path,
    config: WriterConfig,
    native_workers: int,
    reuse: bool,
) -> None:
    """Instrument the native constructor inside an actual spawned child."""
    from dataset_rt.writer import _worker

    def create_runtime(count: int) -> NativeRuntime:
        """Count actual native runtime constructions rather than guessing from PIDs."""
        with (path / f"runtime-{os.getpid()}").open("a") as log:
            log.write("created\n")
        return NativeRuntime(count)

    with patch("dataset_rt.writer.NativeRuntime", side_effect=create_runtime):
        _worker(jobs, results, path, config, native_workers, reuse)


def test_runtime_is_constructed_once_per_child(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More sources than processes reuse the same native instance in each child."""
    import dataset_rt.writer as writer

    monkeypatch.setattr(writer, "_worker", _recording_worker)
    sources: list[CacheSource] = [Source(f"source-{index}", delay=0.05) for index in range(8)]
    results = runtime.write_cache(sources, tmp_path, writer_config=parallel_config)
    assert all(isinstance(result, CacheWriteSuccess) for result in results)
    logs = sorted(tmp_path.glob("runtime-*"))
    assert len(logs) == 2
    assert all(log.read_text() == "created\n" for log in logs)


@dataclass
class SerializationProbe:
    """Record serialization time to detect eager submission beyond available slots."""

    name: str
    observations: Path

    def __reduce__(self) -> tuple[type[SerializationProbe], tuple[str, Path]]:
        """Observe the parent-side boundary without serializing any payload population."""
        (self.observations / self.name).write_text(str(time.monotonic()))
        return type(self), (self.name, self.observations)

    def __iter__(self) -> Iterator[CacheInput]:
        """Make the first source hold the only available worker slot."""
        if self.name == "first":
            time.sleep(0.25)
        yield CacheInput(b"sample", {"index": 0})
        (self.observations / f"{self.name}-finished").write_text(str(time.monotonic()))


def test_source_serialization_is_bounded_by_available_slots(
    runtime: DatasetRuntime,
    tmp_path: Path,
    observations: Path,
) -> None:
    """Later descriptors remain unpickled until the previous source has completed."""
    sources: list[CacheSource] = [
        SerializationProbe(name, observations) for name in ("first", "second", "third")
    ]
    results = runtime.write_cache(
        sources,
        tmp_path / "cache",
        writer_config=WriterConfig(num_processes=1, show_progress=False),
    )
    assert all(isinstance(result, CacheWriteSuccess) for result in results)
    assert float((observations / "second").read_text()) > float(
        (observations / "first-finished").read_text()
    )
    assert float((observations / "third").read_text()) > float(
        (observations / "second-finished").read_text()
    )


def test_relative_result_paths_are_preserved(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-local absolute paths must not change the public caller path shape."""
    monkeypatch.chdir(tmp_path)
    results = runtime.write_cache(Source("relative"), Path("cache"), writer_config=parallel_config)
    assert results == [CacheWriteSuccess("relative", Path("cache/relative"))]


@pytest.mark.parametrize("name", ["", "../escape", "tmp", "Tmp"])
def test_parallel_preflight_rejects_unsafe_destinations(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
    name: str,
) -> None:
    """Source names cannot alias the temporary directory in the process harness."""
    results = runtime.write_cache(Source(name), tmp_path, writer_config=parallel_config)
    assert isinstance(results[0], CacheWriteError)


def test_empty_source_list_starts_no_children(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """The process harness preserves the empty ordered outcome contract."""
    assert runtime.write_cache([], tmp_path, writer_config=parallel_config) == []


@pytest.mark.parametrize(
    "updates",
    [
        {"num_processes": -1},
        {"process_timeout_seconds": 0},
        {"process_timeout_seconds": float("inf")},
    ],
)
def test_process_configuration_is_validated(updates: dict[str, int | float]) -> None:
    """A process count or deadline cannot bypass config validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WriterConfig.model_validate(updates)


def test_parent_interrupt_cleans_children(
    runtime: DatasetRuntime,
    tmp_path: Path,
    observations: Path,
) -> None:
    """Caller Ctrl-C reaps a blocked writer before its owned temporary path is removed."""

    def interrupt() -> None:
        """Signal only after the child has entered its source iterator."""
        deadline = time.monotonic() + 8
        while not (observations / "blocked").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGINT)

    observer = threading.Thread(target=interrupt)
    observer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            runtime.write_cache(
                Source("blocked", behavior="blocked", observations=observations),
                tmp_path / "cache",
                writer_config=WriterConfig(
                    num_processes=1, process_timeout_seconds=10, show_progress=False
                ),
            )
    finally:
        observer.join(timeout=10)
    assert not observer.is_alive()
    assert not multiprocessing.active_children()
    assert not (tmp_path / "cache" / "tmp" / "blocked").exists()


def test_dynamic_assignment_does_not_wait_for_first_source(
    runtime: DatasetRuntime,
    tmp_path: Path,
    observations: Path,
    parallel_config: WriterConfig,
) -> None:
    """A free child starts later sources while the first source is still blocked."""
    sources: list[CacheSource] = [
        Source("slow", delay=1, observations=observations),
        Source("fast", observations=observations),
        Source("later", observations=observations),
    ]
    results = runtime.write_cache(sources, tmp_path / "cache", writer_config=parallel_config)
    assert all(isinstance(result, CacheWriteSuccess) for result in results)
    assert (observations / "later").stat().st_mtime < (
        tmp_path / "cache" / "slow" / "manifest.json"
    ).stat().st_mtime
    assert (observations / "fast").read_text() == (observations / "later").read_text()


def test_reuse_does_not_iterate_source(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """Native reuse bypasses a source that would fail if iterated again."""
    runtime.write_cache(Source("reused"), tmp_path, writer_config=parallel_config)
    result = runtime.from_cache_sources(
        Source("reused", behavior="error"),
        tmp_path,
        writer_config=parallel_config,
        reader_config=ReaderConfig(seed=0, shuffle=False, validate_cache=True),
    )
    assert isinstance(result, CacheSourcesDatasetSuccess)
    assert len(result.dataset) == 8


@pytest.mark.parametrize(
    ("first", "second"), [("same", "same"), ("Case", "case"), ("é", "e\u0301")]
)
def test_duplicate_destinations_fail_before_writes(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
    first: str,
    second: str,
) -> None:
    """Duplicate names cannot race, even when assigned to different processes."""
    results = runtime.write_cache(
        [Source(first), Source(second)], tmp_path, writer_config=parallel_config
    )
    assert len(results) == 2
    assert [result.source_name for result in results] == [first, second]
    for result in results:
        assert isinstance(result, CacheWriteError)
        assert "duplicate generated cache path" in result.message
    assert not (tmp_path / first).exists()


def test_supervisor_bug_propagates_after_cleanup(
    runtime: DatasetRuntime, tmp_path: Path, parallel_config: WriterConfig
) -> None:
    """Unexpected parent bugs stay visible while every spawned child is stopped."""
    with patch("dataset_rt.writer._supervise", side_effect=TypeError("supervisor bug")):
        with pytest.raises(TypeError, match="supervisor bug"):
            runtime.write_cache(Source("bug"), tmp_path, writer_config=parallel_config)
    assert not multiprocessing.active_children()


def test_unowned_temporary_cache_is_preserved(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """Parent cleanup never deletes a temporary path that predates its ownership."""
    temporary = tmp_path / "tmp" / "existing"
    temporary.mkdir(parents=True)
    marker = temporary / "keep"
    marker.write_text("unowned")
    results = runtime.write_cache(Source("existing"), tmp_path, writer_config=parallel_config)
    assert isinstance(results[0], CacheWriteError)
    assert "unowned" in results[0].message
    assert marker.read_text() == "unowned"


@pytest.mark.parametrize("behavior", ["normal", "error", "death"])
def test_symlinked_cache_root_and_temporary_directory(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
    behavior: Literal["normal", "error", "death"],
) -> None:
    """Writing and failure cleanup preserve symlinked parents and unrelated contents."""
    root = tmp_path / "real-cache"
    root.mkdir()
    cache = tmp_path / "cache-link"
    cache.symlink_to(root, target_is_directory=True)
    temporary = tmp_path / "real-temporary"
    temporary.mkdir()
    (root / "tmp").symlink_to(temporary, target_is_directory=True)
    marker = temporary / "unrelated"
    marker.write_text("keep")
    results = runtime.write_cache(
        Source("source", behavior=behavior), cache, writer_config=parallel_config
    )
    if behavior == "normal":
        assert results == [CacheWriteSuccess("source", cache / "source")]
        assert (root / "source" / "manifest.json").is_file()
    else:
        assert isinstance(results[0], CacheWriteError)
        assert not (root / "source").exists()
    assert cache.is_symlink()
    assert (root / "tmp").is_symlink()
    assert temporary.is_dir()
    assert marker.read_text() == "keep"
    assert not (temporary / "source").exists()


@dataclass
class BufferSource:
    """Yield an unpicklable payload that must remain inside its writing child."""

    name: str = "buffers"

    def __iter__(self) -> Iterator[CacheInput]:
        """Native buffer extraction succeeds without serializing sample objects."""
        yield CacheInput(memoryview(b"buffer"), {"index": 0})


def test_sample_payloads_do_not_cross_process_boundaries(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """Memoryview samples prove only source descriptors and outcomes are transported."""
    result = runtime.from_cache_sources(
        BufferSource(),
        tmp_path,
        writer_config=parallel_config,
        reader_config=ReaderConfig(seed=0, shuffle=False),
    )
    assert isinstance(result, CacheSourcesDatasetSuccess)
    assert next(iter(result.dataset)).data == b"buffer"


def _fail_reconstruction(name: str) -> Source:
    """An importable pickle constructor can fail after dispatch in a child."""
    raise ValueError(f"cannot reconstruct {name}")


@dataclass
class BadReconstructionSource:
    """Separate a child deserialization failure from a parent serialization failure."""

    name: str = "reconstruction"

    def __reduce__(self) -> tuple[Callable[[str], Source], tuple[str]]:
        """Parent encoding succeeds; reconstruction fails in the supervised child."""
        return _fail_reconstruction, (self.name,)

    def __iter__(self) -> Iterator[CacheInput]:
        """This body must never be reached by the failed child."""
        yield CacheInput(b"unexpected", {"index": 0})


def test_child_deserialization_failure_stops_without_retry(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """Bad child reconstruction yields bounded broken-pool outcomes and no cache."""
    results = runtime.write_cache(
        BadReconstructionSource(), tmp_path, writer_config=parallel_config
    )
    assert isinstance(results[0], CacheWriteError)
    assert "deserialization" in results[0].message
    assert not (tmp_path / "reconstruction").exists()
    assert not multiprocessing.active_children()


def test_completed_outcomes_survive_later_worker_death(
    runtime: DatasetRuntime,
    tmp_path: Path,
    parallel_config: WriterConfig,
) -> None:
    """Pool failure preserves already-completed healthy caches in input order."""
    sources: list[CacheSource] = [
        Source("death", delay=0.5, behavior="death"),
        Source("healthy"),
        Source("blocked", behavior="blocked"),
    ]
    results = runtime.write_cache(sources, tmp_path, writer_config=parallel_config)
    assert isinstance(results[0], CacheWriteError)
    assert isinstance(results[1], CacheWriteSuccess)
    assert isinstance(results[2], CacheWriteError)
    assert not (tmp_path / "tmp" / "death").exists()
    assert not (tmp_path / "tmp" / "blocked").exists()


def test_profiling_keeps_serial_execution(
    runtime: DatasetRuntime,
    tmp_path: Path,
    observations: Path,
) -> None:
    """A single native profiler report remains serial even with a process count."""
    results = runtime.write_cache(
        Source("profiled", observations=observations),
        tmp_path / "cache",
        writer_config=WriterConfig(
            num_processes=2,
            show_progress=False,
            profiler=WriterProfilerConfig(enabled=True, path=tmp_path / "profile.json"),
        ),
    )
    assert isinstance(results[0], CacheWriteSuccess)
    assert (observations / "profiled").read_text() == str(os.getpid())
    assert (tmp_path / "profile.json").is_file()
