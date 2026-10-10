"""Bounded spawned source writers; Rust retains cache validation and publication."""

from __future__ import annotations

import multiprocessing
import pickle
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias, TypeVar, cast

from dataset_rt._dataset_rt import DatasetRuntime as NativeRuntime
from dataset_rt._dataset_rt import write_cache
from dataset_rt.records import (
    CacheInput,
    CacheSource,
    CacheWriteError,
    CacheWriteResult,
    CacheWriteSuccess,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue

    from dataset_rt.config import WriterConfig


@dataclass
class SourceJob:
    """Freeze destination identity; source iteration stays in the owning child."""

    name: str
    source: CacheSource

    def __iter__(self) -> Iterator[CacheInput]:
        """Delegate serialization without transferring sample payloads to the parent."""
        return iter(self.source)


@dataclass(frozen=True)
class Ready:
    """The child has created its sole reusable native runtime."""


@dataclass(frozen=True)
class Finished:
    """One final native outcome, independent of source completion order."""

    result: CacheWriteResult


@dataclass(frozen=True)
class Interrupted:
    """Preserve process-level control flow rather than treating it as a source error."""

    kind: Literal["keyboard", "exit"]
    message: str
    exit_status: int | str = 0


@dataclass(frozen=True)
class Failed:
    """The worker cannot continue; unfinished jobs must not be retried."""

    message: str


WorkerMessage = Ready | Finished | Interrupted | Failed

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Carry a value accepted at a validation boundary."""

    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    """Carry a recoverable validation error without raising it."""

    error: E


Result: TypeAlias = Ok[T] | Err[E]


def validate_source_name(name: str) -> Result[str, str]:
    """Accept one source destination segment without filesystem access or mutation."""
    if not name or name.casefold() in {".", "..", "tmp"}:
        return Err("CacheSource.name must be a non-reserved plain path segment")
    if "/" in name or "\x00" in name:
        return Err("CacheSource.name must be a plain path segment")
    try:
        encoded = name.encode()
    except UnicodeEncodeError:
        return Err("CacheSource.name must be valid UTF-8")
    if len(encoded) > 255:
        return Err("CacheSource.name exceeds the supported filesystem component limit")
    return Ok(name)


def _diagnostic(error: BaseException) -> str:
    """Bound diagnostic transport even when a source exception formats badly."""
    try:
        return f"{type(error).__name__}: {error}"[:2048]
    except Exception:
        return type(error).__name__


def _exit_status(error: SystemExit) -> int | str:
    """Preserve ordinary exit codes while bounding nonstandard exit diagnostics."""
    match error.code:
        case int():
            return error.code
        case str():
            return error.code[:2048]
        case None:
            return 0
        case _:
            return _diagnostic(error)


def _send(connection: Connection, message: WorkerMessage) -> None:
    """Only bounded lifecycle records cross the result pipe."""
    connection.send_bytes(pickle.dumps(message))


def _receive(connection: Connection) -> Result[WorkerMessage, Failed]:
    """Narrow trusted local worker messages at the pickle boundary."""
    try:
        message = cast("WorkerMessage", pickle.loads(connection.recv_bytes()))
    except (EOFError, OSError, pickle.UnpicklingError) as error:
        return Err(Failed(f"writer transport failed: {_diagnostic(error)}"))
    match message:
        case Ready() | Finished() | Interrupted() | Failed():
            return Ok(message)
        case _:
            return Err(Failed("invalid writer worker message"))


def _run_source(
    runtime: NativeRuntime, job: SourceJob, path: Path, config: WriterConfig, reuse: bool
) -> CacheWriteResult:
    """Pass a single source through unchanged native validation and publication."""
    try:
        records = write_cache(runtime, job, str(path), config, reuse)
    except (ValueError, RuntimeError) as error:
        return CacheWriteError(job.name, _diagnostic(error))
    if len(records) != 1:
        raise ValueError("native single-source writer returned an invalid outcome count")
    status, name, detail = records[0]
    match status:
        case "success":
            return CacheWriteSuccess(name, Path(detail))
        case "error":
            return CacheWriteError(name, detail[:2048])
        case _:
            raise ValueError(f"invalid native writer status: {status}")


def _worker(
    jobs: Queue[bytes],
    results: Connection,
    path: Path,
    config: WriterConfig,
    native_workers: int,
    reuse: bool,
) -> None:
    """Create native state after spawn and reuse it across independent sources."""
    try:
        runtime = NativeRuntime(native_workers)
        _send(results, Ready())
        while True:
            encoded = jobs.get()
            if encoded == b"":
                return
            try:
                job = cast("SourceJob", pickle.loads(encoded))
            except Exception as error:
                _send(results, Failed(f"source deserialization failed: {_diagnostic(error)}"))
                return
            if not isinstance(job, SourceJob):
                raise ValueError("invalid writer job")
            _send(results, Finished(_run_source(runtime, job, path, config, reuse)))
    except KeyboardInterrupt as error:
        _send(results, Interrupted("keyboard", _diagnostic(error)))
    except SystemExit as error:
        _send(results, Interrupted("exit", _diagnostic(error), _exit_status(error)))
    finally:
        results.close()


@dataclass(frozen=True)
class TemporaryDestination:
    """Track a previously absent temporary path owned by one source job."""

    temporary: Path

    def cleanup(self) -> None:
        """Remove the owned temporary cache only after its writer has stopped."""
        if self.temporary.is_dir() and not self.temporary.is_symlink():
            shutil.rmtree(self.temporary)


def _destination_identity(name: str) -> str:
    """Reject case and Unicode aliases before assigning independent source jobs."""
    return unicodedata.normalize("NFC", name.casefold())


def _destination(path: Path, name: str) -> TemporaryDestination | CacheWriteError:
    """Check temporary ownership; callers exclude competing destination writers."""
    temporary = path / "tmp" / name
    if temporary.exists() or temporary.is_symlink():
        return CacheWriteError(name, f"refusing unowned temporary cache: {temporary}")
    return TemporaryDestination(temporary.resolve())


@dataclass(frozen=True)
class Idle:
    """A ready child has no source or destination ownership."""


@dataclass(frozen=True)
class Starting:
    """Child initialization must finish before its deadline."""

    deadline: float


@dataclass(frozen=True)
class Active:
    """Exactly one dispatched source and temporary destination belong to this child."""

    index: int
    deadline: float
    destination: TemporaryDestination


@dataclass
class WriterProcess:
    """One bounded job mailbox, result pipe, and explicitly supervised process."""

    process: BaseProcess
    jobs: Queue[bytes]
    results: Connection
    state: Starting | Idle | Active

    def stop(self) -> None:
        """Bound shutdown before releasing any destination held by this child."""
        if isinstance(self.state, Idle) and self.process.is_alive():
            self.jobs.put_nowait(b"")
            self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2.0)
        if self.process.is_alive():
            raise RuntimeError("writer child could not be stopped; temporary cache retained")
        # A feeder may still hold serialized source bytes after child death.
        self.jobs.cancel_join_thread()
        self.jobs.close()
        self.results.close()
        self.process.close()
        if isinstance(self.state, Active):
            destination = self.state.destination
            self.state = Idle()
            destination.cleanup()


def _start(
    path: Path,
    config: WriterConfig,
    native_workers: int,
    reuse: bool,
) -> WriterProcess:
    """Allocate one child with one bounded mailbox; never inherit native state."""
    context = multiprocessing.get_context("spawn")
    jobs: Queue[bytes] = context.Queue(maxsize=1)
    receiver, sender = context.Pipe(duplex=False)
    child_config = config.model_copy(update={"num_processes": 0, "show_progress": False})
    process = context.Process(
        target=_worker,
        args=(jobs, sender, path, child_config, native_workers, reuse),
    )
    try:
        process.start()
    except BaseException:
        jobs.cancel_join_thread()
        jobs.close()
        receiver.close()
        sender.close()
        raise
    sender.close()
    return WriterProcess(
        process,
        jobs,
        receiver,
        Starting(time.monotonic() + config.process_timeout_seconds),
    )


def _preflight(
    sources: list[CacheSource],
) -> Result[list[SourceJob | CacheWriteError], list[CacheWriteError]]:
    """Reject duplicate destinations before any child starts or source is iterated."""
    jobs: list[SourceJob | CacheWriteError] = []
    for index, source in enumerate(sources):
        try:
            name = source.name
        except Exception as error:
            jobs.append(CacheWriteError(f"source[{index}]", _diagnostic(error)))
            continue
        result = validate_source_name(name)
        match result:
            case Err(error=message):
                jobs.append(CacheWriteError(f"source[{index}]", message))
            case Ok(value=name):
                jobs.append(SourceJob(name, source))
    seen: set[str] = set()
    for job in jobs:
        if isinstance(job, CacheWriteError):
            continue
        identity = _destination_identity(job.name)
        if identity in seen:
            message = f"duplicate generated cache path: {job.name}"
            return Err(
                [
                    item
                    if isinstance(item, CacheWriteError)
                    else CacheWriteError(item.name, message)
                    for item in jobs
                ]
            )
        seen.add(identity)
    return Ok(jobs)


def _dispatch(
    worker: WriterProcess,
    index: int,
    job: SourceJob,
    path: Path,
    timeout: float,
) -> CacheWriteError | Active:
    """Serialize only at an available worker slot with an exclusive source destination."""
    try:
        encoded = pickle.dumps(job)
    except Exception as error:
        return CacheWriteError(job.name, f"source serialization failed: {_diagnostic(error)}")
    try:
        destination = _destination(path, job.name)
    except OSError as error:
        return CacheWriteError(job.name, f"destination preflight failed: {_diagnostic(error)}")
    if isinstance(destination, CacheWriteError):
        return destination
    active = Active(index, time.monotonic() + timeout, destination)
    worker.state = active
    worker.jobs.put_nowait(encoded)
    return active


def _poll(worker: WriterProcess, outcomes: dict[int, CacheWriteResult]) -> Result[None, Failed]:
    """Collect a ready completion before considering child death or its deadline."""
    if worker.results.poll():
        match _receive(worker.results):
            case Err() as failure:
                return failure
            case Ok(value=message):
                pass
        match message:
            case Ready():
                worker.state = Idle()
            case Finished(result=result) if isinstance(worker.state, Active):
                outcomes[worker.state.index] = result
                destination = worker.state.destination
                worker.state = Idle()
                destination.cleanup()
            case Interrupted(kind="keyboard", message=message):
                raise KeyboardInterrupt(message)
            case Interrupted(kind="exit", exit_status=status):
                raise SystemExit(status)
            case Failed(message=message):
                return Err(Failed(message))
            case _:
                return Err(Failed("writer protocol violation"))
    if worker.process.exitcode is not None:
        return Err(Failed(f"writer child exited with code {worker.process.exitcode}"))
    match worker.state:
        case Starting(deadline=deadline) | Active(deadline=deadline) if (
            time.monotonic() >= deadline
        ):
            return Err(Failed("writer process timeout exceeded"))
        case _:
            return Ok(None)


def _supervise(
    workers: list[WriterProcess],
    jobs: list[SourceJob | CacheWriteError],
    path: Path,
    config: WriterConfig,
    outcomes: dict[int, CacheWriteResult],
) -> Result[None, Failed]:
    """Assign one job per idle child without waiting for earlier source outcomes."""
    next_index = 0
    reported = 0
    while len(outcomes) < len(jobs):
        results = [_poll(worker, outcomes) for worker in workers]
        for result in results:
            match result:
                case Err():
                    return result
                case Ok():
                    pass
        for worker in workers:
            while isinstance(worker.state, Idle) and next_index < len(jobs):
                index = next_index
                next_index += 1
                job = jobs[index]
                match job:
                    case CacheWriteError():
                        outcomes[index] = job
                    case SourceJob():
                        result = _dispatch(worker, index, job, path, config.process_timeout_seconds)
                        if isinstance(result, CacheWriteError):
                            outcomes[index] = result
        if config.show_progress and len(outcomes) != reported:
            reported = len(outcomes)
            print(f"DatasetRT sources completed: {reported}/{len(jobs)}", file=sys.stderr)
        if len(outcomes) < len(jobs):
            time.sleep(0.01)
    return Ok(None)


def write_parallel(
    sources: CacheSource | list[CacheSource],
    path: Path,
    config: WriterConfig,
    native_workers: int,
    reuse: bool,
) -> list[CacheWriteResult]:
    """Run bounded independent source jobs, returning the existing ordered outcomes.

    Source failures continue; worker death or timeout stops the pool without
    retrying unfinished jobs. KeyboardInterrupt/SystemExit propagate after
    children stop. Each source has one owning worker. Callers must exclude
    concurrent write invocations targeting the same destinations.
    """
    source_list = sources if isinstance(sources, list) else [sources]
    match _preflight(source_list):
        case Err(error=errors):
            return [error for error in errors]
        case Ok(value=jobs):
            pass
    if not jobs:
        return []
    workers: list[WriterProcess] = []
    outcomes: dict[int, CacheWriteResult] = {}
    failure = "writer pool stopped"
    public_path = path
    try:
        path = path.resolve()
        for _ in range(min(config.num_processes, len(jobs))):
            workers.append(_start(path, config, native_workers, reuse))
        match _supervise(workers, jobs, path, config, outcomes):
            case Err(error=error):
                failure = error.message
            case Ok():
                pass
    except OSError as error:
        failure = f"writer pool failed: {_diagnostic(error)}"
    finally:
        _stop_all(workers)
    return _ordered_outcomes(jobs, outcomes, failure, public_path)


def _stop_all(workers: list[WriterProcess]) -> None:
    """Attempt every child cleanup even if another child's cleanup fails."""
    first_error: Exception | None = None
    for worker in workers:
        try:
            worker.stop()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _ordered_outcomes(
    jobs: list[SourceJob | CacheWriteError],
    outcomes: dict[int, CacheWriteResult],
    failure: str,
    path: Path,
) -> list[CacheWriteResult]:
    """Restore source order and preserve the caller's relative or absolute path edge."""
    results: list[CacheWriteResult] = []
    for index, job in enumerate(jobs):
        name = job.name if isinstance(job, SourceJob) else job.source_name
        result = outcomes.get(
            index, job if isinstance(job, CacheWriteError) else CacheWriteError(name, failure)
        )
        match result:
            case CacheWriteSuccess(source_name=name):
                results.append(CacheWriteSuccess(name, path / name))
            case CacheWriteError():
                results.append(result)
    return results
