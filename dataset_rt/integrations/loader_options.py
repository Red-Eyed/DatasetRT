"""Validate loader settings and resolve explicit modes before exporting metadata."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import isfinite
from multiprocessing import get_all_start_methods
from multiprocessing.context import BaseContext
from typing import Literal

from dataset_rt.integrations.loading import WorkerCounts, resolve_worker_counts

# Torch computes ceil(sample_count / batch_size) through floating-point division.
MAX_SAMPLE_BUDGET = min(sys.maxsize, 1 << 53)


def _validate_count(value: int | Literal["auto"], *, name: str, minimum: int, maximum: int) -> None:
    """Reject unknown modes and coerced integers at the public worker-count boundary."""
    match value:
        case "auto":
            return
        case int() if type(value) is int and minimum <= value <= maximum:
            return
        case _:
            raise ValueError(f"{name} must be 'auto' or an integer between {minimum} and {maximum}")


@dataclass(frozen=True)
class ResolvedWorkers:
    """Worker lifetime settings agree with concrete process/thread counts."""

    counts: WorkerCounts
    persistent: bool


@dataclass(frozen=True)
class LoaderOptions:
    """Validated scalar settings; callbacks and native reader state remain outside this record."""

    shuffle: bool
    seed: int | Literal["random"]
    samples_per_epoch: int | None
    partition: Literal["split", "replicate"] | None
    batch_size: int
    num_workers: int | Literal["auto"]
    native_num_workers: int | Literal["auto"]
    drop_last: bool
    pin_memory: bool
    persistent_workers: bool
    timeout: float
    context: Literal["fork", "spawn", "forkserver"] | BaseContext | None
    prefetch: int | None

    def __post_init__(self) -> None:
        """Validate supplied values before resource detection, export, or worker startup."""
        _validate_count(self.num_workers, name="num_workers", minimum=0, maximum=(1 << 32) - 1)
        _validate_count(
            self.native_num_workers, name="native_num_workers", minimum=1, maximum=sys.maxsize
        )
        for name, value in (
            ("shuffle", self.shuffle),
            ("drop_last", self.drop_last),
            ("pin_memory", self.pin_memory),
            ("persistent_workers", self.persistent_workers),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= sys.maxsize:
            raise ValueError("batch_size must be a positive integer fitting sys.maxsize")
        if self.samples_per_epoch is not None and (
            type(self.samples_per_epoch) is not int
            or not 1 <= self.samples_per_epoch <= MAX_SAMPLE_BUDGET
        ):
            raise ValueError(
                f"samples_per_epoch must be a positive integer <= {MAX_SAMPLE_BUDGET} or None"
            )
        if not isfinite(self.timeout) or self.timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        if self.prefetch is not None and (type(self.prefetch) is not int or self.prefetch < 1):
            raise ValueError("prefetch_factor must be a positive integer")
        self._validate_seed()
        self._validate_context()
        match self.partition:
            case None | "split" | "replicate":
                pass
            case _:
                raise ValueError("worker_partition must be 'split', 'replicate', or None")

    def _validate_seed(self) -> None:
        """Random is a construction-time mode; explicit seeds are finite unsigned values."""
        match self.seed:
            case "random":
                return
            case int() if type(self.seed) is int:
                if not self.shuffle:
                    raise ValueError(
                        "seed requires shuffle=True; use 'random' for sequential reading"
                    )
                if not 0 <= self.seed < 1 << 64:
                    raise ValueError("seed must be 'random' or an unsigned 64-bit integer")
            case _:
                raise ValueError("seed must be 'random' or an unsigned 64-bit integer")

    def _validate_context(self) -> None:
        """Accept only process contexts available on this platform."""
        match self.context:
            case None | BaseContext():
                return
            case str() if self.context in get_all_start_methods():
                return
            case _:
                raise ValueError(
                    "multiprocessing_context must be an available start method or BaseContext"
                )

    def resolve_workers(self, world_size: int) -> ResolvedWorkers:
        """Validate zero-worker constraints after auto selection and normalize persistence."""
        counts = resolve_worker_counts(
            self.num_workers, self.native_num_workers, world_size=world_size
        )
        if counts.processes == 0:
            if self.timeout != 0:
                raise ValueError("timeout must be zero with num_workers=0")
            if self.prefetch is not None:
                raise ValueError("prefetch_factor requires num_workers > 0")
            if self.context is not None:
                raise ValueError("multiprocessing_context requires num_workers > 0")
        return ResolvedWorkers(counts, self.persistent_workers and counts.processes > 0)

    def resolved_seed(self) -> int | None:
        """Translate random selection to the existing internal seed protocol."""
        match self.seed:
            case "random":
                return None
            case int() as seed:
                return seed

    def resolved_partition(self) -> Literal["split", "replicate"]:
        """Retain replicated weighted training and split sequential reading defaults."""
        match self.partition:
            case None:
                return "replicate" if self.shuffle else "split"
            case partition:
                return partition
