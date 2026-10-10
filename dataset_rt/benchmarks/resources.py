"""Process-tree measurements; observer threads never transport dataset samples."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import psutil
from pydantic import BaseModel, ConfigDict


class Resources(BaseModel):
    """Summed process-tree RSS can count shared pages more than once."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rss_bytes: int = 0
    threads: int = 0
    file_descriptors: int = 0
    processes: int = 0


def resource_snapshot(process: psutil.Process) -> Resources:
    """Skip processes that exit between enumeration and measurement."""
    rss = threads = descriptors = count = 0
    for member in (process, *process.children(recursive=True)):
        try:
            with member.oneshot():
                rss += member.memory_info().rss
                threads += member.num_threads()
                descriptors += member.num_fds()
                count += 1
        except psutil.NoSuchProcess:
            continue
    return Resources(rss_bytes=rss, threads=threads, file_descriptors=descriptors, processes=count)


def combine_resources(left: Resources, right: Resources) -> Resources:
    """Keep maxima across snapshots without retaining a time-series population."""
    return Resources(
        rss_bytes=max(left.rss_bytes, right.rss_bytes),
        threads=max(left.threads, right.threads),
        file_descriptors=max(left.file_descriptors, right.file_descriptors),
        processes=max(left.processes, right.processes),
    )


@dataclass
class ResourceObserver:
    """Retain only maxima and at most one observer error, in O(processes) work per tick."""

    stop: threading.Event = field(default_factory=threading.Event)
    peak: Resources = field(default_factory=Resources)
    failures: list[Exception] = field(default_factory=list)

    def run(self) -> None:
        """Include the training rank, loading workers, and their service processes."""
        process = psutil.Process()
        try:
            while not self.stop.is_set():
                current = resource_snapshot(process)
                self.peak = combine_resources(self.peak, current)
                self.stop.wait(0.05)
        except Exception as error:
            self.failures.append(error)
