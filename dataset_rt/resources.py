"""Conservative process CPU budgets from affinity and visible container limits."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True)
class _Membership:
    """Validated process membership in a unified or controller-specific hierarchy."""

    controllers: frozenset[str]
    path: PurePosixPath


@dataclass(frozen=True)
class _Mount:
    """Resolve a visible cgroup subtree without assuming a fixed Docker mount path."""

    root: PurePosixPath
    path: Path
    version: Literal[1, 2]
    controllers: frozenset[str]


def _absolute_path(value: str) -> PurePosixPath:
    """Reject malformed or escaping kernel paths before joining mount locations."""
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid cgroup path")
    return path


def _memberships(text: str) -> tuple[_Membership, ...]:
    """Parse process cgroup records once at the kernel-file boundary."""
    records = []
    for line in text.splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        int(hierarchy)
        records.append(
            _Membership(frozenset(filter(None, controllers.split(","))), _absolute_path(path))
        )
    return tuple(records)


def _unescape_mount(value: str) -> str:
    """Decode the four octal escapes used by proc mountinfo paths."""
    for escaped, character in [(r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")]:
        value = value.replace(escaped, character)
    return value


def _mounts(text: str) -> Iterator[_Mount]:
    """Select CPU-relevant cgroup mounts, including remounted controller subtrees."""
    for line in text.splitlines():
        before, after = line.split(" - ", 1)
        filesystem, _, options = after.split(maxsplit=2)
        if filesystem not in ("cgroup", "cgroup2"):
            continue
        fields = before.split()
        root = _absolute_path(_unescape_mount(fields[3]))
        path = Path(_absolute_path(_unescape_mount(fields[4])))
        controllers = frozenset(options.split(","))
        if filesystem == "cgroup2" or controllers.intersection({"cpu", "cpuset"}):
            yield _Mount(root, path, 2 if filesystem == "cgroup2" else 1, controllers)


def _read_optional(path: Path) -> str | None:
    """Missing controller files mean no limit here; other I/O failures propagate."""
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _cpuset_limit(value: str) -> int | None:
    """Count canonical nonoverlapping CPU ranges without expanding CPU IDs."""
    if not value:
        return None
    count = 0
    previous = -1
    for item in value.split(","):
        ends = item.split("-")
        start = int(ends[0])
        end = int(ends[1]) if len(ends) == 2 else start
        if len(ends) > 2 or start <= previous or end < start:
            raise ValueError("invalid cgroup CPU set")
        count += end - start + 1
        previous = end
    return count


def _directory_limits(path: Path, mount: _Mount) -> Iterator[Fraction]:
    """Retain fractional quota limits and inherited CPU-set constraints."""
    if mount.version == 2:
        maximum = _read_optional(path / "cpu.max")
        if maximum is not None:
            quota, period = maximum.split()
            if int(period) <= 0:
                raise ValueError("invalid cgroup CPU period")
            if quota != "max":
                limit = Fraction(int(quota), int(period))
                if limit <= 0:
                    raise ValueError("invalid cgroup CPU quota")
                yield limit
        cpus = _read_optional(path / "cpuset.cpus.effective")
    else:
        if "cpu" in mount.controllers:
            quota = _read_optional(path / "cpu.cfs_quota_us")
            period = _read_optional(path / "cpu.cfs_period_us")
            if (quota is None) != (period is None):
                raise ValueError("incomplete cgroup CPU quota")
            if quota is not None and period is not None:
                if int(period) <= 0 or int(quota) == 0 or int(quota) < -1:
                    raise ValueError("invalid cgroup CPU quota")
                if int(quota) > 0:
                    yield Fraction(int(quota), int(period))
        cpus = _read_optional(path / "cpuset.cpus") if "cpuset" in mount.controllers else None
    if cpus is not None:
        count = _cpuset_limit(cpus)
        if count is not None:
            yield Fraction(count)


def _linux_limits(proc: Path) -> Iterator[Fraction]:
    """Inspect each process hierarchy up to its visible mount boundary in O(depth)."""
    memberships = _memberships((proc / "cgroup").read_text())
    for mount in _mounts((proc / "mountinfo").read_text()):
        for membership in memberships:
            if mount.version == 2:
                if membership.controllers:
                    continue
            elif not membership.controllers.intersection(mount.controllers):
                continue
            # Namespaced memberships may be relative to the mounted subtree root.
            relative = membership.path.relative_to(
                mount.root if membership.path.is_relative_to(mount.root) else PurePosixPath("/")
            )
            current = mount.path / relative
            while True:
                yield from _directory_limits(current, mount)
                if current == mount.path:
                    break
                current = current.parent


def available_cpu_count() -> int:
    """Return whole CPU capacity bounded by affinity and visible cgroup quotas.

    Fractional quotas are rounded down with a one-thread minimum. Linux inspects
    v1/v2 memberships, controller mounts, CPU sets, and visible ancestor quotas.
    Unreadable or malformed limits warn and return one instead of assuming host
    capacity. Hidden ancestor limits and concurrent workload demand are unknowable.
    """
    limit = Fraction(os.cpu_count() or 1)
    try:
        affinity_reader = cast(
            "Callable[[int], set[int]] | None", getattr(os, "sched_getaffinity", None)
        )
        affinity = 0 if affinity_reader is None else len(affinity_reader(0))
    except OSError as error:
        warnings.warn(
            f"Cannot determine CPU affinity; using one CPU: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1
    if affinity:
        limit = min(limit, Fraction(affinity))
    if sys.platform == "linux":
        try:
            for constraint in _linux_limits(Path("/proc/self")):
                limit = min(limit, constraint)
        except (OSError, ValueError, IndexError) as error:
            warnings.warn(
                f"Cannot determine container CPU limits; using one CPU: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return 1
    return max(1, limit.numerator // limit.denominator)


def native_worker_count(cpu_count: int, *, num_workers: int, local_ranks: int) -> int:
    """Share a conservative CPU budget across consumers, capped at four native threads."""
    if cpu_count < 1 or num_workers < 0 or local_ranks < 1:
        raise ValueError("CPU and rank budgets must be positive; loading workers nonnegative")
    return max(1, min(4, cpu_count // (max(1, num_workers) * local_ranks)))


def loading_worker_count(cpu_count: int, *, local_ranks: int) -> int:
    """Choose up to four loading processes per rank, leaving one CPU for training."""
    if cpu_count < 1 or local_ranks < 1:
        raise ValueError("CPU and rank budgets must be positive")
    return min(4, max(0, cpu_count // local_ranks - 1))
