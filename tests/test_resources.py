"""Container CPU limits use real fixture hierarchies and conservative thread budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from dataset_rt import resources
from dataset_rt.integrations.loading import automatic_native_workers

if TYPE_CHECKING:
    from collections.abc import Iterator
    from fractions import Fraction
    from pathlib import Path


@dataclass(frozen=True)
class Hierarchy:
    """Own one process view and a nested cgroup hierarchy in pytest's temporary tree."""

    proc: Path
    root: Path
    leaf: Path


@pytest.fixture
def hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Hierarchy:
    """Model 32 host CPUs, eight affinity CPUs, and a namespaced v2 mount."""
    proc = tmp_path / "proc"
    root = tmp_path / "cgroup"
    leaf = root / "parent" / "child"
    proc.mkdir()
    leaf.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/parent/child\n")
    (proc / "mountinfo").write_text(f"1 0 0:1 / {root} rw - cgroup2 cgroup rw\n")
    original = resources._linux_limits

    def fixture_limits(_proc: Path) -> Iterator[Fraction]:
        """Redirect only kernel-file discovery while exercising the real parsers."""
        return original(proc)

    monkeypatch.setattr(resources, "_linux_limits", fixture_limits)
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 32)

    def affinity(_pid: int) -> set[int]:
        """Model a process CPU mask independently of its wider host count."""
        return set(range(8))

    monkeypatch.setattr(resources.os, "sched_getaffinity", affinity, raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    return Hierarchy(proc, root, leaf)


@pytest.mark.parametrize(
    ("quota", "expected"),
    [("max 100000", 8), ("200000 100000", 2), ("150000 100000", 1), ("50000 100000", 1)],
)
def test_quota_overrides_host_and_affinity(hierarchy: Hierarchy, quota: str, expected: int) -> None:
    """Docker-style quotas constrain whole CPU capacity, including fractional limits."""
    (hierarchy.leaf / "cpu.max").write_text(quota)
    assert resources.available_cpu_count() == expected


def test_visible_ancestor_quota_is_not_ignored(hierarchy: Hierarchy) -> None:
    """An unlimited leaf cannot override the parent's stricter allocation."""
    (hierarchy.leaf / "cpu.max").write_text("max 100000")
    (hierarchy.leaf.parent / "cpu.max").write_text("200000 100000")
    (hierarchy.root / "cpu.max").write_text("400000 100000")
    assert resources.available_cpu_count() == 2


def test_cpuset_constraints_are_counted_without_expansion(hierarchy: Hierarchy) -> None:
    """Use effective CPU ranges even when the affinity API reports a wider mask."""
    (hierarchy.leaf / "cpuset.cpus.effective").write_text("1-2,7")
    assert resources.available_cpu_count() == 3


@pytest.mark.parametrize("membership", ["/docker/id/parent/child", "/parent/child"])
def test_remounted_and_namespaced_hierarchies(hierarchy: Hierarchy, membership: str) -> None:
    """Resolve both host-relative and namespace-relative memberships under a bind root."""
    (hierarchy.proc / "cgroup").write_text(f"0::{membership}\n")
    (hierarchy.proc / "mountinfo").write_text(
        f"1 0 0:1 /docker/id {hierarchy.root} rw - cgroup2 cgroup rw\n"
    )
    (hierarchy.leaf / "cpu.max").write_text("200000 100000")
    # A sibling file outside the mount must never influence this process.
    (hierarchy.root.parent / "cpu.max").write_text("50000 100000")
    assert resources.available_cpu_count() == 2


@pytest.mark.parametrize(("quota", "expected"), [("-1", 8), ("200000", 2), ("150000", 1)])
def test_legacy_cpu_controller(hierarchy: Hierarchy, quota: str, expected: int) -> None:
    """Apply legacy CFS quota/period limits instead of assuming cgroup v2."""
    (hierarchy.proc / "cgroup").write_text("2:cpu,cpuacct:/parent/child\n")
    (hierarchy.proc / "mountinfo").write_text(
        f"1 0 0:1 / {hierarchy.root} rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    (hierarchy.leaf / "cpu.cfs_quota_us").write_text(quota)
    (hierarchy.leaf / "cpu.cfs_period_us").write_text("100000")
    assert resources.available_cpu_count() == expected


@pytest.mark.parametrize("quota", ["garbage", "0 100000", "200000 0", "-1 100000"])
def test_bad_limits_fall_back_conservatively(hierarchy: Hierarchy, quota: str) -> None:
    """Malformed visible limits must not silently grant the container host capacity."""
    (hierarchy.leaf / "cpu.max").write_text(quota)
    with pytest.warns(RuntimeWarning, match="using one CPU"):
        assert resources.available_cpu_count() == 1


def test_unreadable_process_view_falls_back(hierarchy: Hierarchy) -> None:
    """Missing proc metadata cannot establish a container-safe CPU budget."""
    (hierarchy.proc / "mountinfo").unlink()
    with pytest.warns(RuntimeWarning, match="using one CPU"):
        assert resources.available_cpu_count() == 1


@pytest.mark.parametrize(
    ("cpus", "workers", "ranks", "expected"),
    [(32, 8, 1, 4), (32, 8, 2, 2), (2, 8, 1, 1), (32, 0, 1, 4), (1, 0, 4, 1)],
)
def test_consumer_budget_is_bounded(cpus: int, workers: int, ranks: int, expected: int) -> None:
    """Avoid multiplying a full CPU allocation into every native runtime."""
    assert resources.native_worker_count(cpus, num_workers=workers, local_ranks=ranks) == expected


def test_launcher_local_ranks_share_quota(
    hierarchy: Hierarchy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local launcher rank count, rather than global multi-host size, divides the budget."""
    (hierarchy.leaf / "cpu.max").write_text("800000 100000")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert automatic_native_workers(num_workers=2, world_size=8) == 2
    monkeypatch.delenv("LOCAL_WORLD_SIZE")
    assert automatic_native_workers(num_workers=2, world_size=8) == 1


def test_legacy_cpu_set_and_quota_combine(hierarchy: Hierarchy) -> None:
    """A legacy CPU-set restriction can be tighter than its bandwidth quota."""
    (hierarchy.proc / "cgroup").write_text("2:cpu,cpuset:/parent/child\n")
    (hierarchy.proc / "mountinfo").write_text(
        f"1 0 0:1 / {hierarchy.root} rw - cgroup cgroup rw,cpu,cpuset\n"
    )
    (hierarchy.leaf / "cpu.cfs_quota_us").write_text("400000")
    (hierarchy.leaf / "cpu.cfs_period_us").write_text("100000")
    (hierarchy.leaf / "cpuset.cpus").write_text("2,7")
    assert resources.available_cpu_count() == 2


def test_escaped_controller_mount(hierarchy: Hierarchy) -> None:
    """Decode mountinfo escapes without treating a mount path as separate fields."""
    relocated = hierarchy.root.with_name("cgroup with spaces")
    hierarchy.root.rename(relocated)
    escaped = str(relocated).replace(" ", r"\040")
    (hierarchy.proc / "mountinfo").write_text(f"1 0 0:1 / {escaped} rw - cgroup2 cgroup rw\n")
    (relocated / "parent" / "child" / "cpu.max").write_text("200000 100000")
    assert resources.available_cpu_count() == 2


def test_incomplete_legacy_quota_does_not_grant_host_capacity(hierarchy: Hierarchy) -> None:
    """An unreadable paired period makes the available quota unsafe to ignore."""
    (hierarchy.proc / "cgroup").write_text("2:cpu:/parent/child\n")
    (hierarchy.proc / "mountinfo").write_text(
        f"1 0 0:1 / {hierarchy.root} rw - cgroup cgroup rw,cpu\n"
    )
    (hierarchy.leaf / "cpu.cfs_quota_us").write_text("200000")
    with pytest.warns(RuntimeWarning, match="using one CPU"):
        assert resources.available_cpu_count() == 1


@pytest.mark.parametrize("local_ranks", ["0", "9", "garbage"])
def test_invalid_launcher_local_count_is_rejected(
    hierarchy: Hierarchy, monkeypatch: pytest.MonkeyPatch, local_ranks: str
) -> None:
    """Reject invalid launcher topology instead of inventing an oversubscribed budget."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", local_ranks)
    with pytest.raises(ValueError):
        automatic_native_workers(num_workers=2, world_size=8)


def test_escaping_membership_falls_back(hierarchy: Hierarchy) -> None:
    """Do not read arbitrary parent files for an unresolvable namespace membership."""
    (hierarchy.proc / "cgroup").write_text("0::/../../other\n")
    with pytest.warns(RuntimeWarning, match="using one CPU"):
        assert resources.available_cpu_count() == 1


def test_unreadable_affinity_is_not_ignored(
    hierarchy: Hierarchy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An affinity query failure must not silently select wider host capacity."""

    def unavailable(_pid: int) -> set[int]:
        """Represent an unavailable process CPU mask at the operating-system boundary."""
        raise OSError("affinity unavailable")

    monkeypatch.setattr(resources.os, "sched_getaffinity", unavailable)
    with pytest.warns(RuntimeWarning, match="CPU affinity; using one CPU"):
        assert resources.available_cpu_count() == 1
