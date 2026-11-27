"""Exercise real text/image benchmark workloads without machine-speed assertions."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.bench_storage import (
    Benchmark,
    Measurement,
    Profile,
    TrialSpec,
    compare,
    payloads,
    run_benchmark,
    run_trial,
)

if TYPE_CHECKING:
    from scripts.bench_storage import Codec


@pytest.fixture(params=["text-64", "text-256", "text-4k", "image-64k", "image-1m"])
def profile(request: pytest.FixtureRequest) -> Profile:
    """Validate the matrix label before passing it to typed fixture creation."""
    return TrialSpec(
        profile=request.param,
        codec="none",
        workers=1,
        samples_per_cache=2,
        caches=2,
        prefetch=2,
        pair=0,
        label="candidate",
    ).profile


@pytest.fixture(params=["none", "zstd"])
def measurement(profile: Profile, request: pytest.FixtureRequest, tmp_path: Path) -> Measurement:
    """Write multiple real caches and consume every timed and verified sample."""
    spec = TrialSpec(
        profile=profile,
        codec=request.param,
        workers=2,
        samples_per_cache=8,
        caches=2,
        prefetch=2,
        pair=0,
        label="candidate",
    )
    return run_trial(spec, tmp_path)


def test_real_storage_benchmark(measurement: Measurement) -> None:
    """Small and large payloads exercise real codecs, metadata and shuffled ordering."""
    assert measurement.format_versions == (5, 5)
    assert measurement.verified_samples == 16
    assert measurement.cache_bytes > measurement.shard_bytes > 0
    assert measurement.peak_rss_bytes > 0
    assert measurement.write_seconds > 0
    assert len(measurement.reads) == 4
    for read in measurement.reads:
        assert read.samples == 16
        assert read.payload_bytes == measurement.payload_bytes
        assert read.seconds >= read.first_sample_seconds > 0
        assert read.samples_per_second > 0


def test_profile_payloads_are_domain_data(profile: Profile) -> None:
    """Images are valid decodable PNG streams; text uses exact requested UTF-8 lengths."""
    pool = payloads(profile)
    assert len(pool) == 8
    if profile.startswith("text"):
        expected = {"text-64": 64, "text-256": 256, "text-4k": 4096}[profile]
        assert all(len(item) == expected and " " in item.decode() for item in pool)
        return
    side = 144 if profile == "image-64k" else 592
    for image in pool:
        assert image[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", image[16:24]) == (side, side)
        offset = 8
        compressed = bytearray()
        while offset < len(image):
            length = struct.unpack(">I", image[offset : offset + 4])[0]
            kind = image[offset + 4 : offset + 8]
            data = image[offset + 8 : offset + 8 + length]
            crc = struct.unpack(">I", image[offset + 8 + length : offset + 12 + length])[0]
            assert zlib.crc32(kind + data) == crc
            if kind == b"IDAT":
                compressed.extend(data)
            offset += 12 + length
        assert len(zlib.decompress(compressed)) == side * (side * 3 + 1)


@pytest.fixture
def subprocess_report(tmp_path: Path) -> tuple[Measurement, ...]:
    """The same interpreter exercises isolation without pretending to compare two formats."""
    config = Benchmark(
        baseline_python=Path(sys.executable),
        profiles=("text-256",),
        codecs=("none",),
        workers=(1,),
        samples_per_cache=2,
        caches=2,
        repeats=1,
        quiet=True,
    )
    return run_benchmark(config).measurements


def test_smoke_cannot_claim_performance_acceptance(
    subprocess_report: tuple[Measurement, ...],
) -> None:
    """A single pair of identical formats can never satisfy performance acceptance."""
    assert len(subprocess_report) == 2
    assert all(item.verified_samples == 4 for item in subprocess_report)
    assert all(
        not item.sufficient_repeats and not item.passed for item in compare(subprocess_report, 0.1)
    )


def test_comparison_rejects_different_controls(subprocess_report: tuple[Measurement, ...]) -> None:
    """Prevent invalid cross-release comparisons from producing convincing speed ratios."""
    baseline, candidate = subprocess_report
    changed = candidate.model_copy(
        update={"spec": candidate.spec.model_copy(update={"prefetch": 9})}
    )
    with pytest.raises(ValueError, match="controls differ"):
        compare((baseline, changed), 0.1)


def test_comparison_requires_complete_pair(subprocess_report: tuple[Measurement, ...]) -> None:
    """An interrupted matrix cannot silently turn missing evidence into acceptance."""
    with pytest.raises(ValueError, match="incomplete trial pair"):
        compare(subprocess_report[:1], 0.1)


def test_comparison_requires_matching_dependencies(
    subprocess_report: tuple[Measurement, ...],
) -> None:
    """An unrelated library upgrade must not masquerade as a storage-layout speedup."""
    baseline, candidate = subprocess_report
    changed = candidate.model_copy(
        update={"dependencies": candidate.dependencies | {"polars": "different"}}
    )
    with pytest.raises(ValueError, match="dependency versions differ"):
        compare((baseline, changed), 0.1)


@pytest.mark.parametrize("codec", ["none", "zstd"])
def test_gate_detects_known_read_regression(
    subprocess_report: tuple[Measurement, ...], codec: Codec
) -> None:
    """Synthetic paired times prove a 20% throughput loss fails independently of hardware."""
    baseline, candidate = subprocess_report
    trials = []
    for pair in range(5):
        for trial, layout, seconds in ((baseline, 4, 1.0), (candidate, 5, 1.25)):
            trials.append(
                trial.model_copy(
                    update={
                        "spec": trial.spec.model_copy(update={"pair": pair, "codec": codec}),
                        "format_versions": (layout, layout),
                        "reads": tuple(
                            read.model_copy(update={"seconds": seconds}) for read in trial.reads
                        ),
                    }
                )
            )
    reads = [item for item in compare(tuple(trials), 0.1) if item.metric.startswith("read_")]
    assert len(reads) == 4
    assert all(item.sufficient_repeats and not item.passed for item in reads)
    assert all(item.candidate_over_baseline == pytest.approx(0.8) for item in reads)
