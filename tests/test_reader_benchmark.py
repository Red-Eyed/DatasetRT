"""Bounded real benchmark smoke; throughput thresholds stay outside pytest."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from dataset_rt import CachedSample
from scripts.bench_reader import (
    Benchmark,
    PassMeasurement,
    ReaderCase,
    Resources,
    Transform,
    TrialResult,
    TrialSpec,
    checked_batch,
    create_fixture,
    performance_gates,
    run_trial,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config(tmp_path: Path) -> Benchmark:
    """Use two streamed caches and tiny batches, without timing-based assertions."""
    return Benchmark(
        samples_per_cache=16,
        caches=2,
        batch_size=4,
        transform_rounds=2,
        repeats=1,
        workloads=("heavy",),
        output=tmp_path / "report.json",
    )


@pytest.fixture
def paths(config: Benchmark, tmp_path: Path) -> tuple[Path, ...]:
    """Publish real fixtures before isolated trials create their own native readers."""
    return create_fixture(config, tmp_path)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        ReaderCase(mode="native", workers=0, context="serial"),
        ReaderCase(mode="loader", workers=0, context="serial"),
        ReaderCase(mode="loader", workers=2, context="fork"),
    ],
)
def test_reader_trial(
    config: Benchmark, paths: tuple[Path, ...], case: ReaderCase, shuffle: bool
) -> None:
    """Consume exact counts/bytes and observe actual persistent transform processes."""
    result = run_trial(
        TrialSpec(config=config, paths=paths, case=case, shuffle=shuffle, workload="heavy", pair=0)
    )
    for measurement in (result.cold, result.warm):
        assert measurement.accepted == 32
        assert measurement.bytes_read == 32 * config.payload_bytes
        assert len(measurement.transform_pids) == max(1, case.workers)
        assert measurement.resources.processes >= 1 + case.workers
        assert measurement.resources.rss_bytes > 0
    assert result.cold.transform_pids == result.warm.transform_pids


def test_transform_uses_delivered_bytes() -> None:
    """Verify the workload checksum rather than only counting callback invocations."""
    payload = (3).to_bytes(8, "little") + (7).to_bytes(8, "little") + b"domain payload"
    sample = CachedSample(payload, {}, 3, 7)
    expected = hashlib.sha256(payload).digest()
    expected = hashlib.sha256(expected).digest()
    expected = hashlib.sha256(expected).digest()
    output = Transform(2)(sample)
    assert output.digest == expected and output.bytes_read == len(payload)
    assert checked_batch((output,)) == (output,)


@pytest.mark.parametrize("value", [(), (b"untransformed",), ["wrong batch representation"]])
def test_collator_boundary(value: object) -> None:
    """Reject malformed outputs before they can inflate benchmark work counts."""
    with pytest.raises(TypeError):
        checked_batch(value)


@pytest.mark.parametrize("samples", [1, 3])
def test_exact_work_configuration(samples: int) -> None:
    """Prevent empty steady intervals and partially discarded shuffled batches."""
    with pytest.raises(ValidationError, match="total fixture samples"):
        Benchmark(samples_per_cache=samples, caches=1, batch_size=2)


def test_smoke_is_not_performance_acceptance(config: Benchmark, paths: tuple[Path, ...]) -> None:
    """One real pair cannot satisfy the five-repeat acceptance requirement."""
    results = tuple(
        run_trial(
            TrialSpec(
                config=config, paths=paths, case=case, shuffle=False, workload="heavy", pair=0
            )
        )
        for case in (
            ReaderCase(mode="native", workers=0, context="serial"),
            ReaderCase(mode="loader", workers=0, context="serial"),
        )
    )
    gates = performance_gates(results)
    assert len(gates) == 1
    assert not gates[0].sufficient_repeats and not gates[0].passed


@pytest.fixture
def known_trials() -> tuple[TrialResult, ...]:
    """Use synthetic seconds to test gate arithmetic independently of machine speed."""
    trials = []
    for pair in range(5):
        for case, seconds in (
            (ReaderCase(mode="native", workers=0, context="serial"), 1.0),
            (ReaderCase(mode="loader", workers=0, context="serial"), 1.12),
            (ReaderCase(mode="loader", workers=4, context="fork"), 0.56),
        ):
            measurement = PassMeasurement(
                iterator_seconds=0.0,
                first_batch_seconds=0.1,
                steady_seconds=seconds,
                steady_samples=96,
                accepted=100,
                bytes_read=100 * 1024,
                transform_pids=tuple(range(1, max(1, case.workers) + 1)),
                resources=Resources(),
            )
            trials.append(
                TrialResult(
                    case=case,
                    shuffle=True,
                    workload="heavy",
                    pair=pair,
                    construction_seconds=0.0,
                    cold=measurement,
                    warm=measurement,
                    cleanup_seconds=0.0,
                )
            )
    return tuple(trials)


def test_paired_gate_arithmetic(known_trials: tuple[TrialResult, ...]) -> None:
    """A known 12% regression fails and an exact 2x gain passes after five pairs."""
    regression, speedup = performance_gates(known_trials)
    assert regression.metric == "serial_regression"
    assert regression.median == pytest.approx(0.12)
    assert regression.sufficient_repeats and not regression.passed
    assert speedup.metric == "four_worker_speedup"
    assert speedup.median == pytest.approx(2.0)
    assert speedup.sufficient_repeats and speedup.passed
