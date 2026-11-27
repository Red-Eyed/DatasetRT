"""The current reader must match immutable ground truth captured by each historical producer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from dataset_rt import CacheWriteSuccess, DatasetRuntime, ReaderConfig, WriterConfig
from scripts.generate_storage_fixtures import (
    PINNED_REVISIONS,
    ExpectedSample,
    GroundTruth,
    Identity,
    ManifestHeader,
    SampleMetadata,
    Source,
    codec_cases,
    file_checksums,
)

if TYPE_CHECKING:
    from dataset_rt import CachedSample

ROOT = Path(__file__).parent / "fixtures" / "storage-compatibility"
ORACLES = tuple(sorted(ROOT.glob("v*/oracle.json")))


@dataclass(frozen=True)
class FixtureSet:
    """Keep the saved oracle and its cache paths together, preserving producer path order."""

    directory: Path
    truth: GroundTruth

    @property
    def paths(self) -> tuple[Path, ...]:
        """V2 identities depend on this original path order."""
        return tuple(self.directory / cache.name for cache in self.truth.caches)


@pytest.fixture(params=ORACLES, ids=[path.parent.name for path in ORACLES])
def fixture_set(request: pytest.FixtureRequest) -> FixtureSet:
    """Read committed test assets only; never invoke a producer during ordinary pytest."""
    path = Path(request.param)
    return FixtureSet(path.parent, GroundTruth.model_validate_json(path.read_text()))


@pytest.fixture(params=[1, 4])
def runtime(request: pytest.FixtureRequest) -> DatasetRuntime:
    """Exercise serial and parallel completion against the same historical oracle."""
    return DatasetRuntime(num_workers=int(request.param))


def actual_sample(sample: CachedSample) -> ExpectedSample:
    """Compare complete outputs to saved ground truth without consulting current input generation."""
    return ExpectedSample(
        cache_id=sample.cache_id,
        sample_id=sample.sample_id,
        data_hex=sample.data.hex(),
        metadata=SampleMetadata.model_validate(sample.metadata),
    )


def test_provenance_and_artifacts_are_frozen(fixture_set: FixtureSet) -> None:
    """Changed or missing historical artifacts must fail rather than silently redefine truth."""
    truth = fixture_set.truth
    assert truth.format_version == int(fixture_set.directory.name.removeprefix("v"))
    assert len(truth.producer.source_sha256) == 64
    assert len(truth.producer.revision) == 40
    if truth.format_version in PINNED_REVISIONS:
        assert truth.producer.revision == PINNED_REVISIONS[truth.format_version]
        assert not truth.producer.working_tree
    assert {cache.codec.name for cache in truth.caches} == {
        case.name for case in codec_cases(truth.format_version)
    }
    assert file_checksums(fixture_set.directory) == truth.file_sha256
    for cache, path in zip(truth.caches, fixture_set.paths, strict=True):
        header = ManifestHeader.model_validate_json((path / "manifest.json").read_text())
        assert header.format_version == truth.format_version
        assert (
            sum(sample.metadata.source == cache.name for sample in truth.sequential)
            == cache.sample_count
        )


@pytest.mark.parametrize("validate", [False, True])
def test_historical_sequential_and_random_access(
    fixture_set: FixtureSet, runtime: DatasetRuntime, validate: bool
) -> None:
    """Payloads, all metadata fields and identities must match the producer, without file changes."""
    truth = fixture_set.truth
    dataset = runtime.cached_dataset(
        fixture_set.paths,
        reader_config=ReaderConfig(seed=truth.seed, shuffle=False, validate_cache=validate),
    )
    assert len(dataset) == len(truth.sequential)
    assert tuple(actual_sample(sample) for sample in dataset) == truth.sequential
    for expected in reversed(truth.sequential):
        assert actual_sample(dataset.get_item(expected.cache_id, expected.sample_id)) == expected
    assert file_checksums(fixture_set.directory) == truth.file_sha256


@pytest.mark.parametrize("validate", [False, True])
def test_historical_weighted_epochs(
    fixture_set: FixtureSet, runtime: DatasetRuntime, validate: bool
) -> None:
    """Saved historical weights and consecutive seeded epochs remain valid in the new reader."""
    truth = fixture_set.truth
    dataset = runtime.cached_dataset(
        fixture_set.paths,
        reader_config=ReaderConfig(seed=truth.seed, shuffle=True, validate_cache=validate),
    )
    weights = pl.read_ipc(fixture_set.directory / "weights.arrow")
    dataset.update_metadata(weights)
    expected_samples = {(sample.cache_id, sample.sample_id): sample for sample in truth.sequential}
    dataset.set_epoch_len(len(truth.weighted_epochs[0]))
    for expected_epoch in truth.weighted_epochs:
        actual = tuple(actual_sample(sample) for sample in dataset)
        assert (
            tuple(
                Identity(cache_id=sample.cache_id, sample_id=sample.sample_id) for sample in actual
            )
            == expected_epoch
        )
        for sample in actual:
            assert sample == expected_samples[sample.cache_id, sample.sample_id]
    assert file_checksums(fixture_set.directory) == truth.file_sha256


@pytest.fixture
def all_fixtures() -> tuple[FixtureSet, ...]:
    """Load each format oracle once for mixed-version coverage and the newest-format guard."""
    return tuple(
        FixtureSet(path.parent, GroundTruth.model_validate_json(path.read_text()))
        for path in ORACLES
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_mixed_formats(
    all_fixtures: tuple[FixtureSet, ...],
    runtime: DatasetRuntime,
    reverse: bool,
    validate: bool,
    shuffle: bool,
) -> None:
    """Persistent IDs survive reordering; positional v2 IDs follow the supplied mixed path list."""
    entries = [(fixture, cache) for fixture in all_fixtures for cache in fixture.truth.caches]
    if reverse:
        entries.reverse()
    expected = []
    paths = []
    for position, (fixture, cache) in enumerate(entries):
        paths.append(fixture.directory / cache.name)
        for sample in fixture.truth.sequential:
            if sample.metadata.source == cache.name:
                identity = position if fixture.truth.format_version == 2 else sample.cache_id
                expected.append(sample.model_copy(update={"cache_id": identity}))
    lookup = {(sample.cache_id, sample.sample_id): sample for sample in expected}
    dataset = runtime.cached_dataset(
        paths, reader_config=ReaderConfig(seed=17, shuffle=shuffle, validate_cache=validate)
    )
    actual = tuple(actual_sample(sample) for sample in dataset)
    assert len(actual) == len(expected)
    if not shuffle:
        assert actual == tuple(expected)
    for sample in actual:
        assert sample == lookup[sample.cache_id, sample.sample_id]
    for expected_sample in lookup.values():
        assert (
            actual_sample(dataset.get_item(expected_sample.cache_id, expected_sample.sample_id))
            == expected_sample
        )
    for fixture in all_fixtures:
        assert file_checksums(fixture.directory) == fixture.truth.file_sha256


def test_every_format_through_latest_has_ground_truth(
    all_fixtures: tuple[FixtureSet, ...], tmp_path: Path
) -> None:
    """A new default writer format cannot pass CI until its producer oracle is committed."""
    runtime = DatasetRuntime(num_workers=1)
    result = runtime.write_cache(
        Source("latest-format-probe"), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    latest = ManifestHeader.model_validate_json(
        (result.path / "manifest.json").read_text()
    ).format_version
    covered = {fixture.truth.format_version for fixture in all_fixtures}
    assert covered == set(range(2, latest + 1)), (
        f"format coverage {sorted(covered)} must include v2 through v{latest}; "
        "capture the new producer's cache and ground truth with generate_storage_fixtures.py"
    )
