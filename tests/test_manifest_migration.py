"""Persistent cache identity against genuine v2 artifacts and saved sample tables."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import polars as pl
import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from dataset_rt import (
    AbsentManifestTarget,
    CachedDataset,
    CacheInput,
    CacheSourcesDatasetSuccess,
    CacheWriteSuccess,
    DatasetRuntime,
    Err,
    Ok,
    ReaderConfig,
    WriterConfig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "manifest-v2"
if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection

    from dataset_rt.records import ReaderRecipe


class ExpectedSample(BaseModel):
    """Validate the legacy reader's recorded weighted sequence once at the boundary."""

    cache_id: int
    sample_id: int
    data_hex: str


class Provenance(BaseModel):
    """Keep the legacy fixture producer, checksums and sampling oracle precise."""

    producer_revision: str
    format_version: int
    paths: list[str]
    seed: int
    sample_sequence: list[ExpectedSample]
    file_sha256: dict[str, str]


class ManifestCommon(BaseModel):
    """Retain storage fields when editing only version-specific fixture identity."""

    model_config = ConfigDict(extra="allow", strict=True)
    source_name: str


class ManifestV2(ManifestCommon):
    """Legacy test records have no persistent cache-ID field."""

    format_version: Literal[2] = 2


class ManifestV3(ManifestCommon):
    """Persistent test records require an explicit integer cache ID."""

    format_version: Literal[3] = 3
    cache_id: int


MANIFEST = TypeAdapter(Annotated[ManifestV2 | ManifestV3, Field(discriminator="format_version")])


@dataclass
class NewSource:
    """Produce new-format caches through both serial and spawned writers."""

    name: str = "new-cache"

    def __iter__(self) -> Iterator[CacheInput]:
        """Match legacy schema so old and new caches can form one dataset."""
        for index in range(4):
            yield CacheInput(
                f"new:{index}".encode(),
                {
                    "index": index,
                    "flag": index % 2 == 0,
                    "score": index + 0.25,
                    "label": "échantillon",
                },
            )


@pytest.fixture
def provenance() -> Provenance:
    """Read the immutable oracle without regenerating any legacy cache artifacts."""
    return Provenance.model_validate_json((FIXTURES / "provenance.json").read_text())


@pytest.fixture
def legacy_paths(tmp_path: Path, provenance: Provenance) -> list[Path]:
    """Copy genuine v2 fixtures so migration never modifies the regression oracle."""
    return [Path(shutil.copytree(FIXTURES / name, tmp_path / name)) for name in provenance.paths]


@pytest.fixture
def runtime() -> DatasetRuntime:
    """Use bounded native resources while comparing serial and parallel loads."""
    return DatasetRuntime(num_workers=2)


@pytest.fixture
def legacy_dataset(runtime: DatasetRuntime, legacy_paths: list[Path]) -> CachedDataset:
    """Resolve historical IDs from the original fixture order."""
    return runtime.cached_dataset(legacy_paths, reader_config=ReaderConfig(seed=17, shuffle=False))


@pytest.fixture
def saved_metadata() -> pl.DataFrame:
    """Load reordered, duplicated and weighted rows produced by the legacy reader."""
    return pl.read_ipc(FIXTURES / "saved-metadata.arrow")


def manifest(path: Path) -> ManifestV2 | ManifestV3:
    """Validate manifest edit fields at the test filesystem boundary."""
    return MANIFEST.validate_json((path / "manifest.json").read_text())


def persist_id(path: Path, cache_id: int) -> None:
    """Create a deliberate v3 fixture while preserving its common storage fields."""
    current = manifest(path)
    upgraded = ManifestV3.model_validate(
        current.model_dump() | {"format_version": 3, "cache_id": cache_id}
    )
    (path / "manifest.json").write_text(upgraded.model_dump_json())


def file_bytes(paths: list[Path], *, manifests: bool = True) -> dict[Path, bytes]:
    """Compare tiny fixture artifacts byte-for-byte without production row bridges."""
    return {
        path: path.read_bytes()
        for root in paths
        for path in root.rglob("*")
        if path.is_file() and (manifests or path.name != "manifest.json")
    }


def assert_legacy_sequence(
    dataset: CachedDataset, saved: pl.DataFrame, provenance: Provenance
) -> None:
    """Saved positional tables and seeded draws must remain valid across migration."""
    dataset.update_metadata(saved)
    dataset.set_epoch_len(len(provenance.sample_sequence))
    actual = [
        ExpectedSample(cache_id=s.cache_id, sample_id=s.sample_id, data_hex=s.data.hex())
        for s in dataset
    ]
    assert actual == provenance.sample_sequence


def test_fixture_provenance(provenance: Provenance) -> None:
    """Ensure the producer and all compatibility artifacts remain unchanged."""
    assert provenance.format_version == 2
    assert provenance.producer_revision.startswith("94f2c3e")
    for name, digest in provenance.file_sha256.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("workers", [1, 4])
def test_v2_reader_preserves_saved_tables(
    legacy_paths: list[Path],
    saved_metadata: pl.DataFrame,
    provenance: Provenance,
    validate: bool,
    workers: int,
) -> None:
    """Both loading modes preserve genuine v2 IDs and exact seeded draws without writes."""
    before = file_bytes(legacy_paths)
    runtime = DatasetRuntime(num_workers=workers)
    dataset = runtime.cached_dataset(
        legacy_paths, reader_config=ReaderConfig(seed=17, shuffle=True, validate_cache=validate)
    )
    assert_legacy_sequence(dataset, saved_metadata, provenance)
    assert file_bytes(legacy_paths) == before


def test_migration_preserves_ids_and_sample_files(
    runtime: DatasetRuntime,
    legacy_paths: list[Path],
    legacy_dataset: CachedDataset,
    saved_metadata: pl.DataFrame,
    provenance: Provenance,
) -> None:
    """Persist old IDs once, then reopen in reverse order with the same saved table."""
    before = file_bytes(legacy_paths, manifests=False)
    legacy_dataset.update_metadata(saved_metadata)
    snapshot = legacy_dataset.get_metadata()
    iterator = iter(legacy_dataset)
    first = next(iterator)
    result = legacy_dataset.update_manifests(version=3)
    assert isinstance(result, Ok)
    assert [entry.cache_id for entry in result.value.entries] == [0, 1]
    assert [entry.status for entry in result.value.entries] == ["updated", "updated"]
    assert legacy_dataset.get_metadata().equals(snapshot)
    assert [first, *iterator] == [
        legacy_dataset.get_item(int(row["cache_id"]), int(row["sample_id"]))
        for row in saved_metadata.to_dicts()
    ]
    assert file_bytes(legacy_paths, manifests=False) == before
    for index, path in enumerate(legacy_paths):
        current = manifest(path)
        assert isinstance(current, ManifestV3)
        assert (current.format_version, current.cache_id) == (3, index)
    repeated = legacy_dataset.update_manifests(version=3)
    assert isinstance(repeated, Ok)
    assert all(entry.status == "unchanged" for entry in repeated.value.entries)
    reloaded = runtime.cached_dataset(
        legacy_paths[::-1], reader_config=ReaderConfig(seed=17, shuffle=True, validate_cache=True)
    )
    assert_legacy_sequence(reloaded, saved_metadata, provenance)


def test_unmigrated_v2_keeps_order_semantics(
    runtime: DatasetRuntime, legacy_paths: list[Path]
) -> None:
    """Legacy caches retain positional IDs even when a caller changes their order."""
    dataset = runtime.cached_dataset(
        legacy_paths[::-1], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    assert dataset.get_item(0, 0).data.startswith("legacy-é:0:".encode())
    assert all(manifest(path).format_version == 2 for path in legacy_paths)


@pytest.mark.parametrize("processes", [0, 2])
def test_new_v3_hash_identity(
    runtime: DatasetRuntime,
    tmp_path: Path,
    legacy_dataset: CachedDataset,
    legacy_paths: list[Path],
    processes: int,
) -> None:
    """New name-derived IDs coexist with migrated positional IDs and survive moves."""
    assert isinstance(legacy_dataset.update_manifests(version=3), Ok)
    result = runtime.write_cache(
        NewSource(),
        tmp_path / "new",
        writer_config=WriterConfig(num_processes=processes, show_progress=False),
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    current = manifest(result.path)
    assert isinstance(current, ManifestV3)
    expected = int.from_bytes(hashlib.sha256(b"new-cache").digest()[:8], "big") & ((1 << 63) - 1)
    assert (current.format_version, current.cache_id) == (3, expected)
    moved = tmp_path / "renamed-cache"
    result.path.rename(moved)
    dataset = runtime.cached_dataset(
        [moved, *legacy_paths], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    assert dataset.get_item(expected, 2).data == b"new:2"
    assert {sample.cache_id for sample in dataset} == {expected, 0, 1}
    table = dataset.get_metadata().slice(0, 2)
    dataset.update_metadata(table)
    assert [sample.cache_id for sample in dataset] == [expected, expected]


@pytest.mark.parametrize("version", [2, 4, -1, True, "3", 3.0])
def test_invalid_migration_target(
    legacy_dataset: CachedDataset, legacy_paths: list[Path], version: int | str | float
) -> None:
    """Dynamic callers receive Result errors rather than conversion exceptions."""
    before = file_bytes(legacy_paths)
    result = legacy_dataset.update_manifests(cast("Literal[3]", version))
    assert isinstance(result, Err)
    assert isinstance(result.error.path, AbsentManifestTarget)
    assert result.error.entries == ()
    assert file_bytes(legacy_paths) == before


@pytest.mark.parametrize("corruption", ["missing", "changed", "invalid", "id_changed"])
def test_preflight_failure_leaves_all_manifests_unchanged(
    legacy_dataset: CachedDataset, legacy_paths: list[Path], corruption: str
) -> None:
    """A stale or invalid later target fails before the earlier target is mutated."""
    path = legacy_paths[1] / "manifest.json"
    match corruption:
        case "missing":
            path.unlink()
        case "changed":
            path.write_text(path.read_text().replace("legacy-é", "changed"))
        case "invalid":
            path.write_text("{")
        case "id_changed":
            persist_id(legacy_paths[1], 99)
    before = file_bytes(legacy_paths)
    result = legacy_dataset.update_manifests(version=3)
    assert isinstance(result, Err)
    assert not result.error.entries
    assert file_bytes(legacy_paths) == before


def test_manifest_symlink_is_preserved(
    runtime: DatasetRuntime, legacy_paths: list[Path], tmp_path: Path
) -> None:
    """Atomically replace the symlink target while preserving filesystem topology."""
    link = legacy_paths[0] / "manifest.json"
    target = tmp_path / "external-manifest.json"
    link.rename(target)
    link.symlink_to(target)
    dataset = runtime.cached_dataset(
        legacy_paths, reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    assert isinstance(dataset.update_manifests(version=3), Ok)
    assert link.is_symlink()
    current = manifest(legacy_paths[0])
    assert isinstance(current, ManifestV3)
    assert current.cache_id == 0


@pytest.mark.parametrize("alias", [False, True])
def test_duplicate_legacy_target_rejected_for_migration(
    runtime: DatasetRuntime, legacy_paths: list[Path], alias: bool
) -> None:
    """Legacy duplicate-path reads remain allowed; one manifest cannot persist two IDs."""
    duplicate = legacy_paths[0]
    if alias:
        duplicate = legacy_paths[0].parent / "alias"
        duplicate.symlink_to(legacy_paths[0].name, target_is_directory=True)
    dataset = runtime.cached_dataset(
        [legacy_paths[0], duplicate], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    before = file_bytes(legacy_paths)
    result = dataset.update_manifests(version=3)
    assert isinstance(result, Err)
    assert "duplicate" in result.error.message
    assert file_bytes(legacy_paths) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_unique_mixed_formats_and_migration(
    runtime: DatasetRuntime, legacy_paths: list[Path], reverse: bool
) -> None:
    """Mixed collections keep stored IDs and supplied-position legacy IDs independently."""
    persist_id(legacy_paths[0], 42)
    paths = legacy_paths[::-1] if reverse else legacy_paths
    dataset = runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=0, shuffle=False))
    legacy_id = 0 if reverse else 1
    assert {sample.cache_id for sample in dataset} == {42, legacy_id}
    assert dataset.get_item(42, 0).data.startswith(b"legacy-plain:0:")
    assert dataset.get_item(legacy_id, 0).data.startswith("legacy-é:0:".encode())
    table = dataset.get_metadata()
    assert isinstance(dataset.update_manifests(version=3), Ok)
    reloaded = runtime.cached_dataset(
        paths[::-1], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    reloaded.update_metadata(table)
    assert reloaded.get_metadata().equals(table)
    assert {sample.cache_id for sample in reloaded} == {42, legacy_id}


@pytest.mark.parametrize("change", ["id", "downgrade"])
def test_loaded_v3_manifest_changes_are_rejected(
    runtime: DatasetRuntime, legacy_paths: list[Path], change: str
) -> None:
    """A newly loaded v3 dataset rejects identity changes and version downgrades before writes."""
    for index, path in enumerate(legacy_paths):
        persist_id(path, index)
    dataset = runtime.cached_dataset(legacy_paths, reader_config=ReaderConfig(seed=0))
    if change == "id":
        persist_id(legacy_paths[1], 99)
    else:
        wire = manifest(legacy_paths[1]).model_dump() | {"format_version": 2}
        (legacy_paths[1] / "manifest.json").write_text(json.dumps(wire))
    before = file_bytes(legacy_paths)
    result = dataset.update_manifests(version=3)
    assert isinstance(result, Err)
    assert result.error.entries == ()
    assert file_bytes(legacy_paths) == before


@pytest.mark.parametrize("id_value", [0, 1, (1 << 64) - 1])
def test_stored_integer_ids_are_not_array_positions(
    runtime: DatasetRuntime, legacy_paths: list[Path], id_value: int
) -> None:
    """Accept the full unsigned persisted-ID range without huge allocations."""
    persist_id(legacy_paths[0], id_value)
    dataset = runtime.cached_dataset(
        [legacy_paths[0]], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    assert dataset.get_item(id_value, 3).sample_id == 3
    table = dataset.get_metadata()
    dataset.update_metadata(table)
    assert {sample.cache_id for sample in dataset} == {id_value}


def test_mixed_collision_is_rejected(runtime: DatasetRuntime, legacy_paths: list[Path]) -> None:
    """A persisted ID colliding with a legacy position is never silently reassigned."""
    persist_id(legacy_paths[0], 1)
    with pytest.raises(ValueError, match="duplicate cache_id"):
        runtime.cached_dataset(legacy_paths, reader_config=ReaderConfig(seed=0))


@pytest.mark.parametrize("bad_id", [-1, 1 << 64, None, True, 1.5, "42", "missing"])
def test_invalid_v3_identity_is_rejected(
    runtime: DatasetRuntime, legacy_paths: list[Path], bad_id: int | float | str | None
) -> None:
    """V3 cannot deserialize into a domain record without its required unsigned ID."""
    current = manifest(legacy_paths[0])
    wire = current.model_dump() | {"format_version": 3}
    if bad_id != "missing":
        wire["cache_id"] = bad_id
    (legacy_paths[0] / "manifest.json").write_text(json.dumps(wire))
    with pytest.raises(RuntimeError, match="JSON error"):
        runtime.cached_dataset([legacy_paths[0]], reader_config=ReaderConfig(seed=0))


def test_future_manifest_version_is_rejected(
    runtime: DatasetRuntime, legacy_paths: list[Path]
) -> None:
    """A future schema never silently falls back to legacy identity rules."""
    wire = manifest(legacy_paths[0]).model_dump() | {"format_version": 4, "cache_id": 0}
    (legacy_paths[0] / "manifest.json").write_text(json.dumps(wire))
    with pytest.raises(RuntimeError, match="unsupported format version 4"):
        runtime.cached_dataset([legacy_paths[0]], reader_config=ReaderConfig(seed=0))


def test_unrelated_manifest_fields_survive_migration(
    runtime: DatasetRuntime, legacy_paths: list[Path]
) -> None:
    """Preserve extensions already present at loading, rather than silently discard them."""
    wire = manifest(legacy_paths[0]).model_dump() | {"application_tag": "keep-me"}
    path = legacy_paths[0] / "manifest.json"
    path.write_text(json.dumps(wire))
    dataset = runtime.cached_dataset(legacy_paths, reader_config=ReaderConfig(seed=0))
    assert isinstance(dataset.update_manifests(version=3), Ok)
    current = manifest(legacy_paths[0])
    assert current.model_extra is not None
    assert current.model_extra["application_tag"] == "keep-me"


def test_migrating_subset_preserves_its_resolved_id(
    runtime: DatasetRuntime, legacy_paths: list[Path]
) -> None:
    """A subset has local positional legacy IDs and cannot infer an omitted cache list."""
    dataset = runtime.cached_dataset([legacy_paths[1]], reader_config=ReaderConfig(seed=0))
    assert isinstance(dataset.update_manifests(version=3), Ok)
    current = manifest(legacy_paths[1])
    assert isinstance(current, ManifestV3) and current.cache_id == 0
    assert manifest(legacy_paths[0]).format_version == 2
    with pytest.raises(ValueError, match="duplicate cache_id"):
        runtime.cached_dataset(legacy_paths, reader_config=ReaderConfig(seed=0))


@pytest.mark.parametrize("processes", [0, 2])
def test_v2_cache_reuse_does_not_migrate(
    runtime: DatasetRuntime, legacy_paths: list[Path], processes: int
) -> None:
    """Writer reuse must keep legacy caches byte-identical and avoid source iteration."""
    before = file_bytes(legacy_paths)
    result = runtime.from_cache_sources(
        [NewSource(path.name) for path in legacy_paths],
        legacy_paths[0].parent,
        writer_config=WriterConfig(
            num_processes=processes, show_progress=False, validate_cache=True
        ),
        reader_config=ReaderConfig(seed=0, shuffle=False),
    )
    assert isinstance(result, CacheSourcesDatasetSuccess)
    assert file_bytes(legacy_paths) == before
    assert {sample.cache_id for sample in result.dataset} == {0, 1}


def consume_migrated_recipe(recipe: ReaderRecipe, output: Connection) -> None:
    """Reconstruct native state from paths and accepted IPC in an actual child."""
    from dataset_rt.records import MetadataSnapshot

    try:
        runtime = DatasetRuntime(num_workers=1)
        dataset = runtime.cached_dataset(recipe.cache_paths, reader_config=recipe.reader_config)
        if isinstance(recipe.metadata, MetadataSnapshot):
            dataset._restore_metadata(recipe.metadata)
        output.send([(sample.cache_id, sample.sample_id) for sample in dataset])
    finally:
        output.close()


@pytest.mark.parametrize("context", ["spawn", "fork", "forkserver"])
def test_migrated_reader_reconstructs_in_child(
    runtime: DatasetRuntime,
    legacy_dataset: CachedDataset,
    legacy_paths: list[Path],
    saved_metadata: pl.DataFrame,
    context: Literal["spawn", "fork", "forkserver"],
) -> None:
    """Worker recipes retain migrated IDs and saved tables even after path reordering."""
    assert isinstance(legacy_dataset.update_manifests(version=3), Ok)
    dataset = runtime.cached_dataset(
        legacy_paths[::-1], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    dataset.update_metadata(saved_metadata)
    list(dataset)
    receive, send = multiprocessing.get_context(context).Pipe(duplex=False)
    match context:
        case "spawn":
            child = multiprocessing.get_context("spawn").Process(
                target=consume_migrated_recipe, args=(dataset._reader_recipe(), send)
            )
        case "fork":
            child = multiprocessing.get_context("fork").Process(
                target=consume_migrated_recipe, args=(dataset._reader_recipe(), send)
            )
        case "forkserver":
            child = multiprocessing.get_context("forkserver").Process(
                target=consume_migrated_recipe, args=(dataset._reader_recipe(), send)
            )
    try:
        child.start()
        send.close()
        assert receive.poll(20)
        assert receive.recv() == [(1, 2), (0, 0), (1, 2), (0, 3)]
        child.join(10)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)
        receive.close()
        send.close()


def test_many_caches_and_large_table_remain_columnar(
    runtime: DatasetRuntime, tmp_path: Path
) -> None:
    """Exercise many ID lookups and large updates without Python row-object expansion."""
    paths = [
        Path(shutil.copytree(FIXTURES / "legacy-plain", tmp_path / f"cache-{index}"))
        for index in range(128)
    ]
    dataset = runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=0, shuffle=False))
    assert isinstance(dataset.update_manifests(version=3), Ok)
    dataset = runtime.cached_dataset(paths[::-1], reader_config=ReaderConfig(seed=0, shuffle=False))
    table = pl.concat([dataset.get_metadata()] * 196).head(100000)
    dataset.update_metadata(table)
    assert len(dataset) == 100000
    assert dataset.get_metadata().equals(table)
    assert dataset.get_item(127, 3).sample_id == 3
    assert sum(1 for _ in dataset) == 100000


@pytest.mark.parametrize("kind", ["root", "cache", "relative-cache"])
def test_symlinked_migration_paths(
    runtime: DatasetRuntime, legacy_paths: list[Path], tmp_path: Path, kind: str
) -> None:
    """Resolved target replacement preserves roots, cache links and original file data."""
    link = tmp_path / "linked"
    if kind == "root":
        actual = tmp_path / "actual"
        actual.mkdir()
        for path in legacy_paths:
            path.rename(actual / path.name)
        link.symlink_to(actual, target_is_directory=True)
        paths = [link / path.name for path in legacy_paths]
    else:
        target = legacy_paths[0].name if kind == "relative-cache" else legacy_paths[0]
        link.symlink_to(target, target_is_directory=True)
        paths = [link, legacy_paths[1]]
    before = file_bytes(paths, manifests=False)
    dataset = runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=0))
    assert isinstance(dataset.update_manifests(version=3), Ok)
    assert link.is_symlink()
    assert file_bytes(paths, manifests=False) == before


def test_read_only_target_reports_failure(
    legacy_dataset: CachedDataset, legacy_paths: list[Path]
) -> None:
    """Filesystem write refusal returns an error and leaves publication markers intact."""
    path = legacy_paths[0]
    original_mode = path.stat().st_mode
    before = file_bytes(legacy_paths)
    try:
        path.chmod(0o555)
        result = legacy_dataset.update_manifests(version=3)
        assert isinstance(result, Err)
        assert result.error.entries == ()
        assert file_bytes(legacy_paths) == before
    finally:
        path.chmod(original_mode)


@pytest.mark.parametrize("field", ["cache_id", "format_version"])
def test_duplicate_wire_fields_are_rejected(
    runtime: DatasetRuntime, legacy_paths: list[Path], field: str
) -> None:
    """Ambiguous JSON identity/version fields never silently select the last value."""
    persist_id(legacy_paths[0], 42)
    path = legacy_paths[0] / "manifest.json"
    path.write_text(path.read_text().removesuffix("}") + f',"{field}":3}}')
    with pytest.raises(RuntimeError, match="duplicate manifest field"):
        runtime.cached_dataset([legacy_paths[0]], reader_config=ReaderConfig(seed=0))


def test_colliding_stored_ids_are_rejected(
    runtime: DatasetRuntime, legacy_paths: list[Path]
) -> None:
    """Different source names cannot resolve to the same integer cache identity."""
    for path in legacy_paths:
        persist_id(path, 42)
    with pytest.raises(ValueError, match="duplicate cache_id"):
        runtime.cached_dataset(legacy_paths, reader_config=ReaderConfig(seed=0))


def test_legacy_cache_id_extension_is_not_authoritative(
    runtime: DatasetRuntime, legacy_paths: list[Path]
) -> None:
    """V2 keeps historical positional semantics even if an application added an ID extension."""
    path = legacy_paths[0] / "manifest.json"
    wire = manifest(legacy_paths[0]).model_dump() | {"cache_id": "legacy-application-tag"}
    path.write_text(json.dumps(wire))
    dataset = runtime.cached_dataset(
        legacy_paths, reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    assert dataset.get_item(0, 0).cache_id == 0
    assert isinstance(dataset.update_manifests(version=3), Ok)
    current = manifest(legacy_paths[0])
    assert isinstance(current, ManifestV3) and current.cache_id == 0
    assert isinstance(dataset.update_manifests(version=3), Ok)


def interrupt_migration(parent_pid: int, ready: Connection) -> None:
    """A separate process can signal the parent while native code holds the GIL."""
    try:
        ready.send("ready")
        time.sleep(0.05)
        os.kill(parent_pid, signal.SIGINT)
    finally:
        ready.close()


def test_migration_interrupt_is_control_flow(runtime: DatasetRuntime, tmp_path: Path) -> None:
    """Interrupt an explicit upgrade, then retry readable partial progress safely."""
    paths = [
        Path(shutil.copytree(FIXTURES / "legacy-plain", tmp_path / f"cache-{index}"))
        for index in range(256)
    ]
    dataset = runtime.cached_dataset(paths, reader_config=ReaderConfig(seed=0, shuffle=False))
    receive, send = multiprocessing.get_context("spawn").Pipe(duplex=False)
    child = multiprocessing.get_context("spawn").Process(
        target=interrupt_migration, args=(os.getpid(), send)
    )
    try:
        child.start()
        send.close()
        assert receive.poll(10)
        assert receive.recv() == "ready"
        with pytest.raises(KeyboardInterrupt):
            dataset.update_manifests(version=3)
        child.join(5)
        assert child.exitcode == 0
        reloaded = runtime.cached_dataset(
            paths, reader_config=ReaderConfig(seed=0, shuffle=False, validate_cache=True)
        )
        assert isinstance(reloaded.update_manifests(version=3), Ok)
        assert all(not list(path.glob(".dataset-rt-manifest-*.tmp")) for path in paths)
        assert {sample.cache_id for sample in reloaded} == set(range(256))
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)
        receive.close()
        send.close()
