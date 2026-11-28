from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import polars as pl
import pytest
from pydantic import ValidationError

import dataset_rt.api as dataset_rt_api
import dataset_rt.integrations.torch as torch_integration
from dataset_rt import (
    CacheInput,
    CacheSourcesDatasetError,
    CacheSourcesDatasetSuccess,
    CacheWriteError,
    CacheWriteResult,
    CacheWriteSuccess,
    DatasetRuntime,
    NoCompression,
    ReaderConfig,
    ShardCompression,
    WriterConfig,
    WriterProfilerConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Never

    from dataset_rt.records import CacheSource

RUNTIME = DatasetRuntime(num_workers=4)
TINY_ID = 671_225_450_407_077_731
FIVE_ID = 2_462_074_631_277_705_190
OTHER_ID = 6_424_818_148_262_835_032
MalformedCacheInput: TypeAlias = (
    bytes | tuple[bytes] | tuple[bytes, list[str] | dict[int, str] | dict[str, list[str]]]
)
FIRST_ID = 2_851_758_661_582_890_383
SECOND_ID = 1_600_601_599_791_221_249


class InvalidPayload:
    """Exercise Rust payload validation with a non-buffer value."""


class WorkerSentinel:
    """Signal worker presence without constructing Torch worker state."""


class TinySource:
    name = "tiny"

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream cache inputs for this test scenario."""
        yield CacheInput(b"zero", {"label": "a", "index": 0, "score": 1.5, "kept": True})
        yield CacheInput(bytearray(b"one"), {"label": "b", "index": 1, "score": 2.5, "kept": False})
        yield CacheInput(memoryview(b"two"), {"label": "c", "index": 2, "score": 3.5, "kept": True})


class FiveSource:
    name = "five"

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream cache inputs for this test scenario."""
        for index in range(5):
            yield CacheInput(str(index + 1).encode(), {"index": index, "split": "train"})


def success_paths(results: list[CacheWriteResult]) -> list[Path]:
    paths: list[Path] = []
    for result in results:
        match result:
            case CacheWriteSuccess(path=path):
                paths.append(path)
            case CacheWriteError(source_name=source_name, message=message):
                raise AssertionError(f"{source_name}: {message}")
    return paths


@pytest.fixture
def tiny_cache_paths(tmp_path: Path) -> list[Path]:
    return success_paths(RUNTIME.write_cache(TinySource(), tmp_path / "cache"))


@pytest.fixture
def tiny_dataset(tiny_cache_paths: list[Path]) -> dataset_rt_api.CachedDataset:
    return RUNTIME.cached_dataset(
        tiny_cache_paths,
        reader_config=ReaderConfig(seed=7, shuffle=False),
    )


@pytest.fixture
def shuffled_tiny_dataset(tiny_cache_paths: list[Path]) -> dataset_rt_api.CachedDataset:
    return RUNTIME.cached_dataset(
        tiny_cache_paths,
        reader_config=ReaderConfig(seed=7, shuffle=True),
    )


@pytest.fixture
def five_cache_paths(tmp_path: Path) -> list[Path]:
    return success_paths(RUNTIME.write_cache(FiveSource(), tmp_path / "cache"))


@pytest.fixture
def five_dataset(five_cache_paths: list[Path]) -> dataset_rt_api.CachedDataset:
    return RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=False),
    )


@pytest.fixture
def shuffled_five_dataset(five_cache_paths: list[Path]) -> dataset_rt_api.CachedDataset:
    return RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )


def sample_indices(dataset: dataset_rt_api.CachedDataset) -> list[int]:
    return [cast("int", sample.metadata["index"]) for sample in dataset]


def test_write_and_read_cache(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    results = RUNTIME.write_cache(
        TinySource(),
        base_cache_dir,
        writer_config=WriterConfig(
            prefetch_size=2,
            max_shard_bytes=4,
            shard_compression=ShardCompression(algo="none", ratio=1.0),
            show_progress=False,
        ),
    )
    written = success_paths(results)
    dataset = RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=42))

    assert results == [CacheWriteSuccess("tiny", base_cache_dir / "tiny")]
    assert written[0].parent == base_cache_dir
    assert written[0].name == "tiny"
    assert not (base_cache_dir / "tmp" / "tiny").exists()
    assert len(dataset) == 3
    samples = list(dataset)

    assert len(samples) == 3
    assert {sample.data for sample in samples}.issubset({b"zero", b"one", b"two"})
    assert {sample.cache_id for sample in samples} == {TINY_ID}
    assert all(sample.metadata["label"] in {"a", "b", "c"} for sample in samples)


def test_writer_manifest_records_shard_compression(tmp_path: Path) -> None:
    written = success_paths(
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "cache",
            writer_config=WriterConfig(
                shard_compression=ShardCompression(algo="none", ratio=1.0),
            ),
        )
    )

    manifest = json.loads((written[0] / "manifest.json").read_text())

    assert manifest["shards"][0]["compression"] == {"algo": "none", "ratio": 1.0}
    assert manifest["shards"][0]["byte_len"] > manifest["shards"][0]["uncompressed_byte_len"]


def test_shard_records_contain_only_payloads(tmp_path: Path) -> None:
    written = success_paths(
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "cache",
            writer_config=WriterConfig(shard_compression=NoCompression(), show_progress=False),
        )
    )
    cache_path = written[0]
    manifest = json.loads((cache_path / "manifest.json").read_text())
    index = (cache_path / "index.bin").read_bytes()
    shard = (cache_path / "shards" / manifest["shards"][0]["name"]).read_bytes()
    first_record = shard[: int.from_bytes(index[16:24], "little")]
    assert manifest["format_version"] == 5
    assert first_record == b"zero"
    assert shard == b"zeroonetwo"
    dataset = RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=1, shuffle=False))
    assert next(iter(dataset)).metadata == {"index": 0, "kept": True, "label": "a", "score": 1.5}


def test_writer_lz4_compresses_payloads_and_reads_original_bytes(tmp_path: Path) -> None:
    class CompressibleSource:
        name = "compressible"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            yield CacheInput(b"a" * 10_000, {"label": "first"})
            yield CacheInput(b"b" * 10_000, {"label": "second"})

    written = success_paths(
        RUNTIME.write_cache(
            CompressibleSource(),
            tmp_path / "cache",
            writer_config=WriterConfig(
                shard_compression=ShardCompression(algo="lz4", ratio=2.0),
                show_progress=False,
            ),
        )
    )
    dataset = RUNTIME.cached_dataset(
        written,
        reader_config=ReaderConfig(seed=42, shuffle=False),
    )
    samples = list(dataset)
    manifest = json.loads((written[0] / "manifest.json").read_text())
    shard = manifest["shards"][0]

    assert [sample.data for sample in samples] == [b"a" * 10_000, b"b" * 10_000]
    assert shard["compression"] == {"algo": "lz4", "ratio": 2.0}
    assert shard["byte_len"] < shard["uncompressed_byte_len"]


def test_writer_progress_is_optional(tmp_path: Path) -> None:
    assert WriterConfig().show_progress is True
    assert WriterConfig().validate_cache is False
    assert ReaderConfig(seed=1).validate_cache is False

    written = success_paths(
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "cache",
            writer_config=WriterConfig(show_progress=False),
        )
    )

    assert len(written) == 1


def test_writer_profiler_is_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    written = success_paths(
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "cache",
            writer_config=WriterConfig(show_progress=False),
        )
    )

    assert len(written) == 1
    assert not (tmp_path / "dataset_rt_profile.json").exists()


def test_writer_profiler_records_success_summary(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"

    written = success_paths(
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "cache",
            writer_config=WriterConfig(
                show_progress=False,
                profiler=WriterProfilerConfig(enabled=True, path=profile_path),
            ),
        )
    )
    profile = json.loads(profile_path.read_text())
    tiny_profile = profile["sources"][0]
    stages = {stage["name"]: stage for stage in tiny_profile["stages"]}

    assert len(written) == 1
    assert profile["format_version"] == 1
    assert tiny_profile["source_name"] == "tiny"
    assert stages["python_next"]["calls"] == 4
    assert stages["python_extract"]["bytes"] == 10
    assert stages["compression"]["calls"] == 3
    assert stages["disk_write"]["calls"] == 3
    assert stages["finish_manifest"]["calls"] == 1
    assert stages["publish"]["calls"] == 1


def test_writer_profiler_flushes_on_keyboard_interrupt(tmp_path: Path) -> None:
    class InterruptedSource:
        name = "interrupted_profile"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            yield CacheInput(b"first", {"label": "ok"})
            raise KeyboardInterrupt("stop")

    profile_path = tmp_path / "interrupted-profile.json"

    with pytest.raises(KeyboardInterrupt):
        RUNTIME.write_cache(
            InterruptedSource(),
            tmp_path / "cache",
            writer_config=WriterConfig(
                show_progress=False,
                profiler=WriterProfilerConfig(enabled=True, path=profile_path),
            ),
        )

    profile = json.loads(profile_path.read_text())
    interrupted_profile = profile["sources"][0]
    stages = {stage["name"]: stage for stage in interrupted_profile["stages"]}

    assert interrupted_profile["source_name"] == "interrupted_profile"
    assert stages["python_next"]["calls"] >= 1
    assert stages["python_extract"]["bytes"] == 5


def test_write_multiple_sources_returns_cache_paths(tmp_path: Path) -> None:
    class OtherSource(TinySource):
        name = "other"

    root = tmp_path / "caches"

    written = success_paths(RUNTIME.write_cache([TinySource(), OtherSource()], root))
    dataset = RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=42))

    assert [path.parent for path in written] == [root, root]
    assert [path.name for path in written] == ["tiny", "other"]
    assert len(dataset) == 6


def test_dataset_loads_multiple_caches_in_constructor_order(tmp_path: Path) -> None:
    class OtherSource:
        name = "other"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            yield CacheInput(b"three", {"label": "d", "index": 3, "score": 4.5, "kept": False})
            yield CacheInput(b"four", {"label": "e", "index": 4, "score": 5.5, "kept": True})

    root = tmp_path / "caches"
    written = success_paths(RUNTIME.write_cache([TinySource(), OtherSource()], root))
    dataset = RUNTIME.cached_dataset(
        [written[1], written[0]],
        reader_config=ReaderConfig(seed=42, shuffle=False),
    )

    samples = list(dataset)

    assert [sample.data for sample in samples] == [b"three", b"four", b"zero", b"one", b"two"]
    assert [sample.cache_id for sample in samples] == [
        OTHER_ID,
        OTHER_ID,
        TINY_ID,
        TINY_ID,
        TINY_ID,
    ]

    assert dataset.get_item(OTHER_ID, 1) == samples[1]
    assert dataset.get_item(TINY_ID, 2) == samples[4]


def test_get_item_reads_filtered_sample_without_advancing_cursor(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    tiny_dataset.update_metadata(tiny_dataset.get_metadata().tail(2))

    sample = tiny_dataset.get_item(TINY_ID, 0)

    assert sample == dataset_rt_api.CachedSample(
        b"zero", {"index": 0, "kept": True, "label": "a", "score": 1.5}, TINY_ID, 0
    )
    assert [sample.data for sample in tiny_dataset] == [b"one", b"two"]


def test_get_item_does_not_advance_shuffled_draws(
    shuffled_five_dataset: dataset_rt_api.CachedDataset,
    five_cache_paths: list[Path],
) -> None:
    reference = RUNTIME.cached_dataset(
        five_cache_paths, reader_config=ReaderConfig(seed=11, shuffle=True)
    )

    assert shuffled_five_dataset.get_item(FIVE_ID, 3).data == b"4"
    assert sample_indices(shuffled_five_dataset) == sample_indices(reference)


def test_get_item_reads_sparse_samples_across_many_rows_and_caches(tmp_path: Path) -> None:
    class LargeSource:
        def __init__(self, name: str) -> None:
            self.name = name

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            for sample_id in range(1_024):
                yield CacheInput(sample_id.to_bytes(2, "little"), {"sample_id": sample_id})

    paths = success_paths(
        RUNTIME.write_cache(
            [LargeSource("first"), LargeSource("second")],
            tmp_path / "caches",
            writer_config=WriterConfig(show_progress=False),
        )
    )
    dataset = RUNTIME.cached_dataset(paths, reader_config=ReaderConfig(seed=7, shuffle=False))

    for cache_id, sample_id in [
        (FIRST_ID, 0),
        (FIRST_ID, 1_023),
        (SECOND_ID, 0),
        (SECOND_ID, 1_023),
    ]:
        sample = dataset.get_item(cache_id, sample_id)
        assert sample.data == sample_id.to_bytes(2, "little")
        assert sample.metadata == {"sample_id": sample_id}
        assert (sample.cache_id, sample.sample_id) == (cache_id, sample_id)


@pytest.mark.parametrize(
    "cache_id, sample_id", [(-1, 0), (TINY_ID, -1), (1, 0), (TINY_ID, 3), (2**64, 0)]
)
def test_get_item_rejects_unknown_identity(
    tiny_dataset: dataset_rt_api.CachedDataset, cache_id: int, sample_id: int
) -> None:
    with pytest.raises(IndexError, match="out of range"):
        tiny_dataset.get_item(cache_id, sample_id)


@pytest.mark.parametrize("cache_id, sample_id", [(True, 0), (0, False), ("0", 0), (0, 1.0)])
def test_get_item_rejects_non_integer_identity(
    tiny_dataset: dataset_rt_api.CachedDataset,
    cache_id: int | str | float,
    sample_id: int | str | float,
) -> None:
    with pytest.raises(TypeError, match="must be integers"):
        tiny_dataset.get_item(cast("int", cache_id), cast("int", sample_id))


def test_multi_source_write_rejects_duplicate_names_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "caches"

    with pytest.raises(ValueError, match="duplicate generated cache path"):
        RUNTIME.write_cache([TinySource(), TinySource()], root)

    assert not root.exists()


def test_writer_config_validation_happens_in_rust(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        WriterConfig(prefetch_size=cast("int", 0))

    with pytest.raises(ValidationError):
        WriterConfig.model_validate({"num_threads": 4})

    with pytest.raises(ValueError, match="num_workers must be greater than zero"):
        DatasetRuntime(num_workers=0)

    assert DatasetRuntime(num_workers=1).num_workers == 1

    with pytest.raises(ValueError, match="ratio must be 1.0"):
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "compression",
            writer_config=WriterConfig(
                shard_compression=ShardCompression(algo="none", ratio=2.0),
            ),
        )

    class UnsupportedCompression:
        algo = "unsupported-codec"
        ratio = 2.0

    class UnsupportedWriterConfig:
        prefetch_size = 64
        max_shard_bytes = 64 * 1024 * 1024
        shard_compression = UnsupportedCompression()
        show_progress = False
        validate_cache = False

    with pytest.raises(ValueError, match="unsupported shard compression"):
        RUNTIME.write_cache(
            TinySource(),
            tmp_path / "unsupported-codec",
            writer_config=cast("WriterConfig", UnsupportedWriterConfig()),
        )


def test_dataset_keeps_runtime_pool_alive(tmp_path: Path) -> None:
    def load_dataset() -> dataset_rt_api.CachedDataset:
        """Return a dataset whose local runtime owner has left scope."""
        runtime = DatasetRuntime(num_workers=1)
        written = success_paths(runtime.write_cache(TinySource(), tmp_path / "cache"))
        return runtime.cached_dataset(
            written,
            reader_config=ReaderConfig(seed=7, shuffle=False),
        )

    dataset = load_dataset()

    assert [sample.data for sample in dataset] == [b"zero", b"one", b"two"]


def test_multi_source_write_reports_failure_and_keeps_successes(tmp_path: Path) -> None:
    class LaterSource(TinySource):
        name = "later"

    class BadSource:
        name = "bad"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            yield CacheInput(cast("bytes", InvalidPayload()), {"label": "bad"})

    root = tmp_path / "caches"

    results = RUNTIME.write_cache([TinySource(), BadSource(), LaterSource()], root)

    assert results == [
        CacheWriteSuccess("tiny", root / "tiny"),
        CacheWriteError(
            "bad", f"write source 'bad' to '{root / 'tmp' / 'bad'}': data must be bytes-like"
        ),
        CacheWriteSuccess("later", root / "later"),
    ]
    assert not (root / "bad").exists()
    assert not (root / "tmp" / "bad").exists()


def test_empty_source_is_reported_as_write_error(tmp_path: Path) -> None:
    class EmptySource:
        name = "empty"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            return
            yield CacheInput(b"never", {"label": "empty"})

    root = tmp_path / "caches"

    results = RUNTIME.write_cache([TinySource(), EmptySource()], root)

    assert results == [
        CacheWriteSuccess("tiny", root / "tiny"),
        CacheWriteError(
            "empty",
            f"write source 'empty' to '{root / 'tmp' / 'empty'}': cache source yielded no samples",
        ),
    ]
    assert not (root / "empty").exists()
    assert not (root / "tmp" / "empty").exists()


def test_keyboard_interrupt_is_not_reported_as_write_error(tmp_path: Path) -> None:
    class InterruptedSource:
        name = "interrupted"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            raise KeyboardInterrupt
            yield CacheInput(b"never", {"label": "interrupted"})

    root = tmp_path / "caches"

    with pytest.raises(KeyboardInterrupt):
        RUNTIME.write_cache([TinySource(), InterruptedSource()], root)

    assert (root / "tiny").exists()
    assert not (root / "interrupted").exists()
    assert not (root / "tmp" / "interrupted").exists()


def test_dataset_restarts_deterministically(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    written = success_paths(RUNTIME.write_cache(TinySource(), base_cache_dir))

    first = [
        sample.sample_id
        for sample in RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=7))
    ]
    second = [
        sample.sample_id
        for sample in RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=7))
    ]

    assert first == second


def test_reader_config_controls_workers_prefetch_and_shuffle(tmp_path: Path) -> None:
    written = success_paths(RUNTIME.write_cache(TinySource(), tmp_path / "cache"))
    reader_config = ReaderConfig(seed=7, prefetch_size=2, shuffle=False)

    dataset = RUNTIME.cached_dataset(written, reader_config=reader_config)

    assert [sample.sample_id for sample in dataset] == [0, 1, 2]


def test_set_epoch_len_uses_cyclic_windows_for_non_shuffled_reads(
    five_dataset: dataset_rt_api.CachedDataset,
) -> None:
    five_dataset.set_epoch_len(2)

    first = sample_indices(five_dataset)
    second = sample_indices(five_dataset)
    third = sample_indices(five_dataset)

    assert len(five_dataset) == 2
    assert first == [0, 1]
    assert second == [2, 3]
    assert third == [4, 0]


def test_set_epoch_len_cycles_when_longer_than_active_rows(
    five_dataset: dataset_rt_api.CachedDataset,
) -> None:
    five_dataset.set_epoch_len(7)

    assert len(five_dataset) == 7
    assert sample_indices(five_dataset) == [0, 1, 2, 3, 4, 0, 1]


def test_set_epoch_len_keeps_shuffled_stream_continuous(
    five_cache_paths: list[Path],
) -> None:
    windowed = RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )
    whole = RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )

    windowed.set_epoch_len(2)
    whole.set_epoch_len(6)

    windowed_indices = [index for _ in range(3) for index in sample_indices(windowed)]

    assert windowed_indices == sample_indices(whole)


def test_set_epoch_len_keeps_weighted_shuffled_stream_continuous(
    five_cache_paths: list[Path],
) -> None:
    windowed = RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )
    whole = RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )

    windowed.update_metadata(windowed.get_metadata())
    whole.update_metadata(whole.get_metadata())
    windowed.set_epoch_len(2)
    whole.set_epoch_len(6)

    windowed_indices = [index for _ in range(3) for index in sample_indices(windowed)]

    assert windowed_indices == sample_indices(whole)


def test_update_metadata_resets_epoch_len_and_non_shuffled_cursor(
    five_dataset: dataset_rt_api.CachedDataset,
) -> None:
    five_dataset.set_epoch_len(2)

    assert sample_indices(five_dataset) == [0, 1]

    five_dataset.update_metadata(five_dataset.get_metadata())

    assert len(five_dataset) == 5
    assert sample_indices(five_dataset) == [0, 1, 2, 3, 4]


def test_update_metadata_resets_shuffled_stream(
    five_cache_paths: list[Path],
) -> None:
    advanced = RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )
    reference = RUNTIME.cached_dataset(
        five_cache_paths,
        reader_config=ReaderConfig(seed=11, shuffle=True),
    )

    advanced.set_epoch_len(2)
    reference.set_epoch_len(2)
    reference.update_metadata(reference.get_metadata())
    reference.set_epoch_len(2)
    sample_indices(advanced)
    advanced.update_metadata(advanced.get_metadata())
    advanced.set_epoch_len(2)

    assert sample_indices(advanced) == sample_indices(reference)


def test_set_epoch_len_rejects_non_positive_values(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    with pytest.raises(ValueError, match="epoch_len must be at least 1"):
        tiny_dataset.set_epoch_len(0)


def test_reader_config_validation() -> None:
    with pytest.raises(ValidationError):
        ReaderConfig(seed=7, prefetch_size=cast("int", 0))

    with pytest.raises(ValidationError):
        ReaderConfig.model_validate({"seed": 7, "num_workers": 4})


def test_torch_adapter_requires_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the lazy import boundary even when another test already imported Torch."""
    written = success_paths(RUNTIME.write_cache(TinySource(), tmp_path / "cache"))
    dataset = RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=7))

    def import_without_torch(name: str) -> Never:
        """Simulate the optional dependency being absent at the adapter boundary."""
        assert name == "torch.utils.data"
        raise ImportError("torch is intentionally hidden")

    monkeypatch.setattr(torch_integration, "import_module", import_without_torch)

    with pytest.raises(ImportError, match="requires PyTorch"):
        dataset.to_torch_iterable_dataset()


def test_torch_adapter_rejects_dataloader_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = success_paths(RUNTIME.write_cache(TinySource(), tmp_path / "cache"))
    dataset = RUNTIME.cached_dataset(written, reader_config=ReaderConfig(seed=7))

    class FakeTorchData:
        class IterableDataset:
            pass

        @staticmethod
        def get_worker_info() -> WorkerSentinel:
            """Simulate a worker without native or Torch resources."""
            return WorkerSentinel()

    def import_fake_torch(name: str) -> type[FakeTorchData]:
        """Supply the test adapter without importing the optional dependency."""
        assert name == "torch.utils.data"
        return FakeTorchData

    monkeypatch.setattr(torch_integration, "import_module", import_fake_torch)

    torch_dataset = dataset.to_torch_iterable_dataset()

    with pytest.raises(RuntimeError, match="Use Dataloader with num workers = 0"):
        iter(torch_dataset)


def test_torch_adapter_reports_configured_epoch_len(
    tiny_dataset: dataset_rt_api.CachedDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTorchData:
        class IterableDataset:
            pass

        @staticmethod
        def get_worker_info() -> None:
            return None

    def import_fake_torch(name: str) -> type[FakeTorchData]:
        """Supply the test adapter without importing the optional dependency."""
        assert name == "torch.utils.data"
        return FakeTorchData

    monkeypatch.setattr(torch_integration, "import_module", import_fake_torch)

    tiny_dataset.set_epoch_len(5)
    torch_dataset = tiny_dataset.to_torch_iterable_dataset()

    assert len(torch_dataset) == 5
    assert [sample.data for sample in torch_dataset] == [b"zero", b"one", b"two", b"zero", b"one"]


def test_get_metadata_returns_polars_table_with_metadata(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    metadata = tiny_dataset.get_metadata()

    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == [
        "cache_id",
        "sample_id",
        "index",
        "kept",
        "label",
        "score",
        "weight",
    ]
    assert metadata["weight"].to_list() == [1.0, 1.0, 1.0]


def test_update_metadata_limits_active_samples(tiny_dataset: dataset_rt_api.CachedDataset) -> None:
    tiny_dataset.update_metadata(tiny_dataset.get_metadata().head(2))

    samples = list(tiny_dataset)
    round_trip = tiny_dataset.get_metadata()

    assert len(tiny_dataset) == 2
    assert [sample.data for sample in samples] == [b"zero", b"one"]
    assert round_trip["sample_id"].to_list() == [0, 1]


def test_update_metadata_preserves_table_order_for_non_shuffled_reads(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    tiny_dataset.update_metadata(tiny_dataset.get_metadata().sort("sample_id", descending=True))

    assert [sample.data for sample in tiny_dataset] == [b"two", b"one", b"zero"]


def test_update_metadata_changes_new_iterators_not_existing_iterator(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    existing_iterator = iter(tiny_dataset)

    assert next(existing_iterator).data == b"zero"

    tiny_dataset.update_metadata(tiny_dataset.get_metadata().tail(1))

    assert len(tiny_dataset) == 1
    assert [sample.data for sample in existing_iterator] == [b"one", b"two"]
    assert [sample.data for sample in tiny_dataset] == [b"two"]


def test_update_metadata_limits_shuffled_sampling_to_active_rows(
    shuffled_tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    shuffled_tiny_dataset.update_metadata(
        shuffled_tiny_dataset.get_metadata().filter(pl.col("label") == "c")
    )

    assert len(shuffled_tiny_dataset) == 1
    assert [sample.data for sample in shuffled_tiny_dataset] == [b"two"]


def test_update_metadata_allows_duplicate_rows_in_active_table(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    metadata = tiny_dataset.get_metadata()
    duplicated = pl.concat([metadata.tail(1), metadata.head(1), metadata.tail(1), metadata])

    tiny_dataset.update_metadata(duplicated)

    round_trip = tiny_dataset.get_metadata()

    assert len(tiny_dataset) == 6
    assert [sample.data for sample in tiny_dataset] == [
        b"two",
        b"zero",
        b"two",
        b"zero",
        b"one",
        b"two",
    ]
    assert round_trip["sample_id"].to_list() == [2, 0, 2, 0, 1, 2]


def test_update_metadata_samples_duplicate_rows_when_shuffled(
    shuffled_tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    duplicated = pl.concat(
        [
            shuffled_tiny_dataset.get_metadata().tail(1),
            shuffled_tiny_dataset.get_metadata().tail(1),
        ]
    )

    shuffled_tiny_dataset.update_metadata(duplicated)

    assert len(shuffled_tiny_dataset) == 2
    assert [sample.data for sample in shuffled_tiny_dataset] == [b"two", b"two"]


def test_update_metadata_preserves_optional_columns(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    metadata = (
        tiny_dataset.get_metadata()
        .with_columns(
            pl.when(pl.col("label") == "c").then(8.0).otherwise(1.0).alias("ohem_weight"),
            pl.lit("fast-dev").alias("run_tag"),
        )
        .with_columns(pl.col("ohem_weight").alias("weight"))
    )
    tiny_dataset.update_metadata(metadata)

    round_trip = tiny_dataset.get_metadata()

    assert "ohem_weight" in round_trip.columns
    assert "run_tag" in round_trip.columns
    assert round_trip["weight"].to_list() == [1.0, 1.0, 8.0]
    assert round_trip["ohem_weight"].to_list() == [1.0, 1.0, 8.0]
    assert round_trip["run_tag"].to_list() == ["fast-dev", "fast-dev", "fast-dev"]


def test_update_metadata_rejects_empty_table(tiny_dataset: dataset_rt_api.CachedDataset) -> None:
    """Rejected metadata must preserve the population and explicit epoch length."""
    tiny_dataset.set_epoch_len(2)

    with pytest.raises(ValueError, match="metadata table must include at least one sample"):
        tiny_dataset.update_metadata(tiny_dataset.get_metadata().head(0))

    assert len(tiny_dataset) == 2
    assert tiny_dataset.get_metadata().height == 3
    assert [sample.data for sample in tiny_dataset] == [b"zero", b"one"]


def test_epoch_len_can_be_overridden_after_metadata_update(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    """An explicit window can override the length inferred from updated metadata."""
    tiny_dataset.update_metadata(tiny_dataset.get_metadata().tail(1))
    tiny_dataset.set_epoch_len(2)

    assert len(tiny_dataset) == 2
    assert [sample.data for sample in tiny_dataset] == [b"two", b"two"]


@pytest.mark.parametrize("column", ["cache_id", "sample_id", "weight"])
def test_update_metadata_rejects_missing_required_control_columns(
    tiny_dataset: dataset_rt_api.CachedDataset, column: str
) -> None:
    with pytest.raises(ValueError, match=f"samples metadata missing '{column}' column"):
        tiny_dataset.update_metadata(tiny_dataset.get_metadata().drop(column))


def test_update_metadata_rejects_unknown_identity(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    unknown = tiny_dataset.get_metadata().head(1).with_columns(pl.lit(999).alias("sample_id"))

    with pytest.raises(ValueError, match="unknown samples metadata row identity"):
        tiny_dataset.update_metadata(unknown)


def test_update_metadata_rejects_missing_stored_metadata_column(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    with pytest.raises(ValueError, match="samples metadata missing 'label' column"):
        tiny_dataset.update_metadata(tiny_dataset.get_metadata().drop("label"))


def test_update_metadata_rejects_changed_stored_metadata_column_type(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    changed = tiny_dataset.get_metadata().with_columns(pl.lit(1).alias("label"))

    with pytest.raises(ValueError, match="samples metadata column 'label' must have Arrow type"):
        tiny_dataset.update_metadata(changed)


@pytest.mark.parametrize("column", ["cache_id", "sample_id"])
def test_update_metadata_rejects_null_identity_cells(
    tiny_dataset: dataset_rt_api.CachedDataset, column: str
) -> None:
    invalid = tiny_dataset.get_metadata().with_columns(pl.lit(None, dtype=pl.Int64).alias(column))

    with pytest.raises(ValueError, match=f"samples metadata column '{column}' contains null"):
        tiny_dataset.update_metadata(invalid)


@pytest.mark.parametrize("column", ["cache_id", "sample_id"])
def test_update_metadata_rejects_negative_identity_cells(
    tiny_dataset: dataset_rt_api.CachedDataset, column: str
) -> None:
    invalid = tiny_dataset.get_metadata().with_columns(pl.lit(-1).alias(column))

    with pytest.raises(
        ValueError, match=f"samples metadata column '{column}' must be non-negative"
    ):
        tiny_dataset.update_metadata(invalid)


@pytest.mark.parametrize("column", ["cache_id", "sample_id"])
def test_update_metadata_rejects_non_integer_identity_columns(
    tiny_dataset: dataset_rt_api.CachedDataset, column: str
) -> None:
    invalid = tiny_dataset.get_metadata().with_columns(pl.lit("bad").alias(column))

    with pytest.raises(ValueError, match=f"samples metadata column '{column}' must be an integer"):
        tiny_dataset.update_metadata(invalid)


def test_metadata_weight_validation_happens_in_rust(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    metadata = tiny_dataset.get_metadata()

    with pytest.raises(ValueError, match="positive and finite"):
        tiny_dataset.update_metadata(metadata.with_columns(pl.lit(0.0).alias("weight")))


def test_update_metadata_rejects_null_weight_cells(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    invalid = tiny_dataset.get_metadata().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("weight")
    )

    with pytest.raises(ValueError, match="samples metadata column 'weight' contains null"):
        tiny_dataset.update_metadata(invalid)


def test_update_metadata_rejects_non_numeric_weight_columns(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    invalid = tiny_dataset.get_metadata().with_columns(pl.lit("heavy").alias("weight"))

    with pytest.raises(ValueError, match="samples metadata column 'weight' must be numeric"):
        tiny_dataset.update_metadata(invalid)


@pytest.mark.parametrize("identity_type", [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64])
@pytest.mark.parametrize(
    "weight_type", [pl.Float32, pl.Float64, pl.UInt32, pl.UInt64, pl.Int32, pl.Int64]
)
def test_metadata_numeric_types_preserve_selected_rows(
    tiny_dataset: dataset_rt_api.CachedDataset,
    identity_type: type[pl.DataType],
    weight_type: type[pl.DataType],
) -> None:
    """Supported Arrow numeric types resolve identities without changing row order."""
    selected = (
        tiny_dataset.get_metadata()
        .reverse()
        .with_columns(
            pl.col("sample_id").cast(identity_type),
            pl.col("cache_id").cast(pl.Int64),
            pl.lit(2).cast(weight_type).alias("weight"),
        )
    )
    tiny_dataset.update_metadata(selected)
    assert [sample.data for sample in tiny_dataset] == [b"two", b"one", b"zero"]
    assert tiny_dataset.get_metadata()["weight"].to_list() == [2, 2, 2]


@pytest.mark.parametrize("weight", [-1.0, 0.0, float("nan"), float("inf"), -float("inf")])
def test_rejected_weights_preserve_epoch_cursor(
    tiny_dataset: dataset_rt_api.CachedDataset, weight: float
) -> None:
    """Failed validation cannot replace active rows or reset the cyclic epoch cursor."""
    original = tiny_dataset.get_metadata()
    tiny_dataset.set_epoch_len(1)
    assert [sample.data for sample in tiny_dataset] == [b"zero"]
    invalid = original.reverse().with_columns(pl.lit(weight).alias("weight"))
    with pytest.raises(ValueError, match="positive and finite"):
        tiny_dataset.update_metadata(invalid)
    assert tiny_dataset.get_metadata().equals(original)
    assert len(tiny_dataset) == 1
    assert [sample.data for sample in tiny_dataset] == [b"one"]


@pytest.mark.parametrize("column", ["cache_id", "sample_id"])
def test_unknown_identity_preserves_active_metadata(
    tiny_dataset: dataset_rt_api.CachedDataset, column: str
) -> None:
    """Unknown unsigned identities fail atomically with the rejected identity attached."""
    original = tiny_dataset.get_metadata()
    invalid = original.with_columns(pl.lit(2**64 - 1, dtype=pl.UInt64).alias(column))
    with pytest.raises(ValueError, match=f"{column}={2**64 - 1}"):
        tiny_dataset.update_metadata(invalid)
    assert tiny_dataset.get_metadata().equals(original)
    assert [sample.data for sample in tiny_dataset] == [b"zero", b"one", b"two"]


def test_samples_metadata_and_set_samples_metadata_remain_compatible(
    tiny_dataset: dataset_rt_api.CachedDataset,
) -> None:
    tiny_dataset.set_samples_metadata(tiny_dataset.samples_metadata().tail(1))

    samples = list(tiny_dataset)

    assert len(tiny_dataset) == 1
    assert [sample.data for sample in samples] == [b"two"]


@pytest.mark.parametrize("legacy", [False, True])
def test_native_ipc_metadata_api_preserves_selection_and_rejects_corruption(
    tiny_cache_paths: list[Path], legacy: bool
) -> None:
    """Explicit IPC compatibility calls share Rust validation with the Arrow API."""
    from dataset_rt._dataset_rt import CachedDataset as NativeDataset
    from dataset_rt._dataset_rt import DatasetRuntime as NativeRuntime

    native = NativeDataset(NativeRuntime(1), [str(p) for p in tiny_cache_paths], 7, 2, False, False)
    encoded = native.samples_metadata_ipc() if legacy else native.metadata_ipc()
    original = pl.read_ipc(encoded)
    selected = original.reverse().head(2)
    buffer = selected.write_ipc(None)
    assert buffer is not None
    if legacy:
        native.set_samples_metadata_ipc(buffer.getvalue())
    else:
        native.update_metadata_ipc(buffer.getvalue())
    assert [row[0] for row in native] == [b"two", b"one"]
    with pytest.raises(RuntimeError):
        native.update_metadata_ipc(b"invalid Arrow file")
    assert pl.read_ipc(native.metadata_ipc()).equals(selected)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize(
    ("item", "diagnostic"),
    [
        (b"not a tuple", "CacheInput tuples"),
        ((b"payload",), "contain data and metadata"),
        ((b"payload", []), "metadata must be a dict"),
        ((b"payload", {1: "value"}), "metadata keys must be strings"),
        ((b"payload", {"value": []}), "metadata values must be"),
    ],
)
def test_native_writer_rejects_malformed_python_records(
    tmp_path: Path, item: MalformedCacheInput, diagnostic: str, as_list: bool
) -> None:
    """Rust validates untyped source records in both serial and source-list pipelines."""
    from dataset_rt._dataset_rt import DatasetRuntime as NativeRuntime
    from dataset_rt._dataset_rt import write_cache

    class UntypedSource:
        """Supply deliberately malformed boundary data without using production inputs."""

        name = "malformed"

        def __iter__(self) -> Iterator[MalformedCacheInput]:
            """Yield one external value whose contract the native writer must validate."""
            yield item

    source = cast("CacheSource", UntypedSource())
    outcomes = write_cache(
        NativeRuntime(1),
        [source] if as_list else source,
        str(tmp_path),
        WriterConfig(show_progress=False),
        False,
    )
    assert len(outcomes) == 1
    status, name, detail = outcomes[0]
    assert status == "error"
    assert name == "malformed"
    assert diagnostic in detail
    assert not (tmp_path / "malformed").exists()
    assert not (tmp_path / "tmp" / "malformed").exists()


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("phase", ["name", "iterator"])
@pytest.mark.parametrize("exception", [ValueError, KeyboardInterrupt, SystemExit])
def test_native_writer_preserves_source_setup_failures(
    tmp_path: Path,
    as_list: bool,
    phase: Literal["name", "iterator"],
    exception: type[ValueError] | type[KeyboardInterrupt] | type[SystemExit],
) -> None:
    """Source setup errors remain local; interrupts propagate after temporary-cache cleanup."""
    from dataset_rt._dataset_rt import DatasetRuntime as NativeRuntime
    from dataset_rt._dataset_rt import write_cache

    class FailingSource:
        """Fail before the first input, independently of generator-body exception handling."""

        @property
        def name(self) -> str:
            """Source discovery may fail before any cache destination is available."""
            if phase == "name":
                raise exception("name unavailable")
            return "setup"

        def __iter__(self) -> Iterator[CacheInput]:
            """Fail while obtaining the iterator rather than while advancing it."""
            raise exception("iterator unavailable")

    source = cast("CacheSource", FailingSource())
    runtime = NativeRuntime(1)
    config = WriterConfig(show_progress=False)
    sources = [source] if as_list else source
    if exception is ValueError:
        outcomes = write_cache(runtime, sources, str(tmp_path), config, False)
        assert len(outcomes) == 1
        status, _, detail = outcomes[0]
        assert status == "error"
        assert f"{phase} unavailable" in detail
    else:
        with pytest.raises(exception, match=f"{phase} unavailable"):
            write_cache(runtime, sources, str(tmp_path), config, False)
    assert not (tmp_path / "setup").exists()
    assert not (tmp_path / "tmp" / "setup").exists()

    class RetrySource(TinySource):
        """A failed setup cannot reserve the destination against a healthy retry."""

        name = "setup"

    retry = write_cache(runtime, RetrySource(), str(tmp_path), config, False)
    assert retry[0][0] == "success"
    dataset = RUNTIME.cached_dataset(
        [tmp_path / "setup"], reader_config=ReaderConfig(seed=1, shuffle=False, validate_cache=True)
    )
    assert [sample.data for sample in dataset] == [b"zero", b"one", b"two"]


def test_from_cache_sources_reuses_existing_cache(tmp_path: Path) -> None:
    class ExplodingSource:
        name = "tiny"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            raise AssertionError("existing caches should be reused before iteration")
            yield CacheInput(b"never", {"label": "x", "index": 9, "score": 0.0, "kept": False})

    reader_config = ReaderConfig(seed=42)
    writer_config = WriterConfig()

    first_result = RUNTIME.from_cache_sources(
        TinySource(),
        tmp_path / "cache",
        reader_config=reader_config,
        writer_config=writer_config,
    )
    second_result = RUNTIME.from_cache_sources(
        ExplodingSource(),
        tmp_path / "cache",
        reader_config=reader_config,
        writer_config=writer_config,
    )

    match first_result:
        case CacheSourcesDatasetSuccess(dataset=first):
            pass
        case result:
            raise AssertionError(result)
    match second_result:
        case CacheSourcesDatasetSuccess(dataset=second):
            pass
        case result:
            raise AssertionError(result)

    assert second.cache_paths == first.cache_paths
    assert len(second) == 3


def test_from_cache_sources_returns_dataset_with_write_errors(tmp_path: Path) -> None:
    class EmptySource:
        name = "empty"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            return
            yield CacheInput(b"never", {"label": "empty"})

    result = RUNTIME.from_cache_sources(
        [TinySource(), EmptySource()],
        tmp_path / "cache",
        reader_config=ReaderConfig(seed=42),
    )

    match result:
        case CacheSourcesDatasetSuccess(dataset=dataset, results=results):
            assert len(dataset) == 3
            assert results == [
                CacheWriteSuccess("tiny", tmp_path / "cache" / "tiny"),
                CacheWriteError(
                    "empty",
                    f"write source 'empty' to '{tmp_path / 'cache' / 'tmp' / 'empty'}': cache source yielded no samples",
                ),
            ]
        case result:
            raise AssertionError(result)


def test_from_cache_sources_returns_error_when_no_cache_was_written(tmp_path: Path) -> None:
    class EmptySource:
        name = "empty"

        def __iter__(self) -> Iterator[CacheInput]:
            """Stream cache inputs for this test scenario."""
            return
            yield CacheInput(b"never", {"label": "empty"})

    result = RUNTIME.from_cache_sources(
        EmptySource(),
        tmp_path / "cache",
        reader_config=ReaderConfig(seed=42),
    )

    match result:
        case CacheSourcesDatasetError(results=results, message=message):
            assert results == [CacheWriteError("empty", "cache source yielded no samples")]
            assert message == "no caches were written: empty: cache source yielded no samples"
        case result:
            raise AssertionError(result)


@pytest.mark.parametrize("shuffle", [False, True])
def test_overflowing_weight_total_preserves_dataset(
    tiny_cache_paths: list[Path],
    shuffle: bool,
) -> None:
    """Reject finite weights whose total overflows before replacing authoritative state."""
    config = ReaderConfig(seed=7, shuffle=shuffle)
    tiny_dataset = RUNTIME.cached_dataset(tiny_cache_paths, reader_config=config)
    reference = RUNTIME.cached_dataset(tiny_cache_paths, reader_config=config)
    before = tiny_dataset.samples_metadata()
    with pytest.raises(ValueError, match="weight total must be finite"):
        tiny_dataset.set_samples_metadata(before.with_columns(pl.lit(1e308).alias("weight")))
    assert tiny_dataset.samples_metadata().equals(before)
    assert [sample.sample_id for sample in tiny_dataset] == [
        sample.sample_id for sample in reference
    ]
