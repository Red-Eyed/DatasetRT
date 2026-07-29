from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from pydantic import ValidationError

from dataset_rt import (
    CachedDataset,
    CacheInput,
    ReaderConfig,
    ShardCompression,
    WriterConfig,
    write_cache,
)


class TinySource:
    name = "tiny"

    def __iter__(self):
        yield CacheInput(b"zero", {"label": "a", "index": 0, "score": 1.5, "kept": True})
        yield CacheInput(bytearray(b"one"), {"label": "b", "index": 1, "score": 2.5, "kept": False})
        yield CacheInput(memoryview(b"two"), {"label": "c", "index": 2, "score": 3.5, "kept": True})


def test_write_and_read_cache(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    written = write_cache(
        TinySource(),
        base_cache_dir,
        writer_config=WriterConfig(
            prefetch_size=2,
            num_threads=2,
            max_shard_bytes=4,
            shard_compression=ShardCompression(algo="none", ratio=1.0),
            show_progress=False,
        ),
    )
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=42))

    assert len(written) == 1
    assert written[0].parent == base_cache_dir
    assert written[0].name.startswith("tiny_")
    assert len(dataset) == 3
    samples = list(dataset)

    assert len(samples) == 3
    assert {sample.data for sample in samples}.issubset({b"zero", b"one", b"two"})
    assert {sample.cache_id for sample in samples} == {0}
    assert all(sample.metadata["label"] in {"a", "b", "c"} for sample in samples)


def test_writer_manifest_records_shard_compression(tmp_path: Path) -> None:
    written = write_cache(
        TinySource(),
        tmp_path / "cache",
        writer_config=WriterConfig(
            shard_compression=ShardCompression(algo="none", ratio=1.0),
        ),
    )

    manifest = json.loads((written[0] / "manifest.json").read_text())

    assert manifest["shards"][0]["compression"] == {"algo": "none", "ratio": 1.0}
    assert manifest["shards"][0]["byte_len"] > manifest["shards"][0]["uncompressed_byte_len"]


def test_shard_records_embed_metadata_for_debugging(tmp_path: Path) -> None:
    written = write_cache(
        TinySource(),
        tmp_path / "cache",
        writer_config=WriterConfig(show_progress=False),
    )
    cache_path = written[0]
    manifest = json.loads((cache_path / "manifest.json").read_text())
    index = (cache_path / "index.bin").read_bytes()
    shard = (cache_path / "shards" / manifest["shards"][0]["name"]).read_bytes()
    first_record = shard[: int.from_bytes(index[16:24], "little")]
    metadata_len = int.from_bytes(first_record[:8], "little")
    embedded_metadata = json.loads(first_record[8 : 8 + metadata_len])

    assert manifest["format_version"] == 2
    assert embedded_metadata == {"index": 0, "kept": True, "label": "a", "score": 1.5}


def test_writer_lz4_compresses_payloads_and_reads_original_bytes(tmp_path: Path) -> None:
    class CompressibleSource:
        name = "compressible"

        def __iter__(self):
            yield CacheInput(b"a" * 10_000, {"label": "first"})
            yield CacheInput(b"b" * 10_000, {"label": "second"})

    written = write_cache(
        CompressibleSource(),
        tmp_path / "cache",
        writer_config=WriterConfig(
            shard_compression=ShardCompression(algo="lz4", ratio=2.0),
            show_progress=False,
        ),
    )
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=42, shuffle=False))
    samples = list(dataset)
    manifest = json.loads((written[0] / "manifest.json").read_text())
    shard = manifest["shards"][0]

    assert [sample.data for sample in samples] == [b"a" * 10_000, b"b" * 10_000]
    assert shard["compression"] == {"algo": "lz4", "ratio": 2.0}
    assert shard["byte_len"] < shard["uncompressed_byte_len"]


def test_writer_progress_is_optional(tmp_path: Path) -> None:
    assert WriterConfig().show_progress is True

    written = write_cache(
        TinySource(),
        tmp_path / "cache",
        writer_config=WriterConfig(show_progress=False),
    )

    assert len(written) == 1


def test_write_multiple_sources_returns_cache_paths(tmp_path: Path) -> None:
    class OtherSource(TinySource):
        name = "other"

    root = tmp_path / "caches"

    written = write_cache([TinySource(), OtherSource()], root)
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=42))

    assert [path.parent for path in written] == [root, root]
    assert written[0].name.startswith("tiny_")
    assert written[1].name.startswith("other_")
    assert len(dataset) == 6


def test_writer_config_validation_happens_in_rust(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        WriterConfig(prefetch_size=cast(int, 0))

    with pytest.raises(ValidationError):
        WriterConfig(num_threads=cast(int, 0))

    with pytest.raises(ValueError, match="ratio must be 1.0"):
        write_cache(
            TinySource(),
            tmp_path / "compression",
            writer_config=WriterConfig(
                shard_compression=ShardCompression(algo="none", ratio=2.0),
            ),
        )

    class UnsupportedCompression:
        algo = "zstd"
        ratio = 2.0

    class UnsupportedWriterConfig:
        prefetch_size = 64
        num_threads = 4
        max_shard_bytes = 64 * 1024 * 1024
        shard_compression = UnsupportedCompression()
        show_progress = False

    with pytest.raises(ValueError, match="unsupported shard compression"):
        write_cache(
            TinySource(),
            tmp_path / "zstd",
            writer_config=cast(WriterConfig, UnsupportedWriterConfig()),
        )


def test_multi_source_write_cleans_up_on_failure(tmp_path: Path) -> None:
    class BadSource:
        name = "bad"

        def __iter__(self):
            yield CacheInput(cast(bytes, object()), {"label": "bad"})

    root = tmp_path / "caches"

    with pytest.raises(ValueError, match="bytes-like"):
        write_cache([TinySource(), BadSource()], root)

    assert not list(root.glob("*"))


def test_dataset_restarts_deterministically(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    written = write_cache(TinySource(), base_cache_dir)

    first = [
        sample.sample_id for sample in CachedDataset(written, reader_config=ReaderConfig(seed=7))
    ]
    second = [
        sample.sample_id for sample in CachedDataset(written, reader_config=ReaderConfig(seed=7))
    ]

    assert first == second


def test_reader_config_controls_workers_prefetch_and_shuffle(tmp_path: Path) -> None:
    written = write_cache(TinySource(), tmp_path / "cache")
    reader_config = ReaderConfig(seed=7, prefetch_size=2, num_workers=2, shuffle=False)

    dataset = CachedDataset(written, reader_config=reader_config)

    assert [sample.sample_id for sample in dataset] == [0, 1, 2]


def test_reader_config_validation() -> None:
    with pytest.raises(ValidationError):
        ReaderConfig(seed=7, prefetch_size=cast(int, 0))

    with pytest.raises(ValidationError):
        ReaderConfig(seed=7, num_workers=cast(int, 0))


def test_torch_adapter_requires_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written = write_cache(TinySource(), tmp_path / "cache")
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=7))
    original_import = builtins.__import__

    def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch is intentionally hidden")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    with pytest.raises(ImportError, match="requires PyTorch"):
        dataset.to_torch_iterable_dataset()


def test_weights_are_polars_table_with_metadata(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    written = write_cache(TinySource(), base_cache_dir)
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=7))

    weights = dataset.weight_table()

    assert isinstance(weights, pl.DataFrame)
    assert weights.columns == ["cache_id", "sample_id", "index", "kept", "label", "score", "weight"]
    assert weights["weight"].to_list() == [1.0, 1.0, 1.0]


def test_set_weight_table_accepts_reordered_polars_table(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    written = write_cache(TinySource(), base_cache_dir)
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=7))

    weights = dataset.weight_table()
    updated = weights.with_columns(
        pl.when(pl.col("label") == "b").then(10.0).otherwise(1.0).alias("weight")
    ).sort("sample_id", descending=True)
    dataset.set_weight_table(updated)

    round_trip = dataset.weight_table().sort("sample_id")

    assert round_trip["weight"].to_list() == [1.0, 10.0, 1.0]


def test_weight_validation_happens_in_rust(tmp_path: Path) -> None:
    base_cache_dir = tmp_path / "cache"

    written = write_cache(TinySource(), base_cache_dir)
    dataset = CachedDataset(written, reader_config=ReaderConfig(seed=7))
    weights = dataset.weight_table()

    with pytest.raises(ValueError, match="expected 3 weight rows"):
        dataset.set_weight_table(weights.head(1))

    with pytest.raises(ValueError, match="positive and finite"):
        dataset.set_weight_table(weights.with_columns(pl.lit(0.0).alias("weight")))


def test_from_cache_sources_reuses_existing_cache(tmp_path: Path) -> None:
    class ExplodingSource:
        name = "tiny"

        def __iter__(self):
            raise AssertionError("existing caches should be reused before iteration")
            yield CacheInput(b"never", {"label": "x", "index": 9, "score": 0.0, "kept": False})

    reader_config = ReaderConfig(seed=42)
    writer_config = WriterConfig()

    first = CachedDataset.from_cache_sources(
        TinySource(),
        tmp_path / "cache",
        reader_config=reader_config,
        writer_config=writer_config,
    )
    second = CachedDataset.from_cache_sources(
        ExplodingSource(),
        tmp_path / "cache",
        reader_config=reader_config,
        writer_config=writer_config,
    )

    assert second.cache_paths == first.cache_paths
    assert len(second) == 3
