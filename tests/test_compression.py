"""Record compression variants preserve legacy caches and independent random access."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import BaseModel, ValidationError

from dataset_rt import (
    CacheInput,
    CacheWriteSuccess,
    DatasetRuntime,
    Lz4Fast,
    Lz4HC,
    NoCompression,
    Ok,
    ReaderConfig,
    RecordCompression,
    ShardCompression,
    WriterConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass
class CompressionSource:
    """Stream repeated metadata and small serialized-like payloads for bounded tests."""

    name: str = "compression"
    count: int = 128

    def __iter__(self) -> Iterator[CacheInput]:
        """Keep identities in the payload while exercising metadata-heavy records."""
        for index in range(self.count):
            yield CacheInput(
                index.to_bytes(4, "little") + b"token-array-data" * 32,
                {"image_path": "/repeated/dataset/archive/path/" * 32},
            )


class LegacyPolicy(BaseModel):
    """Inspect legacy wire settings without constructing deprecated user configuration."""

    algo: Literal["none", "lz4"]
    ratio: float


class Shard(BaseModel):
    """Validate compression manifests at the test boundary."""

    name: str
    byte_len: int
    uncompressed_byte_len: int
    compression: RecordCompression | LegacyPolicy


class Manifest(BaseModel):
    """Retain the fields needed to verify record scope and byte accounting."""

    format_version: int
    cache_id: int
    sample_count: int
    shards: list[Shard]


@pytest.fixture
def runtime() -> DatasetRuntime:
    """Own a small native pool independently of other test modules."""
    return DatasetRuntime(num_workers=2)


@pytest.mark.parametrize(
    "policy",
    [
        NoCompression(),
        Lz4Fast(),
        Lz4Fast(acceleration=8),
        Lz4HC(level=1),
        Lz4HC(level=9),
        Lz4HC(level=12),
    ],
)
@pytest.mark.parametrize("processes", [0, 2])
def test_whole_record_round_trip(
    tmp_path: Path, runtime: DatasetRuntime, policy: RecordCompression, processes: int
) -> None:
    """New variants persist their settings and preserve ordered records across source processes."""
    source = CompressionSource()
    config = WriterConfig(
        shard_compression=policy, num_processes=processes, max_shard_bytes=8192, show_progress=False
    )
    assert WriterConfig.model_validate_json(config.model_dump_json()) == config
    assert pickle.loads(pickle.dumps(config)) == config
    result = runtime.write_cache(source, tmp_path, writer_config=config)[0]
    assert isinstance(result, CacheWriteSuccess)
    manifest = Manifest.model_validate_json((result.path / "manifest.json").read_text())
    assert manifest.format_version == 4
    assert manifest.sample_count == source.count
    assert all(shard.compression == policy for shard in manifest.shards)
    dataset = runtime.cached_dataset(
        [result.path], reader_config=ReaderConfig(seed=7, shuffle=False, validate_cache=True)
    )
    for expected, actual in zip(source, dataset, strict=True):
        assert actual.data == expected.data
        assert actual.metadata == expected.metadata
    for index in [127, 0, 63, 1]:
        assert dataset.get_item(manifest.cache_id, index).data.startswith(
            index.to_bytes(4, "little")
        )
    if policy.algo == "lz4":
        assert sum(shard.byte_len for shard in manifest.shards) < sum(
            shard.uncompressed_byte_len for shard in manifest.shards
        )


def test_legacy_warning_and_mixed_caches(tmp_path: Path, runtime: DatasetRuntime) -> None:
    """Legacy constructors warn, retain v3 bytes/settings, and coexist with v4 readers."""
    with pytest.warns(DeprecationWarning, match="ratio never controlled"):
        policy = ShardCompression(algo="lz4", ratio=0.5)
    legacy = runtime.write_cache(
        CompressionSource(name="legacy"),
        tmp_path,
        writer_config=WriterConfig(shard_compression=policy, show_progress=False),
    )[0]
    modern = runtime.write_cache(
        CompressionSource(name="modern"),
        tmp_path,
        writer_config=WriterConfig(shard_compression=Lz4HC(), show_progress=False),
    )[0]
    assert isinstance(legacy, CacheWriteSuccess)
    assert isinstance(modern, CacheWriteSuccess)
    manifest = Manifest.model_validate_json((legacy.path / "manifest.json").read_text())
    assert manifest.format_version == 3
    assert manifest.shards[0].compression == LegacyPolicy(algo="lz4", ratio=0.5)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        dataset = runtime.cached_dataset(
            [legacy.path, modern.path],
            reader_config=ReaderConfig(seed=7, shuffle=False, validate_cache=True),
        )
        assert len(list(dataset)) == 256
    assert not captured
    report = dataset.update_manifests(version=3)
    match report:
        case Ok():
            pass
        case error:
            raise AssertionError(error)
    assert (
        Manifest.model_validate_json((modern.path / "manifest.json").read_text()).format_version
        == 4
    )


@pytest.mark.parametrize(
    "wire",
    [
        '{"algo":"lz4","mode":"hc","level":0}',
        '{"algo":"lz4","mode":"hc","level":13}',
        '{"algo":"lz4","mode":"hc","level":true}',
        '{"algo":"lz4","mode":"fast","acceleration":0}',
        '{"algo":"lz4","mode":"fast","acceleration":65538}',
        '{"algo":"lz4","mode":"fast","level":9}',
        '{"algo":"none","mode":"none","level":9}',
        '{"algo":"none","mode":"hc","level":9}',
    ],
)
def test_invalid_variants_cannot_fall_back_to_legacy(wire: str) -> None:
    """Reject irrelevant parameters and invalid modes even through serialized WriterConfig."""
    with pytest.raises(ValidationError):
        WriterConfig.model_validate_json('{"shard_compression":' + wire + "}")


def test_implicit_default_uses_fast_lz4_without_warnings() -> None:
    """Omitted compression selects the balanced whole-record policy without warnings."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = WriterConfig()
    assert not captured
    assert config.shard_compression == Lz4Fast(acceleration=1)


@pytest.mark.parametrize("replacement", ['"level": 13', '"level": 9, "acceleration": 1'])
def test_invalid_manifest_settings_are_native_errors(
    tmp_path: Path, runtime: DatasetRuntime, replacement: str
) -> None:
    """Hand-edited codec settings are rejected by Rust, independently of Pydantic validation."""
    result = runtime.write_cache(
        CompressionSource(),
        tmp_path,
        writer_config=WriterConfig(shard_compression=Lz4HC(), show_progress=False),
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    path = result.path / "manifest.json"
    original = path.read_text()
    assert '"level": 9' in original
    path.write_text(original.replace('"level": 9', replacement))
    with pytest.raises(RuntimeError):
        runtime.cached_dataset([result.path], reader_config=ReaderConfig(seed=7))


def test_record_scope_cannot_be_relabelled_as_v3(tmp_path: Path, runtime: DatasetRuntime) -> None:
    """A manifest-only edit cannot convert whole-record caches into payload-only caches."""
    result = runtime.write_cache(
        CompressionSource(),
        tmp_path,
        writer_config=WriterConfig(shard_compression=Lz4Fast(), show_progress=False),
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    path = result.path / "manifest.json"
    original = path.read_text()
    assert '"format_version": 4' in original
    path.write_text(original.replace('"format_version": 4', '"format_version": 3'))
    with pytest.raises(RuntimeError, match="compression scope disagrees"):
        runtime.cached_dataset([result.path], reader_config=ReaderConfig(seed=7))


def test_corrupt_record_size_prefix_is_rejected(tmp_path: Path, runtime: DatasetRuntime) -> None:
    """Reject a corrupt decoded size before allocating a whole-record buffer."""
    result = runtime.write_cache(
        CompressionSource(),
        tmp_path,
        writer_config=WriterConfig(shard_compression=Lz4HC(), show_progress=False),
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    with (result.path / "shards" / "000000.bin").open("r+b") as shard:
        shard.write((2**32 - 1).to_bytes(4, "little"))
    dataset = runtime.cached_dataset(
        [result.path], reader_config=ReaderConfig(seed=7, shuffle=False)
    )
    with pytest.raises(ValueError, match="size prefix exceeds"):
        next(iter(dataset))


def test_default_compression_shrinks_zeroes(tmp_path: Path, runtime: DatasetRuntime) -> None:
    """Default writing compresses zero-filled payloads and preserves their complete record."""

    class ZeroSource:
        name = "zeroes"

        def __iter__(self) -> Iterator[CacheInput]:
            """Expose one compressible record without selecting an explicit policy."""
            yield CacheInput(bytes(8192), {"image_path": "/test/zeroes"})

    result = runtime.write_cache(
        ZeroSource(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    manifest = Manifest.model_validate_json((result.path / "manifest.json").read_text())
    assert manifest.format_version == 4
    assert manifest.shards[0].compression == Lz4Fast(acceleration=1)
    assert manifest.shards[0].byte_len < 8192 // 10
    dataset = runtime.cached_dataset(
        [result.path], reader_config=ReaderConfig(seed=7, shuffle=False, validate_cache=True)
    )
    sample = next(iter(dataset))
    assert sample.data == bytes(8192)
    assert sample.metadata == {"image_path": "/test/zeroes"}
