from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ConfigDict

from dataset_rt import (
    CachedDataset,
    CacheInput,
    CacheWriteSuccess,
    DatasetRuntime,
    NoCompression,
    ReaderConfig,
    WriterConfig,
    Zstd,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from dataset_rt.records import Metadata

RUNTIME = DatasetRuntime(num_workers=4)


class ShardManifest(BaseModel):
    """Validate edited shard fields while preserving the remaining native manifest."""

    model_config = ConfigDict(extra="allow", strict=True)

    name: str
    sha256: str


class Manifest(BaseModel):
    """Type corruption-test edits without duplicating Rust's format validation."""

    model_config = ConfigDict(extra="allow", strict=True)

    sample_count: int
    metadata_sha256: str
    index_sha256: str
    shards: list[ShardManifest]


class IntegritySource:
    name = "integrity"

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream cache inputs for this test scenario."""
        yield CacheInput(b"alpha", {"split": "train", "index": 0})
        yield CacheInput(b"beta", {"split": "train", "index": 1})


def write_integrity_cache(tmp_path: Path) -> Path:
    results = RUNTIME.write_cache(
        IntegritySource(),
        tmp_path / "cache",
        writer_config=WriterConfig(shard_compression=NoCompression(), show_progress=False),
    )
    match results[0]:
        case CacheWriteSuccess(path=path):
            return path
        case result:
            raise AssertionError(result)


def load_cache(cache_path: Path) -> CachedDataset:
    return RUNTIME.cached_dataset(
        [cache_path],
        reader_config=ReaderConfig(seed=1, shuffle=False, validate_cache=True),
    )


def read_manifest(cache_path: Path) -> Manifest:
    """Validate the editable fields once when loading a native test artifact."""
    return Manifest.model_validate_json((cache_path / "manifest.json").read_text())


def write_manifest(cache_path: Path, manifest: Manifest) -> None:
    """Retain native fields while publishing deliberate corruption-test edits."""
    (cache_path / "manifest.json").write_text(manifest.model_dump_json())


@pytest.fixture
def zstd_cache(tmp_path: Path) -> Path:
    """Publish independent sized frames for native corruption and shape validation."""
    result = RUNTIME.write_cache(
        IntegritySource(),
        tmp_path,
        writer_config=WriterConfig(shard_compression=Zstd(), show_progress=False),
    )[0]
    match result:
        case CacheWriteSuccess(path=path):
            return path
        case error:
            raise AssertionError(error)


@pytest.mark.parametrize(
    ("corruption", "diagnostic"),
    [("magic", "frame magic"), ("size", "content size exceeds"), ("checksum", "decompress Zstd")],
)
def test_zstd_frame_corruption_is_rejected_without_shard_hashing(
    zstd_cache: Path, corruption: str, diagnostic: str
) -> None:
    """Default reads validate frame sizes and checksums even with full hashing disabled."""
    index = (zstd_cache / "index.bin").read_bytes()
    first_length = int.from_bytes(index[16:24], "little")
    shard_path = zstd_cache / "shards" / "000000.bin"
    data = bytearray(shard_path.read_bytes())
    match corruption:
        case "magic":
            data[0] ^= 1
        case "size":
            # Small single-segment frames store their decoded size in one byte.
            assert data[4] & 0x20 and data[4] >> 6 == 0
            data[5] = 255
        case "checksum":
            data[first_length - 1] ^= 1
    shard_path.write_bytes(data)
    dataset = RUNTIME.cached_dataset(
        [zstd_cache], reader_config=ReaderConfig(seed=0, shuffle=False)
    )
    with pytest.raises(ValueError, match=diagnostic):
        next(iter(dataset))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cache(zstd_cache)


@pytest.mark.parametrize(
    "replacement",
    ['"level": 0', '"level": 23', '"level": 3, "acceleration": 1', '"level": true'],
)
def test_zstd_manifest_parameters_are_validated_in_rust(zstd_cache: Path, replacement: str) -> None:
    """Hand-edited manifests cannot bypass the codec's typed parameter contract."""
    path = zstd_cache / "manifest.json"
    original = path.read_text()
    assert '"level": 3' in original
    path.write_text(original.replace('"level": 3', replacement))
    with pytest.raises(RuntimeError):
        load_cache(zstd_cache)


def read_u64(bytes_: bytes) -> int:
    return int.from_bytes(bytes_, "little")


def replace_first_embedded_metadata(cache_path: Path, metadata: Metadata) -> None:
    """Replace same-length metadata while retaining a valid shard checksum."""
    manifest = read_manifest(cache_path)
    index = (cache_path / "index.bin").read_bytes()
    shard_id = read_u64(index[:8])
    offset = read_u64(index[8:16])
    byte_len = read_u64(index[16:24])
    shard = manifest.shards[shard_id]
    shard_path = cache_path / "shards" / shard.name
    shard_bytes = bytearray(shard_path.read_bytes())
    record = bytes(shard_bytes[offset : offset + byte_len])
    metadata_len = read_u64(record[:8])
    encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()

    assert len(encoded_metadata) == metadata_len
    shard_bytes[offset + 8 : offset + 8 + metadata_len] = encoded_metadata
    shard_path.write_bytes(shard_bytes)
    shard.sha256 = hashlib.sha256(shard_bytes).hexdigest()
    write_manifest(cache_path, manifest)


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    (cache_path / "manifest.json").unlink()

    with pytest.raises(ValueError, match="missing manifest"):
        load_cache(cache_path)


def test_corrupt_manifest_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    (cache_path / "manifest.json").write_text("{")

    with pytest.raises(RuntimeError, match="JSON error"):
        load_cache(cache_path)


def test_metadata_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    manifest = read_manifest(cache_path)
    manifest.metadata_sha256 = "0" * 64
    write_manifest(cache_path, manifest)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cache(cache_path)


def test_checksum_validation_is_optional_for_dataset_load(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    manifest = read_manifest(cache_path)
    manifest.metadata_sha256 = "0" * 64
    write_manifest(cache_path, manifest)

    dataset = RUNTIME.cached_dataset(
        [cache_path],
        reader_config=ReaderConfig(seed=1, shuffle=False),
    )

    assert len(dataset) == 2


def test_index_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    manifest = read_manifest(cache_path)
    manifest.index_sha256 = "0" * 64
    write_manifest(cache_path, manifest)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cache(cache_path)


@pytest.mark.parametrize("validate_cache", [False, True])
@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("short_row", "index length is not divisible"),
        ("missing_row", "index row count does not match"),
        ("empty", "index row count does not match"),
        ("unknown_shard", "index references missing shard"),
    ],
)
def test_mapped_index_shape_is_always_validated(
    tmp_path: Path, validate_cache: bool, corruption: str, error: str
) -> None:
    """Reject malformed mapped rows even when checksum hashing is disabled."""
    cache_path = write_integrity_cache(tmp_path)
    index_path = cache_path / "index.bin"
    original = index_path.read_bytes()
    match corruption:
        case "short_row":
            damaged = original[:-1]
        case "missing_row":
            damaged = original[:24]
        case "empty":
            damaged = b""
        case "unknown_shard":
            damaged = ((1 << 64) - 1).to_bytes(8, "little") + original[8:]
        case _:
            raise AssertionError(corruption)
    index_path.write_bytes(damaged)
    manifest = read_manifest(cache_path)
    manifest.index_sha256 = hashlib.sha256(damaged).hexdigest()
    write_manifest(cache_path, manifest)

    with pytest.raises(ValueError, match=error):
        RUNTIME.cached_dataset(
            [cache_path],
            reader_config=ReaderConfig(seed=1, shuffle=False, validate_cache=validate_cache),
        )


@pytest.mark.parametrize("validate_cache", [False, True])
def test_missing_mapped_index_is_rejected(tmp_path: Path, validate_cache: bool) -> None:
    """Report a missing index as an I/O error on both reader validation paths."""
    cache_path = write_integrity_cache(tmp_path)
    (cache_path / "index.bin").unlink()

    with pytest.raises(RuntimeError, match="I/O error"):
        RUNTIME.cached_dataset(
            [cache_path],
            reader_config=ReaderConfig(seed=1, shuffle=False, validate_cache=validate_cache),
        )


def test_shard_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    shard_path = cache_path / "shards" / "000000.bin"
    shard_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="shard length mismatch|checksum mismatch"):
        load_cache(cache_path)


def test_missing_shard_is_returned_as_runtime_error(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    (cache_path / "shards" / "000000.bin").unlink()

    with pytest.raises(RuntimeError, match="I/O error"):
        load_cache(cache_path)


def test_manifest_sample_count_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    manifest = read_manifest(cache_path)
    manifest.sample_count = 999
    write_manifest(cache_path, manifest)

    with pytest.raises(ValueError, match="row count does not match manifest"):
        load_cache(cache_path)


def test_embedded_metadata_mismatch_is_rejected_during_iteration(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    replace_first_embedded_metadata(cache_path, {"index": 9, "split": "train"})

    with pytest.raises(ValueError, match="embedded metadata does not match"):
        list(load_cache(cache_path))


@pytest.mark.parametrize(("sample_count", "string_bytes"), [(10, 1_000_000), (65_537, 0)])
def test_metadata_byte_batched_cache_round_trips(
    tmp_path: Path, sample_count: int, string_bytes: int
) -> None:
    """Multiple Arrow batches preserve metadata order and redundant-record validation."""
    import polars as pl
    import pyarrow.ipc as ipc

    class LargeMetadataSource:
        name = "byte-batched"

        def __iter__(self) -> Iterator[CacheInput]:
            """Cross the metadata byte budget with a bounded number of records."""
            for index in range(sample_count):
                yield CacheInput(bytes([index % 256]), {"text": "x" * string_bytes, "index": index})

    results = RUNTIME.write_cache(LargeMetadataSource(), tmp_path)
    assert isinstance(results[0], CacheWriteSuccess)
    path = results[0].path
    table = pl.read_ipc(path / "metadata.arrow", rechunk=False)
    with ipc.open_file(path / "metadata.arrow") as reader:
        assert reader.num_record_batches > 1
    assert table["index"].equals(pl.Series("index", range(sample_count)))
    dataset = load_cache(path)
    assert len(dataset) == sample_count
    for index, sample in enumerate(dataset):
        assert sample.data == bytes([index % 256])


@pytest.mark.parametrize("invalid_length", [2**64 - 1, 2**32])
def test_invalid_record_length_returns_value_error(tmp_path: Path, invalid_length: int) -> None:
    """Reject impossible index ranges before allocating payload memory."""
    path = write_integrity_cache(tmp_path)
    index_path = path / "index.bin"
    index = bytearray(index_path.read_bytes())
    index[16:24] = invalid_length.to_bytes(8, "little")
    index_path.write_bytes(index)
    dataset = RUNTIME.cached_dataset([path], reader_config=ReaderConfig(seed=1, shuffle=False))
    with pytest.raises(ValueError, match="record.*(overflow|beyond)"):
        next(iter(dataset))


@pytest.mark.parametrize("invalid_schema", ["missing", "type", "null", "name"])
def test_invalid_metadata_schema_returns_error(tmp_path: Path, invalid_schema: str) -> None:
    """Malformed Arrow shape produces an actionable error without worker panic."""
    import polars as pl

    path = write_integrity_cache(tmp_path)
    metadata_path = path / "metadata.arrow"
    match invalid_schema:
        case "missing":
            table = pl.DataFrame({"index": [0, 1]})
        case "type":
            table = pl.DataFrame({"index": [0.0, 1.0], "split": ["train", "train"]})
        case "null":
            table = pl.DataFrame({"index": [0, None], "split": ["train", "train"]})
        case "name":
            table = pl.DataFrame({"other": [0, 1], "split": ["train", "train"]})
        case _:
            raise AssertionError(invalid_schema)
    table.write_ipc(metadata_path)
    manifest = read_manifest(path)
    manifest.metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="metadata"):
        load_cache(path)


@pytest.fixture
def diagnostic_cache(tmp_path: Path) -> Path:
    """Publish a tiny isolated cache for diagnostic failure checks."""
    return write_integrity_cache(tmp_path)


@pytest.mark.parametrize(
    "filename", ["manifest.json", "index.bin", "metadata.arrow", "shards/000000.bin"]
)
def test_missing_file_error_identifies_path(diagnostic_cache: Path, filename: str) -> None:
    """Every missing required file identifies its path and the failed operation."""
    missing = diagnostic_cache / filename
    missing.unlink()
    error_type = ValueError if filename == "manifest.json" else RuntimeError
    with pytest.raises(error_type) as caught:
        RUNTIME.cached_dataset([diagnostic_cache], reader_config=ReaderConfig(seed=1))
    message = str(caught.value)
    assert str(missing) in message
    assert "load cache" in message
    if filename == "manifest.json":
        assert "Each cache directory must directly contain manifest.json" in message


@pytest.mark.parametrize("bad_path", ["file", "directory"])
def test_manifest_path_type_error_retains_cause(diagnostic_cache: Path, bad_path: str) -> None:
    """An invalid path type must not be misreported as a missing publication marker."""
    manifest_path = diagnostic_cache / "manifest.json"
    if bad_path == "directory":
        manifest_path.unlink()
        manifest_path.mkdir()
        supplied_path = diagnostic_cache
    else:
        supplied_path = manifest_path
    with pytest.raises(RuntimeError) as caught:
        RUNTIME.cached_dataset([supplied_path], reader_config=ReaderConfig(seed=1))
    assert str(manifest_path) in str(caught.value)
    assert "missing manifest" not in str(caught.value)


def test_manifest_parse_error_identifies_file(diagnostic_cache: Path) -> None:
    """Parser details remain attached to the exact broken manifest."""
    manifest_path = diagnostic_cache / "manifest.json"
    manifest_path.write_text("{")
    with pytest.raises(RuntimeError) as caught:
        load_cache(diagnostic_cache)
    assert str(manifest_path) in str(caught.value)
    assert "parse manifest" in str(caught.value)
    assert "JSON error" in str(caught.value)


def test_future_format_error_explains_supported_versions(diagnostic_cache: Path) -> None:
    """Unsupported formats explain which reader versions are understood and the next step."""
    manifest_path = diagnostic_cache / "manifest.json"
    original = manifest_path.read_text()
    manifest_path.write_text(original.replace('"format_version": 4', '"format_version": 99'))
    with pytest.raises(RuntimeError) as caught:
        load_cache(diagnostic_cache)
    message = str(caught.value)
    assert str(manifest_path) in message
    assert "unsupported format version 99" in message
    assert "supported versions: 2, 3, 4" in message
    assert "Upgrade DatasetRT" in message


def test_duplicate_identity_error_identifies_both_paths(
    diagnostic_cache: Path, tmp_path: Path
) -> None:
    """Copied caches retain identities, so conflicts must name both directories."""
    import shutil

    copied = tmp_path / "copied"
    shutil.copytree(diagnostic_cache, copied)
    with pytest.raises(ValueError) as caught:
        RUNTIME.cached_dataset([diagnostic_cache, copied], reader_config=ReaderConfig(seed=1))
    message = str(caught.value)
    assert "duplicate cache_id" in message
    assert str(diagnostic_cache) in message
    assert str(copied) in message


def test_schema_error_identifies_differing_column(diagnostic_cache: Path, tmp_path: Path) -> None:
    """Schema conflicts report the offending column without dumping every column."""

    class OtherSource:
        """Supply a cache with a distinct primitive metadata type."""

        name = "other"

        def __iter__(self) -> Iterator[CacheInput]:
            """Keep the name stable while changing its metadata type."""
            yield CacheInput(b"other", {"index": "text", "split": "train"})

    result = RUNTIME.write_cache(
        OtherSource(), tmp_path, writer_config=WriterConfig(show_progress=False)
    )[0]
    assert isinstance(result, CacheWriteSuccess)
    with pytest.raises(ValueError) as caught:
        RUNTIME.cached_dataset([diagnostic_cache, result.path], reader_config=ReaderConfig(seed=1))
    message = str(caught.value)
    assert str(diagnostic_cache) in message
    assert str(result.path) in message
    assert "index" in message
    assert "Int" in message
    assert "String" in message


def test_record_read_error_identifies_sample_and_address(diagnostic_cache: Path) -> None:
    """Random-read failures retain identity and the exact shard byte range."""
    # Only one read may fail at a time so the reported identity is deterministic.
    dataset = RUNTIME.cached_dataset(
        [diagnostic_cache], reader_config=ReaderConfig(seed=1, shuffle=False, prefetch_size=1)
    )
    shard_path = diagnostic_cache / "shards" / "000000.bin"
    shard_path.write_bytes(b"")
    with pytest.raises(RuntimeError) as caught:
        next(iter(dataset))
    message = str(caught.value)
    assert "cache_id=" in message
    assert "sample_id=0" in message
    assert str(shard_path) in message
    assert "offset 0" in message


def test_checksum_error_reports_expected_and_actual(diagnostic_cache: Path) -> None:
    """Checksum failures identify the file, both digests, and a valid recovery action."""
    manifest = read_manifest(diagnostic_cache)
    actual = manifest.metadata_sha256
    manifest.metadata_sha256 = "0" * 64
    write_manifest(diagnostic_cache, manifest)
    with pytest.raises(ValueError) as caught:
        load_cache(diagnostic_cache)
    message = str(caught.value)
    assert str(diagnostic_cache / "metadata.arrow") in message
    assert f"expected {'0' * 64}" in message
    assert f"actual {actual}" in message
    assert "rebuild the cache" in message


def test_index_count_error_reports_expected_and_actual(diagnostic_cache: Path) -> None:
    """Manifest/index disagreement reports counts with the affected cache path."""
    index_path = diagnostic_cache / "index.bin"
    index_path.write_bytes(b"")
    with pytest.raises(ValueError) as caught:
        RUNTIME.cached_dataset([diagnostic_cache], reader_config=ReaderConfig(seed=1))
    message = str(caught.value)
    assert str(diagnostic_cache) in message
    assert "expected 2, actual 0" in message
