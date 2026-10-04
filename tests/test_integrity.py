from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ConfigDict

from dataset_rt import CachedDataset, CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig

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
    results = RUNTIME.write_cache(IntegritySource(), tmp_path / "cache")
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
