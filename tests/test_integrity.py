from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dataset_rt import CachedDataset, CacheInput, ReaderConfig, write_cache


class IntegritySource:
    name = "integrity"

    def __iter__(self):
        yield CacheInput(b"alpha", {"split": "train", "index": 0})
        yield CacheInput(b"beta", {"split": "train", "index": 1})


def write_integrity_cache(tmp_path: Path) -> Path:
    paths = write_cache(IntegritySource(), tmp_path / "cache")
    return paths[0]


def load_cache(cache_path: Path) -> CachedDataset:
    return CachedDataset([cache_path], reader_config=ReaderConfig(seed=1, shuffle=False))


def read_manifest(cache_path: Path) -> dict[str, object]:
    return json.loads((cache_path / "manifest.json").read_text())


def write_manifest(cache_path: Path, manifest: dict[str, object]) -> None:
    (cache_path / "manifest.json").write_text(json.dumps(manifest))


def read_u64(bytes_: bytes) -> int:
    return int.from_bytes(bytes_, "little")


def replace_first_embedded_metadata(cache_path: Path, metadata: dict[str, object]) -> None:
    manifest = read_manifest(cache_path)
    index = (cache_path / "index.bin").read_bytes()
    shard_id = read_u64(index[:8])
    offset = read_u64(index[8:16])
    byte_len = read_u64(index[16:24])
    shards = manifest["shards"]
    assert isinstance(shards, list)
    shard = shards[shard_id]
    assert isinstance(shard, dict)
    shard_path = cache_path / "shards" / str(shard["name"])
    shard_bytes = bytearray(shard_path.read_bytes())
    record = bytes(shard_bytes[offset : offset + byte_len])
    metadata_len = read_u64(record[:8])
    encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()

    assert len(encoded_metadata) == metadata_len
    shard_bytes[offset + 8 : offset + 8 + metadata_len] = encoded_metadata
    shard_path.write_bytes(shard_bytes)
    shard["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
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
    manifest["metadata_sha256"] = "0" * 64
    write_manifest(cache_path, manifest)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cache(cache_path)


def test_index_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    manifest = read_manifest(cache_path)
    manifest["index_sha256"] = "0" * 64
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
    manifest["sample_count"] = 999
    write_manifest(cache_path, manifest)

    with pytest.raises(ValueError, match="row count does not match manifest"):
        load_cache(cache_path)


def test_embedded_metadata_mismatch_is_rejected_during_iteration(tmp_path: Path) -> None:
    cache_path = write_integrity_cache(tmp_path)
    replace_first_embedded_metadata(cache_path, {"index": 9, "split": "train"})

    with pytest.raises(ValueError, match="embedded metadata does not match"):
        list(load_cache(cache_path))
