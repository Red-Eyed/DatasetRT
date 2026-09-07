from collections.abc import Iterator, Sequence
from typing import NamedTuple

MetadataValue = bool | int | float | str

class CacheWriteRecord(NamedTuple):
    status: str
    source_name: str
    detail: str

class DatasetRuntime:
    def __init__(self, num_workers: int) -> None: ...

class CachedDataset:
    def __init__(
        self,
        runtime: DatasetRuntime,
        paths: Sequence[str],
        seed: int,
        prefetch_size: int,
        shuffle: bool,
        validate_cache: bool,
    ) -> None: ...
    def __iter__(self) -> Iterator[tuple[bytes, dict[str, MetadataValue], int, int]]: ...
    def __len__(self) -> int: ...
    def set_epoch_len(self, epoch_len: int) -> None: ...
    def metadata_ipc(self) -> bytes: ...
    def update_metadata_ipc(self, ipc: bytes) -> None: ...
    def samples_metadata_ipc(self) -> bytes: ...
    def set_samples_metadata_ipc(self, ipc: bytes) -> None: ...

def write_cache(
    runtime: DatasetRuntime,
    sources: object,
    base_cache_dir: str,
    writer_config: object,
    reuse_existing: bool,
) -> list[CacheWriteRecord]: ...
