from collections.abc import Iterator, Sequence
from typing import NamedTuple

MetadataValue = bool | int | float | str

class CacheWriteRecord(NamedTuple):
    status: str
    source_name: str
    detail: str

class CachedDataset:
    def __init__(
        self,
        paths: Sequence[str],
        seed: int,
        prefetch_size: int,
        num_workers: int,
        shuffle: bool,
        validate_cache: bool,
    ) -> None: ...
    def __iter__(self) -> Iterator[tuple[bytes, dict[str, MetadataValue], int, int]]: ...
    def __len__(self) -> int: ...
    def weight_table_ipc(self) -> bytes: ...
    def set_weight_table_ipc(self, ipc: bytes) -> None: ...

def write_cache(
    sources: object,
    base_cache_dir: str,
    writer_config: object,
    reuse_existing: bool,
) -> list[CacheWriteRecord]: ...
