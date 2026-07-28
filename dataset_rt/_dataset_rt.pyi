from collections.abc import Iterator, Sequence

MetadataValue = bool | int | float | str

class CachedDataset:
    def __init__(
        self,
        paths: Sequence[str],
        seed: int,
        prefetch_size: int,
        num_workers: int,
        shuffle: bool,
    ) -> None: ...
    def __iter__(self) -> Iterator[tuple[bytes, dict[str, MetadataValue], int, int]]: ...
    def __len__(self) -> int: ...
    def get_weight_rows(self) -> list[dict[str, MetadataValue]]: ...
    def set_weight_table(self, table: object) -> None: ...

def write_cache(
    sources: object,
    base_cache_dir: str,
    max_shard_bytes: int,
    prefetch_size: int,
    num_threads: int,
    shard_compression: object,
    reuse_existing: bool,
) -> list[str]: ...
