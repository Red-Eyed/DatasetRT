"""Run real nonpersistent spawned readers with unconfigured application logging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

from dataset_rt import CacheInput, CacheWriteSuccess, DatasetRuntime, ReaderConfig, WriterConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dataset_rt import CachedSample


class Settings(BaseSettings):
    """Validate the pytest-owned output directory at the environment boundary."""

    model_config = SettingsConfigDict(env_prefix="DATASETRT_LOG_TEST_")
    cache_root: DirectoryPath


class Report(BaseModel):
    """Return bounded draw sequences for replay checks without sending native state."""

    first: list[int]
    second: list[int]


class Source:
    """Provide a weighted population larger than either short reader pass."""

    name = "logging"

    def __iter__(self) -> Iterator[CacheInput]:
        """Stream independently identifiable small payloads."""
        for index in range(20):
            yield CacheInput(str(index).encode(), {"index": index})


def payload_index(sample: CachedSample) -> int:
    """Transform inside the real loading process using only delivered payload bytes."""
    return int(sample.data)


def main() -> None:
    """Exercise two complete passes so new worker PIDs and reset events are observable."""
    settings = Settings()
    runtime = DatasetRuntime(num_workers=1)
    results = runtime.write_cache(
        Source(), settings.cache_root, writer_config=WriterConfig(show_progress=False)
    )
    match results[0]:
        case CacheWriteSuccess(path=path):
            dataset = runtime.cached_dataset([path], reader_config=ReaderConfig(seed=17))
        case failure:
            raise AssertionError(failure)
    dataset.update_metadata(
        dataset.get_metadata().with_columns((pl.col("index").cast(pl.Float64) + 1).alias("weight"))
    )
    loader = dataset.to_torch_dataloader(
        num_workers=1,
        native_num_workers=1,
        seed=7,
        samples_per_epoch=4,
        multiprocessing_context="spawn",
        persistent_workers=False,
        sample_transform_fn=payload_index,
        verbose=True,
    )
    first = [int(batch.item()) for batch in loader]
    second = [int(batch.item()) for batch in loader]
    print(Report(first=first, second=second).model_dump_json())


if __name__ == "__main__":
    main()
