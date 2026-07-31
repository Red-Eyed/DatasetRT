from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from dataset_rt import (
    CacheInput,
    CacheSourcesDatasetSuccess,
    DatasetRuntime,
    ReaderConfig,
    WriterConfig,
)

SAMPLE_COUNT = 10_000
PAYLOAD_BYTES = 1024


class SyntheticSource:
    name = "synthetic"

    def __iter__(self) -> Iterator[CacheInput]:
        payload = b"x" * PAYLOAD_BYTES
        for index in range(SAMPLE_COUNT):
            yield CacheInput(
                data=payload,
                metadata={"index": index, "split": "train"},
            )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset_rt_bench_") as temp_dir:
        base_cache_dir = Path(temp_dir) / "cache"
        runtime = DatasetRuntime(num_workers=4)
        write_started = time.perf_counter()
        result = runtime.from_cache_sources(
            SyntheticSource(),
            base_cache_dir,
            reader_config=ReaderConfig(seed=42, prefetch_size=128, shuffle=False),
            writer_config=WriterConfig(prefetch_size=128),
        )
        match result:
            case CacheSourcesDatasetSuccess(dataset=dataset):
                pass
            case error:
                raise RuntimeError(error)
        write_seconds = time.perf_counter() - write_started

        read_started = time.perf_counter()
        byte_count = sum(len(sample.data) for sample in dataset)
        read_seconds = time.perf_counter() - read_started

    # JSON keeps this useful for CI logs and future benchmark dashboards.
    print(
        json.dumps(
            {
                "samples": SAMPLE_COUNT,
                "payload_bytes": PAYLOAD_BYTES,
                "total_bytes_read": byte_count,
                "write_seconds": write_seconds,
                "read_seconds": read_seconds,
                "write_samples_per_second": SAMPLE_COUNT / write_seconds,
                "read_samples_per_second": SAMPLE_COUNT / read_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
