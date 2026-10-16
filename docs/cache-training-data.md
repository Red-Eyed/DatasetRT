---
title: Cache Preprocessed Training Data in Python
description: Cache Python dictionaries and preprocessed training records with an iterable DatasetRT source, pickle or dill serialization, and immutable on-disk storage.
---

# Cache preprocessed training data in Python

If every training restart repeats preprocessing, cache the serialized result
once and read it on subsequent runs. Define a named iterable data source that
packs each Python record or dictionary with `pickle` or `dill` and yields its
bytes in `CacheInput`. The record can contain arrays, tensors, and nested data
structures. Your read transform unpacks the payload for training.

## Install and run

Install `dataset-rt` with `uv add dataset-rt`. Save the complete example below as
`cache_text.py` and run `uv run cache_text.py` twice from the same directory.

```python
"""Cache pickled training dictionaries and unpack delivered records."""

import pickle
from collections.abc import Iterator
from pathlib import Path

from pydantic import TypeAdapter
from typing_extensions import TypedDict

import dataset_rt as drt


class TrainingRecord(TypedDict):
    """Keep the serialized record shape explicit at the decoding boundary."""

    text: str
    tokens: list[int]


RECORD_ADAPTER = TypeAdapter(TrainingRecord)


class TextSamples:
    """Keep the source version in its name so reuse has an explicit boundary."""

    name = "text_records_v1"

    def __iter__(self) -> Iterator[drt.CacheInput]:
        """Pack each preprocessed dictionary without retaining the full source."""
        for text, label in [("  Hello world  ", 0), ("Dataset caching", 1)]:
            normalized = text.strip().lower()
            record = TrainingRecord(
                text=normalized, tokens=[ord(char) for char in normalized]
            )
            yield drt.CacheInput(
                data=pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                metadata={"label": label, "split": "train"},
            )


def main() -> None:
    """Reuse complete caches and report failures before consuming samples."""
    runtime = drt.DatasetRuntime(num_workers=2)
    result = runtime.from_cache_sources(
        TextSamples(),
        Path("cache_text"),
        reader_config=drt.ReaderConfig(seed=42, shuffle=False),
    )
    match result:
        case drt.CacheSourcesDatasetSuccess(dataset=dataset, results=results):
            print(results)
        case failure:
            raise RuntimeError(failure)

    for sample in dataset:
        record = RECORD_ADAPTER.validate_python(pickle.loads(sample.data))
        print(record["text"], record["tokens"], sample.metadata["label"])


if __name__ == "__main__":
    main()
```

The first run writes `cache_text/text_records_v1`. The second reuses the
complete cache without iterating the source. Both read `hello world` and
`dataset caching`, with their character-code lists, in physical order.
The dictionary is small here to keep the example runnable; the same iterable
contract applies to large per-sample structures. `TypeAdapter` validates the
decoded dictionary's shape at the application boundary.

A source name identifies a cache; reuse
does **not** detect changes to preprocessing, input files, or labels. Change the
source name or cache root when those change.

## What is persisted

Each cache contains a publication manifest, Arrow metadata, a binary index, and
payload shards. The manifest marks a complete cache. Payloads stay immutable;
metadata filtering and weighting at runtime do not rewrite these files.

Primitive metadata values support columnar inspection. Keep large arrays,
tensors, and domain objects in the serialized payload, and decode them in your
training code. See the [serialization boundary](serialization.md) and
[storage format](storage-format.md).

`pickle` comes with Python. If your objects require `dill`, install it with
`uv add dill` and use matching `dill.dumps()` / `dill.loads()` calls. DatasetRT
does not choose or invoke either serializer for you.

## Load existing caches directly

When source discovery is unnecessary, load known cache directories:

```python
dataset = runtime.cached_dataset(
    [Path("cache_text/text_records_v1")],
    reader_config=drt.ReaderConfig(seed=42, shuffle=False),
)
```

This excerpt uses `runtime`, `Path`, and `drt` from the complete example.
Default loading validates structural requirements without hashing every shard.
Use `ReaderConfig(seed=42, validate_cache=True)` when you explicitly need the
additional integrity checks.

Native queues remain bounded by configured prefetch capacity. Your source must
also stream its inputs; constructing a full Python list of payloads before
yielding would defeat that memory boundary.

Next: [filter and rebalance samples with Polars](polars-sampling.md) or
[build a PyTorch DataLoader](pytorch-dataloader.md).
