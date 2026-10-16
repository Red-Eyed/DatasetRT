# DatasetRT — Dataset Caching for Machine Learning

[Documentation](https://red-eyed.github.io/DatasetRT/) ·
[PyPI](https://pypi.org/project/dataset-rt/) ·
[Issues](https://github.com/Red-Eyed/DatasetRT/issues)

DatasetRT is a Python library with a Rust runtime for caching training datasets
on disk. Reuse preprocessed samples, filter and rebalance data through Polars,
and sample deterministically without rewriting cached payloads.

Install **`dataset-rt`**, import **`dataset_rt`**, and use **DatasetRT** with your
existing training framework.

It keeps payloads immutable on disk, keeps sampling state in Rust, and lets your
Python training code steer the dataset through one Polars metadata table.

```python
metadata = dataset.get_metadata()

dev_run = metadata.head(100)
dataset.update_metadata(dev_run)

for sample in dataset:
    train(sample.data, sample.metadata)
```

Rows in the active metadata table are the rows DatasetRT samples from. Remove
rows for a fast development run, duplicate rows or update `weight` for class
balancing and OHEM, add extra columns for training annotations, and let Rust
validate that every row still maps to an immutable cache sample.

## Why It Exists

Dataset bugs are expensive. A silent shuffle change, corrupt shard, mismatched
metadata row, or weight vector applied to the wrong sample can waste training
runs and make experiments impossible to reproduce.

DatasetRT is for the boring, high-stakes part of ML infrastructure:

- cache expensive preprocessing once and restart quickly
- keep payload bytes immutable while metadata stays editable at runtime
- filter or rebalance samples without rewriting cache files
- sample deterministically from the same cache contents, seed, epoch, and active
  metadata table
- validate metadata and weights against stable `(cache_id, sample_id)` identities
- keep reader and writer queues bounded in Rust

The design rule is simple:

**If it affects correctness, Rust owns it.**

Python remains the ergonomic edge: it describes sources, edits metadata with
Polars, and decodes payload bytes into tensors, images, arrays, or domain
objects.

## The Core Idea

DatasetRT exposes the active dataset as a Polars table:

```text
cache_id | sample_id | <stored metadata columns...> | weight | <extra columns...>
```

That table is the runtime control plane.

```python
import polars as pl

metadata = dataset.get_metadata()

balanced = (
    metadata.join(
        metadata.group_by("label").agg(pl.len().alias("class_count")),
        on="label",
    )
    .with_columns((1.0 / pl.col("class_count")).alias("weight"))
    .drop("class_count")
)

dataset.update_metadata(balanced)
```

`update_metadata()` is runtime-only. It does not rewrite `metadata.arrow`,
`index.bin`, shards, or manifests. Future iterators use the updated active table;
already-created iterators keep their snapshot.

With `drt.ReaderConfig(shuffle=True)`, DatasetRT performs deterministic weighted
multinomial sampling with replacement over the active table. With
`shuffle=False`, it emits active rows in cyclic table order, including
duplicates.
Use `dataset.set_epoch_len(n)` to make future iterators emit a finite window of
`n` samples. Ordered reads continue through the active table cyclically across
iterator boundaries, while shuffled reads continue a deterministic multinomial
draw stream. Updating metadata resets the stream and sets the epoch
length to the new table row count. Call `set_epoch_len` afterward to override it.

## Installation

DatasetRT supports Python 3.10 through 3.13.

```bash
uv add dataset-rt
```

Or with pip:

```bash
pip install dataset-rt
```

## Quickstart

Define a data source as an iterable that packs each Python record with `pickle`
or `dill` and yields `CacheInput`. Put large dictionaries, arrays, and nested
structures in the packed payload; keep fields used for filtering and weighting
in primitive metadata columns.

This complete example caches a dictionary per sample and unpacks it on read. Save it as
`quickstart.py`, then run `uv run quickstart.py` after installing DatasetRT.
Replace the source with your own preprocessing and serialization.

```python
import pickle
from collections.abc import Iterator
from pathlib import Path

from pydantic import TypeAdapter
from typing_extensions import TypedDict

import dataset_rt as drt


class TrainingRecord(TypedDict):
    """Describe the dictionary stored inside each serialized payload."""

    text: str
    tokens: list[int]


RECORD_ADAPTER = TypeAdapter(TrainingRecord)


class TextSamples:
    """Pack preprocessed records, preserving labels as queryable metadata."""

    name = "train_records_v1"

    def __iter__(self) -> Iterator[drt.CacheInput]:
        """Stream payload bytes without building an in-memory dataset."""
        for text, label in [("  Hello world  ", 0), ("Dataset caching", 1)]:
            normalized = text.strip().lower()
            record = TrainingRecord(
                text=normalized, tokens=[ord(char) for char in normalized]
            )
            yield drt.CacheInput(
                data=pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                metadata={"label": label},
            )


runtime = drt.DatasetRuntime(num_workers=4)
result = runtime.from_cache_sources(
    TextSamples(),
    Path("cache"),
    reader_config=drt.ReaderConfig(seed=42, shuffle=True),
)

match result:
    case drt.CacheSourcesDatasetSuccess(dataset, results):
        pass
    case error:
        raise RuntimeError(error)

for sample in dataset:
    record = RECORD_ADAPTER.validate_python(pickle.loads(sample.data))
    print(record["text"], record["tokens"], sample.metadata["label"])
```

Run it again to reuse `cache/train_records_v1` without repeating preprocessing.
Use a new source name or cache root when preprocessing or source contents change;
reuse does not automatically detect those changes. Shuffled reads sample with
replacement, so this example can emit the same record twice.

`pickle` is included with Python. You can use `dill.dumps()` and `dill.loads()`
instead after installing `dill` when your record types require it. The source
packs the record, DatasetRT stores its bytes, and your read transform unpacks it.
See the [serialization guide](https://red-eyed.github.io/DatasetRT/serialization/).

## When To Use It

Use DatasetRT when you care about:

- restart speed after expensive preprocessing
- deterministic training-data order
- class balancing, OHEM, or metadata-driven sampling
- safe development subsets without separate cache copies
- cache integrity and bounded native prefetching

It is not trying to be a training framework, image decoder, tensor format, or
model-specific data pipeline. It is the cache and sampling layer underneath
those pieces.

## Documentation

- [Documentation home](https://red-eyed.github.io/DatasetRT/)
- [Cache training data in Python](https://red-eyed.github.io/DatasetRT/cache-training-data/)
- [Class balancing with Polars](https://red-eyed.github.io/DatasetRT/polars-sampling/)
- [PyTorch DataLoader tutorial](https://red-eyed.github.io/DatasetRT/pytorch-dataloader/)
- [Reader benchmarks](https://red-eyed.github.io/DatasetRT/benchmarks/)
- [Compare dataset tools](https://red-eyed.github.io/DatasetRT/comparisons/)
- [Python API](https://red-eyed.github.io/DatasetRT/python-api/)
- [Runtime guide](https://red-eyed.github.io/DatasetRT/runtime-guide/)
- [Determinism](https://red-eyed.github.io/DatasetRT/determinism/)
- [Storage Format](https://red-eyed.github.io/DatasetRT/storage-format/)
- [Architecture](https://red-eyed.github.io/DatasetRT/overview/)
- [Serialization Boundary](https://red-eyed.github.io/DatasetRT/serialization/)

## Status

DatasetRT is production infrastructure for immutable dataset caching. The core cache lifecycle,
immutable storage format, runtime metadata table, deterministic weighted
sampling, and Rust-owned reader/writer prefetching are in place.

## Citation

```bibtex
@software{stupakov_datasetrt_2026,
  author = {Stupakov, Vadym},
  title = {DatasetRT: A Correctness-First Dataset Cache Runtime},
  year = {2026},
  url = {https://github.com/Red-Eyed/DatasetRT}
}
```
