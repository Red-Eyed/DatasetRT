# DatasetRT

DatasetRT is a correctness-first dataset cache for ML training loops that need
fast restarts, deterministic sampling, and metadata-aware control over what gets
sampled.

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
rows for a fast development run, update `weight` for class balancing or OHEM,
add extra columns for training annotations, and let Rust validate that every row
still maps to the right immutable cache sample.

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

With `ReaderConfig(shuffle=True)`, DatasetRT performs deterministic weighted
multinomial sampling with replacement over the active table. With
`shuffle=False`, it emits active rows once in table order.

## Quickstart

Wrap your existing data source as a Python iterable that yields payload bytes and
primitive metadata.

```python
from pathlib import Path

from dataset_rt import (
    CacheInput,
    CacheSourcesDatasetSuccess,
    DatasetRuntime,
    ReaderConfig,
)


class Images:
    name = "train_images"

    def __iter__(self):
        for image_id, image_bytes, label in load_my_images():
            yield CacheInput(
                data=image_bytes,
                metadata={"image_id": image_id, "label": label},
            )


runtime = DatasetRuntime(num_workers=4)
result = runtime.from_cache_sources(
    Images(),
    Path("cache"),
    reader_config=ReaderConfig(seed=42, shuffle=True),
)

match result:
    case CacheSourcesDatasetSuccess(dataset, results):
        pass
    case error:
        raise RuntimeError(error)

for sample in dataset:
    image = decode_image(sample.data)
    label = sample.metadata["label"]
    train(image, label)
```

The Rust cache stores bytes. DatasetRT does not decode JPEGs, tensors, tokens, or
framework objects in the core; your Python code owns domain decoding.

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

- [Python API](docs/python-api.md)
- [Runtime Model](docs/runtime.md)
- [Determinism](docs/determinism.md)
- [Storage Format](docs/storage-format.md)
- [Architecture](docs/architecture.md)
- [Serialization Boundary](docs/serialization.md)
- [Build Artifacts](docs/build.md)
- [Development](docs/development.md)

## Status

DatasetRT is early production infrastructure. The core cache lifecycle,
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
