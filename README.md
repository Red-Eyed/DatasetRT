# DatasetRT

DatasetRT is a correctness-first dataset cache for ML training loops.

It gives you a deterministic, immutable cache on disk, backed by a Rust runtime and exposed through a small Python API. You keep your model code in PyTorch, JAX, TensorFlow, NumPy, or plain Python; DatasetRT handles cache integrity, metadata, sampling weights, and repeatable iteration without becoming another framework.

## Authorship

Created by Vadym Stupakov <vadim.stupakov@gmail.com>.

## Why ML Users Need This

Dataset bugs are expensive. A silent shuffle change, corrupt shard, mismatched metadata row, or weight vector applied to the wrong sample can waste training runs and make experiments impossible to reproduce.

DatasetRT is built around one rule:

**If it affects correctness, Rust owns it.**

Rust owns:

- immutable cache publication
- manifest and checksum validation
- metadata schema validation
- shard offsets and index generation
- deterministic weighted sampling
- iterator state
- bounded reader/writer prefetch and worker pools
- weight table validation

Python stays thin and ergonomic. It describes your source data and receives bytes plus metadata back.

## Quickstart

```python
from pathlib import Path

import polars as pl

from dataset_rt import (
    CacheInput,
    CachedDataset,
    ReaderConfig,
    ShardCompression,
    WriterConfig,
)


class Images:
    name = "train_images"

    def __iter__(self):
        for sample_id, image_bytes, label in load_my_images():
            yield CacheInput(
                data=image_bytes,
                metadata={"sample_id": sample_id, "label": label},
            )


dataset = CachedDataset.from_cache_sources(
    Images(),
    Path("cache"),
    reader_config=ReaderConfig(seed=42, prefetch_size=64, num_workers=4, shuffle=True),
    writer_config=WriterConfig(
        prefetch_size=64,
        num_threads=4,
        shard_compression=ShardCompression(algo="none", ratio=1.0),
    ),
)

for sample in dataset:
    image = decode_image(sample.data)  # domain decoding stays in Python
    label = sample.metadata["label"]
```

`CachedDataset.from_cache_sources` creates missing caches, reuses valid existing caches, and returns a ready-to-iterate dataset. The cache directory argument is always a base cache directory; Rust writes each source under `base_cache_dir / name_hash`.

## PyTorch

When PyTorch is installed, turn the same DatasetRT object into a sized `IterableDataset`:

```python
torch_dataset = dataset.to_torch_iterable_dataset()
loader = torch.utils.data.DataLoader(torch_dataset, batch_size=None)

for sample in loader:
    image = decode_image(sample.data)
    label = sample.metadata["label"]
```

The adapter does not decode payloads or add a Torch dependency to DatasetRT. It yields `CachedSample` values and reports `len(torch_dataset)`.

## Metadata-Aware Weights

Weights are not a loose list that can drift out of alignment. DatasetRT exposes them as a Polars table with sample identity and metadata:

```python
weights = dataset.weight_table()

rare = weights.with_columns(
    pl.when(pl.col("label") == "rare_class")
    .then(5.0)
    .otherwise(1.0)
    .alias("weight")
)

dataset.set_weight_table(rare)
```

The table contains:

```text
cache_id | sample_id | <metadata columns...> | weight
```

Rust validates that every physical `(cache_id, sample_id)` appears exactly once and that every weight is positive and finite.

## Multiple Sources

```python
dataset = CachedDataset.from_cache_sources(
    [TrainImages(), SyntheticImages(), HardNegatives()],
    Path("cache"),
    reader_config=ReaderConfig(seed=123),
    writer_config=WriterConfig(prefetch_size=128, num_threads=8),
)
```

If one source fails during a multi-source write, DatasetRT cleans up caches created earlier in that call. A loaded dataset means every cache was validated from its manifest.

## Storage Layout

```text
cache/
    train_images_<hash>/
        manifest.json
        metadata.arrow
        index.bin
        shards/
            000000.bin
            000001.bin
```

Metadata is stored separately from payload bytes. This keeps sampling, filtering, auditing, and weight editing independent of domain payload decoding.

## What DatasetRT Does Not Do

DatasetRT does not decode JPEGs, PNGs, tensors, or framework-specific objects in the Rust core.

The core returns payload bytes. Your Python code or optional adapters can decode those bytes into tensors, arrays, images, token sequences, or any other domain object.

DatasetRT v0.1 also intentionally supports only:

- bytes-like payloads: `bytes`, `bytearray`, `memoryview`
- primitive metadata: `bool`, `int`, `float`, `str`
- `ShardCompression(algo="none", ratio=1.0)`

The common Rust `zstd` crate uses C bindings, so zstd compression is not enabled for the first stable version.

## Documentation

- [Architecture](docs/architecture.md)
- [Python API](docs/python-api.md)
- [Storage Format](docs/storage-format.md)
- [Runtime Model](docs/runtime.md)
- [Determinism](docs/determinism.md)
- [Serialization Boundary](docs/serialization.md)
- [Development](docs/development.md)

## Status

DatasetRT is at foundational v0.1 architecture. The core cache lifecycle, immutable storage, metadata, deterministic weighted sampling, Rust-owned reader/writer prefetching, and Polars weight table are in place.
