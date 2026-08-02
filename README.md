# DatasetRT

DatasetRT is a correctness-first dataset cache for ML training loops that need
fast restarts, reproducible sampling, and metadata-aware weighting without
moving dataset state into Python.

It gives you immutable on-disk caches, deterministic weighted sampling, and
bounded Rust-owned read/write pipelines behind a small Python API. You keep your
model code in PyTorch, JAX, TensorFlow, NumPy, or plain Python; DatasetRT handles
cache integrity, metadata, weights, and repeatable iteration without becoming
another framework.

Once a dataset is loaded, balanced sampling is only a few Polars lines:

```python
import polars as pl

metadata = dataset.samples_metadata()

class_counts = metadata.group_by("label").agg(pl.len().alias("class_count"))
balanced = (
    metadata.join(class_counts, on="label")
    .with_columns((1.0 / pl.col("class_count")).alias("weight"))
    .drop("class_count")
)

dataset.set_samples_metadata(balanced)
```

That is the whole balanced-sampling workflow: compute weights with Polars, hand
the table back, and let Rust validate identity, coverage, and weight values
before the next shuffled iterator uses them.

## Why ML Users Need This

Dataset bugs are expensive. A silent shuffle change, corrupt shard, mismatched
metadata row, or weight vector applied to the wrong sample can waste training
runs and make experiments impossible to reproduce.

DatasetRT is for the boring, high-stakes part of training infrastructure:

- cache expensive preprocessing once and restart quickly
- keep payload bytes immutable while metadata stays queryable
- balance imbalanced classes without copying or rewriting the dataset
- sample deterministically from the same seed, epoch, cache contents, and weights
- validate weights against stable `(cache_id, sample_id)` identities before they
  can affect training
- keep queues bounded so fast producers cannot silently grow memory

DatasetRT is built around one rule:

**If it affects correctness, Rust owns it.**

Rust owns:

- immutable cache publication
- manifest and checksum validation
- metadata schema validation
- shard offsets and index generation
- deterministic weighted sampling
- iterator state
- bounded reader/writer prefetch over one reused Rust worker pool
- samples metadata validation

Python stays thin and ergonomic. It describes your source data and receives bytes plus metadata back.

## Quickstart

Wrap your existing data source as a tiny Python iterable. DatasetRT stores the
payload bytes and metadata, then returns cached samples in deterministic order.

```python
from pathlib import Path

from dataset_rt import (
    CacheInput,
    CacheSourcesDatasetError,
    CacheSourcesDatasetSuccess,
    DatasetRuntime,
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


runtime = DatasetRuntime(num_workers=4)
result = runtime.from_cache_sources(
    Images(),
    Path("cache"),
    reader_config=ReaderConfig(
        seed=42,
        prefetch_size=64,
        shuffle=True,
        validate_cache=False,
    ),
    writer_config=WriterConfig(
        prefetch_size=64,
        shard_compression=ShardCompression(algo="none", ratio=1.0),
        show_progress=True,
        validate_cache=False,
    ),
)

match result:
    case CacheSourcesDatasetSuccess(dataset, results):
        pass
    case CacheSourcesDatasetError(results, message):
        raise RuntimeError(message)

for sample in dataset:
    image = decode_image(sample.data)  # domain decoding stays in Python
    label = sample.metadata["label"]
```

`DatasetRuntime` creates exactly the requested number of Rust worker threads once and reuses them for cache loading, reading, and writing. `runtime.from_cache_sources` creates missing caches, reuses existing cache directories, and returns a result containing the loaded dataset plus per-source write outcomes. The cache directory argument is always a base cache directory; Rust writes each source under `base_cache_dir / name`. Cache writing shows committed samples/s and MB/s for the active source and source-count ETA for multi-source writes by default; pass `WriterConfig(show_progress=False)` for quiet jobs. Existing cache checksum validation is opt-in with `validate_cache=True`; by default DatasetRT avoids hashing every payload shard during restart.

## Balanced Sampling With Weights

Class balancing is just a metadata operation. You do not need to duplicate rare
samples, build a Python sampler, or keep a separate weight vector in sync with
the dataset.

For example, imagine this cached dataset:

```text
label | rows
cat   | 900
dog   |  90
fox   |  10
```

If every sample has weight `1.0`, a shuffled epoch mostly follows the original
imbalance. To make the total probability mass of each label equal, give each
sample an inverse-frequency weight:

```python
import polars as pl

metadata = dataset.samples_metadata()

class_counts = metadata.group_by("label").agg(pl.len().alias("class_count"))
balanced = (
    metadata.join(class_counts, on="label")
    .with_columns((1.0 / pl.col("class_count")).alias("weight"))
    .drop("class_count")
)

dataset.set_samples_metadata(balanced)
```

The resulting per-sample weights are:

```text
label | rows | per-sample weight | total label weight
cat   | 900  | 1 / 900           | 1.0
dog   |  90  | 1 / 90            | 1.0
fox   |  10  | 1 / 10            | 1.0
```

With `ReaderConfig(shuffle=True)`, the next iterator snapshots those weights and
uses deterministic weighted multinomial sampling with replacement. Epoch length
is still the physical dataset length; rare samples may appear more than once in
one epoch, and common samples may be skipped.

You can balance on any metadata column or expression:

```python
import polars as pl

metadata = dataset.samples_metadata()

bucketed = metadata.with_columns(
    pl.when(pl.col("source") == "hard_negatives")
    .then(8.0)
    .when(pl.col("split") == "synthetic")
    .then(0.5)
    .otherwise(1.0)
    .alias("weight")
)

dataset.set_samples_metadata(bucketed)
```

Rust accepts only the identity and weight columns from the Polars table, then
validates that every physical `(cache_id, sample_id)` appears exactly once and
that every weight is positive and finite.

## PyTorch

When PyTorch is installed, turn the same DatasetRT object into a sized `IterableDataset`:

```python
torch_dataset = dataset.to_torch_iterable_dataset()
loader = torch.utils.data.DataLoader(torch_dataset, batch_size=None, num_workers=0)

for sample in loader:
    image = decode_image(sample.data)
    label = sample.metadata["label"]
```

The adapter does not decode payloads or add a Torch dependency to DatasetRT. It yields `CachedSample` values and reports `len(torch_dataset)`. Keep PyTorch `DataLoader(num_workers=0)`; use `DatasetRuntime(num_workers=N)` for parallel cache reads.

## Samples Metadata

Weights are not a loose list that can drift out of alignment. DatasetRT exposes
dataset-level samples metadata as a Polars table with stable identity columns,
stored metadata, and editable weights:

```python
import polars as pl

metadata = dataset.samples_metadata()

rare = metadata.with_columns(
    pl.when(pl.col("label") == "rare_class").then(5.0).otherwise(1.0).alias("weight")
)

dataset.set_samples_metadata(rare)
```

The table contains:

```text
cache_id | sample_id | <metadata columns...> | weight
```

Rust validates that every physical `(cache_id, sample_id)` appears exactly once and that every weight is positive and finite.

## Multiple Sources

Pass multiple sources when your training set is assembled from different
origins, such as real images, synthetic images, and hard negatives. Each source
gets its own immutable cache directory, and the loaded dataset has one stable
physical identity space across all successful caches.

```python
large_runtime = DatasetRuntime(num_workers=8)
result = large_runtime.from_cache_sources(
    [TrainImages(), SyntheticImages(), HardNegatives()],
    Path("cache"),
    reader_config=ReaderConfig(seed=123),
    writer_config=WriterConfig(prefetch_size=128, show_progress=False),
)
```

If one source fails during a multi-source write, DatasetRT reports that source as `CacheWriteError` and keeps going. `CacheSourcesDatasetSuccess.results` tells you which sources were loaded and which were missing or malformed. A loaded dataset means every successful cache was validated from its manifest.

If your sources store an origin column in metadata, you can rebalance after
loading:

```python
import polars as pl

metadata = dataset.samples_metadata()

weighted = metadata.with_columns(
    pl.when(pl.col("source") == "hard_negatives").then(4.0).otherwise(1.0).alias("weight")
)

dataset.set_samples_metadata(weighted)
```

## Storage Layout

```text
cache/
    train_images/
        manifest.json
        metadata.arrow
        index.bin
        shards/
            000000.bin
            000001.bin
```

Metadata is stored separately from payload bytes. This keeps sampling, filtering, auditing, and weight editing independent of domain payload decoding. Each shard record also embeds the same metadata redundantly so raw record inspection and visualization can show sample context without joining back through Arrow.

## What DatasetRT Does Not Do

DatasetRT does not decode JPEGs, PNGs, tensors, or framework-specific objects in the Rust core.

The core returns payload bytes. Your Python code or optional adapters can decode those bytes into tensors, arrays, images, token sequences, or any other domain object.

DatasetRT v0.1 also intentionally supports only:

- bytes-like payloads: `bytes`, `bytearray`, `memoryview`
- primitive metadata: `bool`, `int`, `float`, `str`
- shard compression: `ShardCompression(algo="none", ratio=1.0)` or `ShardCompression(algo="lz4", ratio=...)`

LZ4 compression is applied per payload record so random access stays direct. The common Rust `zstd` crate uses C bindings, so zstd compression is not enabled for the first stable version.

## Documentation

- [Architecture](docs/architecture.md)
- [Inversion Analysis](docs/inversion-analysis.md)
- [Python API](docs/python-api.md)
- [Storage Format](docs/storage-format.md)
- [Runtime Model](docs/runtime.md)
- [Determinism](docs/determinism.md)
- [Serialization Boundary](docs/serialization.md)
- [Development](docs/development.md)
- [Build Artifacts](docs/build.md)
- [Changelog](CHANGELOG.md)

## Artifact Builds

Distribution artifacts are built locally with `just build-all`. Linux wheels use Zig cross-compilation; macOS arm64 and x86_64 wheels build on the local host. GitHub Actions stays checks-only.

Wheels use Python's stable ABI (`cp310-abi3`) and support Python 3.10 through 3.13.

## Status

DatasetRT is at foundational v0.1 architecture. The core cache lifecycle, immutable storage, metadata, deterministic weighted sampling, Rust-owned reader/writer prefetching, and Polars samples metadata table are in place.

## Citation

If DatasetRT helps your work, please cite it as:

```bibtex
@software{stupakov_datasetrt_2026,
  author = {Stupakov, Vadym},
  title = {DatasetRT: A Correctness-First Dataset Cache Runtime},
  year = {2026},
  url = {https://github.com/Red-Eyed/DatasetRT}
}
```
