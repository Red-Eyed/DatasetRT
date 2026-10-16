# DatasetRT — Dataset Caching for Machine Learning

[Documentation](https://red-eyed.github.io/DatasetRT/) ·
[PyPI](https://pypi.org/project/dataset-rt/) ·
[Issues](https://github.com/Red-Eyed/DatasetRT/issues)

DatasetRT is a Rust-powered dataset cache for Python machine learning pipelines.
It combines fast native sample reads with a separate metadata table for analyzing,
filtering, and weighting training data without unpacking or rewriting its payloads.

Each dataset has two connected parts:

| Part | Contains | Used for |
| --- | --- | --- |
| Payload store | Serialized bytes, typically dictionaries or large Python structures packed with pickle or dill | Read and unpack the samples needed for training |
| Metadata table | Sample identities, primitive attributes such as labels and splits, and sampling weights | Analyze the population, filter samples, and control how often each sample is drawn |

The Python API exposes the metadata table as a **Polars DataFrame**. The payloads
remain in indexed disk shards, linked to metadata by `(cache_id, sample_id)`.

Install **`dataset-rt`**, import **`dataset_rt`**, and use **DatasetRT** with your
existing training framework.

For example, create a development subset by editing only the metadata:

```python
metadata = dataset.get_metadata()

dev_run = metadata.head(100)
dataset.update_metadata(dev_run)
```

Future reads draw from that subset, and your read transform unpacks `sample.data`.

Rows in the active metadata table are the rows DatasetRT samples from. Remove
rows for a fast development run, duplicate rows or update `weight` for class
balancing and OHEM, add extra columns for training annotations, and let Rust
validate that every row still maps to an immutable cache sample.

## Why It Exists

Training experiments often reuse the same expensive preprocessed records while
changing which samples they use. Keeping metadata separate from large serialized
payloads makes those changes possible without reading every payload:

- **Analyze:** inspect class counts, splits, or stored quality scores using the
  metadata table.
- **Filter:** select development subsets or exclude records without making new
  copies of the cache.
- **Weight:** rebalance classes or prioritize hard examples by changing sampling
  weights while payload bytes remain immutable.
- **Read quickly:** Rust performs indexed payload reads with native workers and
  bounded prefetching; Python unpacks and transforms the delivered records.
- **Save RAM across loading processes:** workers map a shared immutable metadata
  snapshot and cache indexes read-only with `mmap`. The OS can share their backing
  pages on the same host, avoiding a full private copy of those buffers per worker.
- **Reproduce:** repeat sampling from identical cache contents, active metadata,
  seed, and stream history.

See the [reader measurements](https://red-eyed.github.io/DatasetRT/benchmarks/)
for startup, throughput, and memory results. End-to-end speed also depends on
storage, payload size, unpacking, transforms, and worker configuration.
Weights, sampling state, decoded records, and prefetched batches still consume
memory per worker; see the [multiprocessing memory guide](https://red-eyed.github.io/DatasetRT/runtime-guide/#share-metadata-and-indexes-with-mmap).

## The Core Idea

DatasetRT exposes the active **metadata**, including sampling weights, as a
Polars table:

```text
cache_id | sample_id | <stored metadata columns...> | weight | <extra columns...>
```

Each row points to a stored payload; the payload itself is not a DataFrame cell.
Analyze or edit this table without unpickling the sample records. For example,
assign inverse-frequency class weights:

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
