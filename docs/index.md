---
title: Dataset Caching for Machine Learning
description: Read cached training samples with a fast Rust runtime, and analyze, filter, or weight a separate metadata table without unpacking serialized payloads.
---

# DatasetRT — Dataset Caching for Machine Learning

DatasetRT is a Rust-powered dataset cache for Python machine learning pipelines.
It combines fast native sample reads with a separate metadata table for analyzing,
filtering, and weighting training data without unpacking or rewriting its payloads.

## Payloads and metadata

| Part | What it stores | What you do with it |
| --- | --- | --- |
| Payload store | Serialized bytes in indexed disk shards; commonly pickled dictionaries, arrays, or nested Python structures | Read selected samples and unpack them for training |
| Metadata table | Sample identities, primitive attributes, and sampling weights | Analyze class distributions, filter the population, and adjust sampling probabilities |

The metadata table is exposed through a **Polars DataFrame** in Python. Payloads
are stored separately and linked by `(cache_id, sample_id)`. Updating metadata
does not require unpickling the records or rewriting their cached bytes.

Rust owns indexed sample reading, native workers, and bounded prefetching.
See the [reader benchmarks](benchmarks.md) for measured throughput, startup time,
and memory use. Your Python code owns unpacking and training transforms.

## Installation

Install **`dataset-rt`**, import **`dataset_rt`**, and use **DatasetRT** with your
existing training framework.

[Get the package on PyPI](https://pypi.org/project/dataset-rt/) or
[browse the source on GitHub](https://github.com/Red-Eyed/DatasetRT).

```bash
uv add dataset-rt
```

Or install a published wheel through pip:

```bash
pip install dataset-rt
```

Python 3.10–3.13 is supported. PyTorch is optional; install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/) if you use the loader
integration.

## Cache preprocessing once

Define a data source as a named Python iterable. For each sample, pack its
dictionary, arrays, or nested Python structure with `pickle` or `dill`, then
yield `CacheInput(data=packed_record, metadata=...)`. Unpack `sample.data` in your
read transform. Primitive metadata stays available for filtering and sampling
without deserializing the large payload.

DatasetRT writes
immutable shards, an index, columnar metadata, and a manifest. Subsequent runs
reuse complete caches by source name. Change the name or cache root when the
source or preprocessing changes; reuse does not detect those changes for you.

Rust owns validation, sampling, reader and writer worker pools, and bounded
prefetching. Python owns preprocessing, serialization, decoding, and framework
integration.

## Analyze, filter, and weight through metadata

```python
import polars as pl

metadata = dataset.get_metadata()
print(metadata.group_by("label").agg(pl.len().alias("samples")))
training = metadata.filter(pl.col("split") == "train")
training = training.with_columns(
    (1.0 / pl.len().over("label")).alias("weight")
)
dataset.update_metadata(training)
```

This excerpt assumes a loaded dataset with stored `label` and `split` columns.
It analyzes class counts, selects training records, and assigns inverse-frequency
class weights using only metadata. Metadata
updates affect future iterators and leave payload files unchanged. Rust validates
identities and positive, finite weights before accepting changes.

With `shuffle=True`, reads use weighted sampling **with replacement**: a record
can appear more than once while another is absent. Sequential reads follow active
table order and wrap when the requested epoch is longer than the population.
See [deterministic sampling](determinism.md) for stream and epoch semantics.

## Save RAM with multiprocessing loaders

Workers of the same loader use read-only, file-backed `mmap` to access one
immutable metadata snapshot. Cache indexes are also memory-mapped from their
original files. The operating system can share the backing pages between
processes on the same host, reducing duplicated RAM instead of allocating a
complete private copy of these buffers for each worker.

This sharing works with spawned workers too. Weights, sampling state, decoded
records, and batch buffers remain private allocations. See the
[mmap memory guide](runtime-guide.md#share-metadata-and-indexes-with-mmap)
for the boundaries of that sharing.

## Choose your next step

- Read the [complete runnable quickstart](https://github.com/Red-Eyed/DatasetRT#quickstart).
- [Cache preprocessed training data in Python](cache-training-data.md).
- [Filter and rebalance datasets with Polars](polars-sampling.md).
- [Use DatasetRT with a PyTorch DataLoader](pytorch-dataloader.md).
- Inspect the [Python API](python-api.md) and [PyTorch loader contract](runtime-guide.md).
- Understand the [immutable cache format](storage-format.md).
- Run [CPU, GPU, and distributed benchmarks](remote-validation.md).
- Examine [local reader measurements](benchmarks.md) and
  [compare dataset tools](comparisons.md).
- Report a problem in [GitHub Issues](https://github.com/Red-Eyed/DatasetRT/issues)
  or inspect the [changelog](https://github.com/Red-Eyed/DatasetRT/blob/main/CHANGELOG.md).

## Scope and status

DatasetRT is production infrastructure for immutable dataset caching. It stores
payload bytes; it does not decode images or tensors, replace a training framework,
or provide a remote object-storage service. Evaluate the loader topology and
resource budget for your workload using the documented benchmark commands.
