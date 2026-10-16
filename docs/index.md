---
title: Dataset Caching for Machine Learning
description: Cache preprocessed training data in Python with DatasetRT's Rust runtime, immutable disk storage, deterministic weighted sampling, and Polars metadata.
---

# DatasetRT — Dataset Caching for Machine Learning

DatasetRT is a Python library with a Rust runtime for caching training datasets
on disk. Reuse preprocessed samples, filter and rebalance data through Polars,
and sample deterministically without rewriting cached payloads.

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

## Control sampling through metadata

```python
import polars as pl

metadata = dataset.get_metadata()
training = metadata.filter(pl.col("split") == "train")
dataset.update_metadata(training)
```

This excerpt assumes a loaded dataset with a stored `split` column. Metadata
updates affect future iterators and leave payload files unchanged. Rust validates
identities and positive, finite weights before accepting changes.

With `shuffle=True`, reads use weighted sampling **with replacement**: a record
can appear more than once while another is absent. Sequential reads follow active
table order and wrap when the requested epoch is longer than the population.
See [deterministic sampling](determinism.md) for stream and epoch semantics.

## Choose your next step

- Read the [complete runnable quickstart](https://github.com/Red-Eyed/DatasetRT#quickstart).
- [Cache preprocessed training data in Python](cache-training-data.md).
- [Filter and rebalance datasets with Polars](polars-sampling.md).
- [Use DatasetRT with a PyTorch DataLoader](pytorch-dataloader.md).
- Inspect the [Python API](python-api.md) and [PyTorch loader contract](runtime-guide.md).
- Understand the [immutable cache format](storage-format.md).
- Run [CPU, GPU, and distributed benchmarks](remote-validation.md).
- Report a problem in [GitHub Issues](https://github.com/Red-Eyed/DatasetRT/issues)
  or inspect the [changelog](https://github.com/Red-Eyed/DatasetRT/blob/main/CHANGELOG.md).

## Scope and status

DatasetRT is production infrastructure for immutable dataset caching. It stores
payload bytes; it does not decode images or tensors, replace a training framework,
or provide a remote object-storage service. Evaluate the loader topology and
resource budget for your workload using the documented benchmark commands.
