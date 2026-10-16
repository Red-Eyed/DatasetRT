---
title: Use DatasetRT with a PyTorch DataLoader
description: Build a finite PyTorch DataLoader over cached payloads with explicit spawned workers, deterministic weighted sampling, and Python tensor decoding.
---

# Use DatasetRT with a PyTorch DataLoader

DatasetRT caches bytes and metadata; Python transforms turn those bytes into
training inputs. Use `CachedDataset.to_torch_dataloader()` for finite batches and
managed loading processes backed by native reader threads.

For 32 requested samples and batch size 8, one training process receives four
batches. Two loading workers divide that work; they do not multiply it to 64
samples. Native reader threads and Python loading processes are separate settings.

## Complete example

Install `dataset-rt` and the appropriate [PyTorch build](https://pytorch.org/get-started/locally/).
Save this as `train_cached.py` and run `uv run train_cached.py`. Keep the guarded
entry point and module-level transform: spawned workers must import them.

```python
"""Unpack cached training dictionaries in spawned PyTorch loading workers."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import torch
from pydantic import TypeAdapter
from typing_extensions import TypedDict

import dataset_rt as drt


class TrainingRecord(TypedDict):
    """Describe the Python dictionary packed into each cache payload."""

    features: list[float]
    target: int


RECORD_ADAPTER = TypeAdapter(TrainingRecord)


class TrainingSamples:
    """Pack complete records while keeping targets queryable as metadata."""

    name = "training_records_v1"

    def __iter__(self) -> Iterator[drt.CacheInput]:
        """Serialize one preprocessed dictionary at a time."""
        for index in range(32):
            record = TrainingRecord(features=[index / 32, 1.0], target=index % 2)
            yield drt.CacheInput(
                data=pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                metadata={"target": record["target"]},
            )


def decode(sample: drt.CachedSample) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack and validate a record before Torch collates a bounded CPU batch."""
    record = RECORD_ADAPTER.validate_python(pickle.loads(sample.data))
    features = torch.tensor(record["features"])
    target = torch.tensor(record["target"], dtype=torch.long)
    return features, target


def main() -> None:
    """Construct a fixed loader topology and consume one finite sample budget."""
    runtime = drt.DatasetRuntime(num_workers=2)
    result = runtime.from_cache_sources(
        TrainingSamples(),
        Path("cache_torch"),
        reader_config=drt.ReaderConfig(seed=42, shuffle=True),
    )
    match result:
        case drt.CacheSourcesDatasetSuccess(dataset=dataset):
            pass
        case failure:
            raise RuntimeError(failure)

    loader = dataset.to_torch_dataloader(
        sample_transform_fn=decode,
        batch_size=8,
        samples_per_epoch=32,
        num_workers=2,
        native_num_workers=1,
        shuffle=True,
        seed=42,
        multiprocessing_context="spawn",
    )
    for features, targets in loader:
        print(features.shape, targets.shape)


if __name__ == "__main__":
    main()
```

The loader emits four batches with feature shape `[8, 2]` and target shape `[8]`.
The loader's explicit seed controls its shuffled readers. Decode tensors on the
CPU in loading workers, then transfer batches to the training device as needed.

## Sampling, workers, and memory

- Shuffled loading samples by weight with replacement. For reproducibility,
  keep the seed, worker topology, active metadata, and stream history fixed.
- Persistent loading workers are automatic when `num_workers > 0`, allowing
  sampling streams to continue between epochs. Explicitly disabling persistence
  reconstructs readers and repeats their initial stream.
- The default shuffled partition mode replicates the population for global
  weighted frequencies. Each consumer retains private sampling state; memory
  grows with worker count. Explicit split mode changes the weighting tradeoff.
- Apply metadata edits **before** constructing the loader. Existing loaders use
  their exported snapshot. Build a new loader when the active population changes.
- Set `num_workers=0` for a single-process baseline. Native reading remains
  parallel through `native_num_workers`; more Python processes are not always faster.

The legacy `to_torch_iterable_dataset()` adapter requires a Torch DataLoader with
`num_workers=0`. Use the API demonstrated here for multiprocessing.

For distributed global sample budgets, padding, and ordering guarantees, see
the [loader contract](runtime-guide.md) and [API reference](python-api.md). Run the
[portable benchmark](remote-validation.md) on your own hardware before choosing
a worker count.
