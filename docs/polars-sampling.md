---
title: Class-Balanced Sampling with Polars
description: Filter cached training samples and assign inverse-frequency weights with Polars while DatasetRT validates identities and samples deterministically in Rust.
---

# Filter and rebalance datasets with Polars

DatasetRT exposes the active sampling population as a Polars table. You can
filter rows or change their weights while cached payloads stay immutable.

Suppose class 0 has 9 records and class 1 has 3. Uniform per-record weights give
the classes expected shares of 75% and 25%. Assigning weight `1 / class_count`
gives each class total weight 1, so their expected shares become 50% each.
This is a probability statement: individual epochs need not be balanced exactly.

## Complete example

Install `dataset-rt` with `uv add dataset-rt`. Save this as `balance.py` and run
`uv run balance.py`.

```python
"""Apply columnar class weights to an immutable twelve-record cache."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from typing_extensions import TypedDict

import dataset_rt as drt


class TrainingRecord(TypedDict):
    """Describe the sample dictionary carried by the opaque cache payload."""

    index: int


class LabeledSamples:
    """Expose class labels without storing Python objects in payload shards."""

    name = "imbalanced_v1"

    def __iter__(self) -> Iterator[drt.CacheInput]:
        """Stream nine majority and three minority records."""
        for index in range(12):
            yield drt.CacheInput(
                data=pickle.dumps(TrainingRecord(index=index)),
                metadata={"label": int(index >= 9), "split": "train"},
            )


def main() -> None:
    """Validate class weights natively before constructing a read iterator."""
    runtime = drt.DatasetRuntime(num_workers=2)
    result = runtime.from_cache_sources(
        LabeledSamples(),
        Path("cache_balance"),
        reader_config=drt.ReaderConfig(seed=42, shuffle=True),
    )
    match result:
        case drt.CacheSourcesDatasetSuccess(dataset=dataset):
            pass
        case failure:
            raise RuntimeError(failure)

    training = dataset.get_metadata().filter(pl.col("split") == "train")
    balanced = training.with_columns(
        (1.0 / pl.len().over("label")).alias("weight")
    )
    dataset.update_metadata(balanced)
    dataset.set_epoch_len(120)

    print(balanced.group_by("label").agg(pl.col("weight").sum()))
    counts = [0, 0]
    for sample in dataset:
        counts[int(sample.metadata["label"])] += 1
    print("Observed class counts:", counts)


if __name__ == "__main__":
    main()
```

The grouped weight sums are both 1. The 120 emitted samples may contain repeats
and omit some physical samples because `shuffle=True` means weighted multinomial
sampling **with replacement**, rather than a shuffled permutation.

## Identity and validation

Keep `cache_id` and `sample_id`: together they identify an immutable physical
record. Stored metadata columns must also remain consistent with that identity.
You may remove rows, duplicate valid rows, or add annotation columns. Duplicating
a row adds another weighted entry for the same physical record.

Rust validates identities, stored values, and positive finite weights before
mutating the active population. Filtering, window expressions, and transfer stay
columnar; the example's counting loop consumes delivered samples rather than
expanding the full metadata table into Python row objects.

`update_metadata()` resets the sampling stream and sets epoch length to the new
active row count. Call `set_epoch_len()` afterward to override that length.
Already-created iterators retain their snapshots. Metadata updates are runtime
state; they do not persist back into cache files.

## Reproducibility and cost

Fresh datasets with identical cache contents, active rows and weights, seed,
and iteration history reproduce their sample stream. Successive iterators on the
same dataset continue that stream rather than restarting it.

The window count and weighting expression perform work proportional to active
rows, with grouping state proportional to label cardinality. Requesting metadata
materializes a columnar table; do it when you need population edits, rather than
unconditionally on every restart. See [sampling semantics](determinism.md) and
the [Python API](python-api.md) for validation details.
