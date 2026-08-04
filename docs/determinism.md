# Determinism

With `ReaderConfig.shuffle=True`, DatasetRT sampling is deterministic for:

- Cache contents.
- Active metadata table and weight vector.
- Seed.
- Epoch.

Given those inputs, the same physical samples are emitted in the same order.

## Epoch Length

Epoch length equals the number of rows in the active metadata table.

Shuffled sampling is weighted multinomial with replacement. A physical sample can appear more than once in one epoch, and another sample can be absent from that epoch.

With `ReaderConfig.shuffle=False`, DatasetRT emits each active sample once in active table order. Weights are still stored and editable, but they are not applied until a shuffled iterator is created.

## Metadata Table

The active metadata table lives in Rust but is exposed to Python as a Polars `DataFrame`:

```text
cache_id | sample_id | <metadata columns...> | weight
```

The metadata columns make it natural to create weighted subsets in Python, while Rust remains the authority for applying the result.

When `update_metadata` is called, Rust validates:

- At least one physical `(cache_id, sample_id)` appears.
- No unknown physical identity appears.
- No duplicate physical identity appears.
- Every stored metadata column must be present.
- Every weight must be finite.
- Every weight must be positive.

The all-zero case is unrepresentable because zero is not a valid v0.1 weight.
Rows removed from the table are excluded from future iterators.

## Epoch Advancement

Each new shuffled iterator uses the dataset's next epoch number. This means two shuffled iterators created sequentially from the same dataset can produce different deterministic orders because their epoch input differs.

Loading a new dataset with the same cache paths and seed resets the epoch counter. Non-shuffled iterators do not advance the epoch counter because their order does not depend on epoch state.

## Cache Identity

`cache_id` is the position of the cache path passed to `DatasetRuntime.cached_dataset(...)`.

`sample_id` is the physical row within that cache.
