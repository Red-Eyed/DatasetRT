# Determinism

With `ReaderConfig.shuffle=True`, DatasetRT sampling is deterministic for:

- Cache contents.
- Weight vector.
- Seed.
- Epoch.

Given those inputs, the same physical samples are emitted in the same order.

## Epoch Length

Epoch length equals the number of physical samples across all loaded caches.

Shuffled sampling is weighted multinomial with replacement. A physical sample can appear more than once in one epoch, and another sample can be absent from that epoch.

With `ReaderConfig.shuffle=False`, DatasetRT emits each physical sample once in cache order. Weights are still stored and editable, but they are not applied until a shuffled iterator is created.

## Weight Table

Weights live in Rust but are exposed to Python as a Polars `DataFrame`:

```text
cache_id | sample_id | <metadata columns...> | weight
```

The metadata columns make it natural to create weighted subsets in Python, while Rust remains the authority for applying the result.

When `set_samples_metadata` is called, Rust validates:

- Every physical `(cache_id, sample_id)` appears exactly once.
- No unknown physical identity appears.
- No duplicate physical identity appears.
- Every weight must be finite.
- Every weight must be positive.

The all-zero case is unrepresentable because zero is not a valid v0.1 weight.

## Epoch Advancement

Each new shuffled iterator uses the dataset's next epoch number. This means two shuffled iterators created sequentially from the same dataset can produce different deterministic orders because their epoch input differs.

Loading a new dataset with the same cache paths and seed resets the epoch counter. Non-shuffled iterators do not advance the epoch counter because their order does not depend on epoch state.

## Cache Identity

`cache_id` is the position of the cache path passed to `DatasetRuntime.cached_dataset(...)`.

`sample_id` is the physical row within that cache.
