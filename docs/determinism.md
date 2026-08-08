# Determinism

With `ReaderConfig.shuffle=True`, DatasetRT sampling is deterministic for:

- Cache contents.
- Active metadata table and weight vector.
- Seed.
- Epoch.

Given those inputs, the same physical samples are emitted in the same order.

## Epoch Length

By default, epoch length equals the number of rows in the active metadata table.
`CachedDataset.set_epoch_len(n)` changes the number of samples future iterators
emit without changing the active sampling population. `n` must be at least 1.

Shuffled sampling is weighted multinomial with replacement over active metadata rows. A physical sample can appear more than once in one epoch, and another sample can be absent from that epoch.

With `ReaderConfig.shuffle=False`, DatasetRT emits a cyclic window over active
rows in active table order. For active rows `1 2 3 4 5` and
`set_epoch_len(2)`, three sequential iterators emit `1 2`, then `3 4`, then
`5 1`. If the same physical `(cache_id, sample_id)` appears in multiple rows,
that sample is emitted once for each row position. Weights are still stored and
editable, but they are not applied until a shuffled iterator is created.

With `ReaderConfig.shuffle=True`, DatasetRT emits windows from one deterministic
multinomial draw stream. Creating a new iterator does not reset the sampler; it
continues from the next draw. This means short epochs are equivalent to splitting
a longer deterministic stream into smaller windows.

## Metadata Table

The active metadata table lives in Rust but is exposed to Python as a Polars `DataFrame`:

```text
cache_id | sample_id | <metadata columns...> | weight
```

The metadata columns make it natural to create weighted subsets in Python, while Rust remains the authority for applying the result.

When `update_metadata` is called, Rust validates:

- At least one physical `(cache_id, sample_id)` appears.
- No unknown physical identity appears.
- Every stored metadata column must be present.
- Every weight must be finite.
- Every weight must be positive.

The all-zero case is unrepresentable because zero is not a valid v0.1 weight.
Rows removed from the table are excluded from future iterators. Duplicate identities are allowed and make the repeated rows part of the active sampling space. Updating metadata resets the ordered cursor or shuffled draw stream without changing epoch length.

## Epoch Advancement

Each new shuffled iterator reserves the next window from the dataset's shuffled draw stream. This means two shuffled iterators created sequentially from the same dataset can produce different deterministic samples because their stream positions differ.

Loading a new dataset with the same cache paths and seed resets the stream position. Updating metadata also resets the stream position because the active population or weights may have changed.

## Cache Identity

`cache_id` is the position of the cache path passed to `DatasetRuntime.cached_dataset(...)`.

`sample_id` is the physical row within that cache.
