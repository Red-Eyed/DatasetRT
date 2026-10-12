use super::*;
use crate::types::CacheId;

/// Seed a full outstanding window so completion order is independent of disk and thread timing.
fn completed_window(total: usize) -> CacheResult<RuntimeIterator> {
    let (output_sender, output) = bounded(3)?;
    for sequence in [1, 2, 0] {
        output_sender.send(Ok(LoadedResult {
            sequence,
            result: LoadedSample {
                data: vec![0; 1024 * 1024],
                metadata: Vec::new(),
                cache_id: CacheId::from_u64(7),
                sample_id: SampleId::from_position(sequence)?,
            },
        }))?;
    }
    Ok(RuntimeIterator {
        pool: WorkerPool::new(NumWorkers::new(3)?)?,
        caches: Arc::new(Vec::new()),
        cache_offsets: Arc::new(Vec::new()),
        plan: EpochPlan::Sequential {
            len: total,
            start: 0,
            population_len: total,
            physical_indices: None,
        },
        output_sender,
        output,
        cancelled: Arc::new(AtomicBool::new(false)),
        pending: BTreeMap::new(),
        parallelism: 3,
        scheduled: 3,
        emitted: 0,
        total,
    })
}

#[test]
/// A stalled first position cannot trigger replacement reads for completed future positions.
fn delayed_first_completion_does_not_expand_window() -> CacheResult<()> {
    for total in [100, 1_000_000] {
        let mut iterator = completed_window(total)?;
        assert!(matches!(iterator.next(), Some(Ok(sample)) if sample.sample_id.as_u64() == 0));
        assert_eq!(iterator.scheduled, 3);
        assert_eq!(iterator.pending.len(), 2);
        assert_eq!(
            iterator.scheduled - iterator.emitted,
            iterator.pending.len()
        );
        let cancelled = iterator.cancelled.clone();
        drop(iterator);
        assert!(cancelled.load(Ordering::Acquire));
    }
    Ok(())
}

#[test]
/// Buffered completions are emitted once in planned order and finite streams terminate.
fn reordered_results_preserve_order_and_exhaustion() -> CacheResult<()> {
    let mut iterator = completed_window(3)?;
    for expected in 0..3 {
        assert!(
            matches!(iterator.next(), Some(Ok(sample)) if sample.sample_id.as_u64() == expected)
        );
    }
    assert!(iterator.next().is_none());
    assert_eq!(iterator.scheduled, 3);
    assert!(iterator.pending.is_empty());
    Ok(())
}
