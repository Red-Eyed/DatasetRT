use super::*;
use crate::types::CacheId;

/// Duplicate future results must cancel the iterator rather than overwrite buffered data.
#[test]
fn duplicate_completion_cancels_the_reader() -> CacheResult<()> {
    let mut iterator = completed_window(3)?;
    for _ in 0..3 {
        iterator.output.recv()??;
    }
    for _ in 0..2 {
        iterator.output_sender.send(Ok(LoadedResult {
            sequence: 1,
            result: LoadedSample {
                data: Vec::new(),
                metadata: Vec::new(),
                cache_id: CacheId::from_u64(7),
                sample_id: SampleId::from_position(1)?,
            },
        }))?;
    }
    let error = iterator
        .next()
        .ok_or(CacheError::WorkerFailed)?
        .err()
        .ok_or(CacheError::WorkerFailed)?;
    assert!(error.to_string().contains("duplicate result sequence 1"));
    assert!(iterator.cancelled.load(Ordering::Acquire));
    assert!(iterator.next().is_none());
    Ok(())
}

/// Missing physical populations and cache-offset entries return errors instead of indexing.
#[test]
fn invalid_physical_locations_and_plans_are_rejected() -> CacheResult<()> {
    assert!(locate_physical_sample(&[], &[], 0).is_err());
    assert!(locate_physical_sample(&[], &[0], 0).is_err());
    assert!(
        load_sample_by_identity(&[], &std::collections::HashMap::from([(7, 0)]), 7, 0).is_err()
    );
    assert_eq!(wrapping_add_mod(4, usize::MAX, 0), None);
    let mut iterator = completed_window(3)?;
    iterator.scheduled = 3;
    iterator.plan = EpochPlan::Sequential {
        len: 0,
        start: 0,
        population_len: 0,
        physical_indices: None,
    };
    assert!(iterator.schedule_next().is_err());
    Ok(())
}

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

/// Preserve the actual worker cause while identifying the reader result boundary.
#[test]
fn failed_worker_reports_reader_stage() -> CacheResult<()> {
    let mut iterator = completed_window(3)?;
    for _ in 0..3 {
        iterator.output.recv()??;
    }
    iterator.output_sender.send(Err(CacheError::NativeFailure(
        "test worker cause".to_string(),
    )))?;
    let error = iterator
        .next()
        .ok_or_else(|| CacheError::InvalidInput("missing reader error".to_string()))?;
    let error = error
        .err()
        .ok_or_else(|| CacheError::InvalidInput("reader unexpectedly succeeded".to_string()))?;
    assert!(error.to_string().contains("receive reader worker result"));
    assert!(error.to_string().contains("test worker cause"));
    assert!(iterator.cancelled.load(Ordering::Acquire));
    assert!(iterator.next().is_none());
    Ok(())
}
