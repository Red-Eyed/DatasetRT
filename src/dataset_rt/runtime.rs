use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::sampling::EpochSampler;
use crate::storage::{LoadedCache, ShardReaderCache};
use crate::types::{
    CacheError, CacheId, CacheResult, LoadedSample, NumWorkers, PrefetchSize, SampleId,
};
use crate::worker_pool::WorkerPool;

thread_local! {
    static SHARD_READERS: RefCell<ShardReaderCache> = RefCell::new(ShardReaderCache::new());
}

struct LoadedResult {
    sequence: usize,
    result: LoadedSample,
}

pub enum EpochPlan {
    PhysicalOrder { len: usize },
    Shuffled(Box<EpochSampler>),
}

impl EpochPlan {
    /// Return the number of samples the plan will emit without materializing physical order.
    fn len(&self) -> usize {
        match self {
            Self::PhysicalOrder { len } => *len,
            Self::Shuffled(sampler) => sampler.len(),
        }
    }

    /// Return the next physical sample without materializing an epoch-wide task list.
    fn next_physical_index(&mut self, sequence: usize) -> Option<usize> {
        match self {
            Self::PhysicalOrder { len } => (sequence < *len).then_some(sequence),
            Self::Shuffled(sampler) => sampler.next(),
        }
    }
}

pub struct RuntimeIterator {
    pool: Arc<WorkerPool>,
    caches: Arc<Vec<LoadedCache>>,
    cache_offsets: Arc<Vec<usize>>,
    plan: EpochPlan,
    output_sender: Sender<CacheResult<LoadedResult>>,
    output: Receiver<CacheResult<LoadedResult>>,
    cancelled: Arc<AtomicBool>,
    pending: BTreeMap<usize, LoadedSample>,
    parallelism: usize,
    scheduled: usize,
    in_flight: usize,
    next_sequence: usize,
    emitted: usize,
    total: usize,
}

impl RuntimeIterator {
    pub fn start(
        pool: Arc<WorkerPool>,
        caches: Arc<Vec<LoadedCache>>,
        cache_offsets: Arc<Vec<usize>>,
        plan: EpochPlan,
        prefetch_size: PrefetchSize,
        num_workers: NumWorkers,
    ) -> CacheResult<Self> {
        let total = plan.len();
        let parallelism = num_workers
            .as_usize()
            .min(prefetch_size.as_usize())
            .min(total);
        let (output_sender, output) = bounded(parallelism);
        let mut iterator = Self {
            pool,
            caches,
            cache_offsets,
            output_sender,
            output,
            plan,
            cancelled: Arc::new(AtomicBool::new(false)),
            pending: BTreeMap::new(),
            parallelism,
            scheduled: 0,
            in_flight: 0,
            next_sequence: 0,
            emitted: 0,
            total,
        };
        iterator.schedule_available()?;
        Ok(iterator)
    }

    /// Keep only the configured finite number of sample reads active for this iterator.
    fn schedule_available(&mut self) -> CacheResult<()> {
        while self.in_flight < self.parallelism && self.scheduled < self.total {
            self.schedule_next()?;
        }
        Ok(())
    }

    /// Submit one finite read job while preserving its planned output sequence.
    fn schedule_next(&mut self) -> CacheResult<()> {
        let sequence = self.scheduled;
        let physical_index = self
            .plan
            .next_physical_index(sequence)
            .ok_or(CacheError::WorkerFailed)?;
        let caches = self.caches.clone();
        let cache_offsets = self.cache_offsets.clone();
        let output_sender = self.output_sender.clone();
        let cancelled = self.cancelled.clone();

        self.pool.submit(output_sender, move || {
            if cancelled.load(Ordering::Acquire) {
                return Err(CacheError::WorkerFailed);
            }
            let result = SHARD_READERS.with(|readers| {
                load_planned_sample(
                    caches.as_ref(),
                    cache_offsets.as_ref(),
                    physical_index,
                    &mut readers.borrow_mut(),
                )
            })?;
            if cancelled.load(Ordering::Acquire) {
                return Err(CacheError::WorkerFailed);
            }
            Ok(LoadedResult { sequence, result })
        })?;

        self.scheduled += 1;
        self.in_flight += 1;
        Ok(())
    }
}

impl Iterator for RuntimeIterator {
    type Item = CacheResult<LoadedSample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.emitted >= self.total {
            return None;
        }

        loop {
            if let Some(result) = self.pending.remove(&self.next_sequence) {
                self.next_sequence += 1;
                self.emitted += 1;
                return Some(Ok(result));
            }

            let loaded = match self.output.recv() {
                Ok(Ok(loaded)) => loaded,
                Ok(Err(error)) => {
                    self.emitted = self.total;
                    self.cancelled.store(true, Ordering::Release);
                    return Some(Err(error));
                }
                Err(_) => {
                    self.emitted = self.total;
                    self.cancelled.store(true, Ordering::Release);
                    return Some(Err(CacheError::WorkerFailed));
                }
            };
            let Some(in_flight) = self.in_flight.checked_sub(1) else {
                self.emitted = self.total;
                self.cancelled.store(true, Ordering::Release);
                return Some(Err(CacheError::WorkerFailed));
            };
            self.in_flight = in_flight;
            if self.schedule_available().is_err() {
                self.emitted = self.total;
                self.cancelled.store(true, Ordering::Release);
                return Some(Err(CacheError::WorkerFailed));
            }

            // Read workers may finish out of order; Python observes the planned
            // sampler order, so completed future samples wait in a small buffer.
            if loaded.sequence == self.next_sequence {
                self.next_sequence += 1;
                self.emitted += 1;
                return Some(Ok(loaded.result));
            }

            if self
                .pending
                .insert(loaded.sequence, loaded.result)
                .is_some()
            {
                self.emitted = self.total;
                self.cancelled.store(true, Ordering::Release);
                return Some(Err(CacheError::WorkerFailed));
            }
        }
    }
}

impl Drop for RuntimeIterator {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

fn load_planned_sample(
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    physical_index: usize,
    shard_readers: &mut ShardReaderCache,
) -> CacheResult<LoadedSample> {
    let (cache_index, sample_index) =
        locate_physical_sample(caches, cache_offsets, physical_index)?;
    let cache = caches
        .get(cache_index)
        .ok_or_else(|| CacheError::InvalidCache("cache id is out of range".to_string()))?;
    let sample = cache.read_sample_with_cache(sample_index, shard_readers)?;
    Ok(LoadedSample {
        data: sample.data,
        metadata: sample.metadata,
        cache_id: CacheId::from_position(cache_index)?,
        sample_id: SampleId::from_position(sample_index)?,
    })
}

/// Resolve a compact physical index into cache-local identity for sample materialization.
fn locate_physical_sample(
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    physical_index: usize,
) -> CacheResult<(usize, usize)> {
    let cache_index = cache_offsets
        .partition_point(|offset| *offset <= physical_index)
        .checked_sub(1)
        .ok_or_else(|| {
            CacheError::InvalidCache(format!("physical sample {physical_index} is out of range"))
        })?;
    let cache_offset = cache_offsets
        .get(cache_index)
        .copied()
        .ok_or_else(|| CacheError::InvalidCache("cache offset is out of range".to_string()))?;
    let sample_index = physical_index.checked_sub(cache_offset).ok_or_else(|| {
        CacheError::InvalidCache(format!("physical sample {physical_index} is out of range"))
    })?;
    let cache = caches
        .get(cache_index)
        .ok_or_else(|| CacheError::InvalidCache("cache id is out of range".to_string()))?;
    if sample_index >= cache.sample_count() {
        return Err(CacheError::InvalidCache(format!(
            "physical sample {physical_index} is out of range"
        )));
    }
    Ok((cache_index, sample_index))
}
