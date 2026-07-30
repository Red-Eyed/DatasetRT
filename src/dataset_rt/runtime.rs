use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{bounded, select, Receiver, Sender};

use crate::sampling::EpochSampler;
use crate::storage::LoadedCache;
use crate::types::{
    CacheError, CacheId, CacheResult, LoadedSample, NumWorkers, PrefetchSize, SampleId,
};

struct PlannedTask {
    sequence: usize,
    physical_index: usize,
}

struct LoadedResult {
    sequence: usize,
    result: CacheResult<LoadedSample>,
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
}

pub struct RuntimeIterator {
    output: Receiver<LoadedResult>,
    cancel: Option<Sender<()>>,
    handles: Vec<thread::JoinHandle<()>>,
    pending: BTreeMap<usize, CacheResult<LoadedSample>>,
    next_sequence: usize,
    emitted: usize,
    total: usize,
}

impl RuntimeIterator {
    pub fn start(
        caches: Arc<Vec<LoadedCache>>,
        cache_offsets: Arc<Vec<usize>>,
        plan: EpochPlan,
        prefetch_size: PrefetchSize,
        num_workers: NumWorkers,
    ) -> Self {
        let total = plan.len();
        let (task_sender, task_receiver) = bounded(prefetch_size.as_usize());
        let (output_sender, output_receiver) = bounded(prefetch_size.as_usize());
        let (cancel_sender, cancel_receiver) = bounded(1);
        let mut handles = Vec::with_capacity(num_workers.as_usize() + 1);

        handles.push(spawn_scheduler(task_sender, plan, cancel_receiver.clone()));
        handles.extend(spawn_workers(
            caches,
            cache_offsets,
            task_receiver,
            output_sender,
            num_workers,
            cancel_receiver,
        ));

        Self {
            output: output_receiver,
            cancel: Some(cancel_sender),
            handles,
            pending: BTreeMap::new(),
            next_sequence: 0,
            emitted: 0,
            total,
        }
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
                return Some(result);
            }

            let loaded = match self.output.recv() {
                Ok(loaded) => loaded,
                Err(_) => {
                    self.emitted = self.total;
                    return Some(Err(CacheError::WorkerFailed));
                }
            };

            // Read workers may finish out of order; Python observes the planned
            // sampler order, so completed future samples wait in a small buffer.
            if loaded.sequence == self.next_sequence {
                self.next_sequence += 1;
                self.emitted += 1;
                return Some(loaded.result);
            }

            if self
                .pending
                .insert(loaded.sequence, loaded.result)
                .is_some()
            {
                self.emitted = self.total;
                return Some(Err(CacheError::WorkerFailed));
            }
        }
    }
}

impl Drop for RuntimeIterator {
    fn drop(&mut self) {
        let _ = self.cancel.take();
        while let Some(handle) = self.handles.pop() {
            let _ = handle.join();
        }
    }
}

fn spawn_scheduler(
    task_sender: Sender<PlannedTask>,
    plan: EpochPlan,
    cancel: Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || match plan {
        EpochPlan::PhysicalOrder { len } => schedule_physical_order(task_sender, len, cancel),
        EpochPlan::Shuffled(sampler) => schedule_shuffled_order(task_sender, *sampler, cancel),
    })
}

/// Schedule physical-order reads directly from the range of sample indexes.
fn schedule_physical_order(task_sender: Sender<PlannedTask>, len: usize, cancel: Receiver<()>) {
    for physical_index in 0..len {
        if !send_planned_task(&task_sender, physical_index, physical_index, &cancel) {
            break;
        }
    }
}

/// Schedule replacement-sampled shuffled reads from the streaming epoch sampler.
fn schedule_shuffled_order(
    task_sender: Sender<PlannedTask>,
    sampler: EpochSampler,
    cancel: Receiver<()>,
) {
    for (sequence, physical_index) in sampler.enumerate() {
        if !send_planned_task(&task_sender, sequence, physical_index, &cancel) {
            break;
        }
    }
}

/// Send one planned task while respecting iterator cancellation.
fn send_planned_task(
    task_sender: &Sender<PlannedTask>,
    sequence: usize,
    physical_index: usize,
    cancel: &Receiver<()>,
) -> bool {
    let task = PlannedTask {
        sequence,
        physical_index,
    };
    select! {
        send(task_sender, task) -> result => result.is_ok(),
        recv(cancel) -> _ => false,
    }
}

fn spawn_workers(
    caches: Arc<Vec<LoadedCache>>,
    cache_offsets: Arc<Vec<usize>>,
    task_receiver: Receiver<PlannedTask>,
    output_sender: Sender<LoadedResult>,
    num_workers: NumWorkers,
    cancel: Receiver<()>,
) -> Vec<thread::JoinHandle<()>> {
    (0..num_workers.as_usize())
        .map(|_| {
            spawn_worker(
                caches.clone(),
                cache_offsets.clone(),
                task_receiver.clone(),
                output_sender.clone(),
                cancel.clone(),
            )
        })
        .collect()
}

fn spawn_worker(
    caches: Arc<Vec<LoadedCache>>,
    cache_offsets: Arc<Vec<usize>>,
    task_receiver: Receiver<PlannedTask>,
    output_sender: Sender<LoadedResult>,
    cancel: Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        let task = select! {
            recv(task_receiver) -> message => match message {
                Ok(task) => task,
                Err(_) => break,
            },
            recv(cancel) -> _ => break,
        };
        let result = load_planned_sample(&caches, &cache_offsets, task.physical_index);
        let loaded = LoadedResult {
            sequence: task.sequence,
            result,
        };
        select! {
            send(output_sender, loaded) -> result => {
                if result.is_err() {
                    break;
                }
            }
            recv(cancel) -> _ => break,
        }
    })
}

fn load_planned_sample(
    caches: &[LoadedCache],
    cache_offsets: &[usize],
    physical_index: usize,
) -> CacheResult<LoadedSample> {
    let (cache_index, sample_index) =
        locate_physical_sample(caches, cache_offsets, physical_index)?;
    let cache = caches
        .get(cache_index)
        .ok_or_else(|| CacheError::InvalidCache("cache id is out of range".to_string()))?;
    let sample = cache.read_sample(sample_index)?;
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
