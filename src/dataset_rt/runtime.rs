use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{bounded, select, Receiver, Sender};

use crate::storage::LoadedCache;
use crate::types::{
    CacheError, CacheId, CacheResult, LoadedSample, NumWorkers, PhysicalSample, PrefetchSize,
    SampleId,
};

struct PlannedTask {
    sequence: usize,
    physical_index: usize,
}

struct LoadedResult {
    sequence: usize,
    result: CacheResult<LoadedSample>,
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
        physical_samples: Arc<Vec<PhysicalSample>>,
        plan: Vec<usize>,
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
            physical_samples,
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
    plan: Vec<usize>,
    cancel: Receiver<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for (sequence, physical_index) in plan.into_iter().enumerate() {
            let task = PlannedTask {
                sequence,
                physical_index,
            };
            select! {
                send(task_sender, task) -> result => {
                    if result.is_err() {
                        break;
                    }
                }
                recv(cancel) -> _ => break,
            }
        }
    })
}

fn spawn_workers(
    caches: Arc<Vec<LoadedCache>>,
    physical_samples: Arc<Vec<PhysicalSample>>,
    task_receiver: Receiver<PlannedTask>,
    output_sender: Sender<LoadedResult>,
    num_workers: NumWorkers,
    cancel: Receiver<()>,
) -> Vec<thread::JoinHandle<()>> {
    (0..num_workers.as_usize())
        .map(|_| {
            spawn_worker(
                caches.clone(),
                physical_samples.clone(),
                task_receiver.clone(),
                output_sender.clone(),
                cancel.clone(),
            )
        })
        .collect()
}

fn spawn_worker(
    caches: Arc<Vec<LoadedCache>>,
    physical_samples: Arc<Vec<PhysicalSample>>,
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
        let result = load_planned_sample(&caches, &physical_samples, task.physical_index);
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
    physical_samples: &[PhysicalSample],
    physical_index: usize,
) -> CacheResult<LoadedSample> {
    let physical_sample = physical_samples.get(physical_index).ok_or_else(|| {
        CacheError::InvalidCache(format!("physical sample {physical_index} is out of range"))
    })?;
    let cache = caches
        .get(physical_sample.cache_id.as_u64() as usize)
        .ok_or_else(|| {
            CacheError::InvalidCache(format!(
                "cache id {} is out of range",
                physical_sample.cache_id.as_u64()
            ))
        })?;
    let sample = cache.read_sample(physical_sample.sample_id.as_u64() as usize)?;
    Ok(LoadedSample {
        data: sample.data,
        metadata: sample.metadata,
        cache_id: CacheId::from_position(physical_sample.cache_id.as_u64() as usize)?,
        sample_id: SampleId::from_position(physical_sample.sample_id.as_u64() as usize)?,
    })
}
