use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::types::{CacheError, CacheResult, NumWorkers};

type Job = Box<dyn FnOnce() + Send + 'static>;

const QUEUED_JOBS_PER_WORKER: usize = 4;

pub struct WorkerPool {
    sender: Sender<Job>,
}

impl WorkerPool {
    /// Create one fixed worker set owned by a `DatasetRuntime` instance.
    pub fn new(num_workers: NumWorkers) -> CacheResult<Arc<Self>> {
        let worker_count = num_workers.as_usize();
        let queue_capacity = worker_count.saturating_mul(QUEUED_JOBS_PER_WORKER).max(1);
        let (sender, receiver) = bounded(queue_capacity);

        for worker_index in 0..worker_count {
            spawn_worker(worker_index, receiver.clone())?;
        }

        Ok(Arc::new(Self { sender }))
    }

    /// Submit one finite job and guarantee one result even if its implementation unwinds.
    pub fn submit<T>(
        &self,
        result_sender: Sender<CacheResult<T>>,
        job: impl FnOnce() -> CacheResult<T> + Send + 'static,
    ) -> CacheResult<()>
    where
        T: Send + 'static,
    {
        let wrapped = move || {
            let result = catch_unwind(AssertUnwindSafe(job))
                .unwrap_or_else(|_| Err(CacheError::WorkerFailed));
            let _ = result_sender.send(result);
        };
        self.sender
            .send(Box::new(wrapped))
            .map_err(|_| CacheError::WorkerFailed)
    }
}

/// Start one reusable runtime worker and surface thread creation failures.
fn spawn_worker(worker_index: usize, receiver: Receiver<Job>) -> CacheResult<()> {
    let thread_name = format!("dataset-rt-worker-{worker_index}");
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            for job in receiver {
                job();
            }
        })
        .map(|_| ())
        .map_err(|_| CacheError::WorkerFailed)
}
