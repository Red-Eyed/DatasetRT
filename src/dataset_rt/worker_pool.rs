use std::sync::Arc;
use std::thread;

use crate::channel::{bounded, Receiver, Sender};

use crate::types::{native_result, CacheError, CacheResult, NumWorkers};

type Job = Box<dyn FnOnce() + Send + 'static>;

const QUEUED_JOBS_PER_WORKER: usize = 4;

pub struct WorkerPool {
    sender: Sender<Job>,
}

impl WorkerPool {
    /// Create one fixed worker set owned by a `DatasetRuntime` instance.
    pub fn new(num_workers: NumWorkers) -> CacheResult<Arc<Self>> {
        let worker_count = num_workers.as_usize();
        let queue_capacity = worker_count
            .checked_mul(QUEUED_JOBS_PER_WORKER)
            .ok_or_else(|| {
                CacheError::InvalidInput("worker queue capacity overflowed usize".to_string())
            })?;
        let (sender, receiver) = bounded(queue_capacity)?;

        for worker_index in 0..worker_count {
            spawn_worker(worker_index, receiver.clone())?;
        }

        Ok(Arc::new(Self { sender }))
    }

    /// Submit one finite job and guarantee one result even if its implementation unwinds.
    pub(crate) fn submit<T>(
        &self,
        result_sender: Sender<CacheResult<T>>,
        job: impl FnOnce() -> CacheResult<T> + Send + 'static,
    ) -> CacheResult<()>
    where
        T: Send + 'static,
    {
        let wrapped = move || {
            let result = native_result("runtime worker", job);
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
            while let Ok(job) = receiver.recv() {
                job();
            }
        })
        .map(|_| ())
        .map_err(|error| {
            CacheError::NativeFailure(format!(
                "cannot start runtime worker {worker_index}: {error}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One unwinding job reports its message without preventing subsequent work.
    #[test]
    fn panic_diagnostic_survives_and_pool_remains_usable() -> CacheResult<()> {
        let pool = WorkerPool::new(NumWorkers::new(1)?)?;
        let (sender, receiver) = bounded(1)?;
        pool.submit(sender.clone(), || -> CacheResult<usize> {
            std::panic::resume_unwind(Box::new("dependency failure".to_string()))
        })?;
        let result = receiver.recv()?;
        assert!(
            matches!(result, Err(CacheError::NativeFailure(message)) if message.contains("dependency failure"))
        );
        pool.submit(sender, || Ok(7_usize))?;
        assert_eq!(receiver.recv()??, 7);
        Ok(())
    }
}
