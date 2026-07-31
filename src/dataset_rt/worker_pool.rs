use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::types::{CacheError, CacheResult};

type Job = Box<dyn FnOnce() + Send + 'static>;

const QUEUED_JOBS_PER_WORKER: usize = 4;

struct WorkerPool {
    sender: Sender<Job>,
    worker_count: usize,
}

impl WorkerPool {
    /// Create the process-wide fixed worker set and its bounded submission queue.
    fn new(worker_count: usize) -> Result<Self, String> {
        let queue_capacity = worker_count.saturating_mul(QUEUED_JOBS_PER_WORKER).max(1);
        let (sender, receiver) = bounded(queue_capacity);

        for worker_index in 0..worker_count {
            spawn_worker(worker_index, receiver.clone()).map_err(|error| error.to_string())?;
        }

        Ok(Self {
            sender,
            worker_count,
        })
    }
}

static GLOBAL_POOL: OnceLock<Result<WorkerPool, String>> = OnceLock::new();

/// Configure the immutable process-wide thread count and reject inconsistent calls.
pub fn configure(num_workers: usize) -> CacheResult<()> {
    let configured = GLOBAL_POOL.get_or_init(|| WorkerPool::new(num_workers));
    let pool = configured.as_ref().map_err(|_| CacheError::WorkerFailed)?;
    if pool.worker_count != num_workers {
        return Err(CacheError::InvalidInput(format!(
            "global worker pool uses {} workers; this operation requested {num_workers}",
            pool.worker_count
        )));
    }
    Ok(())
}

/// Submit one finite job and guarantee one result even if its implementation unwinds.
pub fn submit<T>(
    result_sender: Sender<CacheResult<T>>,
    job: impl FnOnce() -> CacheResult<T> + Send + 'static,
) -> CacheResult<()>
where
    T: Send + 'static,
{
    let pool = GLOBAL_POOL.get().ok_or_else(|| {
        CacheError::InvalidInput("global worker pool was not initialized".to_string())
    })?;
    let pool = pool.as_ref().map_err(|_| CacheError::WorkerFailed)?;
    let wrapped = move || {
        let result =
            catch_unwind(AssertUnwindSafe(job)).unwrap_or_else(|_| Err(CacheError::WorkerFailed));
        let _ = result_sender.send(result);
    };
    pool.sender
        .send(Box::new(wrapped))
        .map_err(|_| CacheError::WorkerFailed)
}

/// Run finite jobs from the global queue; jobs must never submit nested pool work.
fn spawn_worker(worker_index: usize, receiver: Receiver<Job>) -> std::io::Result<()> {
    let thread_name = format!("dataset-rt-worker-{worker_index}");
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            for job in receiver {
                job();
            }
        })
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::configure;
    use crate::types::CacheError;

    #[test]
    fn global_pool_rejects_inconsistent_worker_counts() {
        assert!(configure(2).is_ok());
        assert!(matches!(configure(3), Err(CacheError::InvalidInput(_))));
    }
}
