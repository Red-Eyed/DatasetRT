//! Bounded FIFO transport whose waits use channel-local synchronization.
//!
//! Fresh runtimes constructed after fork must not park the calling thread on
//! a libdispatch semaphore inherited through Rust's thread-local parker.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::types::{CacheError, CacheResult};

struct State<T> {
    queue: VecDeque<T>,
    senders_open: bool,
    receivers_open: bool,
}

struct Channel<T> {
    capacity: usize,
    state: Mutex<State<T>>,
    readable: Condvar,
    writable: Condvar,
}

struct SenderOwner<T> {
    channel: Arc<Channel<T>>,
}
struct ReceiverOwner<T> {
    channel: Arc<Channel<T>>,
}

/// Cloned senders keep the channel open until the final sender is dropped.
pub(crate) struct Sender<T> {
    owner: Arc<SenderOwner<T>>,
}

/// Cloned receivers share one FIFO and jointly own its consumption lifetime.
pub(crate) struct Receiver<T> {
    owner: Arc<ReceiverOwner<T>>,
}

/// Timed waits distinguish signal-check intervals from terminal channel failures.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RecvTimeoutError {
    Timeout,
    Disconnected,
}

/// Allocate a strictly bounded channel; rendezvous channels are not supported.
pub(crate) fn bounded<T>(capacity: usize) -> CacheResult<(Sender<T>, Receiver<T>)> {
    if capacity == 0 {
        return Err(CacheError::InvalidInput(
            "channel capacity must be positive".to_string(),
        ));
    }
    let channel = Arc::new(Channel {
        capacity,
        state: Mutex::new(State {
            queue: VecDeque::with_capacity(capacity),
            senders_open: true,
            receivers_open: true,
        }),
        readable: Condvar::new(),
        writable: Condvar::new(),
    });
    Ok((
        Sender {
            owner: Arc::new(SenderOwner {
                channel: channel.clone(),
            }),
        },
        Receiver {
            owner: Arc::new(ReceiverOwner { channel }),
        },
    ))
}

impl<T> Clone for Sender<T> {
    /// Share producer lifetime without requiring payloads to be cloneable.
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
        }
    }
}

impl<T> Clone for Receiver<T> {
    /// Share consumer lifetime without allocating another queue.
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
        }
    }
}

impl<T> Sender<T> {
    /// Apply backpressure until space is available or every receiver disconnects.
    pub(crate) fn send(&self, value: T) -> CacheResult<()> {
        let channel = &self.owner.channel;
        let guard = channel.state.lock().map_err(|_| CacheError::WorkerFailed)?;
        let mut guard = channel
            .writable
            .wait_while(guard, |state| {
                state.receivers_open && state.queue.len() == channel.capacity
            })
            .map_err(|_| CacheError::WorkerFailed)?;
        if !guard.receivers_open {
            return Err(CacheError::WorkerFailed);
        }
        guard.queue.push_back(value);
        channel.readable.notify_one();
        Ok(())
    }
}

impl<T> Receiver<T> {
    /// Drain queued values before reporting that every sender has disconnected.
    pub(crate) fn recv(&self) -> CacheResult<T> {
        let channel = &self.owner.channel;
        let guard = channel.state.lock().map_err(|_| CacheError::WorkerFailed)?;
        let mut guard = channel
            .readable
            .wait_while(guard, |state| state.senders_open && state.queue.is_empty())
            .map_err(|_| CacheError::WorkerFailed)?;
        let value = guard.queue.pop_front().ok_or(CacheError::WorkerFailed)?;
        channel.writable.notify_one();
        Ok(value)
    }

    /// Bound the entire wait interval, including spurious wakeups, by one deadline.
    pub(crate) fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvTimeoutError> {
        let channel = &self.owner.channel;
        let started = Instant::now();
        let guard = channel
            .state
            .lock()
            .map_err(|_| RecvTimeoutError::Disconnected)?;
        let (mut guard, _) = channel
            .readable
            .wait_timeout_while(guard, timeout.saturating_sub(started.elapsed()), |state| {
                state.senders_open && state.queue.is_empty()
            })
            .map_err(|_| RecvTimeoutError::Disconnected)?;
        if let Some(value) = guard.queue.pop_front() {
            channel.writable.notify_one();
            return Ok(value);
        }
        if !guard.senders_open {
            return Err(RecvTimeoutError::Disconnected);
        }
        Err(RecvTimeoutError::Timeout)
    }
}

impl<T> Drop for SenderOwner<T> {
    /// Wake receivers only when the last producer handle disappears.
    fn drop(&mut self) {
        let mut guard = match self.channel.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.senders_open = false;
        self.channel.readable.notify_all();
    }
}

impl<T> Drop for ReceiverOwner<T> {
    /// Release queued payloads outside the lock and wake backpressured senders.
    fn drop(&mut self) {
        let mut guard = match self.channel.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.receivers_open = false;
        let queued = std::mem::take(&mut guard.queue);
        self.channel.writable.notify_all();
        drop(guard);
        drop(queued);
    }
}

#[cfg(test)]
#[path = "channel/tests.rs"]
mod tests;
