//! Exercise channel capacity, disconnection, wakeups and payload ownership.

use std::sync::{mpsc, Arc, Barrier};
use std::thread;
use std::time::Duration;

use super::{bounded, RecvTimeoutError};
use crate::types::{CacheError, CacheResult};

const DEADLINE: Duration = Duration::from_secs(5);

/// Closed producers leave already queued values available in FIFO order.
#[test]
fn drains_fifo_after_last_sender_drop() -> CacheResult<()> {
    let (sender, receiver) = bounded(2)?;
    sender.send(1)?;
    sender.send(2)?;
    let clone = sender.clone();
    drop(sender);
    assert_eq!(receiver.recv()?, 1);
    assert_eq!(receiver.recv()?, 2);
    assert_eq!(
        receiver.recv_timeout(Duration::ZERO),
        Err(RecvTimeoutError::Timeout)
    );
    drop(clone);
    assert_eq!(
        receiver.recv_timeout(DEADLINE),
        Err(RecvTimeoutError::Disconnected)
    );
    assert!(receiver.recv().is_err());
    Ok(())
}

/// A receiver clone retains consumption rights after the original is dropped.
#[test]
fn receiver_clones_share_one_queue() -> CacheResult<()> {
    let (sender, receiver) = bounded(2)?;
    let clone = receiver.clone();
    sender.send(1)?;
    sender.send(2)?;
    assert_eq!(receiver.recv()?, 1);
    drop(receiver);
    assert_eq!(clone.recv()?, 2);
    drop(clone);
    assert!(sender.send(3).is_err());
    Ok(())
}

/// Full queues hold back producers until a consumer releases a credit.
#[test]
fn capacity_applies_backpressure() -> CacheResult<()> {
    let (sender, receiver) = bounded(1)?;
    sender.send(1)?;
    let (started, start) = mpsc::channel();
    let (finished, finish) = mpsc::channel();
    let producer = thread::spawn(move || {
        let _ = started.send(());
        let result = sender.send(2);
        let _ = finished.send(result);
    });
    start
        .recv_timeout(DEADLINE)
        .map_err(|_| CacheError::WorkerFailed)?;
    assert!(matches!(
        finish.recv_timeout(Duration::from_millis(20)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(receiver.recv()?, 1);
    finish
        .recv_timeout(DEADLINE)
        .map_err(|_| CacheError::WorkerFailed)??;
    assert_eq!(receiver.recv()?, 2);
    producer.join().map_err(|_| CacheError::WorkerFailed)?;
    Ok(())
}

/// Cancellation releases a sender stalled behind a full result queue.
#[test]
fn last_receiver_drop_wakes_sender() -> CacheResult<()> {
    let (sender, receiver) = bounded(1)?;
    sender.send(1)?;
    let (started, start) = mpsc::channel();
    let (finished, finish) = mpsc::channel();
    let producer = thread::spawn(move || {
        let _ = started.send(());
        let _ = finished.send(sender.send(2));
    });
    start
        .recv_timeout(DEADLINE)
        .map_err(|_| CacheError::WorkerFailed)?;
    drop(receiver);
    assert!(finish
        .recv_timeout(DEADLINE)
        .map_err(|_| CacheError::WorkerFailed)?
        .is_err());
    producer.join().map_err(|_| CacheError::WorkerFailed)?;
    Ok(())
}

/// Every blocked consumer is released when the final producer disappears.
#[test]
fn last_sender_drop_wakes_all_receivers() -> CacheResult<()> {
    let (sender, receiver) = bounded::<u8>(1)?;
    let ready = Arc::new(Barrier::new(3));
    let (finished, finish) = mpsc::channel();
    let mut consumers = Vec::new();
    for _ in 0..2 {
        let receiver = receiver.clone();
        let ready = ready.clone();
        let finished = finished.clone();
        consumers.push(thread::spawn(move || {
            ready.wait();
            let _ = finished.send(receiver.recv_timeout(DEADLINE));
        }));
    }
    ready.wait();
    drop(sender);
    for _ in 0..2 {
        assert_eq!(
            finish
                .recv_timeout(DEADLINE)
                .map_err(|_| CacheError::WorkerFailed)?,
            Err(RecvTimeoutError::Disconnected)
        );
    }
    for consumer in consumers {
        consumer.join().map_err(|_| CacheError::WorkerFailed)?;
    }
    Ok(())
}

/// Timed receives distinguish expiration and eventual arrival without consuming credits.
#[test]
fn timeout_does_not_close_channel() -> CacheResult<()> {
    let (sender, receiver) = bounded(1)?;
    assert_eq!(
        receiver.recv_timeout(Duration::from_millis(1)),
        Err(RecvTimeoutError::Timeout)
    );
    sender.send(7)?;
    assert_eq!(receiver.recv_timeout(Duration::ZERO), Ok(7));
    Ok(())
}

/// Multiple consumers deliver each submitted value once without expanding capacity.
#[test]
fn concurrent_transport_preserves_all_values() -> CacheResult<()> {
    let (sender, receiver) = bounded(3)?;
    let (finished, finish) = mpsc::channel();
    let mut consumers = Vec::new();
    for _ in 0..4 {
        let receiver = receiver.clone();
        let finished = finished.clone();
        consumers.push(thread::spawn(move || {
            let mut sum = 0_u64;
            let mut count = 0_u64;
            while let Ok(value) = receiver.recv() {
                sum += value;
                count += 1;
            }
            let _ = finished.send((sum, count));
        }));
    }
    for value in 0..1000_u64 {
        sender.send(value)?;
        let state = sender
            .owner
            .channel
            .state
            .lock()
            .map_err(|_| CacheError::WorkerFailed)?;
        assert!(state.queue.len() <= 3);
    }
    drop(sender);
    let mut total = (0, 0);
    for _ in 0..4 {
        let (sum, count) = finish
            .recv_timeout(DEADLINE)
            .map_err(|_| CacheError::WorkerFailed)?;
        total.0 += sum;
        total.1 += count;
    }
    assert_eq!(total, (499500, 1000));
    for consumer in consumers {
        consumer.join().map_err(|_| CacheError::WorkerFailed)?;
    }
    Ok(())
}

/// Invalid capacities fail before queue allocation or worker creation.
#[test]
fn rejects_zero_capacity() {
    assert!(bounded::<u8>(0).is_err());
}

/// User payload destructors must never run while the queue mutex is held.
#[test]
fn queued_payloads_drop_outside_lock() -> CacheResult<()> {
    struct Payload {
        channel: Arc<super::Channel<Payload>>,
        finished: mpsc::Sender<bool>,
    }
    impl Drop for Payload {
        /// Check the actual queue lock at destruction without risking deadlock.
        fn drop(&mut self) {
            let lock_available = self.channel.state.try_lock().is_ok();
            let _ = self.finished.send(lock_available);
        }
    }
    let (sender, receiver) = bounded(1)?;
    let (finished, finish) = mpsc::channel();
    sender.send(Payload {
        channel: sender.owner.channel.clone(),
        finished,
    })?;
    drop(receiver);
    assert!(finish
        .recv_timeout(DEADLINE)
        .map_err(|_| CacheError::WorkerFailed)?);
    Ok(())
}
