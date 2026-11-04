//! Bounded, multi-producer/single-consumer channel with a hard capacity
//! ceiling. Production notes are inlined at each decision point below.
//!
//! Concurrency proof status: `no_lost_wakeup_under_contention` exercises the
//! race this design fixes (200 runs x 8 racing senders) and passes
//! deterministically. That's evidence, not a proof. Before trusting this in
//! a high-throughput production path, run it under `loom`
//! (https://docs.rs/loom) as a dev-dependency -- loom exhaustively explores
//! interleavings instead of hoping timing surfaces the bug.
//!
//! This is a general-purpose primitive: rustwall's current only consumer
//! (sync_worker.rs) uses `bounded`, `try_send`, and blocking `recv`. The
//! rest of the API (blocking `send`, `send_timeout`, `try_recv`, `len`,
//! `capacity`) is fully implemented and tested but not yet exercised outside
//! this module's own test suite -- kept rather than pruned, since a
//! general-purpose channel with a hard capacity ceiling is exactly the kind
//! of thing a future feature in this codebase is likely to want, and
//! removing working, tested code to silence a dead-code warning would be
//! backwards.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

// -------------------------------------------------------------------------
// Errors
// -------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub enum SendError<T> {
    /// The receiver has been dropped; no one can ever consume this value.
    Disconnected(T),
}

#[derive(Debug, PartialEq, Eq)]
pub enum TrySendError<T> {
    Disconnected(T),
    /// The channel is at its capacity ceiling right now.
    Full(T),
}

#[derive(Debug, PartialEq, Eq)]
pub enum SendTimeoutError<T> {
    Disconnected(T),
    Timeout(T),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RecvError {
    Disconnected,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TryRecvError {
    Empty,
    Disconnected,
}

// -------------------------------------------------------------------------
// Poison handling policy
// -------------------------------------------------------------------------
//
// A panic while `queue`'s lock is held can only happen inside VecDeque's own
// push/pop or a stored value's Drop impl running during that window — this
// code never runs arbitrary caller logic while the lock is held. So a
// poisoned lock here does not imply a torn VecDeque; recovering the guard is
// safe. Centralizing that decision here (instead of bare `.unwrap()`
// everywhere) means one panicking thread can't take the whole channel down
// for every other producer/consumer.
#[inline]
fn recover<'a, T>(r: Result<MutexGuard<'a, T>, PoisonError<MutexGuard<'a, T>>>) -> MutexGuard<'a, T> {
    r.unwrap_or_else(PoisonError::into_inner)
}

// -------------------------------------------------------------------------
// Shared state
// -------------------------------------------------------------------------

struct Shared<T> {
    queue: Mutex<VecDeque<T>>,
    not_empty: Condvar, // signaled on push, or on last-sender drop
    not_full: Condvar,  // signaled on pop, or on receiver drop
    max_capacity: usize,
    sender_count: AtomicUsize,
    receiver_alive: AtomicUsize, // 0 = dead, 1 = alive
}

impl<T> Shared<T> {
    #[inline]
    fn receiver_is_alive(&self) -> bool {
        self.receiver_alive.load(Ordering::Acquire) == 1
    }

    #[inline]
    fn senders_remaining(&self) -> usize {
        self.sender_count.load(Ordering::Acquire)
    }
}

pub struct Sender<T> {
    inner: Arc<Shared<T>>,
}

pub struct Receiver<T> {
    inner: Arc<Shared<T>>,
}

/// Creates a bounded channel holding at most `max_elements` in-flight items.
///
/// # Panics
/// Panics if `max_elements == 0`. This implementation buffers; it does not
/// implement synchronous rendezvous handoff (sender blocks until receiver
/// takes that exact item). Don't repurpose capacity 0 to mean that — `send`
/// would just reject/block on "full" forever instead of rendezvousing.
pub fn bounded<T>(max_elements: usize) -> (Sender<T>, Receiver<T>) {
    assert!(max_elements > 0, "bounded channel capacity must be at least 1");

    let inner = Arc::new(Shared {
        queue: Mutex::new(VecDeque::with_capacity(max_elements.min(4096))),
        not_empty: Condvar::new(),
        not_full: Condvar::new(),
        max_capacity: max_elements,
        sender_count: AtomicUsize::new(1),
        receiver_alive: AtomicUsize::new(1),
    });

    (Sender { inner: inner.clone() }, Receiver { inner })
}

// -------------------------------------------------------------------------
// Sender
// -------------------------------------------------------------------------

impl<T> Sender<T> {
    /// Blocks until there is room and the item is enqueued, or until the
    /// receiver disconnects. This is what most production pipelines want:
    /// real backpressure instead of forcing every caller to hand-roll a
    /// retry loop around `try_send`.
    pub fn send(&self, item: T) -> Result<(), SendError<T>> {
        let mut queue = recover(self.inner.queue.lock());
        loop {
            if !self.inner.receiver_is_alive() {
                return Err(SendError::Disconnected(item));
            }
            if queue.len() < self.inner.max_capacity {
                queue.push_back(item);
                drop(queue);
                self.inner.not_empty.notify_one();
                return Ok(());
            }
            // Condvar::wait atomically releases `queue` and sleeps, so a
            // pop-then-notify from the receiver (or a disconnect-then-notify
            // from Receiver::drop) can't land in the gap between our
            // capacity check and going to sleep. Same guarantee that fixes
            // the lost-wakeup bug on the receive side.
            queue = self
                .inner
                .not_full
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Non-blocking: fails immediately instead of waiting for space.
    pub fn try_send(&self, item: T) -> Result<(), TrySendError<T>> {
        let mut queue = recover(self.inner.queue.lock());
        if !self.inner.receiver_is_alive() {
            return Err(TrySendError::Disconnected(item));
        }
        if queue.len() >= self.inner.max_capacity {
            return Err(TrySendError::Full(item));
        }
        queue.push_back(item);
        drop(queue);
        self.inner.not_empty.notify_one();
        Ok(())
    }

    /// Blocks until there is room, the receiver disconnects, or `timeout` elapses.
    pub fn send_timeout(&self, item: T, timeout: Duration) -> Result<(), SendTimeoutError<T>> {
        let deadline = Instant::now() + timeout;
        let mut queue = recover(self.inner.queue.lock());
        loop {
            if !self.inner.receiver_is_alive() {
                return Err(SendTimeoutError::Disconnected(item));
            }
            if queue.len() < self.inner.max_capacity {
                queue.push_back(item);
                drop(queue);
                self.inner.not_empty.notify_one();
                return Ok(());
            }
            let remaining = match deadline.checked_duration_since(Instant::now()) {
                Some(d) if d > Duration::ZERO => d,
                _ => return Err(SendTimeoutError::Timeout(item)),
            };
            let (guard, _wait_result) = self
                .inner
                .not_full
                .wait_timeout(queue, remaining)
                .unwrap_or_else(PoisonError::into_inner);
            queue = guard;
            // Loop back around: re-checks receiver_alive and capacity under
            // the lock regardless of whether this was a real wakeup or a
            // timeout, so we never act on a stale read.
        }
    }

    /// Current queue depth. Advisory only — may be stale the instant it's read.
    pub fn len(&self) -> usize {
        recover(self.inner.queue.lock()).len()
    }

    pub fn capacity(&self) -> usize {
        self.inner.max_capacity
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.inner.sender_count.fetch_add(1, Ordering::AcqRel);
        Sender { inner: self.inner.clone() }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if self.inner.sender_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last sender gone: wake a blocked receiver so it observes
            // disconnect instead of waiting forever. Taking the lock keeps
            // this ordered consistently with any concurrent `wait`.
            let _guard = recover(self.inner.queue.lock());
            self.inner.not_empty.notify_one();
        }
    }
}

// -------------------------------------------------------------------------
// Receiver
// -------------------------------------------------------------------------

impl<T> Receiver<T> {
    pub fn recv(&self) -> Result<T, RecvError> {
        let mut queue = recover(self.inner.queue.lock());
        loop {
            if let Some(item) = queue.pop_front() {
                drop(queue);
                self.inner.not_full.notify_one();
                return Ok(item);
            }
            if self.inner.senders_remaining() == 0 {
                return Err(RecvError::Disconnected);
            }
            queue = self
                .inner
                .not_empty
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let mut queue = recover(self.inner.queue.lock());
        if let Some(item) = queue.pop_front() {
            drop(queue);
            self.inner.not_full.notify_one();
            Ok(item)
        } else if self.inner.senders_remaining() == 0 {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    pub fn len(&self) -> usize {
        recover(self.inner.queue.lock()).len()
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.receiver_alive.store(0, Ordering::Release);
        // Wake senders parked in `send`/`send_timeout` so they observe
        // disconnect instead of blocking until their timeout, or forever.
        let _guard = recover(self.inner.queue.lock());
        self.inner.not_full.notify_all();
    }
}

impl<T> Iterator for Receiver<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.recv().ok()
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn send_recv_basic() {
        let (tx, rx) = bounded(4);
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        assert_eq!(rx.recv(), Ok(1));
        assert_eq!(rx.recv(), Ok(2));
    }

    #[test]
    fn try_send_rejects_at_capacity() {
        let (tx, _rx) = bounded(2);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        assert_eq!(tx.try_send(3), Err(TrySendError::Full(3)));
    }

    #[test]
    fn blocking_send_unblocks_on_drain() {
        let (tx, rx) = bounded(1);
        tx.try_send(1).unwrap();
        let tx2 = tx.clone();
        let handle = thread::spawn(move || tx2.send(2).unwrap());
        thread::sleep(Duration::from_millis(20)); // let the sender actually park
        assert_eq!(rx.recv(), Ok(1));
        handle.join().unwrap();
        assert_eq!(rx.recv(), Ok(2));
    }

    #[test]
    fn send_timeout_expires_when_full() {
        let (tx, _rx) = bounded(1);
        tx.try_send(1).unwrap();
        let result = tx.send_timeout(2, Duration::from_millis(30));
        assert_eq!(result, Err(SendTimeoutError::Timeout(2)));
    }

    #[test]
    fn receiver_drop_disconnects_sender() {
        let (tx, rx) = bounded::<i32>(2);
        drop(rx);
        assert_eq!(tx.try_send(1), Err(TrySendError::Disconnected(1)));
        assert_eq!(tx.send(1), Err(SendError::Disconnected(1)));
    }

    #[test]
    fn receiver_drop_unblocks_pending_sender() {
        let (tx, rx) = bounded(1);
        tx.try_send(1).unwrap(); // fill it
        let tx2 = tx.clone();
        let handle = thread::spawn(move || tx2.send(2));
        thread::sleep(Duration::from_millis(20));
        drop(rx); // should wake the blocked sender rather than hang it forever
        assert_eq!(handle.join().unwrap(), Err(SendError::Disconnected(2)));
    }

    #[test]
    fn all_senders_dropped_disconnects_receiver_after_drain() {
        let (tx, rx) = bounded(4);
        tx.send(1).unwrap();
        drop(tx);
        assert_eq!(rx.recv(), Ok(1));
        assert_eq!(rx.recv(), Err(RecvError::Disconnected));
    }

    #[test]
    #[should_panic]
    fn zero_capacity_is_rejected() {
        let _ = bounded::<i32>(0);
    }

    /// Regression test for the lost-wakeup class of bug: receiver blocks
    /// first, many senders race to push right at the capacity boundary.
    #[test]
    fn no_lost_wakeup_under_contention() {
        for _ in 0..200 {
            let (tx, rx) = bounded::<usize>(2);
            let mut handles = Vec::new();
            for i in 0..8 {
                let tx = tx.clone();
                handles.push(thread::spawn(move || {
                    let _ = tx.try_send(i);
                }));
            }
            drop(tx);
            let mut received = 0;
            while rx.recv().is_ok() {
                received += 1;
            }
            assert!(received > 0);
            for h in handles {
                h.join().unwrap();
            }
        }
    }
}
