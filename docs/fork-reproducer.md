# macOS fork crash: dependency-free reproducer

The failure reduces to a used libdispatch semaphore inherited across `fork`.
The standalone [C program](../scripts/repro_macos_fork.c) requires only Apple's
command-line compiler and macOS system libraries. It has no DatasetRT, Python,
PyTorch, Polars, Crossbeam, Rust, cache files, or worker threads.

Build and run from the repository root:

```sh
cc -O2 -Wall -Wextra -Werror scripts/repro_macos_fork.c -o /tmp/repro_macos_fork
/tmp/repro_macos_fork
/tmp/repro_macos_fork --warm-parent
/tmp/repro_macos_fork --warm-parent --fresh-child
```

Verified on macOS 26.6.2 (25G83), ARM64, Apple clang 21.0.0:

| Case | Parent before fork | Child after fork | Result, five runs |
| --- | --- | --- | --- |
| Cold | Creates semaphore; does not wait | Waits on inherited semaphore | 5/5 exit 0 |
| Warm | Creates semaphore and performs a timed wait | Waits on inherited semaphore | 5/5 SIGTRAP (signal 5); supervisor exits 1 |
| Fresh child | Creates semaphore and performs a timed wait | Creates a new semaphore and waits on it | 5/5 exit 0 |

The failure is intentional in the second command. The parent waits for and reaps
the child, reports the signal, and returns a nonzero status. Each semaphore wait
has a one-millisecond timeout. The fresh-child case leaves the inherited object
untouched and uses `_exit` to avoid destructing inherited synchronization state.

## Why this explains the DatasetRT symptom

Apple's [semaphore implementation](https://github.com/apple/swift-corelibs-libdispatch/blob/main/src/semaphore.c)
initializes the backing semaphore lazily on the slow wait path. The
[Darwin backend](https://github.com/apple/swift-corelibs-libdispatch/blob/main/src/shims/lock.c)
creates a Mach semaphore and maps `KERN_INVALID_NAME` to the diagnostic
“Use-after-free of dispatch_semaphore_t or dispatch_group_t”. That diagnostic
therefore does not, by itself, prove an application freed its memory.

The installed Rust 1.97.1 standard library's
`std/src/sys/sync/thread_parking/darwin.rs` uses a libdispatch semaphore for each
thread parker. Crossbeam's thread-local context retains a Rust thread handle.
Creating a fresh DatasetRT runtime creates a new worker pool, but it does not
replace the calling thread's previously initialized parker. The calling thread
is the thread that survives fork and receives the worker's cache-load result.

This source inspection, the native crash stack, and the C controls support the
explanation that the inherited parker's backing semaphore is invalid in the
child. The exact installed macOS implementation is not asserted to be identical
to Apple's public source snapshot.

## Direct kernel confirmation

Removing libdispatch too, a direct `semaphore_create` / `semaphore_timedwait`
probe returns `KERN_OPERATION_TIMED_OUT` (49) in the parent and
`KERN_INVALID_NAME` (15) in the forked child. This confirms the inherited Mach
semaphore name is invalid in the child; libdispatch turns that error into the
observed trap.

This further reduction needs only macOS Mach APIs:

```c
#include <mach/mach.h>
#include <mach/semaphore.h>
#include <mach/sync_policy.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    semaphore_t sem;
    if (semaphore_create(mach_task_self(), &sem, SYNC_POLICY_FIFO, 0) != KERN_SUCCESS)
        return 2;
    mach_timespec_t timeout = {0, 1000000};
    fprintf(stderr, "parent: %d\n", semaphore_timedwait(sem, timeout));
    pid_t pid = fork();
    if (pid < 0) return 2;
    if (pid == 0) {
        kern_return_t result = semaphore_timedwait(sem, timeout);
        fprintf(stderr, "child: %d\n", result);
        _exit(result == KERN_INVALID_NAME ? 0 : 1);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    semaphore_destroy(mach_task_self(), sem);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 2;
}
```

Compile this block with `cc -Wall -Wextra -Werror` as above. Exit 0 means the
probe observed the expected invalid-name error; unlike libdispatch, this probe
reports that error without deliberately trapping.

## Rust standard-library reduction

The same failure also reproduces using Rust's standard library and OS calls only:

```rust
use std::{thread, time::Duration};

unsafe extern "C" {
    fn fork() -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn _exit(status: i32) -> !;
}

fn main() {
    // Remove this line for the passing cold-parent control.
    thread::park_timeout(Duration::from_millis(1));
    let pid = unsafe { fork() };
    if pid < 0 {
        std::process::exit(2);
    }
    if pid == 0 {
        // The already-used calling-thread parker survives fork.
        thread::park_timeout(Duration::from_millis(1));
        unsafe { _exit(0) };
    }
    let mut status = 0;
    if unsafe { waitpid(pid, &mut status, 0) } != pid {
        std::process::exit(2);
    }
    eprintln!("child wait status: {status:#x}");
    std::process::exit(if status == 0 { 0 } else { 1 });
}
```

Save that block as `/tmp/repro.rs` and compile with
`rustc --edition 2021 /tmp/repro.rs -o /tmp/repro-rust`. The warmed case reports
child wait status `0x5`; the cold control reports `0x0`. The captured macOS Rust
crash report has the same libdispatch diagnostic as the DatasetRT failure.

DatasetRT's corrected `__iter__` tests independently confirm that cold-parent
and runtime-only-parent fork cases pass, whereas the previously-used-reader
parent case crashes after the worker completes its fresh runtime construction.
No live native object is serialized or supplied to the worker. Pickling and
copying the native objects are rejected. Fork itself still inherits process
memory; Python copy restrictions cannot prevent that.

This investigation does not fix the runtime or remove fork from the support
contract. It identifies the synchronization mechanism that a fix must address.
