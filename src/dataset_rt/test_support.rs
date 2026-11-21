//! Exclusive temporary directories for native regression fixtures.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::CacheResult;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

pub(crate) struct TestDirectory(PathBuf);

impl TestDirectory {
    /// Reserve a fresh directory without sharing mutable fixture state across tests.
    pub fn new() -> CacheResult<Self> {
        loop {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("dataset-rt-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Borrow the fixture root without transferring cleanup ownership.
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    /// Remove only the directory exclusively reserved by this fixture.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
