//! Explicit, resumable manifest upgrades; physical cache contents remain immutable.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use super::{LoadedCache, Manifest, FORMAT_VERSION};
use crate::types::{CacheError, CacheResult};

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// One cache's observed persistent update, including post-rename sync uncertainty.
pub struct UpdateEntry {
    pub path: PathBuf,
    pub cache_id: u64,
    pub status: &'static str,
}

/// Preserve completed replacements when another target cannot be migrated.
pub struct UpdateError {
    pub path: Option<PathBuf>,
    pub message: String,
    pub entries: Vec<UpdateEntry>,
}

struct Target {
    path: PathBuf,
    cache_id: u64,
    digest: Vec<u8>,
    needs_update: bool,
}

/// Upgrade all loaded caches without changing the identities used by this dataset.
pub fn update_manifests(
    caches: &[LoadedCache],
    version: u32,
    mut check_signals: impl FnMut() -> CacheResult<()>,
) -> Result<Vec<UpdateEntry>, UpdateError> {
    update_with_hook(caches, version, |_, _| check_signals())
}

/// Bound fault injection to durable-write boundaries without changing production policy.
fn update_with_hook(
    caches: &[LoadedCache],
    version: u32,
    mut hook: impl FnMut(Step, &Path) -> CacheResult<()>,
) -> Result<Vec<UpdateEntry>, UpdateError> {
    if version != FORMAT_VERSION {
        return Err(UpdateError {
            path: None,
            message: format!("unsupported manifest migration target {version}; expected 3"),
            entries: Vec::new(),
        });
    }
    let targets = preflight(caches, &mut hook)?;
    let mut entries = Vec::with_capacity(targets.len());
    for target in targets {
        match update_target(&target, &mut hook) {
            Ok(status) => entries.push(UpdateEntry {
                path: target.path,
                cache_id: target.cache_id,
                status,
            }),
            Err((error, replaced)) => {
                if replaced {
                    entries.push(UpdateEntry {
                        path: target.path.clone(),
                        cache_id: target.cache_id,
                        status: "durability_unknown",
                    });
                }
                return Err(UpdateError {
                    path: Some(target.path),
                    message: error.to_string(),
                    entries,
                });
            }
        }
    }
    Ok(entries)
}

/// Validate the entire collection before any replacement; retain only bounded target records.
fn preflight(
    caches: &[LoadedCache],
    hook: &mut impl FnMut(Step, &Path) -> CacheResult<()>,
) -> Result<Vec<Target>, UpdateError> {
    let mut targets = Vec::with_capacity(caches.len());
    let mut paths = HashSet::with_capacity(caches.len());
    let mut ids = HashSet::with_capacity(caches.len());
    for cache in caches {
        let result = hook(Step::Inspect, &cache.path)
            .and_then(|()| inspect_target(cache, &mut paths, &mut ids));
        match result {
            Ok(target) => targets.push(target),
            Err(error) => {
                return Err(UpdateError {
                    path: Some(cache.path.join("manifest.json")),
                    message: error.to_string(),
                    entries: Vec::new(),
                })
            }
        }
    }
    Ok(targets)
}

/// Follow manifest symlinks and reject stale identities, duplicate targets and changed contents.
fn inspect_target(
    cache: &LoadedCache,
    paths: &mut HashSet<PathBuf>,
    ids: &mut HashSet<u64>,
) -> CacheResult<Target> {
    let path = fs::canonicalize(cache.path.join("manifest.json"))?;
    let cache_id = cache.cache_id.as_u64();
    if !paths.insert(path.clone()) || !ids.insert(cache_id) {
        return Err(CacheError::InvalidInput(
            "duplicate migration target or cache_id".to_string(),
        ));
    }
    let bytes = fs::read(&path)?;
    let current: Manifest = serde_json::from_slice(&bytes)?;
    if matches!(&current, Manifest::V3(manifest) if manifest.cache_id != cache_id) {
        return Err(CacheError::InvalidCache(
            "manifest cache_id changed since dataset construction".to_string(),
        ));
    }
    if matches!(cache.manifest, Manifest::V3(_)) && matches!(current, Manifest::V2(_)) {
        return Err(CacheError::InvalidCache(
            "manifest was downgraded since dataset construction".to_string(),
        ));
    }
    let expected = if matches!(current, Manifest::V3(_)) {
        cache.manifest.clone().into_v3(cache_id)
    } else {
        cache.manifest.clone()
    };
    if serde_json::to_value(expected.fields())? != serde_json::to_value(current.fields())? {
        return Err(CacheError::InvalidCache(
            "manifest changed since dataset construction".to_string(),
        ));
    }
    Ok(Target {
        path,
        cache_id,
        digest: Sha256::digest(&bytes).to_vec(),
        needs_update: matches!(current, Manifest::V2(_)),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Inspect,
    Write,
    Flush,
    Rename,
    DirectorySync,
}

/// Never truncate the publication marker; a failed post-rename sync reports replacement.
fn update_target(
    target: &Target,
    hook: &mut impl FnMut(Step, &Path) -> CacheResult<()>,
) -> Result<&'static str, (CacheError, bool)> {
    let mut replaced = false;
    let result = (|| -> CacheResult<&'static str> {
        hook(Step::Inspect, &target.path)?;
        let bytes = fs::read(&target.path)?;
        if Sha256::digest(&bytes).as_slice() != target.digest {
            return Err(CacheError::InvalidCache(
                "manifest changed during migration".to_string(),
            ));
        }
        if !target.needs_update {
            sync_manifest_directory(&target.path)?;
            return Ok("unchanged");
        }
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        let manifest = manifest.into_v3(target.cache_id);
        let temporary = create_temporary(&target.path)?;
        let permissions = fs::metadata(&target.path)?.permissions();
        temporary.file.set_permissions(permissions)?;
        hook(Step::Write, &target.path)?;
        let mut writer = BufWriter::new(&temporary.file);
        serde_json::to_writer_pretty(&mut writer, &manifest)?;
        hook(Step::Flush, &target.path)?;
        writer.flush()?;
        temporary.file.sync_all()?;
        hook(Step::Rename, &target.path)?;
        fs::rename(&temporary.path, &target.path)?;
        replaced = true;
        hook(Step::DirectorySync, &target.path)?;
        sync_manifest_directory(&target.path)?;
        Ok("updated")
    })();
    result.map_err(|error| (error, replaced))
}

/// Retry the durability boundary even when a preceding attempt already renamed the manifest.
fn sync_manifest_directory(manifest: &Path) -> CacheResult<()> {
    let parent = manifest
        .parent()
        .ok_or_else(|| CacheError::InvalidInput("manifest has no parent directory".to_string()))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

struct TemporaryManifest {
    path: PathBuf,
    file: File,
}

impl Drop for TemporaryManifest {
    /// Remove only this operation's temporary publication file, never another writer's.
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Use exclusive creation instead of a lock; callers own manifest mutation exclusively.
fn create_temporary(manifest: &Path) -> CacheResult<TemporaryManifest> {
    let parent = manifest
        .parent()
        .ok_or_else(|| CacheError::InvalidInput("manifest has no parent directory".to_string()))?;
    loop {
        let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".dataset-rt-manifest-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok(TemporaryManifest { path, file }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(CacheError::from(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{cache_id_from_name, load_cache};

    struct Fixture {
        root: PathBuf,
        caches: Vec<LoadedCache>,
    }

    impl Drop for Fixture {
        /// Test artifacts belong exclusively to this fault-injection run.
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    /// Use real v2 files for failures across early, middle and last manifest targets.
    fn fixture() -> CacheResult<Fixture> {
        let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dataset-rt-migration-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let mut fixture = Fixture {
            root,
            caches: Vec::new(),
        };
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/manifest-v2/legacy-plain");
        for position in 0..3 {
            let path = fixture.root.join(format!("cache-{position}"));
            copy_directory(&source, &path)?;
            fixture.caches.push(load_cache(path, true, position)?);
        }
        Ok(fixture)
    }

    /// Copy tiny native fixture files without depending on the current-format writer.
    fn copy_directory(source: &Path, target: &Path) -> CacheResult<()> {
        fs::create_dir(target)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let destination = target.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                copy_directory(&entry.path(), &destination)?;
            } else {
                fs::copy(entry.path(), destination)?;
            }
        }
        Ok(())
    }

    #[test]
    /// Fixed vectors make name hashing portable across implementations and processes.
    fn name_hash_vectors_are_fixed() {
        assert_eq!(cache_id_from_name("source"), 4_742_122_821_122_130_104);
        assert_eq!(cache_id_from_name("é"), 5_375_421_630_974_772_051);
        assert_eq!(cache_id_from_name("e\u{301}"), 4_544_825_244_877_739_698);
    }

    #[test]
    /// Every handled write-boundary failure preserves readable, retryable progress.
    fn failures_are_atomic_and_resumable() -> CacheResult<()> {
        for step in [Step::Write, Step::Flush, Step::Rename, Step::DirectorySync] {
            for failing_target in 0..3 {
                let fixture = fixture()?;
                let mut target_index = 0;
                let result = update_with_hook(&fixture.caches, 3, |current_step, _| {
                    if current_step == step {
                        let fail = target_index == failing_target;
                        target_index += 1;
                        if fail {
                            return Err(CacheError::Io(std::io::Error::other(
                                "injected write-boundary failure",
                            )));
                        }
                    }
                    Ok(())
                });
                let error = result.err().ok_or_else(|| {
                    CacheError::InvalidInput("injected failure was not observed".to_string())
                })?;
                let replaced = step == Step::DirectorySync;
                assert_eq!(error.entries.len(), failing_target + usize::from(replaced));
                if replaced {
                    assert_eq!(
                        error.entries.last().map(|entry| entry.status),
                        Some("durability_unknown")
                    );
                }
                for (position, cache) in fixture.caches.iter().enumerate() {
                    let reopened = load_cache(cache.path.clone(), true, position)?;
                    assert_eq!(reopened.cache_id.as_u64(), position as u64);
                    assert_eq!(
                        matches!(reopened.manifest, Manifest::V3(_)),
                        position < failing_target || (position == failing_target && replaced)
                    );
                    assert!(
                        !fs::read_dir(&cache.path)?.any(|entry| entry.is_ok_and(|entry| entry
                            .file_name()
                            .to_string_lossy()
                            .starts_with(".dataset-rt-manifest-")))
                    );
                }
                let retry = update_manifests(&fixture.caches, 3, || Ok(()))
                    .map_err(|error| CacheError::InvalidInput(error.message))?;
                assert_eq!(retry.len(), 3);
                assert!(retry
                    .iter()
                    .all(|entry| matches!(entry.status, "updated" | "unchanged")));
                let repeated = update_manifests(&fixture.caches, 3, || Ok(()))
                    .map_err(|error| CacheError::InvalidInput(error.message))?;
                assert!(repeated.iter().all(|entry| entry.status == "unchanged"));
            }
        }
        Ok(())
    }

    #[test]
    /// A v3 record requires identity while the v2 serialization contains no ID field.
    fn typed_versions_round_trip_and_require_v3_identity() -> CacheResult<()> {
        let fixture = fixture()?;
        let cache = fixture
            .caches
            .first()
            .ok_or_else(|| CacheError::InvalidInput("missing test fixture".to_string()))?;
        let legacy = serde_json::to_value(&cache.manifest)?;
        assert!(legacy.get("cache_id").is_none());
        let upgraded = cache.manifest.clone().into_v3(42);
        let mut encoded = serde_json::to_value(&upgraded)?;
        assert_eq!(encoded.get("cache_id"), Some(&serde_json::json!(42)));
        assert!(matches!(
            serde_json::from_value::<Manifest>(encoded.clone())?,
            Manifest::V3(_)
        ));
        encoded
            .as_object_mut()
            .ok_or_else(|| CacheError::InvalidInput("manifest was not an object".to_string()))?
            .remove("cache_id");
        assert!(serde_json::from_value::<Manifest>(encoded).is_err());
        Ok(())
    }

    #[test]
    /// Pause a child at one publication boundary so the parent can kill it without unwinding.
    fn kill_boundary_child() -> CacheResult<()> {
        let root = match std::env::var("DATASETRT_KILL_TEST_ROOT") {
            Ok(root) => PathBuf::from(root),
            Err(std::env::VarError::NotPresent) => return Ok(()),
            Err(error) => return Err(CacheError::InvalidInput(error.to_string())),
        };
        let step_name = std::env::var("DATASETRT_KILL_TEST_STEP")
            .map_err(|error| CacheError::InvalidInput(error.to_string()))?;
        let mut caches = Vec::new();
        for position in 0..3 {
            caches.push(load_cache(
                root.join(format!("cache-{position}")),
                true,
                position,
            )?);
        }
        update_with_hook(&caches, 3, |step, _| {
            if format!("{step:?}") == step_name {
                fs::write(root.join("paused"), b"paused at publication boundary")?;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
            Ok(())
        })
        .map_err(|error| CacheError::InvalidInput(error.message))?;
        Err(CacheError::InvalidInput(
            "kill-boundary hook was not reached".to_string(),
        ))
    }

    /// Wait for a child-owned marker so termination occurs at a known publication boundary.
    fn wait_for_pause(child: &mut std::process::Child, root: &Path) -> CacheResult<()> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !root.join("paused").exists() {
            if child.try_wait()?.is_some() || std::time::Instant::now() >= deadline {
                return Err(CacheError::InvalidInput(
                    "kill-boundary child exited or timed out".to_string(),
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Ok(())
    }

    #[test]
    /// Hard termination can orphan a temp file but cannot truncate a published manifest.
    fn abrupt_child_death_preserves_readable_manifests() -> CacheResult<()> {
        for step in [Step::Write, Step::Flush, Step::Rename, Step::DirectorySync] {
            let fixture = fixture()?;
            let mut child = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "storage::migration::tests::kill_boundary_child"])
                .env("DATASETRT_KILL_TEST_ROOT", &fixture.root)
                .env("DATASETRT_KILL_TEST_STEP", format!("{step:?}"))
                .stdout(std::process::Stdio::null())
                .spawn()?;
            let paused = wait_for_pause(&mut child, &fixture.root);
            child.kill()?;
            child.wait()?;
            paused?;
            let mut caches = Vec::new();
            for (position, cache) in fixture.caches.iter().enumerate() {
                let reopened = load_cache(cache.path.clone(), true, position)?;
                assert_eq!(reopened.cache_id.as_u64(), position as u64);
                assert_eq!(
                    matches!(reopened.manifest, Manifest::V3(_)),
                    position == 0 && step == Step::DirectorySync
                );
                caches.push(reopened);
            }
            let retry = update_manifests(&caches, 3, || Ok(()))
                .map_err(|error| CacheError::InvalidInput(error.message))?;
            assert_eq!(retry.len(), 3);
            for (position, cache) in caches.iter().enumerate() {
                let reopened = load_cache(cache.path.clone(), true, position)?;
                assert!(matches!(reopened.manifest, Manifest::V3(_)));
                assert_eq!(reopened.cache_id.as_u64(), position as u64);
            }
        }
        Ok(())
    }
}
