use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crossbeam_channel::{bounded, Sender};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict};

use crate::dataset_runtime::PyDatasetRuntime;
use crate::runtime::{EpochPlan, RuntimeIterator};
use crate::samples_metadata::{
    build_samples_metadata_ipc, extract_metadata_ipc, ActiveMetadataTable, WeightState,
};
use crate::sampling::EpochSampler;
use crate::storage::{load_cache, LoadedCache};
use crate::types::{
    CacheError, CacheResult, EpochLen, MetadataField, MetadataValue, NumWorkers, PrefetchSize,
};
use crate::worker_pool::WorkerPool;

#[pyclass(name = "CachedDataset")]
pub struct PyCachedDataset {
    inner: Arc<DatasetState>,
}

struct DatasetState {
    pool: Arc<WorkerPool>,
    caches: Arc<Vec<LoadedCache>>,
    cache_offsets: Arc<Vec<usize>>,
    total_samples: usize,
    schema: Vec<MetadataField>,
    seed: u64,
    prefetch_size: PrefetchSize,
    num_workers: NumWorkers,
    shuffle: bool,
    mutable: Mutex<MutableDatasetState>,
}

struct MutableDatasetState {
    active: ActiveMetadata,
    epoch_len: EpochLen,
    sequential_offset: usize,
    shuffled_draw_offset: u64,
}

enum ActiveMetadata {
    Full,
    Table(ActiveMetadataTable),
}

#[pymethods]
impl PyCachedDataset {
    #[new]
    fn new(
        runtime: PyRef<'_, PyDatasetRuntime>,
        paths: Vec<String>,
        seed: u64,
        prefetch_size: usize,
        shuffle: bool,
        validate_cache: bool,
    ) -> PyResult<Self> {
        DatasetState::load(
            runtime.pool(),
            runtime.num_workers(),
            paths,
            seed,
            prefetch_size,
            shuffle,
            validate_cache,
        )
        .map(|state| Self {
            inner: Arc::new(state),
        })
        .map_err(CacheError::into_py_err)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner.epoch_len().map_err(CacheError::into_py_err)
    }

    fn __iter__(&self) -> PyResult<PyDatasetIterator> {
        let iterator = self
            .inner
            .start_iterator()
            .map_err(CacheError::into_py_err)?;
        Ok(PyDatasetIterator {
            schema: self.inner.schema.clone(),
            runtime: Mutex::new(iterator),
        })
    }

    /// Return Arrow IPC metadata for the current active sample table.
    fn metadata_ipc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let ipc = self.inner.metadata_ipc().map_err(CacheError::into_py_err)?;
        Ok(PyBytes::new(py, &ipc))
    }

    /// Return Arrow IPC metadata using the compatibility native method name.
    fn samples_metadata_ipc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.metadata_ipc(py)
    }

    /// Replace the active metadata table after Rust validates identities and weights.
    fn update_metadata_ipc(&self, ipc: Bound<'_, PyBytes>) -> PyResult<()> {
        let active = self
            .inner
            .extract_metadata_ipc(ipc.as_bytes())
            .map_err(CacheError::into_py_err)?;
        let mut guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        guard.replace_active(ActiveMetadata::Table(active));
        Ok(())
    }

    /// Replace metadata using the compatibility native method name.
    fn set_samples_metadata_ipc(&self, ipc: Bound<'_, PyBytes>) -> PyResult<()> {
        self.update_metadata_ipc(ipc)
    }

    /// Set the finite number of samples emitted by future iterators.
    fn set_epoch_len(&self, epoch_len: usize) -> PyResult<()> {
        self.inner
            .set_epoch_len(epoch_len)
            .map_err(CacheError::into_py_err)
    }
}

impl DatasetState {
    fn load(
        pool: Arc<WorkerPool>,
        num_workers: NumWorkers,
        paths: Vec<String>,
        seed: u64,
        prefetch_size: usize,
        shuffle: bool,
        validate_cache: bool,
    ) -> CacheResult<Self> {
        if paths.is_empty() {
            return Err(CacheError::InvalidInput(
                "CachedDataset requires at least one cache path".to_string(),
            ));
        }

        let prefetch_size = PrefetchSize::new(prefetch_size)?;
        let caches = load_caches(pool.clone(), paths, validate_cache, num_workers)?;
        let schema = common_schema(&caches)?;
        let (cache_offsets, total_samples) = collect_cache_offsets(&caches)?;
        if total_samples == 0 {
            return Err(CacheError::InvalidCache(
                "dataset contains no physical samples".to_string(),
            ));
        }

        Ok(Self {
            pool,
            caches: Arc::new(caches),
            cache_offsets: Arc::new(cache_offsets),
            total_samples,
            schema,
            seed,
            prefetch_size,
            num_workers,
            shuffle,
            mutable: Mutex::new(MutableDatasetState {
                active: ActiveMetadata::Full,
                epoch_len: EpochLen::new(total_samples)?,
                sequential_offset: 0,
                shuffled_draw_offset: 0,
            }),
        })
    }

    fn start_iterator(&self) -> CacheResult<RuntimeIterator> {
        let plan = self.epoch_plan()?;
        RuntimeIterator::start(
            self.pool.clone(),
            self.caches.clone(),
            self.cache_offsets.clone(),
            plan,
            self.prefetch_size,
            self.num_workers,
        )
    }

    /// Snapshot the active table and construct the epoch plan used by one iterator.
    fn epoch_plan(&self) -> CacheResult<EpochPlan> {
        let mut guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
        let epoch_len = guard.epoch_len.as_usize();
        if !self.shuffle {
            let population_len = active_len(&guard.active, self.total_samples);
            let start = guard.sequential_offset;
            guard.sequential_offset = advance_cyclic_offset(start, epoch_len, population_len)?;
            return Ok(EpochPlan::Sequential {
                len: epoch_len,
                start,
                population_len,
                physical_indices: active_physical_indices(&guard.active),
            });
        }

        let start_draw = guard.shuffled_draw_offset;
        guard.shuffled_draw_offset = advance_draw_offset(start_draw, epoch_len)?;
        match &guard.active {
            ActiveMetadata::Full => Ok(EpochPlan::Shuffled {
                sampler: Box::new(EpochSampler::uniform(
                    self.total_samples,
                    self.seed,
                    start_draw,
                    epoch_len,
                )?),
                physical_indices: None,
            }),
            ActiveMetadata::Table(active) => Ok(EpochPlan::Shuffled {
                sampler: Box::new(EpochSampler::weighted(
                    &active.weights,
                    self.seed,
                    start_draw,
                    epoch_len,
                )?),
                physical_indices: Some(active.physical_indices.clone()),
            }),
        }
    }

    /// Return the finite number of samples emitted by future dataset iterators.
    fn epoch_len(&self) -> CacheResult<usize> {
        let guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
        Ok(guard.epoch_len.as_usize())
    }

    /// Set the future iterator length without changing the active sampling population.
    fn set_epoch_len(&self, epoch_len: usize) -> CacheResult<()> {
        let epoch_len = EpochLen::new(epoch_len)?;
        let mut guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
        guard.epoch_len = epoch_len;
        Ok(())
    }

    /// Build or return the active metadata table IPC without touching immutable cache files.
    fn metadata_ipc(&self) -> CacheResult<Vec<u8>> {
        let guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
        match &guard.active {
            ActiveMetadata::Full => build_samples_metadata_ipc(
                self.caches.as_ref(),
                self.cache_offsets.as_ref(),
                &self.schema,
                &WeightState::Uniform,
            ),
            ActiveMetadata::Table(active) => Ok(active.ipc.clone()),
        }
    }

    /// Parse an active metadata table and validate every included sample identity.
    fn extract_metadata_ipc(&self, ipc: &[u8]) -> CacheResult<ActiveMetadataTable> {
        extract_metadata_ipc(
            ipc,
            self.caches.as_ref(),
            self.cache_offsets.as_ref(),
            &self.schema,
        )
    }
}

impl MutableDatasetState {
    /// Replace active rows and reset stream positions without changing epoch length.
    fn replace_active(&mut self, active: ActiveMetadata) {
        self.active = active;
        self.sequential_offset = 0;
        self.shuffled_draw_offset = 0;
    }
}

fn active_len(active: &ActiveMetadata, total_samples: usize) -> usize {
    match active {
        ActiveMetadata::Full => total_samples,
        ActiveMetadata::Table(active) => active.physical_indices.len(),
    }
}

fn active_physical_indices(active: &ActiveMetadata) -> Option<Arc<Vec<usize>>> {
    match active {
        ActiveMetadata::Full => None,
        ActiveMetadata::Table(active) => Some(active.physical_indices.clone()),
    }
}

fn advance_cyclic_offset(start: usize, step: usize, population_len: usize) -> CacheResult<usize> {
    if population_len == 0 {
        return Err(CacheError::InvalidInput(
            "cannot iterate an empty active population".to_string(),
        ));
    }
    let step = step % population_len;
    let distance_to_wrap = population_len.checked_sub(step).ok_or_else(|| {
        CacheError::InvalidInput("active population length overflowed usize".to_string())
    })?;
    if start >= distance_to_wrap {
        Ok(start - distance_to_wrap)
    } else {
        Ok(start + step)
    }
}

fn advance_draw_offset(start: u64, step: usize) -> CacheResult<u64> {
    let step = u64::try_from(step)
        .map_err(|_| CacheError::InvalidInput("epoch_len does not fit in u64".to_string()))?;
    start
        .checked_add(step)
        .ok_or_else(|| CacheError::InvalidInput("shuffled draw offset overflowed u64".to_string()))
}

/// Build prefix offsets that map each cache to its first physical sample index.
fn cache_offsets(caches: &[LoadedCache]) -> CacheResult<Vec<usize>> {
    let mut offsets = Vec::with_capacity(caches.len());
    let mut next_offset = 0_usize;
    for cache in caches {
        offsets.push(next_offset);
        next_offset = next_offset
            .checked_add(cache.sample_count())
            .ok_or_else(|| {
                CacheError::InvalidInput("total sample count overflowed usize".to_string())
            })?;
    }
    Ok(offsets)
}

type PySampleTuple<'py> = (Bound<'py, PyBytes>, Bound<'py, PyDict>, u64, u64);

#[pyclass]
pub struct PyDatasetIterator {
    schema: Vec<MetadataField>,
    runtime: Mutex<RuntimeIterator>,
}

#[pymethods]
impl PyDatasetIterator {
    fn __iter__(self_: PyRef<'_, Self>) -> PyRef<'_, Self> {
        self_
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<PySampleTuple<'py>>> {
        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        match guard.next() {
            Some(Ok(sample)) => sample_to_python(py, &self.schema, sample).map(Some),
            Some(Err(error)) => Err(error.into_py_err()),
            None => Ok(None),
        }
    }
}

struct CacheLoadResult {
    position: usize,
    cache: LoadedCache,
}

/// Load cache directories with bounded concurrency while preserving constructor path order.
fn load_caches(
    pool: Arc<WorkerPool>,
    paths: Vec<String>,
    validate_cache: bool,
    num_workers: NumWorkers,
) -> CacheResult<Vec<LoadedCache>> {
    let total = paths.len();
    let parallelism = num_workers.as_usize().min(total);
    let (result_sender, result_receiver) = bounded(parallelism);
    let mut paths = paths.into_iter().enumerate();
    let mut scheduled = 0_usize;

    while scheduled < parallelism {
        submit_next_cache_load(&pool, &mut paths, &result_sender, validate_cache)?;
        scheduled += 1;
    }

    let mut results = Vec::with_capacity(total);
    while results.len() < total {
        results.push(
            result_receiver
                .recv()
                .map_err(|_| CacheError::WorkerFailed)??,
        );
        if scheduled < total {
            submit_next_cache_load(&pool, &mut paths, &result_sender, validate_cache)?;
            scheduled += 1;
        }
    }

    order_loaded_caches(results)
}

/// Submit one cache load while retaining its constructor identity position.
fn submit_next_cache_load(
    pool: &WorkerPool,
    paths: &mut impl Iterator<Item = (usize, String)>,
    result_sender: &Sender<CacheResult<CacheLoadResult>>,
    validate_cache: bool,
) -> CacheResult<()> {
    let Some((position, path)) = paths.next() else {
        return Err(CacheError::WorkerFailed);
    };
    let path = PathBuf::from(path);
    let result_sender = result_sender.clone();
    pool.submit(result_sender, move || {
        let cache = load_cache(path, validate_cache)?;
        Ok(CacheLoadResult { position, cache })
    })
}

/// Restore constructor path order so `cache_id` remains stable under parallel loading.
fn order_loaded_caches(results: Vec<CacheLoadResult>) -> CacheResult<Vec<LoadedCache>> {
    let mut ordered = results
        .into_iter()
        .map(|result| (result.position, result.cache))
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(position, _)| *position);

    Ok(ordered.into_iter().map(|(_, cache)| cache).collect())
}

fn common_schema(caches: &[LoadedCache]) -> CacheResult<Vec<MetadataField>> {
    let first = caches
        .first()
        .ok_or_else(|| CacheError::InvalidInput("no caches provided".to_string()))?;
    for cache in caches.iter().skip(1) {
        if cache.manifest.metadata_schema.len() != first.manifest.metadata_schema.len() {
            return Err(CacheError::InvalidCache(
                "all caches must share the same metadata schema".to_string(),
            ));
        }
        for (left, right) in cache
            .manifest
            .metadata_schema
            .iter()
            .zip(first.manifest.metadata_schema.iter())
        {
            if left.name != right.name || left.kind != right.kind {
                return Err(CacheError::InvalidCache(
                    "all caches must share the same metadata schema".to_string(),
                ));
            }
        }
    }
    Ok(first.manifest.metadata_schema.clone())
}

/// Build compact dataset identity state from loaded cache sample counts.
fn collect_cache_offsets(caches: &[LoadedCache]) -> CacheResult<(Vec<usize>, usize)> {
    let offsets = cache_offsets(caches)?;
    let total = caches.iter().try_fold(0_usize, |total, cache| {
        total.checked_add(cache.sample_count()).ok_or_else(|| {
            CacheError::InvalidInput("total sample count overflowed usize".to_string())
        })
    })?;
    Ok((offsets, total))
}

fn sample_to_python<'py>(
    py: Python<'py>,
    schema: &[MetadataField],
    sample: crate::types::LoadedSample,
) -> PyResult<PySampleTuple<'py>> {
    let metadata = metadata_to_dict(py, schema, &sample.metadata)?;
    Ok((
        PyBytes::new(py, &sample.data),
        metadata,
        sample.cache_id.as_u64(),
        sample.sample_id.as_u64(),
    ))
}

fn metadata_to_dict<'py>(
    py: Python<'py>,
    schema: &[MetadataField],
    metadata: &[MetadataValue],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (field, value) in schema.iter().zip(metadata.iter()) {
        match value {
            MetadataValue::Bool(value) => dict.set_item(&field.name, *value)?,
            MetadataValue::Int(value) => dict.set_item(&field.name, *value)?,
            MetadataValue::Float(value) => dict.set_item(&field.name, *value)?,
            MetadataValue::String(value) => dict.set_item(&field.name, value)?,
        }
    }
    Ok(dict)
}
