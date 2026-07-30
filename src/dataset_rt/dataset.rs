use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict};

use crate::runtime::{EpochPlan, RuntimeIterator};
use crate::sampling::EpochSampler;
use crate::storage::{load_cache, LoadedCache};
use crate::types::{
    CacheError, CacheResult, MetadataField, MetadataValue, NumWorkers, PrefetchSize,
};
use crate::weight_table::{build_weight_table_ipc, extract_weight_table_ipc, WeightState};

#[pyclass(name = "CachedDataset")]
pub struct PyCachedDataset {
    inner: Arc<DatasetState>,
}

struct DatasetState {
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
    weights: WeightState,
    next_epoch: u64,
}

#[pymethods]
impl PyCachedDataset {
    #[new]
    fn new(
        paths: Vec<String>,
        seed: u64,
        prefetch_size: usize,
        num_workers: usize,
        shuffle: bool,
        validate_cache: bool,
    ) -> PyResult<Self> {
        DatasetState::load(
            paths,
            seed,
            prefetch_size,
            num_workers,
            shuffle,
            validate_cache,
        )
        .map(|state| Self {
            inner: Arc::new(state),
        })
        .map_err(CacheError::into_py_err)
    }

    fn __len__(&self) -> usize {
        self.inner.total_samples
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

    /// Return an Arrow IPC weight table built by Rust from cache metadata and weights.
    fn weight_table_ipc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let ipc = self
            .inner
            .weight_table_ipc()
            .map_err(CacheError::into_py_err)?;
        Ok(PyBytes::new(py, &ipc))
    }

    /// Replace weights from a columnar Arrow IPC payload containing identity and weight columns.
    fn set_weight_table_ipc(&self, ipc: Bound<'_, PyBytes>) -> PyResult<()> {
        let weights = self
            .inner
            .extract_weight_table_ipc(ipc.as_bytes())
            .map_err(CacheError::into_py_err)?;
        let mut guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        guard.weights = WeightState::Custom(weights);
        Ok(())
    }
}

impl DatasetState {
    fn load(
        paths: Vec<String>,
        seed: u64,
        prefetch_size: usize,
        num_workers: usize,
        shuffle: bool,
        validate_cache: bool,
    ) -> CacheResult<Self> {
        if paths.is_empty() {
            return Err(CacheError::InvalidInput(
                "CachedDataset requires at least one cache path".to_string(),
            ));
        }

        let caches = load_caches(paths, validate_cache)?;
        let schema = common_schema(&caches)?;
        let (cache_offsets, total_samples) = collect_cache_offsets(&caches)?;
        if total_samples == 0 {
            return Err(CacheError::InvalidCache(
                "dataset contains no physical samples".to_string(),
            ));
        }

        Ok(Self {
            caches: Arc::new(caches),
            cache_offsets: Arc::new(cache_offsets),
            total_samples,
            schema,
            seed,
            prefetch_size: PrefetchSize::new(prefetch_size)?,
            num_workers: NumWorkers::new(num_workers)?,
            shuffle,
            mutable: Mutex::new(MutableDatasetState {
                weights: WeightState::Uniform,
                next_epoch: 0,
            }),
        })
    }

    fn start_iterator(&self) -> CacheResult<RuntimeIterator> {
        let plan = if self.shuffle {
            self.shuffled_plan()?
        } else {
            EpochPlan::PhysicalOrder {
                len: self.total_samples,
            }
        };
        Ok(RuntimeIterator::start(
            self.caches.clone(),
            self.cache_offsets.clone(),
            plan,
            self.prefetch_size,
            self.num_workers,
        ))
    }

    /// Snapshot the current weight state and construct a replacement-sampling epoch plan.
    fn shuffled_plan(&self) -> CacheResult<EpochPlan> {
        let mut guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
        let epoch = guard.next_epoch;
        guard.next_epoch += 1;

        let sampler = match &guard.weights {
            WeightState::Uniform => EpochSampler::uniform(self.total_samples, self.seed, epoch)?,
            WeightState::Custom(weights) => EpochSampler::weighted(weights, self.seed, epoch)?,
        };
        Ok(EpochPlan::Shuffled(Box::new(sampler)))
    }

    /// Build one Arrow IPC table containing identity, metadata, and current weights.
    fn weight_table_ipc(&self) -> CacheResult<Vec<u8>> {
        let guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
        build_weight_table_ipc(
            self.caches.as_ref(),
            self.cache_offsets.as_ref(),
            &self.schema,
            &guard.weights,
        )
    }

    /// Parse a columnar weight update and validate that it covers the dataset exactly once.
    fn extract_weight_table_ipc(&self, ipc: &[u8]) -> CacheResult<Vec<f64>> {
        extract_weight_table_ipc(
            ipc,
            self.caches.as_ref(),
            self.cache_offsets.as_ref(),
            self.total_samples,
        )
    }
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

fn load_caches(paths: Vec<String>, validate_cache: bool) -> CacheResult<Vec<LoadedCache>> {
    paths
        .into_iter()
        .map(|path| load_cache(PathBuf::from(path), validate_cache))
        .collect()
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
