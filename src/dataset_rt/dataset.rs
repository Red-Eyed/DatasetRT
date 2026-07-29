use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyList};

use crate::runtime::RuntimeIterator;
use crate::sampling::{plan_epoch, validate_weights};
use crate::storage::{load_cache, LoadedCache};
use crate::types::{
    CacheError, CacheId, CacheResult, MetadataField, MetadataValue, NumWorkers, PhysicalSample,
    PrefetchSize, SampleId,
};

#[pyclass(name = "CachedDataset")]
pub struct PyCachedDataset {
    inner: Arc<DatasetState>,
}

struct DatasetState {
    caches: Arc<Vec<LoadedCache>>,
    physical_samples: Arc<Vec<PhysicalSample>>,
    schema: Vec<MetadataField>,
    seed: u64,
    prefetch_size: PrefetchSize,
    num_workers: NumWorkers,
    shuffle: bool,
    mutable: Mutex<MutableDatasetState>,
}

struct MutableDatasetState {
    weights: Vec<f64>,
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
        self.inner.physical_samples.len()
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

    fn get_weight_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        self.inner
            .weight_rows(py, &guard.weights)
            .map_err(CacheError::into_py_err)
    }

    fn set_weight_table(&self, table: Bound<'_, PyAny>) -> PyResult<()> {
        let weights = self
            .inner
            .extract_weight_table(table)
            .map_err(CacheError::into_py_err)?;
        let mut guard = self
            .inner
            .mutable
            .lock()
            .map_err(|_| CacheError::WorkerFailed.into_py_err())?;
        guard.weights = weights;
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
        let physical_samples = collect_physical_samples(&caches)?;
        if physical_samples.is_empty() {
            return Err(CacheError::InvalidCache(
                "dataset contains no physical samples".to_string(),
            ));
        }

        Ok(Self {
            caches: Arc::new(caches),
            physical_samples: Arc::new(physical_samples.clone()),
            schema,
            seed,
            prefetch_size: PrefetchSize::new(prefetch_size)?,
            num_workers: NumWorkers::new(num_workers)?,
            shuffle,
            mutable: Mutex::new(MutableDatasetState {
                weights: vec![1.0; physical_samples.len()],
                next_epoch: 0,
            }),
        })
    }

    fn start_iterator(&self) -> CacheResult<RuntimeIterator> {
        let plan = if self.shuffle {
            self.shuffled_plan()?
        } else {
            physical_order_plan(self.physical_samples.len())
        };
        Ok(RuntimeIterator::start(
            self.caches.clone(),
            self.physical_samples.clone(),
            plan,
            self.prefetch_size,
            self.num_workers,
        ))
    }

    fn shuffled_plan(&self) -> CacheResult<Vec<usize>> {
        let (weights, epoch) = {
            let mut guard = self.mutable.lock().map_err(|_| CacheError::WorkerFailed)?;
            let snapshot = guard.weights.clone();
            let epoch = guard.next_epoch;
            guard.next_epoch += 1;
            (snapshot, epoch)
        };
        plan_epoch(&weights, self.seed, epoch)
    }

    fn weight_rows<'py>(
        &self,
        py: Python<'py>,
        weights: &[f64],
    ) -> CacheResult<Bound<'py, PyList>> {
        validate_weights(weights, self.physical_samples.len())?;
        let rows = PyList::empty(py);

        for (physical_index, physical_sample) in self.physical_samples.iter().enumerate() {
            let cache = self
                .caches
                .get(physical_sample.cache_id.as_u64() as usize)
                .ok_or_else(|| CacheError::InvalidCache("cache id is out of range".to_string()))?;
            let metadata = cache
                .metadata_rows
                .get(physical_sample.sample_id.as_u64() as usize)
                .ok_or_else(|| {
                    CacheError::InvalidCache("metadata row is out of range".to_string())
                })?;
            let weight = weights.get(physical_index).copied().ok_or_else(|| {
                CacheError::InvalidCache("weight row is out of range".to_string())
            })?;
            let row = weight_row_to_dict(py, &self.schema, physical_sample, metadata, weight)?;
            rows.append(row).map_err(py_error)?;
        }

        Ok(rows)
    }

    fn extract_weight_table(&self, table: Bound<'_, PyAny>) -> CacheResult<Vec<f64>> {
        // Metadata columns are for user filtering; sample identity is only cache_id + sample_id.
        let selected = select_weight_columns(&table)?;
        let iterator = iter_named_rows(&selected)?;
        let updates = iterator
            .map(|row| {
                let row = row.map_err(py_error)?;
                extract_weight_update(&row)
            })
            .collect::<CacheResult<Vec<_>>>()?;
        self.validated_weight_updates(updates)
    }

    fn validated_weight_updates(&self, updates: Vec<WeightUpdate>) -> CacheResult<Vec<f64>> {
        if updates.len() != self.physical_samples.len() {
            return Err(CacheError::InvalidInput(format!(
                "expected {} weight rows, got {}",
                self.physical_samples.len(),
                updates.len()
            )));
        }

        let mut weights = vec![0.0; self.physical_samples.len()];
        let mut seen = vec![false; self.physical_samples.len()];

        for update in updates {
            // Rows may be reordered by Polars, so weight updates are applied by identity.
            let physical_index = self.physical_index(update.cache_id, update.sample_id)?;
            let already_seen = seen.get(physical_index).copied().ok_or_else(|| {
                CacheError::InvalidCache("weight seen index is out of range".to_string())
            })?;
            if already_seen {
                return Err(CacheError::InvalidInput(format!(
                    "duplicate weight row for cache_id={} sample_id={}",
                    update.cache_id, update.sample_id
                )));
            }
            let seen_slot = seen.get_mut(physical_index).ok_or_else(|| {
                CacheError::InvalidCache("weight seen index is out of range".to_string())
            })?;
            *seen_slot = true;
            let weight_slot = weights.get_mut(physical_index).ok_or_else(|| {
                CacheError::InvalidCache("weight index is out of range".to_string())
            })?;
            *weight_slot = update.weight;
        }

        if seen.iter().any(|value| !value) {
            return Err(CacheError::InvalidInput(
                "weight table must include every physical sample exactly once".to_string(),
            ));
        }

        validate_weights(&weights, self.physical_samples.len())?;
        Ok(weights)
    }

    fn physical_index(&self, cache_id: u64, sample_id: u64) -> CacheResult<usize> {
        self.physical_samples
            .iter()
            .position(|sample| {
                sample.cache_id.as_u64() == cache_id && sample.sample_id.as_u64() == sample_id
            })
            .ok_or_else(|| {
                CacheError::InvalidInput(format!(
                    "unknown weight row identity cache_id={cache_id} sample_id={sample_id}"
                ))
            })
    }
}

fn physical_order_plan(sample_count: usize) -> Vec<usize> {
    (0..sample_count).collect()
}

struct WeightUpdate {
    cache_id: u64,
    sample_id: u64,
    weight: f64,
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

fn collect_physical_samples(caches: &[LoadedCache]) -> CacheResult<Vec<PhysicalSample>> {
    let mut physical_samples = Vec::new();
    for (cache_index, cache) in caches.iter().enumerate() {
        let cache_id = CacheId::from_position(cache_index)?;
        for sample_index in 0..cache.sample_count() {
            physical_samples.push(PhysicalSample {
                cache_id,
                sample_id: SampleId::from_position(sample_index)?,
            });
        }
    }
    Ok(physical_samples)
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

fn weight_row_to_dict<'py>(
    py: Python<'py>,
    schema: &[MetadataField],
    physical_sample: &PhysicalSample,
    metadata: &[MetadataValue],
    weight: f64,
) -> CacheResult<Bound<'py, PyDict>> {
    let dict = metadata_to_dict(py, schema, metadata).map_err(py_error)?;
    dict.set_item("cache_id", physical_sample.cache_id.as_u64())
        .map_err(py_error)?;
    dict.set_item("sample_id", physical_sample.sample_id.as_u64())
        .map_err(py_error)?;
    dict.set_item("weight", weight).map_err(py_error)?;
    Ok(dict)
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

fn select_weight_columns<'py>(table: &Bound<'py, PyAny>) -> CacheResult<Bound<'py, PyAny>> {
    let columns = PyList::new(table.py(), ["cache_id", "sample_id", "weight"]).map_err(py_error)?;
    table.call_method1("select", (columns,)).map_err(py_error)
}

fn iter_named_rows<'py>(table: &Bound<'py, PyAny>) -> CacheResult<Bound<'py, PyIterator>> {
    let kwargs = PyDict::new(table.py());
    kwargs.set_item("named", true).map_err(py_error)?;
    let rows = table
        .call_method("iter_rows", (), Some(&kwargs))
        .map_err(py_error)?;
    PyIterator::from_object(&rows).map_err(py_error)
}

fn extract_weight_update(row: &Bound<'_, PyAny>) -> CacheResult<WeightUpdate> {
    let dict = row
        .cast::<PyDict>()
        .map_err(|_| CacheError::InvalidInput("weight rows must be dictionaries".to_string()))?;
    Ok(WeightUpdate {
        cache_id: extract_required_u64(dict, "cache_id")?,
        sample_id: extract_required_u64(dict, "sample_id")?,
        weight: extract_required_f64(dict, "weight")?,
    })
}

fn extract_required_u64(dict: &Bound<'_, PyDict>, key: &str) -> CacheResult<u64> {
    dict.get_item(key)
        .map_err(py_error)?
        .ok_or_else(|| CacheError::InvalidInput(format!("weight table missing '{key}' column")))?
        .extract::<u64>()
        .map_err(py_error)
}

fn extract_required_f64(dict: &Bound<'_, PyDict>, key: &str) -> CacheResult<f64> {
    dict.get_item(key)
        .map_err(py_error)?
        .ok_or_else(|| CacheError::InvalidInput(format!("weight table missing '{key}' column")))?
        .extract::<f64>()
        .map_err(py_error)
}

fn py_error(error: PyErr) -> CacheError {
    CacheError::Python(error.to_string())
}
