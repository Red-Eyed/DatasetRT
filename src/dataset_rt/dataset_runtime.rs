use std::sync::Arc;

use pyo3::prelude::*;

use crate::types::{CacheError, NumWorkers};
use crate::worker_pool::WorkerPool;

#[pyclass(name = "DatasetRuntime")]
pub struct PyDatasetRuntime {
    pool: Arc<WorkerPool>,
    num_workers: NumWorkers,
}

#[pymethods]
impl PyDatasetRuntime {
    #[new]
    fn new(num_workers: usize) -> PyResult<Self> {
        let num_workers = NumWorkers::new(num_workers).map_err(CacheError::into_py_err)?;
        let pool = WorkerPool::new(num_workers).map_err(CacheError::into_py_err)?;
        Ok(Self { pool, num_workers })
    }
}

impl PyDatasetRuntime {
    /// Share the runtime-owned worker pool with one reader or writer operation.
    pub fn pool(&self) -> Arc<WorkerPool> {
        self.pool.clone()
    }

    /// Return the fixed worker count used for operation-local task windows.
    pub fn num_workers(&self) -> NumWorkers {
        self.num_workers
    }
}
