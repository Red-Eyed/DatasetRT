use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PySystemExit, PyValueError};
use pyo3::PyErr;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    InvalidCache(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("Python error: {0}")]
    Python(String),
    #[error("{0}")]
    KeyboardInterrupt(String),
    #[error("{0}")]
    SystemExit(String),
    #[error("runtime worker failed")]
    WorkerFailed,
}

impl CacheError {
    pub fn into_py_err(self) -> PyErr {
        match self {
            Self::InvalidInput(message) | Self::InvalidCache(message) => {
                PyValueError::new_err(message)
            }
            Self::KeyboardInterrupt(message) => PyKeyboardInterrupt::new_err(message),
            Self::SystemExit(message) => PySystemExit::new_err(message),
            Self::Io(_) | Self::Json(_) | Self::Arrow(_) | Self::Python(_) | Self::WorkerFailed => {
                PyRuntimeError::new_err(self.to_string())
            }
        }
    }
}

pub type CacheResult<T> = Result<T, CacheError>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CacheId(u64);

impl CacheId {
    pub fn from_position(position: usize) -> CacheResult<Self> {
        let id = u64::try_from(position)
            .map_err(|_| CacheError::InvalidInput("cache id does not fit in u64".to_string()))?;
        Ok(Self(id))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SampleId(u64);

impl SampleId {
    pub fn from_position(position: usize) -> CacheResult<Self> {
        let id = u64::try_from(position)
            .map_err(|_| CacheError::InvalidInput("sample id does not fit in u64".to_string()))?;
        Ok(Self(id))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaxShardBytes(u64);

impl MaxShardBytes {
    pub fn new(value: u64) -> CacheResult<Self> {
        if value == 0 {
            return Err(CacheError::InvalidInput(
                "max_shard_bytes must be greater than zero".to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrefetchSize(usize);

impl PrefetchSize {
    pub fn new(value: usize) -> CacheResult<Self> {
        if value == 0 {
            return Err(CacheError::InvalidInput(
                "prefetch_size must be greater than zero".to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NumWorkers(usize);

impl NumWorkers {
    pub fn new(value: usize) -> CacheResult<Self> {
        if value == 0 {
            return Err(CacheError::InvalidInput(
                "num_workers must be greater than zero".to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionAlgo {
    None,
    Lz4,
}

impl CompressionAlgo {
    pub fn from_name(value: &str) -> CacheResult<Self> {
        match value {
            "none" => Ok(Self::None),
            "lz4" => Ok(Self::Lz4),
            _ => Err(CacheError::InvalidInput(format!(
                "unsupported shard compression algorithm '{value}'"
            ))),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardCompression {
    pub algo: CompressionAlgo,
    pub ratio: f64,
}

impl ShardCompression {
    pub fn new(algo: CompressionAlgo, ratio: f64) -> CacheResult<Self> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(CacheError::InvalidInput(
                "shard_compression.ratio must be positive and finite".to_string(),
            ));
        }
        if algo == CompressionAlgo::None && ratio != 1.0 {
            return Err(CacheError::InvalidInput(
                "shard_compression.ratio must be 1.0 when algo is 'none'".to_string(),
            ));
        }
        Ok(Self { algo, ratio })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataKind {
    Bool,
    Int,
    Float,
    String,
}

impl fmt::Display for MetadataKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => write!(formatter, "bool"),
            Self::Int => write!(formatter, "int"),
            Self::Float => write!(formatter, "float"),
            Self::String => write!(formatter, "string"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl MetadataValue {
    pub fn kind(&self) -> MetadataKind {
        match self {
            Self::Bool(_) => MetadataKind::Bool,
            Self::Int(_) => MetadataKind::Int,
            Self::Float(_) => MetadataKind::Float,
            Self::String(_) => MetadataKind::String,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetadataField {
    pub name: String,
    pub kind: MetadataKind,
}

#[derive(Clone, Debug)]
pub struct CacheSample {
    pub data: Vec<u8>,
    pub metadata: Vec<MetadataValue>,
}

#[derive(Clone, Debug)]
pub struct LoadedSample {
    pub data: Vec<u8>,
    pub metadata: Vec<MetadataValue>,
    pub cache_id: CacheId,
    pub sample_id: SampleId,
}
