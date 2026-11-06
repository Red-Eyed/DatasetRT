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
    #[error("native operation failed: {0}")]
    NativeFailure(String),
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
            Self::Io(_)
            | Self::Json(_)
            | Self::Arrow(_)
            | Self::Python(_)
            | Self::NativeFailure(_)
            | Self::WorkerFailed => PyRuntimeError::new_err(self.to_string()),
        }
    }
}

pub type CacheResult<T> = Result<T, CacheError>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CacheId(u64);

impl CacheId {
    /// Accept a persisted identity without treating it as an array position.
    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }

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
pub struct EpochLen(usize);

impl EpochLen {
    pub fn new(value: usize) -> CacheResult<Self> {
        if value == 0 {
            return Err(CacheError::InvalidInput(
                "epoch_len must be at least 1".to_string(),
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
    Brotli,
}

impl CompressionAlgo {
    pub fn from_name(value: &str) -> CacheResult<Self> {
        match value {
            "none" => Ok(Self::None),
            "lz4" => Ok(Self::Lz4),
            "brotli" => Ok(Self::Brotli),
            _ => Err(CacheError::InvalidInput(format!(
                "unsupported shard compression algorithm '{value}'"
            ))),
        }
    }
}

/// Legacy payload-only settings retain their exact manifest representation.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyCompression {
    pub algo: CompressionAlgo,
    pub ratio: f64,
}

/// Record scope and supported codec parameters are selected structurally.
#[derive(Clone, Debug)]
pub enum ShardCompression {
    Legacy(LegacyCompression),
    None,
    Lz4Fast { acceleration: i32 },
    Lz4HC { level: i32 },
    Brotli { level: i32 },
}

/// Modern wire modes reject parameters belonging to other variants.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum RecordOptions {
    None,
    Fast { acceleration: i32 },
    Hc { level: i32 },
    Brotli { level: i32 },
}

/// Parse historical and modern settings once, without guessing from numeric values.
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum CompressionWire {
    Record {
        algo: CompressionAlgo,
        #[serde(flatten)]
        options: RecordOptions,
    },
    Legacy(LegacyCompression),
}

impl ShardCompression {
    /// Preserve legacy ratios as metadata while validating their historical contract.
    pub fn new(algo: CompressionAlgo, ratio: f64) -> CacheResult<Self> {
        if algo == CompressionAlgo::Brotli {
            return Err(CacheError::InvalidInput(
                "Brotli requires the typed whole-record policy".to_string(),
            ));
        }
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
        Ok(Self::Legacy(LegacyCompression { algo, ratio }))
    }

    /// Reject upstream clamping so requested acceleration is never silently changed.
    pub fn fast(acceleration: i32) -> CacheResult<Self> {
        if !(1..=65_537).contains(&acceleration) {
            return Err(CacheError::InvalidInput(
                "LZ4 acceleration must be between 1 and 65537".to_string(),
            ));
        }
        Ok(Self::Lz4Fast { acceleration })
    }

    /// Validate HC effort before it reaches the native codec.
    pub fn hc(level: i32) -> CacheResult<Self> {
        if !(1..=12).contains(&level) {
            return Err(CacheError::InvalidInput(
                "LZ4 HC level must be between 1 and 12".to_string(),
            ));
        }
        Ok(Self::Lz4HC { level })
    }

    /// Reject unsupported effort levels before codec allocation or manifest acceptance.
    pub fn brotli(level: i32) -> CacheResult<Self> {
        if !(0..=11).contains(&level) {
            return Err(CacheError::InvalidInput(
                "Brotli level must be between 0 and 11".to_string(),
            ));
        }
        Ok(Self::Brotli { level })
    }

    /// Only explicit modern variants change the on-disk record scope.
    pub fn whole_record(&self) -> bool {
        !matches!(self, Self::Legacy(_))
    }

    /// Uncompressed variants bypass codec allocation.
    pub fn compressed(&self) -> bool {
        !matches!(
            self,
            Self::None
                | Self::Legacy(LegacyCompression {
                    algo: CompressionAlgo::None,
                    ..
                })
        )
    }
}

impl Serialize for ShardCompression {
    /// Retain legacy fields and make modern codec/mode discriminators explicit.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            Self::Legacy(settings) => CompressionWire::Legacy(settings.clone()),
            Self::None => CompressionWire::Record {
                algo: CompressionAlgo::None,
                options: RecordOptions::None,
            },
            Self::Lz4Fast { acceleration } => CompressionWire::Record {
                algo: CompressionAlgo::Lz4,
                options: RecordOptions::Fast {
                    acceleration: *acceleration,
                },
            },
            Self::Brotli { level } => CompressionWire::Record {
                algo: CompressionAlgo::Brotli,
                options: RecordOptions::Brotli { level: *level },
            },
            Self::Lz4HC { level } => CompressionWire::Record {
                algo: CompressionAlgo::Lz4,
                options: RecordOptions::Hc { level: *level },
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ShardCompression {
    /// Reject invalid codec/mode combinations even for hand-edited manifests.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let result = match CompressionWire::deserialize(deserializer)? {
            CompressionWire::Legacy(settings) => Self::new(settings.algo, settings.ratio),
            CompressionWire::Record {
                algo: CompressionAlgo::None,
                options: RecordOptions::None,
            } => Ok(Self::None),
            CompressionWire::Record {
                algo: CompressionAlgo::Lz4,
                options: RecordOptions::Fast { acceleration },
            } => Self::fast(acceleration),
            CompressionWire::Record {
                algo: CompressionAlgo::Lz4,
                options: RecordOptions::Hc { level },
            } => Self::hc(level),
            CompressionWire::Record {
                algo: CompressionAlgo::Brotli,
                options: RecordOptions::Brotli { level },
            } => Self::brotli(level),
            _ => Err(CacheError::InvalidInput(
                "compression algorithm and mode disagree".to_string(),
            )),
        };
        result.map_err(D::Error::custom)
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

/// Retain dependency panic diagnostics; callers must discard mutated state on failure.
pub fn native_result<T>(context: &str, action: impl FnOnce() -> CacheResult<T>) -> CacheResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic payload");
        Err(CacheError::NativeFailure(format!("{context}: {message}")))
    })
}
