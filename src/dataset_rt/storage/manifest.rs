//! Versioned records retain legacy identities and distinguish whole-record compression.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::ShardManifest;
use crate::types::{CacheId, CacheResult, MetadataField};

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Storage fields shared by supported versions, including preserved application extensions.
pub struct ManifestFields {
    pub source_name: String,
    pub sample_count: u64,
    pub metadata_schema: Vec<MetadataField>,
    pub metadata_sha256: String,
    pub index_sha256: String,
    pub shards: Vec<ShardManifest>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Legacy wire shape has no persistent identity; the reader supplies its list position.
pub struct ManifestV2 {
    #[serde(flatten)]
    pub fields: ManifestFields,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Persistent-identity wire shape shared by v3 and v4 record layouts.
pub struct ManifestV3 {
    pub cache_id: u64,
    #[serde(flatten)]
    pub fields: ManifestFields,
}

#[derive(Clone, Debug)]
/// A validated version chooses its complete schema rather than optional feature fields.
pub enum Manifest {
    V2(ManifestV2),
    V3(ManifestV3),
    V4(ManifestV3),
}

impl Manifest {
    /// Storage shape and checksums are common without optional version-specific fields.
    pub fn fields(&self) -> &ManifestFields {
        match self {
            Self::V2(manifest) => &manifest.fields,
            Self::V3(manifest) | Self::V4(manifest) => &manifest.fields,
        }
    }

    /// Only v4 encloses metadata and payload in the same compression block.
    pub fn whole_record(&self) -> bool {
        matches!(self, Self::V4(_))
    }

    /// Resolve legacy positions only for v2; persisted identities are authoritative in v3.
    pub fn resolve_cache_id(&self, position: usize) -> CacheResult<CacheId> {
        match self {
            Self::V2(_) => CacheId::from_position(position),
            Self::V3(manifest) | Self::V4(manifest) => Ok(CacheId::from_u64(manifest.cache_id)),
        }
    }

    /// Upgrade legacy identity without downgrading a newer record layout.
    pub fn into_v3(self, cache_id: u64) -> Self {
        match self {
            Self::V2(mut manifest) => {
                // V2 readers ignored extensions; a reserved extension cannot duplicate the v3 field.
                manifest.fields.extra.remove("cache_id");
                Self::V3(ManifestV3 {
                    cache_id,
                    fields: manifest.fields,
                })
            }
            Self::V3(manifest) => Self::V3(manifest),
            Self::V4(manifest) => Self::V4(manifest),
        }
    }
}

#[derive(Serialize)]
struct Versioned<'a, T: Serialize> {
    format_version: u32,
    #[serde(flatten)]
    data: &'a T,
}

impl Serialize for Manifest {
    /// Keep the established numeric version tag and flat JSON manifest shape.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::V2(manifest) => Versioned {
                format_version: 2,
                data: manifest,
            }
            .serialize(serializer),
            Self::V4(manifest) => Versioned {
                format_version: super::FORMAT_VERSION,
                data: manifest,
            }
            .serialize(serializer),
            Self::V3(manifest) => Versioned {
                format_version: 3,
                data: manifest,
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Manifest {
    /// Dispatch at the untyped JSON boundary; unsupported versions never become domain records.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let mut fields = unique_fields(deserializer)?;
        let version = fields
            .remove("format_version")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| D::Error::custom("manifest requires integer format_version"))?;
        let body = Value::Object(fields.into_iter().collect());
        let manifest = match version {
            2 => serde_json::from_value(body)
                .map(Self::V2)
                .map_err(D::Error::custom),
            3 => serde_json::from_value(body)
                .map(Self::V3)
                .map_err(D::Error::custom),
            4 => serde_json::from_value(body)
                .map(Self::V4)
                .map_err(D::Error::custom),
            _ => Err(D::Error::custom(format!(
                "unsupported format version {version}"
            ))),
        }?;
        if manifest
            .fields()
            .shards
            .iter()
            .any(|shard| shard.compression.whole_record() != manifest.whole_record())
        {
            return Err(D::Error::custom(
                "compression scope disagrees with manifest format version",
            ));
        }
        Ok(manifest)
    }
}

/// Reject ambiguous duplicate JSON fields before selecting a version-specific record.
fn unique_fields<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, Value>, D::Error> {
    struct Fields;
    impl<'de> serde::de::Visitor<'de> for Fields {
        type Value = BTreeMap<String, Value>;
        /// Only a flat manifest object can select a supported version.
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a manifest object with unique field names")
        }
        /// Keep one untyped boundary map while rejecting duplicate identity/version declarations.
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            use serde::de::Error;
            let mut fields = BTreeMap::new();
            while let Some((name, value)) = map.next_entry::<String, Value>()? {
                if fields.insert(name.clone(), value).is_some() {
                    return Err(M::Error::custom(format!("duplicate manifest field {name}")));
                }
            }
            Ok(fields)
        }
    }
    deserializer.deserialize_map(Fields)
}
