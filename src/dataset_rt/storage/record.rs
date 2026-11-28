//! Closed record layouts own envelope framing; codec algorithms remain shared.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::{read_u64, record_range, ShardManifest};
use crate::compression::{compress_payload, decompress_payload};
use crate::types::{
    CacheError, CacheResult, CacheSample, MetadataField, MetadataValue, ShardCompression,
};

const RECORD_METADATA_LEN_BYTES: usize = 8;

/// Manifest versions resolve once into their physical byte layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RecordLayout {
    /// V2/v3 compress only the payload following the uncompressed JSON envelope.
    LegacyEnvelope,
    /// V4 applies its codec to metadata and payload together.
    RecordEnvelope,
    /// V5 keeps metadata exclusively in Arrow and applies its codec only to payload bytes.
    PayloadOnly,
}

/// Decoding preserves metadata provenance; orchestration never guesses from an empty vector.
pub(super) enum DecodedRecord {
    Embedded(CacheSample),
    ArrowPayload(Vec<u8>),
}

/// One bounded encoded record retains profiler counters for the actual codec input.
pub(super) struct EncodedRecord {
    pub(super) bytes: Vec<u8>,
    pub(super) uncompressed_byte_len: u64,
    pub(super) compression: Duration,
    pub(super) record_encode: Duration,
}

impl RecordLayout {
    /// Legacy policies retain v3 output; explicit modern policies select payload-only writes.
    pub(super) fn for_write(compression: &ShardCompression) -> Self {
        match compression {
            ShardCompression::Legacy(_) => Self::LegacyEnvelope,
            ShardCompression::None
            | ShardCompression::Lz4Fast { .. }
            | ShardCompression::Lz4HC { .. }
            | ShardCompression::Brotli { .. }
            | ShardCompression::Zstd { .. } => Self::PayloadOnly,
        }
    }

    /// Reject wire-policy/layout conflicts at the manifest boundary, before reading samples.
    pub(super) fn validate_compression(self, compression: &ShardCompression) -> CacheResult<()> {
        match (self, compression) {
            (Self::LegacyEnvelope, ShardCompression::Legacy(_)) => Ok(()),
            (
                Self::RecordEnvelope | Self::PayloadOnly,
                ShardCompression::None
                | ShardCompression::Lz4Fast { .. }
                | ShardCompression::Lz4HC { .. }
                | ShardCompression::Brotli { .. }
                | ShardCompression::Zstd { .. },
            ) => Ok(()),
            (Self::LegacyEnvelope, _)
            | (Self::RecordEnvelope | Self::PayloadOnly, ShardCompression::Legacy(_)) => {
                Err(CacheError::InvalidCache(
                    "compression scope disagrees with manifest format version".to_string(),
                ))
            }
        }
    }

    /// Apply the selected framing once, preserving per-layout size-accounting semantics.
    pub(super) fn encode(
        self,
        data: Vec<u8>,
        schema: &[MetadataField],
        metadata: &[MetadataValue],
        compression: &ShardCompression,
    ) -> CacheResult<EncodedRecord> {
        match self {
            Self::PayloadOnly => encode_unit(data, compression, Duration::ZERO),
            Self::RecordEnvelope => {
                let started = Instant::now();
                let record = encode_sample_record(&data, schema, metadata)?;
                drop(data);
                encode_unit(record, compression, started.elapsed())
            }
            Self::LegacyEnvelope => {
                let uncompressed_byte_len = input_len(&data)?;
                let started = Instant::now();
                let payload = compress_payload(data, compression)?;
                let compression = started.elapsed();
                let started = Instant::now();
                let bytes = encode_sample_record(&payload, schema, metadata)?;
                Ok(EncodedRecord {
                    bytes,
                    uncompressed_byte_len,
                    compression,
                    record_encode: started.elapsed(),
                })
            }
        }
    }

    /// Decode one addressed unit and make its metadata source explicit in the result type.
    pub(super) fn decode(
        self,
        record: Vec<u8>,
        schema: &[MetadataField],
        shard: &ShardManifest,
    ) -> CacheResult<DecodedRecord> {
        match self {
            Self::PayloadOnly => Ok(DecodedRecord::ArrowPayload(decode_unit(record, shard)?)),
            Self::RecordEnvelope => {
                let mut decoded = decode_unit(record, shard)?;
                let (metadata, payload) = split_sample_record(&decoded, schema)?;
                let envelope_len = decoded.len() - payload.len();
                decoded.drain(..envelope_len);
                Ok(DecodedRecord::Embedded(CacheSample {
                    data: decoded,
                    metadata,
                }))
            }
            Self::LegacyEnvelope => {
                let (metadata, payload) = split_sample_record(&record, schema)?;
                let data =
                    decompress_payload(payload, &shard.compression, shard.uncompressed_byte_len)?;
                Ok(DecodedRecord::Embedded(CacheSample { data, metadata }))
            }
        }
    }
}

/// Count the codec input fallibly before moving it into compression.
fn input_len(data: &[u8]) -> CacheResult<u64> {
    u64::try_from(data.len())
        .map_err(|_| CacheError::InvalidInput("record length does not fit u64".to_string()))
}

/// Share modern codec framing while keeping metadata-encoding time separate from compression.
fn encode_unit(
    data: Vec<u8>,
    compression: &ShardCompression,
    record_encode: Duration,
) -> CacheResult<EncodedRecord> {
    let uncompressed_byte_len = input_len(&data)?;
    let started = Instant::now();
    let bytes = compress_payload(data, compression)?;
    Ok(EncodedRecord {
        bytes,
        uncompressed_byte_len,
        compression: started.elapsed(),
        record_encode,
    })
}

/// Uncompressed indexed units transfer ownership without another payload allocation.
fn decode_unit(record: Vec<u8>, shard: &ShardManifest) -> CacheResult<Vec<u8>> {
    if shard.compression.compressed() {
        decompress_payload(&record, &shard.compression, shard.uncompressed_byte_len)
    } else {
        Ok(record)
    }
}

/// Encode the historical schema-ordered JSON envelope without changing its wire bytes.
fn encode_sample_record(
    stored_payload: &[u8],
    schema: &[MetadataField],
    metadata_row: &[MetadataValue],
) -> CacheResult<Vec<u8>> {
    let metadata_object = metadata_row_to_object(schema, metadata_row)?;
    let metadata_json = serde_json::to_vec(&metadata_object)?;
    let metadata_len = u64::try_from(metadata_json.len()).map_err(|_| {
        CacheError::InvalidInput("metadata record length does not fit in u64".to_string())
    })?;
    let capacity = RECORD_METADATA_LEN_BYTES
        .checked_add(metadata_json.len())
        .and_then(|len| len.checked_add(stored_payload.len()))
        .ok_or_else(|| {
            CacheError::InvalidInput("record byte length overflowed usize".to_string())
        })?;
    let mut record = Vec::new();
    record.try_reserve_exact(capacity).map_err(|error| {
        CacheError::InvalidInput(format!("cannot allocate sample record: {error}"))
    })?;
    record.extend_from_slice(&metadata_len.to_le_bytes());
    record.extend_from_slice(&metadata_json);
    record.extend_from_slice(stored_payload);
    Ok(record)
}

/// Validate the metadata prefix before exposing the addressed payload slice.
fn split_sample_record<'a>(
    record: &'a [u8],
    schema: &[MetadataField],
) -> CacheResult<(Vec<MetadataValue>, &'a [u8])> {
    let metadata_len = read_u64(record_range(record, 0, RECORD_METADATA_LEN_BYTES)?)?;
    let metadata_len = usize::try_from(metadata_len).map_err(|_| {
        CacheError::InvalidCache("metadata record length does not fit in usize".to_string())
    })?;
    let payload_start = RECORD_METADATA_LEN_BYTES
        .checked_add(metadata_len)
        .ok_or_else(|| CacheError::InvalidCache("metadata record length overflowed".to_string()))?;
    let metadata_json = record_range(record, RECORD_METADATA_LEN_BYTES, payload_start)?;
    let metadata_object = serde_json::from_slice(metadata_json)?;
    let metadata = metadata_object_to_row(schema, metadata_object)?;
    let stored_payload = record.get(payload_start..).ok_or_else(|| {
        CacheError::InvalidCache("payload record is missing stored payload".to_string())
    })?;
    Ok((metadata, stored_payload))
}

/// Preserve historical key ordering and reject row/schema length drift.
fn metadata_row_to_object(
    schema: &[MetadataField],
    row: &[MetadataValue],
) -> CacheResult<BTreeMap<String, MetadataValue>> {
    if row.len() != schema.len() {
        return Err(CacheError::InvalidInput(
            "metadata row length does not match schema".to_string(),
        ));
    }

    schema
        .iter()
        .zip(row.iter())
        .map(|(field, value)| Ok((field.name.clone(), value.clone())))
        .collect()
}

/// Restore manifest column order while rejecting missing or mistyped embedded fields.
fn metadata_object_to_row(
    schema: &[MetadataField],
    metadata: BTreeMap<String, MetadataValue>,
) -> CacheResult<Vec<MetadataValue>> {
    if metadata.len() != schema.len() {
        return Err(CacheError::InvalidCache(
            "embedded metadata keys do not match schema".to_string(),
        ));
    }

    schema
        .iter()
        .map(|field| match metadata.get(&field.name) {
            Some(value) if value.kind() == field.kind => Ok(value.clone()),
            Some(value) => Err(CacheError::InvalidCache(format!(
                "embedded metadata field '{}' expected {}, got {}",
                field.name,
                field.kind,
                value.kind()
            ))),
            None => Err(CacheError::InvalidCache(format!(
                "embedded metadata field '{}' is missing",
                field.name
            ))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompressionAlgo, MetadataKind};

    /// Envelope parsing rejects truncation, schema drift, and invalid declared lengths.
    #[test]
    fn malformed_envelopes_fail_before_payload_decode() -> CacheResult<()> {
        let schema = vec![MetadataField {
            name: "value".to_string(),
            kind: MetadataKind::Int,
        }];
        assert!(metadata_row_to_object(&schema, &[]).is_err());
        for metadata in [
            BTreeMap::new(),
            BTreeMap::from([("other".to_string(), MetadataValue::Int(1))]),
            BTreeMap::from([("value".to_string(), MetadataValue::Bool(true))]),
        ] {
            assert!(metadata_object_to_row(&schema, metadata).is_err());
        }
        for bytes in [vec![], vec![0; 7], u64::MAX.to_le_bytes().to_vec()] {
            assert!(split_sample_record(&bytes, &schema).is_err());
        }
        let mut invalid_json = 1_u64.to_le_bytes().to_vec();
        invalid_json.push(b'{');
        assert!(split_sample_record(&invalid_json, &schema).is_err());
        let valid = encode_sample_record(b"payload", &schema, &[MetadataValue::Int(8)])?;
        let (metadata, payload) = split_sample_record(&valid, &schema)?;
        assert_eq!(metadata, vec![MetadataValue::Int(8)]);
        assert_eq!(payload, b"payload");
        Ok(())
    }

    /// Each layout preserves bytes and metadata provenance across its supported codecs.
    #[test]
    fn layouts_preserve_provenance_and_accounting() -> CacheResult<()> {
        let schema = vec![MetadataField {
            name: "index".to_string(),
            kind: MetadataKind::Int,
        }];
        let metadata = vec![MetadataValue::Int(7)];
        let modern = vec![
            ShardCompression::None,
            ShardCompression::fast(1)?,
            ShardCompression::hc(9)?,
            ShardCompression::brotli(5)?,
            ShardCompression::zstd(3)?,
        ];
        let cases = [
            (
                RecordLayout::LegacyEnvelope,
                vec![
                    ShardCompression::new(CompressionAlgo::None, 1.0)?,
                    ShardCompression::new(CompressionAlgo::Lz4, 1.0)?,
                ],
            ),
            (RecordLayout::RecordEnvelope, modern.clone()),
            (RecordLayout::PayloadOnly, modern),
        ];
        for (layout, codecs) in cases {
            for compression in codecs {
                layout.validate_compression(&compression)?;
                for data in [Vec::new(), b"a bounded payload".repeat(128)] {
                    let encoded = layout.encode(data.clone(), &schema, &metadata, &compression)?;
                    match layout {
                        RecordLayout::RecordEnvelope => {
                            assert!(encoded.uncompressed_byte_len > data.len() as u64)
                        }
                        RecordLayout::LegacyEnvelope | RecordLayout::PayloadOnly => {
                            assert_eq!(encoded.uncompressed_byte_len, data.len() as u64)
                        }
                    }
                    let shard = ShardManifest {
                        name: "fixture".to_string(),
                        uncompressed_byte_len: encoded.uncompressed_byte_len,
                        byte_len: encoded.bytes.len() as u64,
                        compression: compression.clone(),
                        sha256: String::new(),
                    };
                    match layout.decode(encoded.bytes, &schema, &shard)? {
                        DecodedRecord::Embedded(sample) => {
                            assert_ne!(layout, RecordLayout::PayloadOnly);
                            assert_eq!(sample.data, data);
                            assert_eq!(sample.metadata, metadata);
                        }
                        DecodedRecord::ArrowPayload(payload) => {
                            assert_eq!(layout, RecordLayout::PayloadOnly);
                            assert_eq!(payload, data);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Existing 0.5.9 bytes constrain the extracted v4 encoder independently of its decoder.
    #[test]
    fn v4_framing_matches_original_writer_fixtures() -> CacheResult<()> {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/format-v4");
        for name in ["plain", "lz4", "zstd"] {
            let cache = super::super::load_cache(root.join(name), true, 0)?;
            let entry = cache.index.get(0)?;
            let shard = cache
                .manifest
                .fields()
                .shards
                .first()
                .ok_or(CacheError::WorkerFailed)?;
            let encoded = RecordLayout::RecordEnvelope.encode(
                b"alpha".to_vec(),
                &cache.manifest.fields().metadata_schema,
                &[
                    MetadataValue::Int(0),
                    MetadataValue::String("train".to_string()),
                ],
                &shard.compression,
            )?;
            let stored = std::fs::read(cache.path.join("shards").join(&shard.name))?;
            let end = usize::try_from(entry.byte_len).map_err(|_| CacheError::WorkerFailed)?;
            assert_eq!(encoded.bytes, record_range(&stored, 0, end)?);
        }
        Ok(())
    }

    /// Modern and legacy policy discriminators cannot be combined with the opposite layout.
    #[test]
    fn incompatible_layout_policies_are_rejected() -> CacheResult<()> {
        let legacy = ShardCompression::new(CompressionAlgo::None, 1.0)?;
        assert!(RecordLayout::LegacyEnvelope
            .validate_compression(&ShardCompression::None)
            .is_err());
        for layout in [RecordLayout::RecordEnvelope, RecordLayout::PayloadOnly] {
            assert!(layout.validate_compression(&legacy).is_err());
        }
        Ok(())
    }
}
