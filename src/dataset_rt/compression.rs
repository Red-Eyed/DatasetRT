use lz4_flex::block::{compress_into, decompress_into, get_maximum_output_size};

use crate::types::{native_result, CacheError, CacheResult, CompressionAlgo, ShardCompression};

/// Reject payload lengths that cannot round-trip through the LZ4 u32 size prefix.
pub fn compress_payload(data: Vec<u8>, compression: &ShardCompression) -> CacheResult<Vec<u8>> {
    match compression {
        ShardCompression::None
        | ShardCompression::Legacy(crate::types::LegacyCompression {
            algo: CompressionAlgo::None,
            ..
        }) => Ok(data),
        ShardCompression::Lz4Fast { acceleration } => {
            compress_record(data, lz4::block::CompressionMode::FAST(*acceleration))
        }
        ShardCompression::Lz4HC { level } => {
            compress_record(data, lz4::block::CompressionMode::HIGHCOMPRESSION(*level))
        }
        ShardCompression::Legacy(_) => {
            let len = u32::try_from(data.len()).map_err(|_| {
                CacheError::InvalidInput("LZ4 payload exceeds u32 size-prefix limit".to_string())
            })?;
            let capacity = get_maximum_output_size(data.len())
                .checked_add(4)
                .filter(|capacity| *capacity >= data.len())
                .ok_or_else(|| {
                    CacheError::InvalidInput("LZ4 output capacity overflowed usize".to_string())
                })?;
            let mut output = byte_buffer(capacity)?;
            let (prefix, body) = output.split_at_mut_checked(4).ok_or_else(|| {
                CacheError::InvalidInput("LZ4 output prefix is missing".to_string())
            })?;
            prefix.copy_from_slice(&len.to_le_bytes());
            let written = native_result("LZ4 compression", || {
                compress_into(&data, body).map_err(|error| {
                    CacheError::InvalidInput(format!("failed to compress LZ4 payload: {error}"))
                })
            })?;
            output.truncate(written + 4);
            Ok(output)
        }
    }
}

/// Encode a complete record with a validated upstream mode into a caller-owned buffer.
fn compress_record(data: Vec<u8>, mode: lz4::block::CompressionMode) -> CacheResult<Vec<u8>> {
    let capacity = lz4::block::compress_bound(data.len())
        .map_err(|error| CacheError::InvalidInput(format!("LZ4 record input rejected: {error}")))?
        .checked_add(4)
        .ok_or_else(|| CacheError::InvalidInput("LZ4 record capacity overflow".to_string()))?;
    let mut output = byte_buffer(capacity)?;
    let written = native_result("LZ4 record compression", || {
        lz4::block::compress_to_buffer(&data, Some(mode), true, &mut output).map_err(|error| {
            CacheError::InvalidInput(format!("failed to compress LZ4 record: {error}"))
        })
    })?;
    output.truncate(written);
    Ok(output)
}

/// Allocate through a fallible reservation before initializing payload bytes.
fn byte_buffer(len: usize) -> CacheResult<Vec<u8>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|error| {
        CacheError::NativeFailure(format!(
            "cannot allocate payload buffer of {len} bytes: {error}"
        ))
    })?;
    output.resize(len, 0);
    Ok(output)
}

/// Validate compression envelopes before decoding into a fallibly allocated buffer.
pub fn decompress_payload(
    data: &[u8],
    compression: &ShardCompression,
    max_uncompressed_len: u64,
) -> CacheResult<Vec<u8>> {
    match compression.compressed() {
        false => {
            let mut output = Vec::new();
            output.try_reserve_exact(data.len()).map_err(|error| {
                CacheError::NativeFailure(format!("cannot allocate payload buffer: {error}"))
            })?;
            output.extend_from_slice(data);
            Ok(output)
        }
        true => {
            let (prefix, body) = data.split_at_checked(4).ok_or_else(|| {
                CacheError::InvalidCache("LZ4 payload is missing its size prefix".to_string())
            })?;
            let prefix = prefix
                .try_into()
                .map_err(|_| CacheError::InvalidCache("invalid LZ4 size prefix".to_string()))?;
            let len = u32::from_le_bytes(prefix) as usize;
            if len as u64 > max_uncompressed_len {
                return Err(CacheError::InvalidCache(
                    "LZ4 size prefix exceeds shard uncompressed length".to_string(),
                ));
            }
            let mut output = byte_buffer(len)?;
            let written = native_result("LZ4 decompression", || {
                decompress_into(body, &mut output).map_err(|error| {
                    CacheError::InvalidCache(format!("failed to decompress lz4 payload: {error}"))
                })
            })?;
            if written != len {
                return Err(CacheError::InvalidCache(
                    "LZ4 decoded length does not match size prefix".to_string(),
                ));
            }
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caller-owned buffers preserve valid LZ4 records and reject false size prefixes.
    #[test]
    fn lz4_round_trip_and_invalid_envelopes() -> CacheResult<()> {
        let compression = ShardCompression::new(CompressionAlgo::Lz4, 1.0)?;
        for input in [Vec::new(), vec![42; 4096], (0..=255).collect()] {
            let stored = compress_payload(input.clone(), &compression)?;
            assert_eq!(
                decompress_payload(&stored, &compression, input.len() as u64)?,
                input
            );
            if !input.is_empty() {
                assert!(decompress_payload(&stored, &compression, 0).is_err());
            }
            let mut invalid = stored;
            let prefix = invalid.get_mut(..4).ok_or(CacheError::WorkerFailed)?;
            prefix.copy_from_slice(&8192_u32.to_le_bytes());
            assert!(decompress_payload(&invalid, &compression, 8192).is_err());
        }
        assert!(decompress_payload(&[0, 1], &compression, 0).is_err());
        assert!(byte_buffer(usize::MAX).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod record_mode_tests {
    use super::*;

    /// Independent mode blocks remain decodable after typed manifest round-trips.
    #[test]
    fn record_modes_round_trip_through_wire_settings() -> CacheResult<()> {
        let data = b"{\"image_path\":\"/repeated/path\"}serialized-token-data".repeat(256);
        for mode in [
            ShardCompression::None,
            ShardCompression::fast(1)?,
            ShardCompression::fast(65_537)?,
            ShardCompression::hc(1)?,
            ShardCompression::hc(12)?,
        ] {
            let wire = serde_json::to_vec(&mode)?;
            let decoded_mode: ShardCompression = serde_json::from_slice(&wire)?;
            let compressed = compress_payload(data.clone(), &decoded_mode)?;
            assert_eq!(
                decompress_payload(&compressed, &decoded_mode, data.len() as u64)?,
                data
            );
        }
        assert!(ShardCompression::fast(0).is_err());
        assert!(ShardCompression::fast(65_538).is_err());
        assert!(ShardCompression::hc(0).is_err());
        assert!(ShardCompression::hc(13).is_err());
        Ok(())
    }
}
