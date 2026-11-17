//! Independent checksummed frames keep random access and decoded allocation bounded.

use ::zstd::{bulk, zstd_safe};

use super::byte_buffer;
use crate::types::{native_result, CacheError, CacheResult};

/// Encode one complete record with an explicit content size and no codec-owned threads.
pub(super) fn compress(data: Vec<u8>, level: i32) -> CacheResult<Vec<u8>> {
    let capacity = zstd_safe::compress_bound(data.len());
    if capacity == 0 || capacity < data.len() {
        return Err(CacheError::InvalidInput(
            "Zstd record exceeds compression capacity limit".to_string(),
        ));
    }
    let mut output = byte_buffer(capacity)?;
    let written = native_result("Zstd compression", || {
        let mut encoder = bulk::Compressor::new(level)?;
        encoder.set_parameter(zstd_safe::CParameter::ContentSizeFlag(true))?;
        encoder.set_parameter(zstd_safe::CParameter::ChecksumFlag(true))?;
        Ok(encoder.compress_to_buffer(&data, output.as_mut_slice())?)
    })?;
    output.truncate(written);
    Ok(output)
}

/// Validate a single sized frame before allocating its exact, caller-bounded output.
pub(super) fn decompress(data: &[u8], max_uncompressed_len: u64) -> CacheResult<Vec<u8>> {
    let len = decoded_len(data, max_uncompressed_len)?;
    let mut output = byte_buffer(len)?;
    let written = native_result("Zstd decompression", || {
        bulk::decompress_to_buffer(data, &mut output).map_err(|error| {
            CacheError::InvalidCache(format!("failed to decompress Zstd record: {error}"))
        })
    })?;
    if written != len {
        return Err(CacheError::InvalidCache(
            "Zstd decoded length does not match frame content size".to_string(),
        ));
    }
    Ok(output)
}

/// Reject unknown sizes, skippable/concatenated frames, and trailing or truncated bytes.
fn decoded_len(data: &[u8], max_uncompressed_len: u64) -> CacheResult<usize> {
    if !data.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        return Err(CacheError::InvalidCache(
            "Zstd record is missing its frame magic".to_string(),
        ));
    }
    let len = zstd_safe::get_frame_content_size(data)
        .map_err(|error| CacheError::InvalidCache(format!("invalid Zstd frame header: {error}")))?
        .ok_or_else(|| {
            CacheError::InvalidCache("Zstd record is missing its frame content size".to_string())
        })?;
    if len > max_uncompressed_len {
        return Err(CacheError::InvalidCache(
            "Zstd frame content size exceeds shard uncompressed length".to_string(),
        ));
    }
    let frame_len = zstd_safe::find_frame_compressed_size(data).map_err(|error| {
        CacheError::InvalidCache(format!("invalid or truncated Zstd record: {error}"))
    })?;
    if frame_len != data.len() {
        return Err(CacheError::InvalidCache(
            "Zstd record has trailing bytes or multiple frames".to_string(),
        ));
    }
    usize::try_from(len)
        .map_err(|_| CacheError::InvalidCache("Zstd content size does not fit usize".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard frames restore empty, text, binary, and repetitive records at effort extremes.
    #[test]
    fn records_round_trip_and_reject_corruption() -> CacheResult<()> {
        for level in [1, 3, 22] {
            for input in [
                Vec::new(),
                "Український текст. Deutsche Wörter. Plain text."
                    .as_bytes()
                    .to_vec(),
                (0..=255).collect(),
                vec![42; 4096],
            ] {
                let stored = compress(input.clone(), level)?;
                assert_eq!(decompress(&stored, input.len() as u64)?, input);
                for end in 0..stored.len() {
                    assert!(decompress(
                        stored.get(..end).ok_or(CacheError::WorkerFailed)?,
                        input.len() as u64
                    )
                    .is_err());
                }
                let mut trailing = stored.clone();
                trailing.extend_from_slice(&[0, 1]);
                assert!(decompress(&trailing, input.len() as u64).is_err());
                let mut multiple = stored.clone();
                multiple.extend_from_slice(&stored);
                assert!(decompress(&multiple, input.len() as u64 * 2).is_err());
                let mut corrupted = stored;
                *corrupted.last_mut().ok_or(CacheError::WorkerFailed)? ^= 1;
                assert!(decompress(&corrupted, input.len() as u64).is_err());
            }
        }
        Ok(())
    }

    /// Frame metadata cannot trigger unbounded allocation or bypass exact decoded lengths.
    #[test]
    fn declared_sizes_are_validated_before_decode() -> CacheResult<()> {
        let input = vec![42; 40];
        let stored = compress(input.clone(), 3)?;
        assert!(decompress(&stored, 39).is_err());
        let mut false_size = stored;
        let size_byte = false_size.get_mut(5).ok_or(CacheError::WorkerFailed)?;
        assert_eq!(*size_byte, 40);
        *size_byte = 41;
        assert!(decompress(&false_size, 41).is_err());
        *false_size.get_mut(5).ok_or(CacheError::WorkerFailed)? = 39;
        assert!(decompress(&false_size, 40).is_err());

        let mut encoder = bulk::Compressor::new(3)?;
        encoder.set_parameter(zstd_safe::CParameter::ContentSizeFlag(false))?;
        let unknown_size = encoder.compress(&input)?;
        assert!(decompress(&unknown_size, 40).is_err());
        assert!(decompress(&[0x50, 0x2a, 0x4d, 0x18, 0, 0, 0, 0], 0).is_err());
        Ok(())
    }
}
