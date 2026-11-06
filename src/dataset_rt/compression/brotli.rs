use std::io::Cursor;

use ::brotli::enc::{encode::BrotliEncoderMaxCompressedSize, BrotliEncoderParams, StandardAlloc};
use ::brotli::{BrotliDecompressStream, BrotliResult, BrotliState};

use super::byte_buffer;
use crate::types::{native_result, CacheError, CacheResult};

/// Encode one record into a bounded buffer, prefixed by its exact decoded size.
pub(super) fn compress(data: Vec<u8>, level: i32) -> CacheResult<Vec<u8>> {
    let len = u32::try_from(data.len()).map_err(|_| {
        CacheError::InvalidInput("Brotli record exceeds u32 size-prefix limit".to_string())
    })?;
    let capacity = BrotliEncoderMaxCompressedSize(data.len())
        .checked_add(4)
        .filter(|capacity| *capacity > data.len())
        .ok_or_else(|| CacheError::InvalidInput("Brotli output capacity overflow".to_string()))?;
    let mut output = byte_buffer(capacity)?;
    let (prefix, body) = output
        .split_at_mut_checked(4)
        .ok_or_else(|| CacheError::InvalidInput("Brotli output prefix is missing".to_string()))?;
    prefix.copy_from_slice(&len.to_le_bytes());
    let params = BrotliEncoderParams {
        quality: level,
        lgwin: 20,
        ..Default::default()
    };
    let mut writer = Cursor::new(body);
    native_result("Brotli compression", || {
        ::brotli::BrotliCompress(&mut data.as_slice(), &mut writer, &params).map_err(|error| {
            CacheError::InvalidInput(format!("failed to compress Brotli record: {error}"))
        })
    })?;
    let written = usize::try_from(writer.position()).map_err(|_| {
        CacheError::InvalidInput("Brotli encoded size does not fit usize".to_string())
    })?;
    output.truncate(written + 4);
    Ok(output)
}

/// Bound output before decoding and reject false decoded lengths without growing a Vec.
pub(super) fn decompress(data: &[u8], max_uncompressed_len: u64) -> CacheResult<Vec<u8>> {
    let (prefix, body) = data.split_at_checked(4).ok_or_else(|| {
        CacheError::InvalidCache("Brotli record is missing its size prefix".to_string())
    })?;
    let prefix = prefix
        .try_into()
        .map_err(|_| CacheError::InvalidCache("invalid Brotli size prefix".to_string()))?;
    let len = u32::from_le_bytes(prefix) as usize;
    if len as u64 > max_uncompressed_len {
        return Err(CacheError::InvalidCache(
            "Brotli size prefix exceeds shard uncompressed length".to_string(),
        ));
    }
    let mut output = byte_buffer(len)?;
    native_result("Brotli decompression", || decode_into(body, &mut output))?;
    Ok(output)
}

/// Accept one standard-window stream, consuming its entire input and exact declared output.
fn decode_into(body: &[u8], output: &mut [u8]) -> CacheResult<()> {
    let mut state = BrotliState::new_strict(
        StandardAlloc::default(),
        StandardAlloc::default(),
        StandardAlloc::default(),
    );
    let mut available_in = body.len();
    let mut input_offset = 0;
    let mut available_out = output.len();
    let mut output_offset = 0;
    let mut total_out = 0;
    let result = BrotliDecompressStream(
        &mut available_in,
        &mut input_offset,
        body,
        &mut available_out,
        &mut output_offset,
        output,
        &mut total_out,
        &mut state,
    );
    match result {
        BrotliResult::ResultSuccess if available_in == 0 && output_offset == output.len() => Ok(()),
        BrotliResult::ResultSuccess => Err(CacheError::InvalidCache(
            "Brotli record has trailing bytes or decoded length mismatch".to_string(),
        )),
        BrotliResult::NeedsMoreInput => Err(CacheError::InvalidCache(
            "truncated Brotli record".to_string(),
        )),
        BrotliResult::NeedsMoreOutput => Err(CacheError::InvalidCache(
            "Brotli decoded length exceeds size prefix".to_string(),
        )),
        BrotliResult::ResultFailure => Err(CacheError::InvalidCache(
            "invalid Brotli record".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty, Unicode, binary, and repeated records round-trip at supported effort extremes.
    #[test]
    fn records_round_trip_and_reject_corruption() -> CacheResult<()> {
        for level in [0, 5, 11] {
            for input in [
                Vec::new(),
                "Український текст. Deutsche Wörter. Plain text."
                    .as_bytes()
                    .repeat(12),
                (0..=255).collect(),
            ] {
                let stored = compress(input.clone(), level)?;
                assert_eq!(decompress(&stored, input.len() as u64)?, input);
                assert!(decompress(
                    stored
                        .get(..stored.len() - 1)
                        .ok_or(CacheError::WorkerFailed)?,
                    input.len() as u64
                )
                .is_err());
                let mut trailing = stored.clone();
                trailing.extend_from_slice(&[0, 1]);
                assert!(decompress(&trailing, input.len() as u64).is_err());
                let mut wrong_size = stored;
                let prefix = wrong_size.get_mut(..4).ok_or(CacheError::WorkerFailed)?;
                prefix.copy_from_slice(&(input.len() as u32 + 1).to_le_bytes());
                assert!(decompress(&wrong_size, input.len() as u64 + 1).is_err());
            }
        }
        assert!(decompress(&[0, 1, 2], 0).is_err());
        Ok(())
    }
}
