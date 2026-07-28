use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};

use crate::types::{CacheError, CacheResult, CompressionAlgo, ShardCompression};

pub fn compress_payload(data: Vec<u8>, compression: &ShardCompression) -> Vec<u8> {
    match compression.algo {
        CompressionAlgo::None => data,
        CompressionAlgo::Lz4 => compress_prepend_size(&data),
    }
}

pub fn decompress_payload(data: &[u8], compression: &ShardCompression) -> CacheResult<Vec<u8>> {
    match compression.algo {
        CompressionAlgo::None => Ok(data.to_vec()),
        CompressionAlgo::Lz4 => decompress_lz4_payload(data),
    }
}

fn decompress_lz4_payload(data: &[u8]) -> CacheResult<Vec<u8>> {
    decompress_size_prepended(data).map_err(|error| {
        CacheError::InvalidCache(format!("failed to decompress lz4 payload: {error}"))
    })
}
