use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::{read_index_entry, IndexEntry, INDEX_ROW_BYTES};
use crate::types::{CacheError, CacheResult};

/// Own an immutable file mapping and decode individual index rows without a heap table.
#[derive(Debug)]
pub struct MappedIndex {
    mapping: Option<Mmap>,
}

impl MappedIndex {
    /// Validate fixed-width shape before exposing the published cache's read-only index.
    pub fn open(path: &Path) -> CacheResult<Self> {
        let file = File::open(path)?;
        let byte_len = file.metadata()?.len();
        if byte_len % INDEX_ROW_BYTES as u64 != 0 {
            return Err(CacheError::InvalidCache(
                "index length is not divisible by row size".to_string(),
            ));
        }
        let mapping = match byte_len {
            0 => None,
            // Published cache files remain immutable for the mapping's lifetime.
            // The mapping owns its pages independently of this file handle.
            _ => Some(unsafe { Mmap::map(&file)? }),
        };
        Ok(Self { mapping })
    }

    /// Return the physical row count without decoding any entries.
    pub fn len(&self) -> usize {
        self.bytes().len() / INDEX_ROW_BYTES
    }

    /// Decode one little-endian row after checking its position and byte range.
    pub fn get(&self, position: usize) -> CacheResult<IndexEntry> {
        let row = position
            .checked_mul(INDEX_ROW_BYTES)
            .and_then(|start| self.bytes().get(start..))
            .and_then(|remaining| remaining.get(..INDEX_ROW_BYTES))
            .ok_or_else(|| {
                CacheError::InvalidCache(format!("sample index {position} is out of range"))
            })?;
        read_index_entry(row)
    }

    /// Scan entries in physical order while retaining only the current decoded row.
    pub fn iter(&self) -> impl Iterator<Item = CacheResult<IndexEntry>> + '_ {
        self.bytes()
            .as_chunks::<INDEX_ROW_BYTES>()
            .0
            .iter()
            .map(|row| read_index_entry(row))
    }

    /// Treat a valid zero-row index as an empty byte view without mapping an empty file.
    fn bytes(&self) -> &[u8] {
        self.mapping.as_deref().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise checked addressing on a real immutable legacy-format mapping.
    #[test]
    fn mapped_rows_survive_closed_file_and_reject_overflow() -> CacheResult<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/manifest-v2/legacy-plain/index.bin");
        let index = MappedIndex::open(&path)?;
        assert!(index.len() > 0);
        let first = index.get(0)?;
        let scanned = index.iter().next().ok_or_else(|| {
            CacheError::InvalidCache("fixture index must contain a row".to_string())
        })??;
        assert_eq!(first.shard_id, scanned.shard_id);
        assert_eq!(first.offset, scanned.offset);
        assert_eq!(first.byte_len, scanned.byte_len);
        assert_eq!(index.iter().count(), index.len());
        assert!(matches!(
            index.get(index.len()),
            Err(CacheError::InvalidCache(_))
        ));
        assert!(matches!(
            index.get(usize::MAX),
            Err(CacheError::InvalidCache(_))
        ));
        Ok(())
    }

    /// Preserve every unsigned field bit without native-layout casts or alignment assumptions.
    #[test]
    fn little_endian_decoder_preserves_full_width() -> CacheResult<()> {
        let mut bytes = Vec::new();
        for field in [u64::MAX, 0x0123_4567_89ab_cdef, u64::MAX - 1] {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        let entry = read_index_entry(&bytes)?;
        assert_eq!(entry.shard_id, u64::MAX);
        assert_eq!(entry.offset, 0x0123_4567_89ab_cdef);
        assert_eq!(entry.byte_len, u64::MAX - 1);
        assert!(read_index_entry(&[]).is_err());
        Ok(())
    }

    /// A valid empty index needs no mapping and exposes no addressable rows.
    #[test]
    fn empty_index_has_no_rows() {
        let index = MappedIndex { mapping: None };
        assert_eq!(index.len(), 0);
        assert_eq!(index.iter().count(), 0);
        assert!(index.get(0).is_err());
    }
}
